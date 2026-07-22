#!/usr/bin/env python3
"""Plot one RMPFlow proof-run CSV as a multi-panel PNG.

The column names in this script are the schema-v1 names written by
``scripts/rmpflow_trace_logger.py``.  In particular, the dual-solve message
already contains the Escape-ON and Escape-OFF results evaluated at the same
robot state.  This script never substitutes unrelated trace columns when those
measurements are absent.

This is an experimental certificate report, not a hard collision-safety
certificate.  ``stability_conditional_nonincrease`` is valid only under the
runtime assumptions represented by the other certificate flags.
"""

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


TIME_FIELD = "time_ros_s"

CERTIFICATE_FIELDS = [
    "stability_schema_version",
    "stability_base_gds_structural",
    "stability_environment_static",
    "stability_external_rmp_empty",
    "stability_guard_enabled",
    "stability_conditional_nonincrease",
    "stability_escape_scale",
    "stability_tank_energy",
    "stability_tank_capacity",
    "stability_escape_power_requested",
    "stability_escape_power_applied",
    "stability_solve_power",
    "stability_clamp_power",
    "stability_numerical_power",
    "stability_positive_escape_energy",
    "stability_negative_escape_energy",
    "stability_net_escape_energy",
    "stability_sample_count",
    "stability_escape_metric_trace_requested",
    "stability_escape_metric_trace_applied",
    "stability_escape_force_norm_requested",
    "stability_escape_force_norm_applied",
    "stability_raw_qdd_norm",
    "stability_command_qdd_norm",
    "stability_clamp_active",
    "stability_root_solve_offset",
    "stability_escape_active",
    "stability_tank_identity_residual",
    "stability_energy_bound_violation",
    "stability_nonincrease_upper_bound",
    "stability_initial_energy",
]

DUAL_QDD_PREFIXES = [
    "tangent_escape_qdd_with",
    "tangent_escape_qdd_without",
    "tangent_escape_delta_qdd",
]
DUAL_TASK_PREFIXES = [
    "tangent_escape_tcp_accel_with",
    "tangent_escape_tcp_accel_without",
    "tangent_escape_delta_tcp_accel",
    "tangent_escape_cp_accel_with",
    "tangent_escape_cp_accel_without",
    "tangent_escape_delta_cp_accel",
]
DUAL_VECTOR_FIELDS = [
    *(f"{prefix}_{index}_rad_s2" for prefix in DUAL_QDD_PREFIXES for index in range(1, 7)),
    *(f"{prefix}_{axis}_m_s2" for prefix in DUAL_TASK_PREFIXES for axis in ("x", "y", "z")),
]
DUAL_NORM_FIELDS = [
    *(f"{prefix}_norm" for prefix in DUAL_QDD_PREFIXES),
    *(f"{prefix}_norm_m_s2" for prefix in DUAL_TASK_PREFIXES),
]
DUAL_SCALAR_FIELDS = [
    "tangent_escape_dual_solve_active",
    "tangent_escape_delta_tcp_accel_dot_tangent_m_s2",
    "tangent_escape_delta_tcp_accel_dot_normal_m_s2",
    "tangent_escape_delta_cp_accel_dot_tangent_m_s2",
    "tangent_escape_delta_cp_accel_dot_normal_m_s2",
    "tangent_escape_dual_activation",
    "tangent_escape_dual_effective_metric_scalar",
]
DUAL_FIELDS = DUAL_VECTOR_FIELDS + DUAL_NORM_FIELDS + DUAL_SCALAR_FIELDS

ORTHOGONALITY_FIELD = "tangent_escape_rmp_normal_dot_tangent"

FLAG_FIELDS = [
    ("stability_base_gds_structural", "base GDS"),
    ("stability_environment_static", "environment static"),
    ("stability_external_rmp_empty", "external RMP empty"),
    ("stability_guard_enabled", "energy guard"),
    ("stability_conditional_nonincrease", "conditional nonincrease"),
]


class ProofPlotError(RuntimeError):
    """Raised when a proof trace cannot be interpreted without guessing."""


@dataclass(frozen=True)
class Sample:
    time_s: float
    values: Mapping[str, float]


@dataclass(frozen=True)
class TraceData:
    path: Path
    row_count: int
    duration_s: float
    certificate: Sequence[Sample]
    dual: Sequence[Sample]
    dual_active: Sequence[Sample]
    orthogonality: Sequence[Tuple[float, float]]
    warnings: Sequence[str]


@dataclass(frozen=True)
class SeriesStats:
    count: int
    minimum: float
    maximum: float
    mean: float
    p95: float


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save a stability-certificate and same-state Escape ON/OFF "
            "dual-solve report as a multi-panel PNG."
        )
    )
    parser.add_argument("csv", type=Path, help="Proof-run rmpflow_trace CSV")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PNG (default: <input_stem>_proof.png beside the CSV)",
    )
    parser.add_argument(
        "--title",
        default="RMPFlow Stability Certificate and Same-State Escape Dual Solve",
        help="Figure title",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=160,
        help="Output resolution in dots per inch (default: 160)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output PNG",
    )
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be > 0")
    return args


def finite_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    try:
        value = float(stripped)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _present(row: Mapping[str, str], field: str) -> bool:
    value = row.get(field)
    return value is not None and bool(value.strip())


def _complete_sample(
    row: Mapping[str, str],
    fields: Sequence[str],
    time_origin: float,
) -> Optional[Sample]:
    stamp = finite_float(row.get(TIME_FIELD))
    if stamp is None:
        return None
    values: Dict[str, float] = {}
    for field in fields:
        value = finite_float(row.get(field))
        if value is None:
            return None
        values[field] = value
    return Sample(time_s=stamp - time_origin, values=values)


def _find_time_origin(rows: Sequence[Mapping[str, str]]) -> Tuple[float, float]:
    stamps = [
        value
        for row in rows
        if (value := finite_float(row.get(TIME_FIELD))) is not None
    ]
    if not stamps:
        raise ProofPlotError(f"Column {TIME_FIELD!r} has no finite values")
    return stamps[0], max(stamps) - min(stamps)


def _validate_headers(fieldnames: Sequence[str]) -> None:
    duplicates = sorted(
        field for field in set(fieldnames) if fieldnames.count(field) > 1
    )
    if duplicates:
        raise ProofPlotError(
            "CSV contains duplicate headers and cannot be interpreted safely: "
            + ", ".join(duplicates)
        )
    required = [TIME_FIELD, *CERTIFICATE_FIELDS, *DUAL_FIELDS]
    missing = [field for field in required if field not in fieldnames]
    if missing:
        raise ProofPlotError(
            "CSV does not match the current proof-run logger schema. Missing "
            "columns: " + ", ".join(missing)
        )


def load_trace(path: Path) -> TraceData:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ProofPlotError(f"Input CSV does not exist or is not a file: {path}")
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            _validate_headers(fieldnames)
            rows = list(reader)
    except (OSError, csv.Error) as error:
        raise ProofPlotError(f"Could not read CSV {path}: {error}") from error
    if not rows:
        raise ProofPlotError(f"CSV contains a header but no data rows: {path}")

    time_origin, duration_s = _find_time_origin(rows)
    warnings: List[str] = []

    certificate: List[Sample] = []
    certificate_partial = 0
    unsupported_schema: List[float] = []
    for row in rows:
        if not _present(row, "stability_schema_version"):
            continue
        sample = _complete_sample(row, CERTIFICATE_FIELDS, time_origin)
        if sample is None:
            certificate_partial += 1
            continue
        schema = sample.values["stability_schema_version"]
        if abs(schema - 1.0) > 1.0e-9:
            unsupported_schema.append(schema)
            continue
        certificate.append(sample)
    if unsupported_schema:
        versions = ", ".join(f"{value:g}" for value in sorted(set(unsupported_schema)))
        raise ProofPlotError(
            f"Unsupported stability certificate schema version(s): {versions}; "
            "this script supports schema v1 only"
        )
    if not certificate:
        raise ProofPlotError(
            "Stability certificate columns exist but contain no complete finite "
            "schema-v1 samples. Record with "
            "publish_stability_certificate_data:=true and verify "
            "/rmp_stability_certificate."
        )
    if certificate_partial:
        warnings.append(
            f"dropped {certificate_partial} partially populated/non-finite "
            "stability certificate row(s)"
        )

    dual: List[Sample] = []
    dual_partial = 0
    for row in rows:
        if not _present(row, "tangent_escape_dual_solve_active"):
            continue
        sample = _complete_sample(row, DUAL_FIELDS, time_origin)
        if sample is None:
            dual_partial += 1
            continue
        dual.append(sample)
    if not dual:
        raise ProofPlotError(
            "Dual-solve columns exist but contain no complete finite samples. "
            "They must not be interpreted as zero. Record with "
            "publish_tangent_escape_dual_solve_data:=true and verify "
            "/tangent_escape_dual_solve."
        )
    if dual_partial:
        warnings.append(
            f"dropped {dual_partial} partially populated/non-finite dual-solve row(s)"
        )

    dual_active = [
        sample
        for sample in dual
        if sample.values["tangent_escape_dual_solve_active"] >= 0.5
    ]
    if not dual_active:
        raise ProofPlotError(
            "Dual-solve data is present, but no row has "
            "tangent_escape_dual_solve_active >= 0.5. Run a scenario that "
            "actually activates Escape before making an ON/OFF proof plot."
        )

    orthogonality: List[Tuple[float, float]] = []
    if ORTHOGONALITY_FIELD in fieldnames:
        for row in rows:
            stamp = finite_float(row.get(TIME_FIELD))
            dot = finite_float(row.get(ORTHOGONALITY_FIELD))
            active = finite_float(row.get("tangent_escape_rmp_active"))
            if stamp is None or dot is None or active is None or active < 0.5:
                continue
            orthogonality.append((stamp - time_origin, dot))
    if not orthogonality:
        warnings.append(
            f"optional active-row {ORTHOGONALITY_FIELD} data is unavailable; "
            "the orthogonality overlay will be omitted"
        )

    return TraceData(
        path=path,
        row_count=len(rows),
        duration_s=duration_s,
        certificate=certificate,
        dual=dual,
        dual_active=dual_active,
        orthogonality=orthogonality,
        warnings=warnings,
    )


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def series_stats(
    samples: Sequence[Sample],
    field: str,
    *,
    absolute: bool = False,
) -> SeriesStats:
    values = [sample.values[field] for sample in samples]
    if absolute:
        values = [abs(value) for value in values]
    if not values:
        raise ValueError(f"no samples available for {field}")
    return SeriesStats(
        count=len(values),
        minimum=min(values),
        maximum=max(values),
        mean=statistics.fmean(values),
        p95=percentile(values, 0.95),
    )


def flag_ratio(samples: Sequence[Sample], field: str) -> float:
    if not samples:
        return float("nan")
    return sum(sample.values[field] >= 0.5 for sample in samples) / len(samples)


def values(samples: Sequence[Sample], field: str) -> List[float]:
    return [sample.values[field] for sample in samples]


def times(samples: Sequence[Sample]) -> List[float]:
    return [sample.time_s for sample in samples]


def _active_spans(
    samples: Sequence[Sample],
    field: str,
) -> List[Tuple[float, float]]:
    if not samples:
        return []
    sample_times = times(samples)
    gaps = [
        end - start
        for start, end in zip(sample_times, sample_times[1:])
        if end > start
    ]
    fallback_gap = statistics.median(gaps) if gaps else 0.01
    spans: List[Tuple[float, float]] = []
    start: Optional[float] = None
    for index, sample in enumerate(samples):
        active = sample.values[field] >= 0.5
        if active and start is None:
            start = sample.time_s
        if start is not None and not active:
            spans.append((start, sample.time_s))
            start = None
        if start is not None and index == len(samples) - 1:
            spans.append((start, sample.time_s + fallback_gap))
    return spans


def shade_active(axes: Iterable[object], spans: Sequence[Tuple[float, float]]) -> None:
    for axis in axes:
        for start, end in spans:
            axis.axvspan(start, end, color="#f59e0b", alpha=0.08, linewidth=0)


def _combined_legend(axis: object, secondary: Optional[object] = None) -> None:
    handles, labels = axis.get_legend_handles_labels()
    if secondary is not None:
        other_handles, other_labels = secondary.get_legend_handles_labels()
        handles += other_handles
        labels += other_labels
    if handles:
        axis.legend(handles, labels, loc="best", fontsize=8, ncol=2)


def _stats_text(label: str, stats: SeriesStats, unit: str) -> str:
    return (
        f"{label}: max={stats.maximum:.6g}, mean={stats.mean:.6g}, "
        f"p95={stats.p95:.6g} {unit}"
    )


def console_summary(trace: TraceData, output: Path) -> str:
    certificate = trace.certificate
    active = trace.dual_active
    qdd = series_stats(active, "tangent_escape_delta_qdd_norm")
    tcp = series_stats(active, "tangent_escape_delta_tcp_accel_norm_m_s2")
    cp = series_stats(active, "tangent_escape_delta_cp_accel_norm_m_s2")
    tcp_tangent = series_stats(
        active,
        "tangent_escape_delta_tcp_accel_dot_tangent_m_s2",
        absolute=True,
    )
    tcp_normal = series_stats(
        active,
        "tangent_escape_delta_tcp_accel_dot_normal_m_s2",
        absolute=True,
    )
    cp_tangent = series_stats(
        active,
        "tangent_escape_delta_cp_accel_dot_tangent_m_s2",
        absolute=True,
    )
    cp_normal = series_stats(
        active,
        "tangent_escape_delta_cp_accel_dot_normal_m_s2",
        absolute=True,
    )
    violation = series_stats(certificate, "stability_energy_bound_violation")
    identity = series_stats(
        certificate,
        "stability_tank_identity_residual",
        absolute=True,
    )
    solve = series_stats(certificate, "stability_solve_power", absolute=True)
    clamp = series_stats(certificate, "stability_clamp_power", absolute=True)
    scale = series_stats(certificate, "stability_escape_scale")
    tank_start = certificate[0].values["stability_tank_energy"]
    tank_end = certificate[-1].values["stability_tank_energy"]
    positive_start = certificate[0].values["stability_positive_escape_energy"]
    positive_end = certificate[-1].values["stability_positive_escape_energy"]
    negative_start = certificate[0].values["stability_negative_escape_energy"]
    negative_end = certificate[-1].values["stability_negative_escape_energy"]
    net_start = certificate[0].values["stability_net_escape_energy"]
    net_end = certificate[-1].values["stability_net_escape_energy"]

    lines = [
        f"source: {trace.path}",
        f"output: {output}",
        (
            f"rows={trace.row_count}, duration={trace.duration_s:.6f} s, "
            f"certificate_valid={len(certificate)}, dual_valid={len(trace.dual)}, "
            f"dual_active={len(active)}"
        ),
        (
            "flags: "
            f"structural={100.0 * flag_ratio(certificate, 'stability_base_gds_structural'):.2f}%, "
            f"environment_static={100.0 * flag_ratio(certificate, 'stability_environment_static'):.2f}%, "
            f"external_empty={100.0 * flag_ratio(certificate, 'stability_external_rmp_empty'):.2f}%, "
            f"guard={100.0 * flag_ratio(certificate, 'stability_guard_enabled'):.2f}%, "
            f"conditional={100.0 * flag_ratio(certificate, 'stability_conditional_nonincrease'):.2f}%"
        ),
        (
            f"energy: violation_max={violation.maximum:.6g}, "
            f"|identity_residual|_max={identity.maximum:.6g}, "
            f"|solve_power|_max={solve.maximum:.6g}, "
            f"|clamp_power|_max={clamp.maximum:.6g}, "
            f"scale_min={scale.minimum:.6g}"
        ),
        (
            f"tank: {tank_start:.9g} -> {tank_end:.9g}; "
            f"delta_positive={positive_end - positive_start:.9g}, "
            f"delta_negative={negative_end - negative_start:.9g}, "
            f"delta_net={net_end - net_start:.9g}"
        ),
        _stats_text("active delta qdd norm", qdd, "rad/s^2"),
        _stats_text("active delta TCP accel norm", tcp, "m/s^2"),
        _stats_text("active delta CP accel norm", cp, "m/s^2"),
        _stats_text("active |delta TCP tangent|", tcp_tangent, "m/s^2"),
        _stats_text("active |delta TCP normal|", tcp_normal, "m/s^2"),
        _stats_text("active |delta CP tangent|", cp_tangent, "m/s^2"),
        _stats_text("active |delta CP normal|", cp_normal, "m/s^2"),
    ]
    if trace.orthogonality:
        orth_values = [abs(value) for _, value in trace.orthogonality]
        lines.append(
            "active tangent orthogonality: "
            f"|n dot t| max={max(orth_values):.6g}, "
            f"mean={statistics.fmean(orth_values):.6g}, "
            f"p95={percentile(orth_values, 0.95):.6g}"
        )
    return "\n".join(lines)


def _import_matplotlib() -> object:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except (ImportError, ModuleNotFoundError) as error:
        raise ProofPlotError(
            "matplotlib is required to create the PNG but is not available. "
            "Install a compatible matplotlib package (for example the ROS host "
            "distribution's python3-matplotlib package) and run again."
        ) from error
    return pyplot


def plot_trace(trace: TraceData, output: Path, title: str, dpi: int) -> None:
    pyplot = _import_matplotlib()
    figure, axes_grid = pyplot.subplots(4, 2, figsize=(18, 20), sharex=True)
    axes = [axis for row in axes_grid for axis in row]
    certificate = trace.certificate
    dual = trace.dual
    cert_times = times(certificate)
    dual_times = times(dual)
    cert_spans = _active_spans(certificate, "stability_escape_active")
    dual_spans = _active_spans(dual, "tangent_escape_dual_solve_active")
    shade_active(axes[:4], cert_spans)
    shade_active(axes[4:], dual_spans)

    flag_axis = axes[0]
    flag_count = len(FLAG_FIELDS)
    for index, (field, label) in enumerate(FLAG_FIELDS):
        offset = float(flag_count - index - 1)
        ratio = 100.0 * flag_ratio(certificate, field)
        plotted = [offset + 0.75 * value for value in values(certificate, field)]
        flag_axis.step(
            cert_times,
            plotted,
            where="post",
            linewidth=1.5,
            label=f"{label} ({ratio:.1f}%)",
        )
    flag_axis.set_yticks(
        [float(flag_count - index - 1) + 0.375 for index in range(flag_count)],
        [label for _, label in FLAG_FIELDS],
    )
    flag_axis.set_ylim(-0.2, flag_count - 0.05)
    flag_axis.set_title("Certificate assumptions and result")
    flag_axis.set_ylabel("0 / 1 flags")
    _combined_legend(flag_axis)

    power_axis = axes[1]
    power_axis.plot(
        cert_times,
        values(certificate, "stability_escape_power_requested"),
        color="#dc2626",
        linewidth=1.2,
        label="Escape power requested",
    )
    power_axis.plot(
        cert_times,
        values(certificate, "stability_escape_power_applied"),
        color="#2563eb",
        linestyle="--",
        linewidth=1.2,
        label="Escape power applied",
    )
    power_axis.axhline(0.0, color="#111827", linewidth=0.8)
    scale_axis = power_axis.twinx()
    scale_axis.plot(
        cert_times,
        values(certificate, "stability_escape_scale"),
        color="#16a34a",
        linewidth=1.0,
        label="applied scale",
    )
    scale_axis.set_ylim(-0.05, 1.05)
    scale_axis.set_ylabel("scale")
    power_axis.set_title("Escape interconnection power and guard scale")
    power_axis.set_ylabel("power")
    _combined_legend(power_axis, scale_axis)

    energy_axis = axes[2]
    positive = values(certificate, "stability_positive_escape_energy")
    negative = values(certificate, "stability_negative_escape_energy")
    net = values(certificate, "stability_net_escape_energy")
    tank = values(certificate, "stability_tank_energy")
    energy_axis.plot(
        cert_times,
        [value - positive[0] for value in positive],
        color="#dc2626",
        label="positive Escape energy increase",
    )
    energy_axis.plot(
        cert_times,
        [value - negative[0] for value in negative],
        color="#16a34a",
        label="dissipative Escape energy increase",
    )
    energy_axis.plot(
        cert_times,
        [value - net[0] for value in net],
        color="#7c3aed",
        label="net Escape energy change",
    )
    energy_axis.plot(
        cert_times,
        [tank[0] - value for value in tank],
        color="#ea580c",
        linestyle="--",
        label="tank depletion from first sample",
    )
    energy_axis.axhline(0.0, color="#111827", linewidth=0.8)
    energy_axis.set_title("Energy ledger and tank")
    energy_axis.set_ylabel("energy change")
    _combined_legend(energy_axis)

    residual_axis = axes[3]
    residual_fields = [
        ("stability_solve_power", "solve power", "#2563eb"),
        ("stability_clamp_power", "clamp power", "#dc2626"),
        (
            "stability_nonincrease_upper_bound",
            "nonincrease upper bound",
            "#7c3aed",
        ),
        (
            "stability_tank_identity_residual",
            "tank identity residual",
            "#ea580c",
        ),
        (
            "stability_energy_bound_violation",
            "energy-bound violation",
            "#111827",
        ),
    ]
    for field, label, color in residual_fields:
        residual_axis.plot(
            cert_times,
            values(certificate, field),
            linewidth=1.0,
            color=color,
            label=label,
        )
    residual_axis.axhline(0.0, color="#111827", linewidth=0.8)
    residual_axis.set_yscale("symlog", linthresh=1.0e-12)
    residual_axis.set_title("Numerical residual and violation audit")
    residual_axis.set_ylabel("signed value (symlog)")
    _combined_legend(residual_axis)

    qdd_axis = axes[4]
    qdd_axis.plot(
        dual_times,
        values(dual, "tangent_escape_qdd_with_norm"),
        color="#2563eb",
        label="qdd norm with Escape",
    )
    qdd_axis.plot(
        dual_times,
        values(dual, "tangent_escape_qdd_without_norm"),
        color="#6b7280",
        linestyle="--",
        label="qdd norm without Escape",
    )
    qdd_axis.plot(
        dual_times,
        values(dual, "tangent_escape_delta_qdd_norm"),
        color="#dc2626",
        linewidth=1.5,
        label="delta qdd norm",
    )
    qdd_axis.set_title("Same-state joint acceleration: Escape ON versus OFF")
    qdd_axis.set_ylabel("rad/s^2")
    _combined_legend(qdd_axis)

    task_axis = axes[5]
    task_axis.plot(
        dual_times,
        values(dual, "tangent_escape_delta_tcp_accel_norm_m_s2"),
        color="#2563eb",
        linewidth=1.5,
        label="delta TCP accel norm",
    )
    task_axis.plot(
        dual_times,
        values(dual, "tangent_escape_delta_cp_accel_norm_m_s2"),
        color="#dc2626",
        linewidth=1.5,
        label="delta CP accel norm",
    )
    activation_axis = task_axis.twinx()
    activation_axis.plot(
        dual_times,
        values(dual, "tangent_escape_dual_activation"),
        color="#16a34a",
        alpha=0.8,
        linewidth=1.0,
        label="dual activation",
    )
    activation_axis.set_ylabel("activation")
    task_axis.set_title("Same-state task acceleration difference")
    task_axis.set_ylabel("m/s^2")
    _combined_legend(task_axis, activation_axis)

    projection_axis = axes[6]
    projection_fields = [
        (
            "tangent_escape_delta_tcp_accel_dot_tangent_m_s2",
            "TCP delta dot tangent",
            "#2563eb",
            "-",
        ),
        (
            "tangent_escape_delta_tcp_accel_dot_normal_m_s2",
            "TCP delta dot normal",
            "#2563eb",
            "--",
        ),
        (
            "tangent_escape_delta_cp_accel_dot_tangent_m_s2",
            "CP delta dot tangent",
            "#dc2626",
            "-",
        ),
        (
            "tangent_escape_delta_cp_accel_dot_normal_m_s2",
            "CP delta dot normal",
            "#dc2626",
            "--",
        ),
    ]
    for field, label, color, linestyle in projection_fields:
        projection_axis.plot(
            dual_times,
            values(dual, field),
            color=color,
            linestyle=linestyle,
            linewidth=1.2,
            label=label,
        )
    projection_axis.axhline(0.0, color="#111827", linewidth=0.8)
    projection_axis.set_title("Projection of the same-state ON/OFF difference")
    projection_axis.set_ylabel("m/s^2")
    _combined_legend(projection_axis)

    orthogonal_axis = axes[7]
    orthogonal_axis.plot(
        dual_times,
        values(dual, "tangent_escape_dual_activation"),
        color="#16a34a",
        linewidth=1.2,
        label="dual activation",
    )
    if trace.orthogonality:
        orthogonal_axis.plot(
            [stamp for stamp, _ in trace.orthogonality],
            [dot for _, dot in trace.orthogonality],
            color="#7c3aed",
            marker=".",
            markersize=4,
            linewidth=1.0,
            label="normal dot tangent",
        )
        orthogonal_axis.axhline(0.2, color="#dc2626", linestyle=":", linewidth=0.8)
        orthogonal_axis.axhline(-0.2, color="#dc2626", linestyle=":", linewidth=0.8)
    metric_axis = orthogonal_axis.twinx()
    metric_axis.plot(
        dual_times,
        values(dual, "tangent_escape_dual_effective_metric_scalar"),
        color="#ea580c",
        alpha=0.75,
        linewidth=1.0,
        label="effective Escape metric",
    )
    orthogonal_axis.set_ylim(-1.05, 1.05)
    orthogonal_axis.set_ylabel("activation / normal dot tangent")
    metric_axis.set_ylabel("metric scalar")
    orthogonal_axis.set_title("Escape activation, metric, and tangent orthogonality")
    _combined_legend(orthogonal_axis, metric_axis)

    for axis in axes:
        axis.grid(True, alpha=0.25, linewidth=0.6)
        axis.set_xlim(0.0, max(trace.duration_s, 1.0e-9))
    for axis in axes[-2:]:
        axis.set_xlabel("time from first CSV row (s)")

    active = trace.dual_active
    qdd_stats = series_stats(active, "tangent_escape_delta_qdd_norm")
    tcp_stats = series_stats(active, "tangent_escape_delta_tcp_accel_norm_m_s2")
    cp_stats = series_stats(active, "tangent_escape_delta_cp_accel_norm_m_s2")
    violation_max = max(
        values(certificate, "stability_energy_bound_violation")
    )
    conditional = 100.0 * flag_ratio(
        certificate,
        "stability_conditional_nonincrease",
    )
    subtitle = (
        f"{trace.path.name} | rows {trace.row_count}, certificate "
        f"{len(certificate)}, dual {len(dual)}, dual-active {len(active)} | "
        f"conditional {conditional:.1f}%, violation max {violation_max:.3g}\n"
        f"active max/mean/p95: delta qdd "
        f"{qdd_stats.maximum:.3g}/{qdd_stats.mean:.3g}/{qdd_stats.p95:.3g} rad/s^2; "
        f"TCP {tcp_stats.maximum:.3g}/{tcp_stats.mean:.3g}/{tcp_stats.p95:.3g}; "
        f"CP {cp_stats.maximum:.3g}/{cp_stats.mean:.3g}/{cp_stats.p95:.3g} m/s^2"
    )
    figure.suptitle(f"{title}\n{subtitle}", fontsize=14)
    figure.subplots_adjust(
        left=0.08,
        right=0.92,
        top=0.91,
        bottom=0.05,
        hspace=0.36,
        wspace=0.25,
    )
    try:
        figure.savefig(
            output,
            dpi=dpi,
            bbox_inches="tight",
            metadata={
                "Title": title,
                "Description": (
                    "Runtime stability certificate and same-state Tangent "
                    "Escape RMP ON/OFF dual-solve report"
                ),
                "Source": str(trace.path),
            },
        )
    except OSError as error:
        raise ProofPlotError(f"Could not write PNG {output}: {error}") from error
    finally:
        pyplot.close(figure)


def resolve_output(args: argparse.Namespace) -> Path:
    output = args.output
    if output is None:
        output = args.csv.with_name(f"{args.csv.stem}_proof.png")
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".png":
        raise ProofPlotError(f"Output must use a .png suffix: {output}")
    if output.exists() and not args.overwrite:
        raise ProofPlotError(
            f"Output already exists: {output}; pass --overwrite to replace it"
        )
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ProofPlotError(
            f"Could not create output directory {output.parent}: {error}"
        ) from error
    return output


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        trace = load_trace(args.csv)
        output = resolve_output(args)
        plot_trace(trace, output, args.title, args.dpi)
        for warning in trace.warnings:
            print(f"warning: {warning}", file=sys.stderr)
        print(console_summary(trace, output))
    except ProofPlotError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
