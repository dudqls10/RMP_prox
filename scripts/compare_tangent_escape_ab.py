#!/usr/bin/env python3
"""Compare canonical Tangent Escape RMP OFF/ON trace CSV files.

The comparison is intentionally dependency-free.  It reports measurements, not
a global stability proof: the canonical velocity leaf is a bounded hybrid RMP,
and its end-to-end safety/stability still has to be interpreted together with
the collision leaves and the controller limits.
"""

import argparse
import csv
import math
import re
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


QDD_FIELDS = [f"rmp_joint_accel_{index}_rad_s2" for index in range(1, 7)]
QVEL_FIELDS = [f"joint_{index}_vel_rad_s" for index in range(1, 7)]
TCP_ACCEL_FIELDS = [
    "rmp_tcp_accel_x_m_s2",
    "rmp_tcp_accel_y_m_s2",
    "rmp_tcp_accel_z_m_s2",
]
GOAL_FIELDS = ["controller_goal_x", "controller_goal_y", "controller_goal_z"]
EE_FIELDS = ["rmp_ee_x", "rmp_ee_y", "rmp_ee_z"]
TANGENT_FIELDS = [
    "tangent_escape_rmp_tangent_x",
    "tangent_escape_rmp_tangent_y",
    "tangent_escape_rmp_tangent_z",
]

STATE_NAMES = {
    0: "OFF",
    1: "PREVENT",
    2: "RECOVERY",
    3: "RELEASE",
    4: "RESELECT",
}
HANDOFF_REASON_NAMES = {
    0: "none",
    1: "blockage_clear",
    2: "pair_switch",
    3: "goal_change",
    4: "tangent_or_sector",
    5: "blocked_branch",
    6: "progress_recovered",
}
NONFINITE_TOKENS = {
    "nan",
    "+nan",
    "-nan",
    "inf",
    "+inf",
    "-inf",
    "infinity",
    "+infinity",
    "-infinity",
}


class InputError(RuntimeError):
    """Raised when a trace cannot be interpreted."""


@dataclass
class Trace:
    path: Path
    fields: List[str]
    rows: List[Dict[str, str]]
    times: List[Optional[float]]
    sensor_fields: Dict[str, str]
    nonfinite_cells: int


@dataclass
class GoalSegment:
    goal: Tuple[float, float, float]
    duration_s: float
    reached: bool
    minimum_error_m: Optional[float]


@dataclass
class ChangeAudit:
    total: int = 0
    cp_changes: int = 0
    direction_changes: int = 0
    combined_changes: int = 0
    observed_zero_effect: int = 0
    violations: int = 0
    unobservable: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare OFF and ON canonical RMPFlow trace CSVs without pandas/numpy."
        )
    )
    parser.add_argument("off_csv", type=Path, help="Escape-disabled trace CSV")
    parser.add_argument("on_csv", type=Path, help="Escape-enabled trace CSV")
    parser.add_argument(
        "--reach-tolerance",
        type=float,
        default=0.005,
        help="Manual controller-goal/EE reach tolerance in metres (default: 0.005)",
    )
    parser.add_argument(
        "--risk-distance",
        type=float,
        default=0.29,
        help="Effective fake-sensor distance defining a risk row (default: 0.29)",
    )
    parser.add_argument(
        "--global-accel-cap",
        type=float,
        default=10.0,
        help="Global per-joint acceleration cap in rad/s^2 (default: 10)",
    )
    parser.add_argument(
        "--cap-tolerance",
        type=float,
        default=1.0e-3,
        help="Tolerance below the acceleration cap counted as at-cap (default: 0.001)",
    )
    parser.add_argument(
        "--fresh-dt",
        type=float,
        default=0.1,
        help="Maximum adjacent sample/data age for observable handoff (default: 0.1 s)",
    )
    parser.add_argument(
        "--lambda-tolerance",
        type=float,
        default=1.0e-3,
        help="Lambda zero-effect tolerance (default: 0.001)",
    )
    parser.add_argument(
        "--metric-tolerance",
        type=float,
        default=1.0e-3,
        help="Effective-metric zero-effect tolerance (default: 0.001)",
    )
    parser.add_argument(
        "--direction-dot-threshold",
        type=float,
        default=0.999,
        help="Adjacent tangent dot below this denotes a direction change (default: 0.999)",
    )
    parser.add_argument(
        "--jerk-min-dt",
        type=float,
        default=1.0e-5,
        help="Minimum dt used for numerical jerk (default: 1e-5 s)",
    )
    parser.add_argument(
        "--jerk-max-dt",
        type=float,
        default=1.0,
        help="Maximum dt used for numerical jerk and row-time integration (default: 1.0 s)",
    )
    parser.add_argument(
        "--sensor-equality-tolerance",
        type=float,
        default=1.0e-9,
        help="Absolute tolerance for row-aligned sensor equality (default: 1e-9 m)",
    )
    args = parser.parse_args()
    positive = [
        ("reach tolerance", args.reach_tolerance),
        ("risk distance", args.risk_distance),
        ("global acceleration cap", args.global_accel_cap),
        ("fresh dt", args.fresh_dt),
        ("jerk min dt", args.jerk_min_dt),
        ("jerk max dt", args.jerk_max_dt),
    ]
    for label, value in positive:
        if not math.isfinite(value) or value <= 0.0:
            parser.error(f"{label} must be finite and > 0")
    if args.jerk_max_dt < args.jerk_min_dt:
        parser.error("--jerk-max-dt must be >= --jerk-min-dt")
    if not -1.0 <= args.direction_dot_threshold <= 1.0:
        parser.error("--direction-dot-threshold must be in [-1, 1]")
    for label, value in [
        ("cap tolerance", args.cap_tolerance),
        ("lambda tolerance", args.lambda_tolerance),
        ("metric tolerance", args.metric_tolerance),
        ("sensor equality tolerance", args.sensor_equality_tolerance),
    ]:
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"{label} must be finite and >= 0")
    return args


def as_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        result = float(text)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def vector(row: Dict[str, str], fields: Sequence[str]) -> Optional[Tuple[float, ...]]:
    values = tuple(as_float(row.get(field)) for field in fields)
    if any(value is None for value in values):
        return None
    return tuple(float(value) for value in values if value is not None)


def norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def distance(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    return norm([lhs[index] - rhs[index] for index in range(len(lhs))])


def rms(values: Iterable[float]) -> Optional[float]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None
    return math.sqrt(sum(value * value for value in finite) / len(finite))


def percentile(values: Iterable[float], fraction: float) -> Optional[float]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return None
    index = max(0, min(len(finite) - 1, math.ceil(fraction * len(finite)) - 1))
    return finite[index]


def fmt(value: Optional[float], digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}g}"


def sensor_key(field: str) -> str:
    base = field[: -len("_effective_m")]
    match = re.search(r"(?:distance|tof)[^0-9]*(\d+)$", base, re.IGNORECASE)
    if match:
        return f"sensor_{int(match.group(1)):02d}"
    match = re.search(r"(\d+)$", base)
    if match:
        return f"sensor_{int(match.group(1)):02d}"
    return base


def discover_sensor_fields(fields: Sequence[str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for field in fields:
        if not field.endswith("_effective_m"):
            continue
        if field.startswith("tangent_escape_"):
            continue
        key = sensor_key(field)
        if key in result:
            raise InputError(
                f"ambiguous effective-distance columns for {key}: "
                f"{result[key]!r}, {field!r}"
            )
        result[key] = field
    return dict(sorted(result.items()))


def load_trace(path: Path) -> Trace:
    if not path.is_file():
        raise InputError(f"file not found: {path}")
    try:
        handle = path.open("r", newline="", encoding="utf-8-sig")
    except OSError as error:
        raise InputError(f"cannot open {path}: {error}") from error
    with handle:
        try:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise InputError(f"missing CSV header: {path}")
            fields = list(reader.fieldnames)
            if len(set(fields)) != len(fields):
                raise InputError(f"duplicate CSV header field: {path}")
            required = (
                ["time_ros_s"]
                + QDD_FIELDS
                + QVEL_FIELDS
                + TCP_ACCEL_FIELDS
                + GOAL_FIELDS
                + EE_FIELDS
            )
            missing = [field for field in required if field not in fields]
            if missing:
                raise InputError(
                    f"{path} is not a current RMPFlow trace; missing: "
                    + ", ".join(missing)
                )
            rows: List[Dict[str, str]] = []
            nonfinite_cells = 0
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    raise InputError(
                        f"row {line_number} has more cells than the header: {path}"
                    )
                normalized = {
                    field: (row.get(field) or "").strip()
                    for field in fields
                }
                nonfinite_cells += sum(
                    value.lower() in NONFINITE_TOKENS
                    for value in normalized.values()
                )
                rows.append(normalized)
        except csv.Error as error:
            raise InputError(f"CSV parse error in {path}: {error}") from error
    if not rows:
        raise InputError(f"CSV has no data rows: {path}")
    times = [as_float(row.get("time_ros_s")) for row in rows]
    if not any(time is not None for time in times):
        raise InputError(f"CSV has no finite ROS timestamps: {path}")
    sensor_fields = discover_sensor_fields(fields)
    if not sensor_fields:
        raise InputError(f"CSV has no effective fake-sensor distance columns: {path}")
    return Trace(path, fields, rows, times, sensor_fields, nonfinite_cells)


def positive_dts(trace: Trace) -> List[float]:
    result = []
    for previous, current in zip(trace.times, trace.times[1:]):
        if previous is not None and current is not None and current > previous:
            result.append(current - previous)
    return result


def trace_duration(trace: Trace) -> float:
    finite = [time for time in trace.times if time is not None]
    return max(0.0, finite[-1] - finite[0]) if len(finite) >= 2 else 0.0


def integrated_row_time(
    trace: Trace,
    predicate: Sequence[bool],
    max_dt: float,
) -> float:
    total = 0.0
    for index in range(min(len(predicate), len(trace.rows) - 1)):
        previous = trace.times[index]
        current = trace.times[index + 1]
        if (
            predicate[index]
            and previous is not None
            and current is not None
            and 0.0 < current - previous <= max_dt
        ):
            total += current - previous
    return total


def goal_segments(trace: Trace, tolerance: float) -> List[GoalSegment]:
    segment_rows: List[Tuple[int, Tuple[float, float, float]]] = []
    previous_goal: Optional[Tuple[float, float, float]] = None
    for index, row in enumerate(trace.rows):
        goal = vector(row, GOAL_FIELDS)
        if goal is None:
            continue
        goal3 = (goal[0], goal[1], goal[2])
        if previous_goal is None or distance(goal3, previous_goal) > 1.0e-6:
            segment_rows.append((index, goal3))
            previous_goal = goal3
    result = []
    for segment_index, (start, goal) in enumerate(segment_rows):
        stop = (
            segment_rows[segment_index + 1][0]
            if segment_index + 1 < len(segment_rows)
            else len(trace.rows)
        )
        errors = []
        reached_time: Optional[float] = None
        for row_index, row in enumerate(trace.rows[start:stop], start=start):
            ee = vector(row, EE_FIELDS)
            current_goal = vector(row, GOAL_FIELDS)
            if ee is not None and current_goal is not None:
                error = distance(ee, current_goal)
                errors.append(error)
                if error <= tolerance and reached_time is None:
                    reached_time = trace.times[row_index]
        start_time = trace.times[start]
        end_time = trace.times[max(start, stop - 1)]
        duration_end = reached_time if reached_time is not None else end_time
        duration = (
            max(0.0, duration_end - start_time)
            if start_time is not None and duration_end is not None
            else 0.0
        )
        minimum_error = min(errors) if errors else None
        result.append(
            GoalSegment(
                goal=goal,
                duration_s=duration,
                reached=minimum_error is not None and minimum_error <= tolerance,
                minimum_error_m=minimum_error,
            )
        )
    return result


def distinct_goal_count(segments: Sequence[GoalSegment]) -> int:
    return len(
        {
            tuple(round(component, 6) for component in segment.goal)
            for segment in segments
        }
    )


def final_goal_error(trace: Trace) -> Optional[float]:
    for row in reversed(trace.rows):
        goal = vector(row, GOAL_FIELDS)
        ee = vector(row, EE_FIELDS)
        if goal is not None and ee is not None:
            return distance(goal, ee)
    return None


def ee_path_length(rows: Sequence[Dict[str, str]]) -> float:
    total = 0.0
    previous: Optional[Tuple[float, ...]] = None
    for row in rows:
        current = vector(row, EE_FIELDS)
        if current is None:
            continue
        if previous is not None:
            total += distance(previous, current)
        previous = current
    return total


def vector_norms(
    rows: Sequence[Dict[str, str]],
    fields: Sequence[str],
) -> List[float]:
    result = []
    for row in rows:
        values = vector(row, fields)
        if values is not None:
            result.append(norm(values))
    return result


def jerk_norms(trace: Trace, minimum_dt: float, maximum_dt: float) -> List[float]:
    result = []
    for index in range(1, len(trace.rows)):
        previous_time = trace.times[index - 1]
        current_time = trace.times[index]
        previous = vector(trace.rows[index - 1], QDD_FIELDS)
        current = vector(trace.rows[index], QDD_FIELDS)
        if (
            previous_time is None
            or current_time is None
            or previous is None
            or current is None
        ):
            continue
        dt = current_time - previous_time
        if not minimum_dt <= dt <= maximum_dt:
            continue
        result.append(
            norm(
                [
                    (current[joint] - previous[joint]) / dt
                    for joint in range(6)
                ]
            )
        )
    return result


def scalar_values(rows: Sequence[Dict[str, str]], field: str) -> List[float]:
    result = []
    for row in rows:
        value = as_float(row.get(field))
        if value is not None:
            result.append(value)
    return result


def risk_mask(trace: Trace, threshold: float) -> List[bool]:
    fields = list(trace.sensor_fields.values())
    result = []
    for row in trace.rows:
        values = [as_float(row.get(field)) for field in fields]
        result.append(any(value is not None and value <= threshold for value in values))
    return result


def minimum_sensor_distance(trace: Trace) -> Optional[float]:
    values = []
    for field in trace.sensor_fields.values():
        values.extend(scalar_values(trace.rows, field))
    return min(values) if values else None


def print_input_equality(
    off: Trace,
    on: Trace,
    risk_distance: float,
    equality_tolerance: float,
) -> None:
    print("\n[input equality: fake effective distances]")
    print(
        "note=Row-aligned equality is diagnostic only; independent launches/replays "
        "can have a phase offset. Aggregate minima and risk-row counts are also shown."
    )
    keys = sorted(set(off.sensor_fields) | set(on.sensor_fields))
    all_aggregate_equal = True
    for key in keys:
        off_field = off.sensor_fields.get(key)
        on_field = on.sensor_fields.get(key)
        off_values = scalar_values(off.rows, off_field) if off_field else []
        on_values = scalar_values(on.rows, on_field) if on_field else []
        paired = min(len(off.rows), len(on.rows))
        comparable = 0
        equal = 0
        max_difference = 0.0
        if off_field and on_field:
            for index in range(paired):
                off_value = as_float(off.rows[index].get(off_field))
                on_value = as_float(on.rows[index].get(on_field))
                if off_value is None or on_value is None:
                    continue
                comparable += 1
                difference = abs(off_value - on_value)
                max_difference = max(max_difference, difference)
                equal += difference <= equality_tolerance
        off_min = min(off_values) if off_values else None
        on_min = min(on_values) if on_values else None
        off_risk = sum(value <= risk_distance for value in off_values)
        on_risk = sum(value <= risk_distance for value in on_values)
        aggregate_equal = (
            off_min is not None
            and on_min is not None
            and abs(off_min - on_min) <= equality_tolerance
            and off_risk == on_risk
            and len(off_values) == len(on_values)
        )
        all_aggregate_equal = all_aggregate_equal and aggregate_equal
        print(
            f"{key}: n={len(off_values)}/{len(on_values)} "
            f"min_m={fmt(off_min)}/{fmt(on_min)} "
            f"risk_rows={off_risk}/{on_risk} "
            f"row_equal={equal}/{comparable} max_abs_diff_m={fmt(max_difference)} "
            f"aggregate_equal={str(aggregate_equal).lower()}"
        )
    print(f"all_sensor_aggregates_equal={str(all_aggregate_equal).lower()}")


def print_goal_summary(label: str, trace: Trace, reach_tolerance: float) -> None:
    segments = goal_segments(trace, reach_tolerance)
    reached = sum(segment.reached for segment in segments)
    compact = ",".join(
        f"{fmt(segment.duration_s, 4)}{'R' if segment.reached else 'X'}"
        for segment in segments
    )
    print(
        f"{label}: distinct_goals={distinct_goal_count(segments)} "
        f"segments={len(segments)} reached={reached}/{len(segments)} "
        f"final_goal_error_m={fmt(final_goal_error(trace))} "
        f"ee_path_m={fmt(ee_path_length(trace.rows))}"
    )
    print(
        f"{label}: segment_time_to_reach_or_observed_s=[{compact}] "
        f"(R=first manual error <= {reach_tolerance:g} m, X=not observed)"
    )


def min_optional(values: Sequence[float]) -> Optional[float]:
    return min(values) if values else None


def max_optional(values: Sequence[float]) -> Optional[float]:
    return max(values) if values else None


def print_dynamics_summary(
    label: str,
    trace: Trace,
    global_cap: float,
    cap_tolerance: float,
    minimum_dt: float,
    maximum_dt: float,
) -> None:
    qdd_norms = vector_norms(trace.rows, QDD_FIELDS)
    tcp_accel_norms = vector_norms(trace.rows, TCP_ACCEL_FIELDS)
    joint_velocity_norms = vector_norms(trace.rows, QVEL_FIELDS)
    jerks = jerk_norms(trace, minimum_dt, maximum_dt)
    component_parts = []
    for index, field in enumerate(QDD_FIELDS, start=1):
        values = [abs(value) for value in scalar_values(trace.rows, field)]
        component_parts.append(
            f"j{index}(peak={fmt(max_optional(values))},"
            f"rms={fmt(rms(values))},p95={fmt(percentile(values, 0.95))})"
        )

    cap_threshold = max(0.0, global_cap - cap_tolerance)
    at_cap = []
    for row in trace.rows:
        qdd = vector(row, QDD_FIELDS)
        at_cap.append(
            qdd is not None and any(abs(value) >= cap_threshold for value in qdd)
        )
    clearance = scalar_values(trace.rows, "min_external_clearance_m")
    min_z = [
        (as_float(row.get("min_z_safety_triggered")) or 0.0) >= 0.5
        for row in trace.rows
    ]
    dts = positive_dts(trace)
    print(
        f"{label}: rows={len(trace.rows)} duration_s={fmt(trace_duration(trace))} "
        f"median_dt_s={fmt(statistics.median(dts) if dts else None)} "
        f"nonfinite_cells={trace.nonfinite_cells}"
    )
    print(
        f"{label}: min_external_clearance_m={fmt(min_optional(clearance))} "
        f"min_sensor_distance_m={fmt(minimum_sensor_distance(trace))} "
        f"min_z_trigger_rows={sum(min_z)} "
        f"min_z_trigger_time_s={fmt(integrated_row_time(trace, min_z, maximum_dt))}"
    )
    print(f"{label}: qdd_components_rad_s2 " + " ".join(component_parts))
    print(
        f"{label}: qdd_norm_rad_s2 peak={fmt(max_optional(qdd_norms))} "
        f"rms={fmt(rms(qdd_norms))}; "
        f"jerk_norm_rad_s3 peak={fmt(max_optional(jerks))} rms={fmt(rms(jerks))} "
        f"valid_pairs={len(jerks)} dt_guard=[{minimum_dt:g},{maximum_dt:g}]"
    )
    print(
        f"{label}: accel_cap={global_cap:g} threshold={cap_threshold:g} "
        f"rows_at_cap={sum(at_cap)} "
        f"time_at_cap_s={fmt(integrated_row_time(trace, at_cap, maximum_dt))}; "
        f"joint_velocity_norm_peak_rad_s={fmt(max_optional(joint_velocity_norms))}; "
        f"tcp_accel_norm_m_s2 peak={fmt(max_optional(tcp_accel_norms))} "
        f"rms={fmt(rms(tcp_accel_norms))}"
    )


def normalized_dot(
    lhs: Sequence[float],
    rhs: Sequence[float],
) -> Optional[float]:
    lhs_norm = norm(lhs)
    rhs_norm = norm(rhs)
    if lhs_norm <= 1.0e-12 or rhs_norm <= 1.0e-12:
        return None
    return sum(a * b for a, b in zip(lhs, rhs)) / (lhs_norm * rhs_norm)


def canonical_change_audit(
    trace: Trace,
    fresh_dt: float,
    lambda_tolerance: float,
    metric_tolerance: float,
    direction_dot_threshold: float,
) -> ChangeAudit:
    audit = ChangeAudit()
    for index in range(1, len(trace.rows)):
        previous_row = trace.rows[index - 1]
        current_row = trace.rows[index]
        previous_cp = as_float(
            previous_row.get("tangent_escape_rmp_control_point_index")
        )
        current_cp = as_float(
            current_row.get("tangent_escape_rmp_control_point_index")
        )
        previous_tangent = vector(previous_row, TANGENT_FIELDS)
        current_tangent = vector(current_row, TANGENT_FIELDS)
        cp_changed = (
            previous_cp is not None
            and current_cp is not None
            and int(round(previous_cp)) != int(round(current_cp))
        )
        tangent_dot = (
            normalized_dot(previous_tangent, current_tangent)
            if previous_tangent is not None and current_tangent is not None
            else None
        )
        direction_changed = (
            tangent_dot is not None and tangent_dot < direction_dot_threshold
        )
        if not cp_changed and not direction_changed:
            continue
        audit.total += 1
        audit.cp_changes += int(cp_changed)
        audit.direction_changes += int(direction_changed)
        audit.combined_changes += int(cp_changed and direction_changed)

        previous_time = trace.times[index - 1]
        current_time = trace.times[index]
        previous_age = as_float(previous_row.get("tangent_escape_rmp_age_s"))
        current_age = as_float(current_row.get("tangent_escape_rmp_age_s"))
        fresh = (
            previous_time is not None
            and current_time is not None
            and 0.0 < current_time - previous_time <= fresh_dt
            and previous_age is not None
            and current_age is not None
            and previous_age <= fresh_dt
            and current_age <= fresh_dt
        )
        if not fresh:
            audit.unobservable += 1
            continue

        previous_lambda = as_float(previous_row.get("tangent_escape_rmp_lambda"))
        current_lambda = as_float(current_row.get("tangent_escape_rmp_lambda"))
        previous_metric = as_float(
            previous_row.get("tangent_escape_rmp_effective_metric_scalar")
        )
        current_metric = as_float(
            current_row.get("tangent_escape_rmp_effective_metric_scalar")
        )
        if any(
            value is None
            for value in [
                previous_lambda,
                current_lambda,
                previous_metric,
                current_metric,
            ]
        ):
            audit.unobservable += 1
            continue
        previous_effective = (
            previous_lambda > lambda_tolerance
            and previous_metric > metric_tolerance
        )
        current_effective = (
            current_lambda > lambda_tolerance
            and current_metric > metric_tolerance
        )
        if previous_effective and current_effective:
            audit.violations += 1
        else:
            audit.observed_zero_effect += 1
    return audit


def count_discrete_states(
    trace: Trace,
    field: str,
    inactive_default: Optional[int] = None,
) -> Tuple[Counter, int]:
    values: List[Optional[int]] = []
    for row in trace.rows:
        value = as_float(row.get(field))
        if value is None and inactive_default is not None:
            active = as_float(row.get("tangent_escape_rmp_active"))
            if active is not None and active < 0.5:
                value = float(inactive_default)
        values.append(int(round(value)) if value is not None else None)
    counts = Counter(value for value in values if value is not None)
    transitions = sum(
        previous is not None and current is not None and previous != current
        for previous, current in zip(values, values[1:])
    )
    return counts, transitions


def handoff_event_counts(trace: Trace) -> Counter:
    result: Counter = Counter()
    previous = 0
    for row in trace.rows:
        value = as_float(row.get("tangent_escape_rmp_handoff_reason_id"))
        current = int(round(value)) if value is not None else 0
        if current != 0 and current != previous:
            result[current] += 1
        previous = current
    return result


def named_counts(counts: Counter, names: Dict[int, str]) -> str:
    keys = sorted(set(counts) | set(names))
    return ",".join(f"{names.get(key, str(key))}:{counts.get(key, 0)}" for key in keys)


def print_canonical_on_summary(trace: Trace, args: argparse.Namespace) -> None:
    active = [
        (as_float(row.get("tangent_escape_rmp_active")) or 0.0) >= 0.5
        for row in trace.rows
    ]
    intervals = sum(current and not previous for previous, current in zip([False] + active, active))
    task_accel = [
        abs(value)
        for value in scalar_values(
            trace.rows, "tangent_escape_rmp_desired_tangent_accel_m_s2"
        )
    ]
    metrics = scalar_values(
        trace.rows, "tangent_escape_rmp_effective_metric_scalar"
    )
    normal_dot = [
        abs(value)
        for value in scalar_values(
            trace.rows, "tangent_escape_rmp_normal_dot_tangent"
        )
    ]
    normal_dot_engage_drive = []
    normal_dot_transition_metric = []
    for row in trace.rows:
        metric = as_float(row.get("tangent_escape_rmp_effective_metric_scalar"))
        dot_value = as_float(row.get("tangent_escape_rmp_normal_dot_tangent"))
        phase = as_float(row.get("tangent_escape_rmp_handoff_phase_id"))
        if dot_value is None or phase is None:
            continue
        if int(round(phase)) in (1, 2):
            normal_dot_engage_drive.append(abs(dot_value))
        elif metric is not None and metric > 1.0:
            normal_dot_transition_metric.append(abs(dot_value))
    velocity_error = [
        abs(value)
        for value in scalar_values(
            trace.rows, "tangent_escape_rmp_velocity_error_m_s"
        )
    ]
    states, state_transitions = count_discrete_states(
        trace, "tangent_escape_rmp_state_mode_id", inactive_default=0
    )
    phase_counts, phase_transitions = count_discrete_states(
        trace, "tangent_escape_rmp_handoff_phase_id"
    )
    handoff_events = handoff_event_counts(trace)
    audit = canonical_change_audit(
        trace,
        args.fresh_dt,
        args.lambda_tolerance,
        args.metric_tolerance,
        args.direction_dot_threshold,
    )

    print("\n[ON canonical leaf]")
    print(
        f"active_intervals={intervals} active_rows={sum(active)} "
        f"active_time_s={fmt(integrated_row_time(trace, active, args.jerk_max_dt))} "
        f"max_abs_task_accel_m_s2={fmt(max_optional(task_accel))}"
    )
    negative_count = sum(value < 0.0 for value in metrics)
    negative_metrics = sum(value < -args.metric_tolerance for value in metrics)
    print(
        f"metric_scalar min={fmt(min_optional(metrics))} max={fmt(max_optional(metrics))} "
        f"nonnegative={str(negative_count == 0).lower()} "
        f"negative_count={negative_count} negative_beyond_tolerance={negative_metrics}; "
        f"max_abs_normal_dot_all_phases={fmt(max_optional(normal_dot))} "
        f"max_engage_drive={fmt(max_optional(normal_dot_engage_drive))} "
        f"max_transition_when_metric_gt_1={fmt(max_optional(normal_dot_transition_metric))}"
    )
    print(
        f"velocity_error_m_s rms={fmt(rms(velocity_error))} "
        f"peak={fmt(max_optional(velocity_error))}"
    )
    print(
        f"state_rows={named_counts(states, STATE_NAMES)} "
        f"state_transitions={state_transitions}; "
        f"phase_rows={named_counts(phase_counts, {})} phase_transitions={phase_transitions}"
    )
    print(
        f"handoff_events={named_counts(handoff_events, HANDOFF_REASON_NAMES)}"
    )
    print(
        "direction_cp_change_audit "
        f"total={audit.total} cp={audit.cp_changes} direction={audit.direction_changes} "
        f"combined={audit.combined_changes} "
        f"observed_zero_effect={audit.observed_zero_effect} "
        f"violations={audit.violations} unobservable={audit.unobservable} "
        f"(fresh_dt<={args.fresh_dt:g}, lambda_tol={args.lambda_tolerance:g}, "
        f"metric_tol={args.metric_tolerance:g})"
    )


def risk_rows(trace: Trace, threshold: float) -> List[Dict[str, str]]:
    mask = risk_mask(trace, threshold)
    return [row for row, selected in zip(trace.rows, mask) if selected]


def print_risk_summary(label: str, trace: Trace, args: argparse.Namespace) -> None:
    mask = risk_mask(trace, args.risk_distance)
    rows = [row for row, selected in zip(trace.rows, mask) if selected]
    qdd = vector_norms(rows, QDD_FIELDS)
    velocity = vector_norms(rows, QVEL_FIELDS)
    tcp_accel = vector_norms(rows, TCP_ACCEL_FIELDS)
    goal_errors = []
    for row in rows:
        goal = vector(row, GOAL_FIELDS)
        ee = vector(row, EE_FIELDS)
        if goal is not None and ee is not None:
            goal_errors.append(distance(goal, ee))
    clearances = scalar_values(rows, "min_external_clearance_m")
    cap_threshold = max(0.0, args.global_accel_cap - args.cap_tolerance)
    cap_rows = 0
    for row in rows:
        qdd_values = vector(row, QDD_FIELDS)
        if (
            qdd_values is not None and
            any(abs(value) >= cap_threshold for value in qdd_values)
        ):
            cap_rows += 1
    sensor_distances = []
    for field in trace.sensor_fields.values():
        sensor_distances.extend(scalar_values(rows, field))
    active = [
        (as_float(row.get("tangent_escape_rmp_active")) or 0.0) >= 0.5
        for row in trace.rows
    ]
    selected_active = [
        selected and active_value
        for selected, active_value in zip(mask, active)
    ]
    print(
        f"{label}: risk_rows={sum(mask)} "
        f"risk_time_s={fmt(integrated_row_time(trace, mask, args.jerk_max_dt))} "
        f"min_clearance_m={fmt(min_optional(clearances))} "
        f"min_sensor_distance_m={fmt(min_optional(sensor_distances))} "
        f"goal_error_m median={fmt(statistics.median(goal_errors) if goal_errors else None)} "
        f"max={fmt(max_optional(goal_errors))} "
        f"qdd_norm_rad_s2 peak={fmt(max_optional(qdd))} rms={fmt(rms(qdd))} "
        f"accel_cap_rows={cap_rows}/{len(rows)} "
        f"joint_vel_norm_peak_rad_s={fmt(max_optional(velocity))} "
        f"tcp_accel_norm_m_s2 peak={fmt(max_optional(tcp_accel))} rms={fmt(rms(tcp_accel))} "
        f"escape_active_rows={sum(selected_active)}"
    )


def main() -> int:
    args = parse_args()
    try:
        off = load_trace(args.off_csv)
        on = load_trace(args.on_csv)
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print("Tangent Escape canonical OFF/ON comparison")
    print(f"OFF={off.path.resolve()}")
    print(f"ON={on.path.resolve()}")
    print_dynamics_summary(
        "OFF",
        off,
        args.global_accel_cap,
        args.cap_tolerance,
        args.jerk_min_dt,
        args.jerk_max_dt,
    )
    print_dynamics_summary(
        "ON ",
        on,
        args.global_accel_cap,
        args.cap_tolerance,
        args.jerk_min_dt,
        args.jerk_max_dt,
    )

    print("\n[route]")
    print_goal_summary("OFF", off, args.reach_tolerance)
    print_goal_summary("ON ", on, args.reach_tolerance)
    print_input_equality(
        off,
        on,
        args.risk_distance,
        args.sensor_equality_tolerance,
    )

    print(f"\n[risk-conditioned: any effective fake distance <= {args.risk_distance:g} m]")
    print_risk_summary("OFF", off, args)
    print_risk_summary("ON ", on, args)
    print_canonical_on_summary(on, args)

    serious_nonfinite = off.nonfinite_cells + on.nonfinite_cells
    if serious_nonfinite:
        print(
            f"\nSERIOUS: {serious_nonfinite} explicit NaN/Inf CSV cells detected.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
