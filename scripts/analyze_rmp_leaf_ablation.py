#!/usr/bin/env python3

import argparse
import csv
import html
import math
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


LEAF_GROUPS = [
    "cspace_target",
    "joint_limits",
    "joint_velocity_cap",
    "target",
    "collision",
    "tangent_escape",
    "damping",
    "body_target",
    "axis_target_x",
    "axis_target_y",
    "axis_target_z",
    "wrist_axis_target",
    "external",
    "other",
]

LEAF_LABELS = {
    "cspace_target": "C-space Target",
    "joint_limits": "Joint Limit",
    "joint_velocity_cap": "Joint Velocity Cap",
    "target": "TCP Target",
    "collision": "Collision",
    "tangent_escape": "Tangent Escape",
    "damping": "Joint Damping",
    "body_target": "Body Target",
    "axis_target_x": "Axis Target X",
    "axis_target_y": "Axis Target Y",
    "axis_target_z": "Axis Target Z",
    "wrist_axis_target": "Wrist Axis Target",
    "external": "External RMP",
    "other": "Other",
}


def finite_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(max(fraction, 0.0), 1.0) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return (1.0 - ratio) * ordered[lower] + ratio * ordered[upper]


def latest_compatible_csv(directory: Path) -> Path:
    candidates = sorted(
        directory.glob("*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.reader(stream)
                header = next(reader, [])
        except (OSError, csv.Error):
            continue
        if "leaf_ablation_schema_version" in header:
            return path
    raise RuntimeError(f"No leaf-ablation CSV found in {directory}")


def load_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if "leaf_ablation_schema_version" not in fieldnames:
        raise RuntimeError(
            "CSV has no leaf-ablation columns. Rebuild and record with "
            "publish_leaf_ablation_data:=true."
        )
    compatible = [
        row
        for row in rows
        if finite_float(row.get("leaf_ablation_schema_version")) == 1.0
    ]
    if not compatible:
        raise RuntimeError(
            "CSV contains the columns but no schema-v1 samples. Confirm that "
            "/rmp_leaf_ablation is being published."
        )
    active_values = [finite_float(row.get("rmp_active")) for row in compatible]
    if any(value == 1.0 for value in active_values):
        compatible = [
            row
            for row, active in zip(compatible, active_values)
            if active == 1.0
        ]
    return fieldnames, compatible


@dataclass
class LeafJointStats:
    active_rows: int = 0
    saturated_samples: int = 0
    relief_count: int = 0
    top_cause_count: int = 0
    aligned_deltas: List[float] = field(default_factory=list)
    absolute_deltas: List[float] = field(default_factory=list)
    excess_reductions: List[float] = field(default_factory=list)


@dataclass
class JointSummary:
    saturation_count: int = 0
    max_abs_raw: float = 0.0
    max_excess: float = 0.0
    top_leaf: str = "none"
    top_leaf_count: int = 0


@dataclass
class Analysis:
    path: Path
    rows: int
    duration_s: float
    saturation_rows: int
    longest_saturation_s: float
    minimum_clip_cosine: float
    median_clip_cosine: float
    joint_summaries: List[JointSummary]
    leaf_stats: Dict[str, List[LeafJointStats]]
    timeline: List[Tuple[float, float, float]]


def row_time(row: Dict[str, str], fallback: float) -> float:
    return finite_float(row.get("time_ros_s")) or fallback


def analyze(path: Path, rows: List[Dict[str, str]]) -> Analysis:
    times = [row_time(row, float(index)) for index, row in enumerate(rows)]
    start_time = times[0]
    relative_times = [value - start_time for value in times]
    duration_s = max(relative_times[-1], 0.0)
    positive_steps = [
        relative_times[index] - relative_times[index - 1]
        for index in range(1, len(relative_times))
        if relative_times[index] > relative_times[index - 1]
    ]
    nominal_dt = statistics.median(positive_steps) if positive_steps else 0.01
    interval_gap = max(3.0 * nominal_dt, 0.05)

    joint_summaries = [JointSummary() for _ in range(6)]
    leaf_stats = {
        leaf: [LeafJointStats() for _ in range(6)] for leaf in LEAF_GROUPS
    }
    clip_cosines: List[float] = []
    saturation_rows = 0
    current_start: Optional[float] = None
    previous_saturated_time: Optional[float] = None
    longest_saturation_s = 0.0
    timeline: List[Tuple[float, float, float]] = []

    for relative_time, row in zip(relative_times, rows):
        limit = abs(finite_float(row.get("leaf_ablation_max_joint_accel_rad_s2")) or 0.0)
        raw = [
            finite_float(row.get(f"leaf_ablation_raw_qdd_{joint + 1}_rad_s2")) or 0.0
            for joint in range(6)
        ]
        saturated = [abs(value) > limit + 1e-9 for value in raw]
        any_saturated = any(saturated)
        max_abs_raw = max(abs(value) for value in raw)
        timeline.append((relative_time, max_abs_raw, limit))

        cosine = finite_float(row.get("leaf_ablation_clip_direction_cosine"))
        if cosine is not None:
            clip_cosines.append(cosine)

        if any_saturated:
            saturation_rows += 1
            if (
                current_start is None
                or previous_saturated_time is None
                or relative_time - previous_saturated_time > interval_gap
            ):
                current_start = relative_time
            previous_saturated_time = relative_time
            longest_saturation_s = max(
                longest_saturation_s,
                relative_time - current_start + nominal_dt,
            )
        else:
            current_start = None
            previous_saturated_time = None

        per_joint_leaf_reduction: List[Dict[str, float]] = [dict() for _ in range(6)]
        for leaf in LEAF_GROUPS:
            prefix = f"leaf_ablation_{leaf}"
            active = (finite_float(row.get(f"{prefix}_active")) or 0.0) > 0.5
            for joint in range(6):
                stats = leaf_stats[leaf][joint]
                if active:
                    stats.active_rows += 1
                delta = finite_float(
                    row.get(f"{prefix}_delta_raw_qdd_{joint + 1}_rad_s2")
                )
                if delta is None or not saturated[joint]:
                    continue
                without = raw[joint] - delta
                full_excess = max(abs(raw[joint]) - limit, 0.0)
                without_excess = max(abs(without) - limit, 0.0)
                reduction = full_excess - without_excess
                aligned = math.copysign(delta, raw[joint]) if raw[joint] != 0.0 else 0.0
                stats.saturated_samples += 1
                stats.aligned_deltas.append(aligned)
                stats.absolute_deltas.append(abs(delta))
                stats.excess_reductions.append(reduction)
                if abs(without) <= limit + 1e-9:
                    stats.relief_count += 1
                per_joint_leaf_reduction[joint][leaf] = reduction

        for joint in range(6):
            summary = joint_summaries[joint]
            summary.max_abs_raw = max(summary.max_abs_raw, abs(raw[joint]))
            summary.max_excess = max(summary.max_excess, abs(raw[joint]) - limit)
            if not saturated[joint]:
                continue
            summary.saturation_count += 1
            reductions = per_joint_leaf_reduction[joint]
            if not reductions:
                continue
            top_leaf, top_reduction = max(reductions.items(), key=lambda item: item[1])
            if top_reduction <= 1e-9:
                continue
            leaf_stats[top_leaf][joint].top_cause_count += 1

    for joint, summary in enumerate(joint_summaries):
        ranked = sorted(
            (
                (leaf, leaf_stats[leaf][joint].top_cause_count)
                for leaf in LEAF_GROUPS
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked and ranked[0][1] > 0:
            summary.top_leaf, summary.top_leaf_count = ranked[0]

    max_points = 1200
    if len(timeline) > max_points:
        stride = int(math.ceil(len(timeline) / max_points))
        timeline = timeline[::stride]

    return Analysis(
        path=path,
        rows=len(rows),
        duration_s=duration_s,
        saturation_rows=saturation_rows,
        longest_saturation_s=longest_saturation_s,
        minimum_clip_cosine=min(clip_cosines) if clip_cosines else 1.0,
        median_clip_cosine=statistics.median(clip_cosines) if clip_cosines else 1.0,
        joint_summaries=joint_summaries,
        leaf_stats=leaf_stats,
        timeline=timeline,
    )


def print_report(result: Analysis) -> None:
    percentage = 100.0 * result.saturation_rows / max(result.rows, 1)
    print(f"Input CSV: {result.path}")
    print(f"Rows: {result.rows}")
    print(f"Duration: {result.duration_s:.3f} s")
    print(
        f"Saturation rows: {result.saturation_rows}/{result.rows} "
        f"({percentage:.1f}%)"
    )
    print(f"Longest saturation interval: {result.longest_saturation_s:.3f} s")
    print(
        "Clip direction cosine: "
        f"median={result.median_clip_cosine:.6f} "
        f"min={result.minimum_clip_cosine:.6f}"
    )
    print("\nPer-joint attribution:")
    for index, summary in enumerate(result.joint_summaries, start=1):
        top_label = LEAF_LABELS.get(summary.top_leaf, summary.top_leaf)
        print(
            f"  J{index}: saturated={summary.saturation_count}/{result.rows}, "
            f"max|raw|={summary.max_abs_raw:.4g}, "
            f"max_excess={max(summary.max_excess, 0.0):.4g}, "
            f"top={top_label} ({summary.top_leaf_count})"
        )

    print("\nLeaf removal effects on saturated joint samples:")
    for leaf in LEAF_GROUPS:
        stats = result.leaf_stats[leaf]
        top_count = sum(item.top_cause_count for item in stats)
        relief_count = sum(item.relief_count for item in stats)
        samples = sum(item.saturated_samples for item in stats)
        reductions = [value for item in stats for value in item.excess_reductions]
        if samples == 0 or (top_count == 0 and relief_count == 0 and not any(reductions)):
            continue
        positive_reduction = sum(max(value, 0.0) for value in reductions)
        print(
            f"  {LEAF_LABELS[leaf]}: top={top_count}, relieved={relief_count}, "
            f"total_excess_reduction={positive_reduction:.4g}"
        )


def heat_color(value: float, maximum: float) -> str:
    if maximum <= 1e-12 or value <= 0.0:
        return "#f3f5f7"
    ratio = min(value / maximum, 1.0)
    red = 255
    green = int(246 - 130 * ratio)
    blue = int(240 - 160 * ratio)
    return f"rgb({red},{green},{blue})"


def timeline_svg(result: Analysis, width: int = 1000, height: int = 230) -> str:
    if not result.timeline:
        return ""
    margin_left = 52
    margin_right = 20
    margin_top = 18
    margin_bottom = 34
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_time = max(result.timeline[-1][0], 1e-9)
    max_value = max(max(raw, limit) for _, raw, limit in result.timeline)
    max_value = max(max_value, 1e-9)

    def point(time_s: float, value: float) -> str:
        x = margin_left + plot_width * time_s / max_time
        y = margin_top + plot_height * (1.0 - value / max_value)
        return f"{x:.2f},{y:.2f}"

    raw_points = " ".join(point(time_s, raw) for time_s, raw, _ in result.timeline)
    limit_points = " ".join(point(time_s, limit) for time_s, _, limit in result.timeline)
    return f"""
<svg viewBox="0 0 {width} {height}" role="img" aria-label="Maximum raw joint acceleration over time">
  <rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#fbfcfd" stroke="#d9dee5"/>
  <polyline points="{limit_points}" fill="none" stroke="#2979a8" stroke-width="2" stroke-dasharray="7 5"/>
  <polyline points="{raw_points}" fill="none" stroke="#c9362b" stroke-width="2"/>
  <text x="{margin_left}" y="{height - 8}" font-size="12">0 s</text>
  <text x="{width - margin_right}" y="{height - 8}" text-anchor="end" font-size="12">{max_time:.1f} s</text>
  <text x="8" y="{margin_top + 4}" font-size="12">{max_value:.1f}</text>
  <text x="8" y="{margin_top + plot_height}" font-size="12">0</text>
</svg>"""


def write_html(result: Analysis, output_path: Path) -> None:
    saturation_percentage = 100.0 * result.saturation_rows / max(result.rows, 1)
    matrix_values: Dict[Tuple[str, int], float] = {}
    for leaf in LEAF_GROUPS:
        for joint in range(6):
            matrix_values[(leaf, joint)] = sum(
                max(value, 0.0)
                for value in result.leaf_stats[leaf][joint].excess_reductions
            )
    maximum_matrix_value = max(matrix_values.values(), default=0.0)

    joint_rows = []
    for joint, summary in enumerate(result.joint_summaries, start=1):
        top_label = LEAF_LABELS.get(summary.top_leaf, summary.top_leaf)
        joint_rows.append(
            "<tr>"
            f"<td>J{joint}</td>"
            f"<td>{summary.saturation_count}</td>"
            f"<td>{100.0 * summary.saturation_count / max(result.rows, 1):.1f}%</td>"
            f"<td>{summary.max_abs_raw:.3f}</td>"
            f"<td>{max(summary.max_excess, 0.0):.3f}</td>"
            f"<td>{html.escape(top_label)}</td>"
            f"<td>{summary.top_leaf_count}</td>"
            "</tr>"
        )

    heat_rows = []
    for leaf in LEAF_GROUPS:
        active_rows = max(item.active_rows for item in result.leaf_stats[leaf])
        if active_rows == 0:
            continue
        cells = []
        for joint in range(6):
            value = matrix_values[(leaf, joint)]
            cells.append(
                f'<td style="background:{heat_color(value, maximum_matrix_value)}">'
                f"{value:.3f}</td>"
            )
        heat_rows.append(
            "<tr>"
            f"<td>{html.escape(LEAF_LABELS[leaf])}</td>"
            f"<td>{active_rows}</td>"
            + "".join(cells)
            + "</tr>"
        )

    document = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RMP leaf 관절 가속도 포화 분석</title>
<style>
body {{ margin:0; font-family:Arial, sans-serif; color:#1c252d; background:#fff; }}
header {{ background:#17242d; color:#fff; padding:24px max(24px, calc((100% - 1180px)/2)); }}
main {{ max-width:1180px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 8px; font-size:25px; letter-spacing:0; }}
h2 {{ margin:30px 0 10px; font-size:19px; letter-spacing:0; }}
.meta {{ color:#cbd6dc; overflow-wrap:anywhere; }}
.summary {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border:1px solid #d8dde2; }}
.summary div {{ padding:14px; border-right:1px solid #d8dde2; }}
.summary div:last-child {{ border-right:0; }}
.value {{ display:block; font-size:23px; font-weight:700; }}
.label {{ display:block; margin-top:4px; color:#57636d; font-size:13px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th,td {{ border:1px solid #d8dde2; padding:8px 9px; text-align:right; }}
th {{ background:#eef2f4; }}
th:first-child,td:first-child {{ text-align:left; }}
.legend {{ display:flex; gap:18px; font-size:13px; color:#57636d; }}
.swatch {{ display:inline-block; width:18px; height:3px; margin-right:6px; vertical-align:middle; }}
.note {{ padding:12px 14px; border-left:4px solid #2979a8; background:#eef6fa; line-height:1.55; }}
svg {{ width:100%; height:auto; }}
@media(max-width:760px) {{ .summary {{ grid-template-columns:1fr 1fr; }} .summary div {{ border-bottom:1px solid #d8dde2; }} main {{ padding:14px; overflow-x:auto; }} }}
</style>
</head>
<body>
<header>
  <h1>RMP leaf 관절 가속도 포화 분석</h1>
  <div class="meta">{html.escape(str(result.path))}</div>
</header>
<main>
  <section class="summary">
    <div><span class="value">{result.rows}</span><span class="label">분석 샘플</span></div>
    <div><span class="value">{saturation_percentage:.1f}%</span><span class="label">포화 행 비율</span></div>
    <div><span class="value">{result.longest_saturation_s:.3f}s</span><span class="label">최장 연속 포화</span></div>
    <div><span class="value">{result.minimum_clip_cosine:.4f}</span><span class="label">최소 방향 보존값</span></div>
  </section>

  <h2>시간에 따른 최대 raw 관절 가속도</h2>
  <div class="legend"><span><i class="swatch" style="background:#c9362b"></i>max |raw qdd|</span><span><i class="swatch" style="background:#2979a8"></i>관절 가속도 제한</span></div>
  {timeline_svg(result)}

  <h2>관절별 가장 큰 원인</h2>
  <table>
    <thead><tr><th>관절</th><th>포화 횟수</th><th>포화율</th><th>최대 |raw|</th><th>최대 초과량</th><th>가장 자주 초과를 줄인 leaf</th><th>선정 횟수</th></tr></thead>
    <tbody>{''.join(joint_rows)}</tbody>
  </table>

  <h2>Leaf 제거 시 포화 초과량 감소 합계</h2>
  <p class="note">값이 클수록 해당 leaf를 같은 상태에서 제거했을 때 그 관절의 제한 초과량이 많이 줄었다는 뜻입니다. 각 leaf의 qdd를 단순 분해한 값이 아니라, 전체 root metric을 다시 푼 국소 반사실 비교입니다.</p>
  <table>
    <thead><tr><th>Leaf</th><th>활성 행</th><th>J1</th><th>J2</th><th>J3</th><th>J4</th><th>J5</th><th>J6</th></tr></thead>
    <tbody>{''.join(heat_rows)}</tbody>
  </table>
</main>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze same-state per-leaf RMP acceleration saturation diagnostics."
    )
    parser.add_argument("csv_path", nargs="?", help="Trace CSV; latest compatible file by default.")
    parser.add_argument(
        "--directory",
        default="~/ros2_ws/log/rmpflow_trace",
        help="Directory searched when csv_path is omitted.",
    )
    parser.add_argument("--output", help="HTML output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.csv_path:
        input_path = Path(os.path.expanduser(args.csv_path)).resolve()
    else:
        input_path = latest_compatible_csv(
            Path(os.path.expanduser(args.directory)).resolve()
        )
    _, rows = load_rows(input_path)
    result = analyze(input_path, rows)
    print_report(result)

    output_path = (
        Path(os.path.expanduser(args.output)).resolve()
        if args.output
        else input_path.with_name(f"{input_path.stem}_leaf_ablation_report.html")
    )
    write_html(result, output_path)
    print(f"Saved report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
