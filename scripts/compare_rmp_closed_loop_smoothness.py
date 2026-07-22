#!/usr/bin/env python3

import argparse
import bisect
import csv
import html
import math
import os
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


JOINT_COUNT = 6
SETTLE_ERROR_M = 0.005
SETTLE_DWELL_S = 0.3
LATE_WINDOW_S = 2.0
COLORS = ["#266a8f", "#bd4b3b", "#39845a", "#76549a", "#b67b21"]


@dataclass
class Segment:
    time_s: List[float]
    goal_error_m: List[float]
    joint_speed_rad_s: List[float]
    joint_accel_rad_s2: List[float]
    initial_distance_m: float
    overshoot_m: float
    settling_time_s: Optional[float]
    late_error_p95_m: float
    late_speed_p95_rad_s: float
    late_accel_p95_rad_s2: float
    late_jerk_p95_rad_s3: float


@dataclass
class Case:
    label: str
    path: Path
    rows: int
    duration_s: float
    saturation_rate: float
    longest_saturation_s: float
    minimum_clip_cosine: float
    segments: List[Segment]


def parse_case(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must use LABEL=CSV_PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    path = Path(os.path.expanduser(raw_path.strip())).resolve()
    if not label:
        raise argparse.ArgumentTypeError("case label must not be empty")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"CSV does not exist: {path}")
    return label, path


def as_float(value: Optional[str]) -> float:
    try:
        return float(value) if value is not None else math.nan
    except (TypeError, ValueError):
        return math.nan


def norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def subtract(left: Sequence[float], right: Sequence[float]) -> List[float]:
    return [a - b for a, b in zip(left, right)]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def percentile(values: Sequence[float], percentage: float) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return math.nan
    position = (len(finite) - 1) * percentage / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    fraction = position - lower
    return finite[lower] * (1.0 - fraction) + finite[upper] * fraction


def settling_time(time_s: Sequence[float], error_m: Sequence[float]) -> Optional[float]:
    start: Optional[int] = None
    for index, error in enumerate(error_m):
        inside = error <= SETTLE_ERROR_M
        if inside and start is None:
            start = index
        if start is not None and ((not inside) or index == len(error_m) - 1):
            stop = index if not inside else index + 1
            if time_s[stop - 1] - time_s[start] >= SETTLE_DWELL_S:
                return time_s[start]
            start = None
    return None


def analyze_segment(
    time_s: Sequence[float],
    ee: Sequence[Sequence[float]],
    goal: Sequence[float],
    joint_speed: Sequence[float],
    joint_accel: Sequence[float],
    joint_accel_vectors: Sequence[Sequence[float]],
) -> Segment:
    local_time = [value - time_s[0] for value in time_s]
    start_to_goal = subtract(goal, ee[0])
    distance = norm(start_to_goal)
    direction = [value / distance for value in start_to_goal]
    progress = [dot(subtract(position, ee[0]), direction) for position in ee]
    error = [norm(subtract(position, goal)) for position in ee]
    overshoot = max(0.0, max(progress) - distance)

    late_start = max(0.0, local_time[-1] - LATE_WINDOW_S)
    late_indices = [index for index, value in enumerate(local_time) if value >= late_start]
    late_errors = [error[index] for index in late_indices]
    late_speeds = [joint_speed[index] for index in late_indices]
    late_accels = [joint_accel[index] for index in late_indices]
    late_jerk: List[float] = []
    for previous, current in zip(late_indices, late_indices[1:]):
        dt = local_time[current] - local_time[previous]
        if 0.005 <= dt <= 0.05:
            late_jerk.append(
                norm(subtract(joint_accel_vectors[current], joint_accel_vectors[previous]))
                / dt
            )

    return Segment(
        time_s=local_time,
        goal_error_m=error,
        joint_speed_rad_s=list(joint_speed),
        joint_accel_rad_s2=list(joint_accel),
        initial_distance_m=distance,
        overshoot_m=overshoot,
        settling_time_s=settling_time(local_time, error),
        late_error_p95_m=percentile(late_errors, 95.0),
        late_speed_p95_rad_s=percentile(late_speeds, 95.0),
        late_accel_p95_rad_s2=percentile(late_accels, 95.0),
        late_jerk_p95_rad_s3=percentile(late_jerk, 95.0),
    )


def load_case(label: str, path: Path) -> Case:
    required = {
        "time_ros_s",
        "controller_goal_x",
        "controller_goal_y",
        "controller_goal_z",
        "rmp_ee_x",
        "rmp_ee_y",
        "rmp_ee_z",
        "leaf_ablation_saturated_joint_count",
        "leaf_ablation_clip_direction_cosine",
    }
    required.update(f"joint_{joint}_vel_rad_s" for joint in range(1, JOINT_COUNT + 1))
    required.update(
        f"rmp_joint_accel_{joint}_rad_s2" for joint in range(1, JOINT_COUNT + 1)
    )

    samples: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {sorted(missing)}")
        for row in reader:
            time_value = as_float(row["time_ros_s"])
            goal = [as_float(row[f"controller_goal_{axis}"]) for axis in "xyz"]
            ee = [as_float(row[f"rmp_ee_{axis}"]) for axis in "xyz"]
            velocity = [
                as_float(row[f"joint_{joint}_vel_rad_s"])
                for joint in range(1, JOINT_COUNT + 1)
            ]
            acceleration = [
                as_float(row[f"rmp_joint_accel_{joint}_rad_s2"])
                for joint in range(1, JOINT_COUNT + 1)
            ]
            values = [time_value, *goal, *ee, *velocity, *acceleration]
            if not all(math.isfinite(value) for value in values):
                continue
            samples.append(
                {
                    "time": time_value,
                    "goal": goal,
                    "ee": ee,
                    "joint_speed": norm(velocity),
                    "joint_accel": norm(acceleration),
                    "joint_accel_vector": acceleration,
                    "saturated": as_float(row["leaf_ablation_saturated_joint_count"]) > 0.0,
                    "clip_cosine": as_float(row["leaf_ablation_clip_direction_cosine"]),
                }
            )

    if len(samples) < 2:
        raise ValueError(f"{path}: not enough complete rows")
    initial_time = float(samples[0]["time"])
    for sample in samples:
        sample["time"] = float(sample["time"]) - initial_time

    switches = [0]
    for index in range(1, len(samples)):
        previous = samples[index - 1]["goal"]
        current = samples[index]["goal"]
        if norm(subtract(current, previous)) > 0.05:
            switches.append(index)
    switches.append(len(samples))

    segments: List[Segment] = []
    for start, stop in zip(switches, switches[1:]):
        if stop - start < 10:
            continue
        segment_samples = samples[start:stop]
        distance = norm(
            subtract(segment_samples[0]["goal"], segment_samples[0]["ee"])
        )
        if distance <= 0.05:
            continue
        segments.append(
            analyze_segment(
                [float(sample["time"]) for sample in segment_samples],
                [sample["ee"] for sample in segment_samples],
                segment_samples[0]["goal"],
                [float(sample["joint_speed"]) for sample in segment_samples],
                [float(sample["joint_accel"]) for sample in segment_samples],
                [sample["joint_accel_vector"] for sample in segment_samples],
            )
        )

    saturated = [bool(sample["saturated"]) for sample in samples]
    longest = 0.0
    interval_start: Optional[int] = None
    for index, active in enumerate(saturated):
        if active and interval_start is None:
            interval_start = index
        if interval_start is not None and ((not active) or index == len(samples) - 1):
            interval_stop = index if not active else index
            longest = max(
                longest,
                float(samples[interval_stop]["time"])
                - float(samples[interval_start]["time"]),
            )
            interval_start = None
    clip_values = [
        float(sample["clip_cosine"])
        for sample in samples
        if sample["saturated"] and math.isfinite(float(sample["clip_cosine"]))
    ]
    return Case(
        label=label,
        path=path,
        rows=len(samples),
        duration_s=float(samples[-1]["time"]),
        saturation_rate=sum(saturated) / len(saturated),
        longest_saturation_s=longest,
        minimum_clip_cosine=min(clip_values) if clip_values else math.nan,
        segments=segments,
    )


def interpolate(time_s: Sequence[float], values: Sequence[float], query: float) -> float:
    if query <= time_s[0]:
        return values[0]
    if query >= time_s[-1]:
        return values[-1]
    upper = bisect.bisect_right(time_s, query)
    lower = upper - 1
    span = time_s[upper] - time_s[lower]
    fraction = 0.0 if span <= 0.0 else (query - time_s[lower]) / span
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def average_series(case: Case, field: str, duration_s: float) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for index in range(201):
        query = duration_s * index / 200.0
        values = []
        for segment in case.segments:
            if segment.time_s[-1] < query:
                continue
            values.append(interpolate(segment.time_s, getattr(segment, field), query))
        if values:
            points.append((query, statistics.fmean(values)))
    return points


def svg_plot(
    title: str,
    y_label: str,
    series: Sequence[Tuple[str, Sequence[Tuple[float, float]]]],
    log_y: bool = False,
) -> str:
    width, height = 1040, 380
    left, right, top, bottom = 76, 24, 38, 100
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_points = [point for _, points in series for point in points]
    x_max = max((point[0] for point in all_points), default=1.0)
    raw_y = [point[1] for point in all_points]
    if log_y:
        transformed = [math.log10(max(value, 1e-7)) for value in raw_y]
    else:
        transformed = raw_y
    y_min = min(transformed, default=0.0)
    y_max = max(transformed, default=1.0)
    if not log_y:
        y_min = 0.0
    if y_max - y_min < 1e-12:
        y_max = y_min + 1.0

    def sx(value: float) -> float:
        return left + plot_width * value / max(x_max, 1e-9)

    def sy(value: float) -> float:
        transformed_value = math.log10(max(value, 1e-7)) if log_y else value
        return top + plot_height * (y_max - transformed_value) / (y_max - y_min)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="22" class="plot-title">{html.escape(title)}</text>',
    ]
    for index in range(6):
        fraction = index / 5.0
        x = left + plot_width * fraction
        y = top + plot_height * fraction
        x_value = x_max * fraction
        y_value = y_max - (y_max - y_min) * fraction
        y_text = f"{10.0 ** y_value:.1e}" if log_y else f"{y_value:.3g}"
        parts.extend(
            [
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" class="grid"/>',
                f'<text x="{x:.1f}" y="{top + plot_height + 22}" class="tick" text-anchor="middle">{x_value:.1f}</text>',
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>',
                f'<text x="{left - 9}" y="{y + 4:.1f}" class="tick" text-anchor="end">{y_text}</text>',
            ]
        )
    parts.append(
        f'<text x="{left + plot_width / 2:.1f}" y="{top + plot_height + 47}" class="axis" text-anchor="middle">segment time (s)</text>'
    )
    parts.append(
        f'<text x="17" y="{top + plot_height / 2:.1f}" class="axis" text-anchor="middle" transform="rotate(-90 17 {top + plot_height / 2:.1f})">{html.escape(y_label)}</text>'
    )
    for index, (label, points) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        coordinates = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        legend_x = left + index * 230
        parts.extend(
            [
                f'<line x1="{legend_x}" y1="{height - 23}" x2="{legend_x + 24}" y2="{height - 23}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 30}" y="{height - 19}" class="legend">{html.escape(label)}</text>',
            ]
        )
    parts.append("</svg>")
    return "".join(parts)


def mean_optional(values: Sequence[Optional[float]]) -> float:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return statistics.fmean(finite) if finite else math.nan


def mean_field(segments: Sequence[Segment], field: str) -> float:
    values = [float(getattr(segment, field)) for segment in segments]
    return statistics.fmean(values) if values else math.nan


def format_value(value: float, digits: int = 3) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.{digits}f}"


def write_html(cases: Sequence[Case], output_path: Path) -> None:
    duration = min(
        (segment.time_s[-1] for case in cases for segment in case.segments),
        default=1.0,
    )
    goal_series = [
        (case.label, average_series(case, "goal_error_m", duration)) for case in cases
    ]
    speed_series = [
        (case.label, average_series(case, "joint_speed_rad_s", duration))
        for case in cases
    ]
    accel_series = [
        (case.label, average_series(case, "joint_accel_rad_s2", duration))
        for case in cases
    ]
    rows = []
    for case in cases:
        settle = mean_optional([segment.settling_time_s for segment in case.segments])
        overshoot = max((segment.overshoot_m for segment in case.segments), default=math.nan)
        rows.append(
            "<tr>"
            f"<td>{html.escape(case.label)}</td>"
            f"<td>{len(case.segments)}</td>"
            f"<td>{format_value(settle)}</td>"
            f"<td>{format_value(overshoot * 1000.0)}</td>"
            f"<td>{format_value(mean_field(case.segments, 'late_error_p95_m') * 1000.0, 4)}</td>"
            f"<td>{format_value(mean_field(case.segments, 'late_speed_p95_rad_s'), 6)}</td>"
            f"<td>{format_value(mean_field(case.segments, 'late_accel_p95_rad_s2'), 6)}</td>"
            f"<td>{format_value(mean_field(case.segments, 'late_jerk_p95_rad_s3'), 6)}</td>"
            f"<td>{100.0 * case.saturation_rate:.2f}%</td>"
            f"<td>{case.longest_saturation_s:.3f}</td>"
            f"<td>{format_value(case.minimum_clip_cosine, 6)}</td>"
            "</tr>"
        )
    sources = "".join(
        f"<li><strong>{html.escape(case.label)}</strong>: {html.escape(str(case.path))}</li>"
        for case in cases
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RMP closed-loop smoothness comparison</title>
<style>
body {{ margin:0; font-family:Arial,sans-serif; color:#1b252c; background:#fff; }}
header {{ padding:24px max(22px,calc((100% - 1180px)/2)); color:#fff; background:#18272f; }}
main {{ max-width:1180px; margin:auto; padding:24px; }}
h1 {{ margin:0 0 7px; font-size:25px; letter-spacing:0; }}
h2 {{ margin:28px 0 10px; font-size:18px; letter-spacing:0; }}
.meta {{ color:#c8d3d9; }}
.table-wrap {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ border:1px solid #d5dce0; padding:8px; text-align:right; white-space:nowrap; }}
th {{ background:#eef2f4; }}
th:first-child,td:first-child {{ text-align:left; }}
.note {{ padding:11px 13px; background:#eef6f9; border-left:4px solid #266a8f; line-height:1.5; }}
.plot {{ border:1px solid #d5dce0; margin:12px 0; overflow-x:auto; }}
svg {{ display:block; width:100%; min-width:760px; }}
.grid {{ stroke:#dce2e5; stroke-width:1; }}
.tick,.legend {{ fill:#53616a; font-size:11px; }}
.axis {{ fill:#37444c; font-size:12px; }}
.plot-title {{ fill:#1b252c; font-size:15px; font-weight:700; }}
.sources {{ color:#58656e; font-size:13px; line-height:1.6; overflow-wrap:anywhere; }}
</style>
</head>
<body>
<header><h1>RMP closed-loop smoothness comparison</h1><div class="meta">Fixed goals, no obstacle, moving segments only</div></header>
<main>
<p class="note">Settling requires goal error at or below 5 mm for at least 0.3 s. Late metrics use the final 2 s of each moving segment. Overshoot is displacement beyond the goal plane along the initial start-to-goal direction.</p>
<h2>Summary</h2>
<div class="table-wrap"><table>
<thead><tr><th>Case</th><th>Segments</th><th>Settle mean (s)</th><th>Max overshoot (mm)</th><th>Late error p95 (mm)</th><th>Late |qd| p95</th><th>Late |qdd| p95</th><th>Late jerk p95</th><th>Raw saturation</th><th>Longest (s)</th><th>Min clip cosine</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table></div>
<h2>Goal convergence</h2><div class="plot">{svg_plot('Mean moving-segment goal error', 'goal error (m, log scale)', goal_series, True)}</div>
<h2>Joint speed</h2><div class="plot">{svg_plot('Mean joint velocity norm', '|qd| (rad/s)', speed_series)}</div>
<h2>Joint acceleration</h2><div class="plot">{svg_plot('Mean commanded joint acceleration norm', '|qdd| (rad/s^2)', accel_series)}</div>
<h2>Inputs</h2><ul class="sources">{sources}</ul>
</main>
</body>
</html>"""
    output_path.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare settling, overshoot, jitter, and qdd limiting across traces."
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        type=parse_case,
        metavar="LABEL=CSV",
    )
    parser.add_argument(
        "--output",
        default="~/ros2_ws/log/rmpflow_trace/rmp_closed_loop_smoothness.html",
    )
    args = parser.parse_args()
    if len(args.case) < 2:
        parser.error("provide at least two --case arguments")

    cases = [load_case(label, path) for label, path in args.case]
    for case in cases:
        if not case.segments:
            raise ValueError(f"{case.path}: no moving goal segments found")
    output_path = Path(os.path.expanduser(args.output)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_html(cases, output_path)
    print(f"Saved comparison: {output_path}")
    for case in cases:
        settle = mean_optional([segment.settling_time_s for segment in case.segments])
        overshoot = max(segment.overshoot_m for segment in case.segments)
        print(
            f"  {case.label}: settle={format_value(settle)}s, "
            f"overshoot={format_value(overshoot * 1000.0)}mm, "
            f"late_qd_p95={format_value(mean_field(case.segments, 'late_speed_p95_rad_s'), 6)}, "
            f"saturation={100.0 * case.saturation_rate:.2f}%, "
            f"clip_cos_min={format_value(case.minimum_clip_cosine, 6)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
