#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_DATA_DIR = Path("~/ros2_ws/data/rmp_datasets").expanduser()


def is_finite_number(text: object) -> bool:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value)


def parse_float(row: Dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if not is_finite_number(value):
        return float("nan")
    return float(value)


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    raw_lines: List[str] = []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            raw_lines.append(line)

    reader = csv.DictReader(raw_lines)
    if not reader.fieldnames:
        raise ValueError(f"CSV file has no header row: {path}")
    rows = [row for row in reader]
    if not rows:
        raise ValueError(f"CSV file has no data rows: {path}")
    return list(reader.fieldnames), rows


def latest_csv_in_directory(directory: Path) -> Path:
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")

    candidates = sorted(directory.glob("*.csv"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in: {directory}")
    return candidates[-1]


def resolve_input_path(csv_path: Optional[str], latest_dir: str) -> Path:
    if csv_path:
        path = Path(csv_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        return path
    return latest_csv_in_directory(Path(latest_dir).expanduser())


def vector(row: Dict[str, str], prefix: str) -> Tuple[float, float, float]:
    return (
        parse_float(row, f"{prefix}_x"),
        parse_float(row, f"{prefix}_y"),
        parse_float(row, f"{prefix}_z"),
    )


def goal_points(rows: Sequence[Dict[str, str]]) -> List[Tuple[float, float, float]]:
    goals: List[Tuple[float, float, float]] = []
    for row in rows:
        if all(key in row for key in ("goal_pose_x", "goal_pose_y", "goal_pose_z")):
            goal = vector(row, "goal_pose")
        else:
            goal = vector(row, "goal")
        if not all(math.isfinite(value) for value in goal):
            continue
        if not goals or distance_3d(goal, goals[-1]) > 1e-4:
            goals.append(goal)
    return goals


def vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def normalize(values: Sequence[float]) -> Tuple[float, float, float]:
    norm = vector_norm(values)
    if norm <= 1e-12:
        return (float("nan"), float("nan"), float("nan"))
    return (values[0] / norm, values[1] / norm, values[2] / norm)


def dot(first: Sequence[float], second: Sequence[float]) -> float:
    return sum(first[index] * second[index] for index in range(3))


def distance_3d(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((float(first[index]) - float(second[index])) ** 2 for index in range(3)))


def angle_degrees(first: Sequence[float], second: Sequence[float]) -> float:
    first_norm = normalize(first)
    second_norm = normalize(second)
    if not all(math.isfinite(value) for value in (*first_norm, *second_norm)):
        return float("nan")
    cosine = max(-1.0, min(1.0, dot(first_norm, second_norm)))
    return math.degrees(math.acos(cosine))


def sample_indices(count: int, max_count: int) -> List[int]:
    if count <= max_count:
        return list(range(count))
    step = (count - 1) / float(max_count - 1)
    return sorted({round(index * step) for index in range(max_count)})


def set_axes_equal(ax, points: Sequence[Sequence[float]]) -> None:
    finite_points = [
        point for point in points
        if len(point) >= 3 and all(math.isfinite(value) for value in point[:3])
    ]
    if not finite_points:
        return

    xs = [point[0] for point in finite_points]
    ys = [point[1] for point in finite_points]
    zs = [point[2] for point in finite_points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)
    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-3)
    half = max_range * 0.55
    x_mid = (x_min + x_max) * 0.5
    y_mid = (y_min + y_max) * 0.5
    z_mid = (z_min + z_max) * 0.5
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)


def require_columns(fieldnames: Sequence[str]) -> None:
    required = (
        "timestamp_unix",
        "tangent_escape_active",
        "tangent_escape_cp_x",
        "tangent_escape_cp_y",
        "tangent_escape_cp_z",
        "tangent_escape_obstacle_x",
        "tangent_escape_obstacle_y",
        "tangent_escape_obstacle_z",
        "tangent_escape_normal_x",
        "tangent_escape_normal_y",
        "tangent_escape_normal_z",
        "tangent_escape_tangent_x",
        "tangent_escape_tangent_y",
        "tangent_escape_tangent_z",
        "tangent_escape_raw_cp_accel_x_m_s2",
        "tangent_escape_raw_cp_accel_y_m_s2",
        "tangent_escape_raw_cp_accel_z_m_s2",
        "tangent_escape_filtered_cp_accel_x_m_s2",
        "tangent_escape_filtered_cp_accel_y_m_s2",
        "tangent_escape_filtered_cp_accel_z_m_s2",
        "tangent_escape_clearance",
        "tangent_escape_activation",
    )
    missing = [column for column in required if column not in fieldnames]
    if missing:
        raise ValueError(
            "Missing tangent escape columns. Record again after rebuilding/running the updated "
            f"controller and recorder. Missing: {missing}"
        )


def active_rows(rows: Sequence[Dict[str, str]], max_age: float) -> List[Dict[str, str]]:
    selected: List[Dict[str, str]] = []
    for row in rows:
        active = parse_float(row, "tangent_escape_active")
        age = parse_float(row, "tangent_escape_age_s")
        if not math.isfinite(active) or active < 0.5:
            continue
        if math.isfinite(age) and age > max_age:
            continue
        cp = vector(row, "tangent_escape_cp")
        obstacle = vector(row, "tangent_escape_obstacle")
        tangent = vector(row, "tangent_escape_tangent")
        raw = (
            parse_float(row, "tangent_escape_raw_cp_accel_x_m_s2"),
            parse_float(row, "tangent_escape_raw_cp_accel_y_m_s2"),
            parse_float(row, "tangent_escape_raw_cp_accel_z_m_s2"),
        )
        filtered = (
            parse_float(row, "tangent_escape_filtered_cp_accel_x_m_s2"),
            parse_float(row, "tangent_escape_filtered_cp_accel_y_m_s2"),
            parse_float(row, "tangent_escape_filtered_cp_accel_z_m_s2"),
        )
        if all(math.isfinite(value) for value in (*cp, *obstacle, *tangent, *raw, *filtered)):
            selected.append(row)
    return selected


def quiver_from(
    ax,
    origin: Sequence[float],
    direction: Sequence[float],
    length: float,
    color: str,
    label: Optional[str],
    linewidth: float = 1.4,
    alpha: float = 0.9,
) -> Optional[Tuple[float, float, float]]:
    unit = normalize(direction)
    if not all(math.isfinite(value) for value in unit):
        return None
    ax.quiver(
        [origin[0]],
        [origin[1]],
        [origin[2]],
        [unit[0]],
        [unit[1]],
        [unit[2]],
        length=length,
        normalize=True,
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
    )
    return (
        origin[0] + unit[0] * length,
        origin[1] + unit[1] * length,
        origin[2] + unit[2] * length,
    )


def plot_3d(
    path: Path,
    rows: Sequence[Dict[str, str]],
    output_path: Path,
    max_samples: int,
    arrow_length: float,
) -> int:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9.5, 7.5))
    ax = fig.add_subplot(111, projection="3d")

    sampled = [rows[index] for index in sample_indices(len(rows), max_samples)]
    cp_points: List[Tuple[float, float, float]] = []
    obstacle_points: List[Tuple[float, float, float]] = []
    axis_points: List[Sequence[float]] = []

    for row in rows:
        cp_points.append(vector(row, "tangent_escape_cp"))
        obstacle_points.append(vector(row, "tangent_escape_obstacle"))
    goals = goal_points(rows)

    ax.plot(
        [point[0] for point in cp_points],
        [point[1] for point in cp_points],
        [point[2] for point in cp_points],
        color="#222222",
        linewidth=1.0,
        alpha=0.55,
        label="active cp path",
    )
    ax.scatter(
        [point[0] for point in obstacle_points],
        [point[1] for point in obstacle_points],
        [point[2] for point in obstacle_points],
        color="#ff8c00",
        s=34,
        alpha=0.7,
        label="obstacle",
    )
    if goals:
        ax.scatter(
            [point[0] for point in goals],
            [point[1] for point in goals],
            [point[2] for point in goals],
            color="#ffd700",
            edgecolors="black",
            marker="*",
            s=170,
            linewidth=0.8,
            label="goal",
        )
        axis_points.extend(goals)

    labels_used = set()
    for row in sampled:
        cp = vector(row, "tangent_escape_cp")
        obstacle = vector(row, "tangent_escape_obstacle")
        normal = vector(row, "tangent_escape_normal")
        tangent = vector(row, "tangent_escape_tangent")
        raw = (
            parse_float(row, "tangent_escape_raw_cp_accel_x_m_s2"),
            parse_float(row, "tangent_escape_raw_cp_accel_y_m_s2"),
            parse_float(row, "tangent_escape_raw_cp_accel_z_m_s2"),
        )
        filtered = (
            parse_float(row, "tangent_escape_filtered_cp_accel_x_m_s2"),
            parse_float(row, "tangent_escape_filtered_cp_accel_y_m_s2"),
            parse_float(row, "tangent_escape_filtered_cp_accel_z_m_s2"),
        )

        ax.scatter(
            [cp[0]],
            [cp[1]],
            [cp[2]],
            color="white",
            edgecolors="black",
            s=45,
            linewidth=0.8,
            label="active control point" if "cp" not in labels_used else None,
        )
        labels_used.add("cp")
        ax.plot(
            [obstacle[0], cp[0]],
            [obstacle[1], cp[1]],
            [obstacle[2], cp[2]],
            color="#888888",
            linewidth=0.7,
            alpha=0.35,
        )
        axis_points.extend([cp, obstacle])
        for direction, color, key, label, scale in (
            (normal, "#d62728", "normal", "obstacle normal", 0.8),
            (tangent, "#2ca02c", "tangent", "selected escape tangent", 1.0),
            (raw, "#1f77b4", "raw", "raw cp accel", 1.0),
            (filtered, "#17becf", "filtered", "filtered cp accel", 1.0),
        ):
            end = quiver_from(
                ax,
                cp,
                direction,
                arrow_length * scale,
                color,
                label if key not in labels_used else None,
            )
            labels_used.add(key)
            if end is not None:
                axis_points.append(end)

    set_axes_equal(ax, axis_points)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"Tangent escape filter vectors\n{path.name}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return len(sampled)


def plot_timeseries(
    path: Path,
    rows: Sequence[Dict[str, str]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    times: List[float] = []
    clearances: List[float] = []
    activations: List[float] = []
    raw_dot_tangent: List[float] = []
    filtered_dot_tangent: List[float] = []
    raw_angle_tangent: List[float] = []
    filtered_angle_tangent: List[float] = []
    raw_norms: List[float] = []
    filtered_norms: List[float] = []
    first_time: Optional[float] = None

    for index, row in enumerate(rows):
        stamp = parse_float(row, "timestamp_unix")
        if not math.isfinite(stamp):
            stamp = float(index)
        if first_time is None:
            first_time = stamp
        times.append(stamp - first_time)
        clearances.append(parse_float(row, "tangent_escape_clearance"))
        activations.append(parse_float(row, "tangent_escape_activation"))
        tangent = vector(row, "tangent_escape_tangent")
        raw = (
            parse_float(row, "tangent_escape_raw_cp_accel_x_m_s2"),
            parse_float(row, "tangent_escape_raw_cp_accel_y_m_s2"),
            parse_float(row, "tangent_escape_raw_cp_accel_z_m_s2"),
        )
        filtered = (
            parse_float(row, "tangent_escape_filtered_cp_accel_x_m_s2"),
            parse_float(row, "tangent_escape_filtered_cp_accel_y_m_s2"),
            parse_float(row, "tangent_escape_filtered_cp_accel_z_m_s2"),
        )
        raw_dot_tangent.append(dot(raw, normalize(tangent)))
        filtered_dot_tangent.append(dot(filtered, normalize(tangent)))
        raw_angle_tangent.append(angle_degrees(raw, tangent))
        filtered_angle_tangent.append(angle_degrees(filtered, tangent))
        raw_norms.append(vector_norm(raw))
        filtered_norms.append(vector_norm(filtered))

    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(times, clearances, color="#ff8c00", label="clearance")
    axes[0].set_ylabel("clearance [m]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(times, activations, color="#2ca02c", label="activation")
    axes[1].set_ylabel("activation")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(times, raw_dot_tangent, color="#1f77b4", label="raw dot tangent")
    axes[2].plot(times, filtered_dot_tangent, color="#17becf", label="filtered dot tangent")
    axes[2].set_ylabel("m/s^2")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    axes[3].plot(times, raw_angle_tangent, color="#1f77b4", label="raw angle to tangent")
    axes[3].plot(times, filtered_angle_tangent, color="#17becf", label="filtered angle to tangent")
    axes[3].plot(times, raw_norms, color="#1f77b4", linestyle="--", alpha=0.45, label="raw accel norm")
    axes[3].plot(
        times,
        filtered_norms,
        color="#17becf",
        linestyle="--",
        alpha=0.45,
        label="filtered accel norm",
    )
    axes[3].set_ylabel("deg / m/s^2")
    axes[3].set_xlabel("active-filter time [s]")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="best")

    fig.suptitle(f"Tangent escape filter change\n{path.name}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot tangent escape filter debug vectors from an RMP recorder CSV."
    )
    parser.add_argument("csv_path", nargs="?", help="Recorder CSV path. Defaults to latest CSV.")
    parser.add_argument(
        "--latest-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory used when no CSV path is provided.",
    )
    parser.add_argument("--save-3d", help="Output PNG path for the 3D vector plot.")
    parser.add_argument("--save-timeseries", help="Output PNG path for the time-series plot.")
    parser.add_argument("--show", action="store_true", help="Show matplotlib windows.")
    parser.add_argument("--max-samples", type=int, default=16, help="Max 3D vector samples.")
    parser.add_argument("--arrow-length", type=float, default=0.08, help="3D arrow length in meters.")
    parser.add_argument(
        "--max-age",
        type=float,
        default=0.2,
        help="Ignore cached debug samples older than this many seconds.",
    )
    args = parser.parse_args()

    if args.max_samples < 1:
        parser.error("--max-samples must be positive.")
    if args.arrow_length <= 0.0:
        parser.error("--arrow-length must be greater than 0.")

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    path = resolve_input_path(args.csv_path, args.latest_dir)
    fieldnames, rows = read_csv_rows(path)
    require_columns(fieldnames)
    selected_rows = active_rows(rows, args.max_age)
    if not selected_rows:
        raise ValueError(f"No active tangent escape rows found in: {path}")

    output_3d = (
        Path(args.save_3d).expanduser()
        if args.save_3d
        else path.with_name(f"{path.stem}_tangent_escape_3d.png")
    )
    output_timeseries = (
        Path(args.save_timeseries).expanduser()
        if args.save_timeseries
        else path.with_name(f"{path.stem}_tangent_escape_timeseries.png")
    )
    output_3d.parent.mkdir(parents=True, exist_ok=True)
    output_timeseries.parent.mkdir(parents=True, exist_ok=True)

    plotted_samples = plot_3d(path, selected_rows, output_3d, args.max_samples, args.arrow_length)
    plot_timeseries(path, selected_rows, output_timeseries)

    print(f"Input CSV: {path}")
    print(f"Active rows: {len(selected_rows)}")
    print(f"3D vector samples plotted: {plotted_samples}")
    print(f"Saved 3D plot: {output_3d}")
    print(f"Saved time-series plot: {output_timeseries}")

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
