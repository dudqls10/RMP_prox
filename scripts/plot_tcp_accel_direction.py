#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_DATA_DIR = Path("~/ros2_ws/data/rmp_datasets").expanduser()
EE_COLUMNS = ("ee_pose_x", "ee_pose_y", "ee_pose_z")
GOAL_COLUMNS = ("goal_pose_x", "goal_pose_y", "goal_pose_z")
TCP_ACCEL_COLUMN_SETS = (
    ("rmp_tcp_accel_x_m_s2", "rmp_tcp_accel_y_m_s2", "rmp_tcp_accel_z_m_s2"),
    ("rmp_tcp_accel_x", "rmp_tcp_accel_y", "rmp_tcp_accel_z"),
)
TCP_ACCEL_DIR_COLUMNS = (
    "rmp_tcp_accel_dir_x",
    "rmp_tcp_accel_dir_y",
    "rmp_tcp_accel_dir_z",
)
COLOR_CYCLE = (
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#17becf",
    "#7f7f7f",
)


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


def resolve_input_paths(csv_paths: Sequence[str], latest_dir: str) -> List[Path]:
    if csv_paths:
        paths: List[Path] = []
        for csv_path in csv_paths:
            path = Path(csv_path).expanduser()
            if not path.exists():
                raise FileNotFoundError(f"CSV file not found: {path}")
            paths.append(path)
        return paths
    return [latest_csv_in_directory(Path(latest_dir).expanduser())]


def make_labels(paths: Sequence[Path], labels: Optional[Sequence[str]]) -> List[str]:
    if labels:
        return [str(label) for label in labels]
    return [path.stem for path in paths]


def validate_labels(labels: Optional[Sequence[str]], path_count: int) -> None:
    if labels and len(labels) != path_count:
        raise ValueError(f"--labels count ({len(labels)}) must match CSV count ({path_count}).")


def find_tcp_accel_columns(fieldnames: Sequence[str]) -> Tuple[str, str, str]:
    field_set = set(fieldnames)
    for columns in TCP_ACCEL_COLUMN_SETS:
        if all(column in field_set for column in columns):
            return columns
    raise ValueError(
        "Missing TCP accel columns. Expected one of: "
        + ", ".join("/".join(columns) for columns in TCP_ACCEL_COLUMN_SETS)
    )


def vector_norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def normalize(values: Sequence[float]) -> Tuple[float, float, float]:
    norm = vector_norm(values)
    if norm <= 1e-12:
        return (float("nan"), float("nan"), float("nan"))
    return (values[0] / norm, values[1] / norm, values[2] / norm)


def sample_indices(count: int, max_count: int) -> List[int]:
    if count <= max_count:
        return list(range(count))
    step = (count - 1) / float(max_count - 1)
    return sorted({round(index * step) for index in range(max_count)})


def load_tcp_accel_direction(
    path: Path,
    min_norm: float,
    max_arrows: int,
) -> Tuple[
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[Tuple[float, float, float]],
]:
    fieldnames, rows = read_csv_rows(path)
    missing_ee = [column for column in EE_COLUMNS if column not in fieldnames]
    if missing_ee:
        raise ValueError(f"Missing EE pose columns in {path}: {missing_ee}")

    accel_columns = find_tcp_accel_columns(fieldnames)
    has_direction_columns = all(column in fieldnames for column in TCP_ACCEL_DIR_COLUMNS)
    has_goal = all(column in fieldnames for column in GOAL_COLUMNS)
    has_time = "timestamp_unix" in fieldnames
    first_time: Optional[float] = None

    traj_x: List[float] = []
    traj_y: List[float] = []
    traj_z: List[float] = []
    arrow_records: List[Tuple[float, float, float, float, float, float, float]] = []
    goals: List[Tuple[float, float, float]] = []

    for index, row in enumerate(rows):
        point = [parse_float(row, key) for key in EE_COLUMNS]
        if not all(math.isfinite(value) for value in point):
            continue

        traj_x.append(point[0])
        traj_y.append(point[1])
        traj_z.append(point[2])

        accel = [parse_float(row, key) for key in accel_columns]
        if not all(math.isfinite(value) for value in accel):
            continue
        norm = vector_norm(accel)
        if norm < min_norm:
            continue

        if has_direction_columns:
            direction = [parse_float(row, key) for key in TCP_ACCEL_DIR_COLUMNS]
            if not all(math.isfinite(value) for value in direction):
                direction = list(normalize(accel))
        else:
            direction = list(normalize(accel))
        if not all(math.isfinite(value) for value in direction):
            continue

        if has_time:
            stamp = parse_float(row, "timestamp_unix")
            if not math.isfinite(stamp):
                stamp = float(index)
            if first_time is None:
                first_time = stamp
            relative_time = stamp - first_time
        else:
            relative_time = float(index)

        arrow_records.append((
            point[0],
            point[1],
            point[2],
            direction[0],
            direction[1],
            direction[2],
            relative_time,
        ))

        if has_goal:
            goal = tuple(parse_float(row, key) for key in GOAL_COLUMNS)
            if all(math.isfinite(value) for value in goal):
                if not goals or distance_3d(goal, goals[-1]) > 1e-4:
                    goals.append(goal)

    if not traj_x:
        raise ValueError(f"No finite EE pose samples found in: {path}")
    if not arrow_records:
        raise ValueError(
            f"No TCP accel direction samples found in {path}. "
            "The file may have been recorded before rmp_tcp_accel columns were added."
        )

    selected = [arrow_records[index] for index in sample_indices(len(arrow_records), max_arrows)]
    arrow_x = [record[0] for record in selected]
    arrow_y = [record[1] for record in selected]
    arrow_z = [record[2] for record in selected]
    dir_x = [record[3] for record in selected]
    dir_y = [record[4] for record in selected]
    dir_z = [record[5] for record in selected]
    times = [record[6] for record in selected]
    return traj_x, traj_y, traj_z, arrow_x, arrow_y, arrow_z, dir_x, dir_y, dir_z, times, goals


def distance_3d(first: Sequence[float], second: Sequence[float]) -> float:
    return math.sqrt(sum((float(first[i]) - float(second[i])) ** 2 for i in range(3)))


def set_axes_equal(ax, xs: Sequence[float], ys: Sequence[float], zs: Sequence[float]) -> None:
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-3)
    half = max_range * 0.5
    x_mid = (x_min + x_max) * 0.5
    y_mid = (y_min + y_max) * 0.5
    z_mid = (z_min + z_max) * 0.5

    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)


def default_output_path(paths: Sequence[Path]) -> Path:
    if len(paths) == 1:
        return paths[0].with_name(f"{paths[0].stem}_tcp_accel_direction_3d.png")
    return paths[-1].with_name("tcp_accel_direction_compare_3d.png")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot TCP acceleration direction arrows from recorded RMP CSV files."
    )
    parser.add_argument("csv_paths", nargs="*", help="Recorder CSV path(s).")
    parser.add_argument(
        "--latest-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory used when no CSV path is provided.",
    )
    parser.add_argument("--labels", nargs="+", help="Labels matching the CSV paths.")
    parser.add_argument("--save", help="Output PNG path.")
    parser.add_argument("--show", action="store_true", help="Show a matplotlib window.")
    parser.add_argument(
        "--max-arrows",
        type=int,
        default=160,
        help="Maximum direction arrows per CSV.",
    )
    parser.add_argument(
        "--min-norm",
        type=float,
        default=0.001,
        help="Ignore TCP accel samples below this norm.",
    )
    parser.add_argument(
        "--arrow-length",
        type=float,
        default=0.06,
        help="Fixed arrow length in meters for unit direction vectors.",
    )
    parser.add_argument("--title", help="Optional plot title.")
    args = parser.parse_args()

    if args.max_arrows < 2:
        parser.error("--max-arrows must be at least 2.")
    if args.arrow_length <= 0.0:
        parser.error("--arrow-length must be greater than 0.")

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    paths = resolve_input_paths(args.csv_paths, args.latest_dir)
    try:
        validate_labels(args.labels, len(paths))
    except ValueError as exc:
        parser.error(str(exc))
    labels = make_labels(paths, args.labels)
    output_path = Path(args.save).expanduser() if args.save else default_output_path(paths)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    all_xs: List[float] = []
    all_ys: List[float] = []
    all_zs: List[float] = []
    arrow_count = 0

    for index, (path, label) in enumerate(zip(paths, labels)):
        try:
            (
                traj_x,
                traj_y,
                traj_z,
                arrow_x,
                arrow_y,
                arrow_z,
                dir_x,
                dir_y,
                dir_z,
                _times,
                goals,
            ) = load_tcp_accel_direction(path, args.min_norm, args.max_arrows)
        except ValueError as exc:
            parser.error(str(exc))
        color = COLOR_CYCLE[index % len(COLOR_CYCLE)]
        ax.plot(traj_x, traj_y, traj_z, color=color, linewidth=1.2, alpha=0.75, label=label)
        ax.quiver(
            arrow_x,
            arrow_y,
            arrow_z,
            dir_x,
            dir_y,
            dir_z,
            length=args.arrow_length,
            normalize=True,
            color=color,
            linewidth=0.8,
            alpha=0.75,
        )
        ax.scatter(traj_x[0], traj_y[0], traj_z[0], color=color, s=60, marker="o", edgecolors="black")
        ax.scatter(traj_x[-1], traj_y[-1], traj_z[-1], color=color, s=60, marker="X", edgecolors="black")
        if goals:
            ax.scatter(
                [goal[0] for goal in goals],
                [goal[1] for goal in goals],
                [goal[2] for goal in goals],
                color=color,
                s=60,
                marker="*",
                edgecolors="black",
                label=f"{label} goal",
            )
            all_xs.extend(goal[0] for goal in goals)
            all_ys.extend(goal[1] for goal in goals)
            all_zs.extend(goal[2] for goal in goals)

        all_xs.extend(traj_x)
        all_ys.extend(traj_y)
        all_zs.extend(traj_z)
        arrow_count += len(arrow_x)

    set_axes_equal(ax, all_xs, all_ys, all_zs)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(args.title or "TCP acceleration direction")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)

    for index, path in enumerate(paths, start=1):
        print(f"Input CSV {index}: {path}")
    print(f"Direction arrows plotted: {arrow_count}")
    print(f"Saved plot: {output_path}")

    if args.show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
