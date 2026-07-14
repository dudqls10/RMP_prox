#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = REPO_ROOT.parents[1]
DEFAULT_BASE_PARAMS = REPO_ROOT / "config" / "params.yaml"
DEFAULT_OUTPUT_DIR = Path("~/ros2_ws/log/rmpflow_trace/tuning").expanduser()


SCENARIOS = [
    {
        "name": "single_tof6_1_R",
        "timeout_s": 18,
        "launch_args": {
            "fake_scenario": "approach_retreat",
            "fake_sensor_name": "tof6_1_R",
            "fake_range_m": "0.05",
            "fake_inactive_range_m": "0.90",
            "fake_start_s": "1.0",
            "fake_duration_s": "10.0",
            "fake_period_s": "10.0",
            "fake_hold_s": "6.0",
        },
    },
    {
        "name": "random_pair",
        "timeout_s": 24,
        "launch_args": {
            "fake_scenario": "random",
            "fake_random_count": "8",
            "fake_random_sensor_count": "2",
            "fake_random_seed": "31",
            "fake_random_allow_repeats": "true",
            "fake_period_s": "2.0",
            "fake_hold_s": "1.3",
            "fake_range_m": "0.05",
            "fake_inactive_range_m": "0.90",
        },
    },
]


CANDIDATES = [
    {
        "name": "baseline",
        "params": {
            "tangent_escape_rmp_metric_scalar": 1000.0,
            "tangent_escape_rmp_damping_gain": 1.0,
            "tangent_escape_rmp_position_gain": 20.0,
            "tangent_escape_rmp_max_accel": 1.0,
            "tangent_escape_rmp_softmax_beta": 5.0,
            "tangent_escape_rmp_stuck_metric_boost": 1.5,
            "tangent_escape_rmp_stuck_accel_boost": 1.2,
        },
    },
    {
        "name": "soft_250_10_4_04",
        "params": {
            "tangent_escape_rmp_metric_scalar": 250.0,
            "tangent_escape_rmp_damping_gain": 4.0,
            "tangent_escape_rmp_position_gain": 10.0,
            "tangent_escape_rmp_max_accel": 0.4,
            "tangent_escape_rmp_softmax_beta": 3.0,
            "tangent_escape_rmp_stuck_metric_boost": 1.2,
            "tangent_escape_rmp_stuck_accel_boost": 1.05,
        },
    },
    {
        "name": "soft_300_12_4_05",
        "params": {
            "tangent_escape_rmp_metric_scalar": 300.0,
            "tangent_escape_rmp_damping_gain": 4.0,
            "tangent_escape_rmp_position_gain": 12.0,
            "tangent_escape_rmp_max_accel": 0.5,
            "tangent_escape_rmp_softmax_beta": 4.0,
            "tangent_escape_rmp_stuck_metric_boost": 1.2,
            "tangent_escape_rmp_stuck_accel_boost": 1.05,
        },
    },
    {
        "name": "damped_400_10_6_05",
        "params": {
            "tangent_escape_rmp_metric_scalar": 400.0,
            "tangent_escape_rmp_damping_gain": 6.0,
            "tangent_escape_rmp_position_gain": 10.0,
            "tangent_escape_rmp_max_accel": 0.5,
            "tangent_escape_rmp_softmax_beta": 3.0,
            "tangent_escape_rmp_stuck_metric_boost": 1.15,
            "tangent_escape_rmp_stuck_accel_boost": 1.0,
        },
    },
    {
        "name": "metric_150_16_4_06",
        "params": {
            "tangent_escape_rmp_metric_scalar": 150.0,
            "tangent_escape_rmp_damping_gain": 4.0,
            "tangent_escape_rmp_position_gain": 16.0,
            "tangent_escape_rmp_max_accel": 0.6,
            "tangent_escape_rmp_softmax_beta": 4.0,
            "tangent_escape_rmp_stuck_metric_boost": 1.2,
            "tangent_escape_rmp_stuck_accel_boost": 1.05,
        },
    },
    {
        "name": "stable_120_18_5_05",
        "params": {
            "tangent_escape_rmp_metric_scalar": 120.0,
            "tangent_escape_rmp_damping_gain": 5.0,
            "tangent_escape_rmp_position_gain": 18.0,
            "tangent_escape_rmp_max_accel": 0.5,
            "tangent_escape_rmp_softmax_beta": 4.0,
            "tangent_escape_rmp_stuck_metric_boost": 1.1,
            "tangent_escape_rmp_stuck_accel_boost": 1.0,
        },
    },
    {
        "name": "stable_100_20_6_05",
        "params": {
            "tangent_escape_rmp_metric_scalar": 100.0,
            "tangent_escape_rmp_damping_gain": 6.0,
            "tangent_escape_rmp_position_gain": 20.0,
            "tangent_escape_rmp_max_accel": 0.5,
            "tangent_escape_rmp_softmax_beta": 4.0,
            "tangent_escape_rmp_stuck_metric_boost": 1.1,
            "tangent_escape_rmp_stuck_accel_boost": 1.0,
        },
    },
    {
        "name": "balanced_200_14_5_05",
        "params": {
            "tangent_escape_rmp_metric_scalar": 200.0,
            "tangent_escape_rmp_damping_gain": 5.0,
            "tangent_escape_rmp_position_gain": 14.0,
            "tangent_escape_rmp_max_accel": 0.5,
            "tangent_escape_rmp_softmax_beta": 3.5,
            "tangent_escape_rmp_stuck_metric_boost": 1.15,
            "tangent_escape_rmp_stuck_accel_boost": 1.0,
        },
    },
    {
        "name": "low_80_16_6_04",
        "params": {
            "tangent_escape_rmp_metric_scalar": 80.0,
            "tangent_escape_rmp_damping_gain": 6.0,
            "tangent_escape_rmp_position_gain": 16.0,
            "tangent_escape_rmp_max_accel": 0.4,
            "tangent_escape_rmp_softmax_beta": 3.5,
            "tangent_escape_rmp_stuck_metric_boost": 1.1,
            "tangent_escape_rmp_stuck_accel_boost": 1.0,
        },
    },
    {
        "name": "low_80_16_8_03",
        "params": {
            "tangent_escape_rmp_metric_scalar": 80.0,
            "tangent_escape_rmp_damping_gain": 8.0,
            "tangent_escape_rmp_position_gain": 16.0,
            "tangent_escape_rmp_max_accel": 0.3,
            "tangent_escape_rmp_softmax_beta": 3.5,
            "tangent_escape_rmp_stuck_metric_boost": 1.1,
            "tangent_escape_rmp_stuck_accel_boost": 1.0,
        },
    },
    {
        "name": "low_60_14_8_03",
        "params": {
            "tangent_escape_rmp_metric_scalar": 60.0,
            "tangent_escape_rmp_damping_gain": 8.0,
            "tangent_escape_rmp_position_gain": 14.0,
            "tangent_escape_rmp_max_accel": 0.3,
            "tangent_escape_rmp_softmax_beta": 3.0,
            "tangent_escape_rmp_stuck_metric_boost": 1.1,
            "tangent_escape_rmp_stuck_accel_boost": 1.0,
        },
    },
]


def parse_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def finite_values(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}g}"


def norm(values: Sequence[float]) -> float:
    if any(not math.isfinite(value) for value in values):
        return float("nan")
    return math.sqrt(sum(value * value for value in values))


def qdd_vector(row: Dict[str, str]) -> List[float]:
    return [parse_float(row, f"rmp_joint_accel_{index}_rad_s2") for index in range(1, 7)]


def qdd_norm(row: Dict[str, str]) -> float:
    value = parse_float(row, "rmp_joint_accel_norm")
    if math.isfinite(value):
        return value
    return norm(qdd_vector(row))


def row_time(row: Dict[str, str]) -> float:
    return parse_float(row, "time_ros_s")


def max_jerk(rows: Sequence[Dict[str, str]]) -> float:
    best = float("nan")
    previous_qdd: Optional[List[float]] = None
    previous_time = float("nan")
    for row in rows:
        current_qdd = qdd_vector(row)
        current_time = row_time(row)
        if not all(math.isfinite(value) for value in current_qdd) or not math.isfinite(current_time):
            continue
        if previous_qdd is not None and math.isfinite(previous_time):
            dt = current_time - previous_time
            if dt > 1e-6:
                value = norm([
                    (current_qdd[joint] - previous_qdd[joint]) / dt
                    for joint in range(6)
                ])
                best = value if not math.isfinite(best) else max(best, value)
        previous_qdd = current_qdd
        previous_time = current_time
    return best


def load_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def update_params(base_text: str, values: Dict[str, float]) -> str:
    result = base_text
    for key, value in values.items():
        pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*).*$", re.MULTILINE)
        replacement = rf"\g<1>{value}"
        result, count = pattern.subn(replacement, result)
        if count == 0:
            raise ValueError(f"Parameter not found in base params: {key}")
    return result


def active_rows(rows: Sequence[Dict[str, str]]) -> List[Tuple[int, Dict[str, str]]]:
    return [
        (index, row) for index, row in enumerate(rows)
        if parse_float(row, "tangent_escape_rmp_active") >= 0.5
    ]


def lyapunov(row: Dict[str, str], position_gain: float) -> float:
    metric = parse_float(row, "tangent_escape_rmp_effective_metric_scalar")
    velocity = parse_float(row, "tangent_escape_rmp_scalar_velocity_m_s")
    error = parse_float(row, "tangent_escape_rmp_scalar_error_m")
    if not all(math.isfinite(value) for value in (metric, velocity, error)):
        return float("nan")
    return 0.5 * metric * velocity * velocity + 0.5 * metric * position_gain * error * error


def positive_vdot_ratio(
    indexed_active_rows: Sequence[Tuple[int, Dict[str, str]]],
    position_gain: float,
) -> Tuple[float, int, int]:
    positive = 0
    pairs = 0
    for previous, current in zip(indexed_active_rows, indexed_active_rows[1:]):
        previous_index, previous_row = previous
        current_index, current_row = current
        if current_index != previous_index + 1:
            continue
        same_cp = (
            round(parse_float(previous_row, "tangent_escape_rmp_control_point_index")) ==
            round(parse_float(current_row, "tangent_escape_rmp_control_point_index"))
        )
        same_slot = (
            round(parse_float(previous_row, "tangent_escape_rmp_selected_candidate_index")) ==
            round(parse_float(current_row, "tangent_escape_rmp_selected_candidate_index"))
        )
        same_mode = (
            round(parse_float(previous_row, "tangent_escape_rmp_supervisor_mode_id")) ==
            round(parse_float(current_row, "tangent_escape_rmp_supervisor_mode_id"))
        )
        if not (same_cp and same_slot and same_mode):
            continue
        t0 = row_time(previous_row)
        t1 = row_time(current_row)
        if not (math.isfinite(t0) and math.isfinite(t1) and t1 > t0):
            continue
        v0 = lyapunov(previous_row, position_gain)
        v1 = lyapunov(current_row, position_gain)
        if not (math.isfinite(v0) and math.isfinite(v1)):
            continue
        pairs += 1
        if (v1 - v0) / (t1 - t0) > 1e-9:
            positive += 1
    ratio = positive / pairs if pairs else float("nan")
    return ratio, positive, pairs


def summarize_rows(rows: Sequence[Dict[str, str]], params: Dict[str, float]) -> Dict[str, float]:
    indexed_active = active_rows(rows)
    active = [row for _, row in indexed_active]
    scalar_s = finite_values(parse_float(row, "tangent_escape_rmp_scalar_s_m") for row in active)
    scalar_targets = finite_values(parse_float(row, "tangent_escape_rmp_scalar_target_m") for row in active)
    progress_ratio = float("nan")
    if scalar_s and scalar_targets and abs(scalar_targets[0]) > 1e-9:
        progress_ratio = max(scalar_s) / scalar_targets[0]

    clearances = finite_values(parse_float(row, "tangent_escape_rmp_clearance_m") for row in active)
    normal_dots = finite_values(
        abs(parse_float(row, "tangent_escape_rmp_normal_dot_tangent")) for row in active
    )
    metrics = finite_values(parse_float(row, "tangent_escape_rmp_effective_metric_scalar") for row in active)
    branch_weight_sums = finite_values(parse_float(row, "tangent_escape_rmp_branch_weight_sum") for row in active)
    qdd_norms = finite_values(qdd_norm(row) for row in rows)
    active_qdd_norms = finite_values(qdd_norm(row) for row in active)
    threshold = 9.5
    sat_active = 0
    for row in active:
        qdd = qdd_vector(row)
        if all(math.isfinite(value) for value in qdd) and max(abs(value) for value in qdd) >= threshold:
            sat_active += 1

    branch_keys: List[Tuple[int, int]] = []
    for row in active:
        cp = parse_float(row, "tangent_escape_rmp_control_point_index")
        slot = parse_float(row, "tangent_escape_rmp_selected_candidate_index")
        if math.isfinite(cp) and math.isfinite(slot):
            branch_keys.append((int(round(cp)), int(round(slot))))
    branch_switches = sum(1 for lhs, rhs in zip(branch_keys, branch_keys[1:]) if lhs != rhs)
    positive_ratio, positive_count, vdot_pairs = positive_vdot_ratio(
        indexed_active,
        params["tangent_escape_rmp_position_gain"],
    )
    closure_pass = (
        bool(active) and
        metrics and min(metrics) >= -1e-9 and
        (not normal_dots or max(normal_dots) <= 0.05) and
        (not branch_weight_sums or max(abs(value - 1.0) for value in branch_weight_sums) <= 0.02)
    )

    min_clearance = min(clearances) if clearances else float("nan")
    score = 0.0
    score += sat_active * 20.0
    if math.isfinite(max_jerk(rows)):
        score += max(0.0, max_jerk(rows) - 100.0) * 0.1
    if math.isfinite(positive_ratio):
        score += positive_ratio * 25.0
    if math.isfinite(min_clearance):
        score += max(0.0, 0.02 - min_clearance) * 1000.0
    score += branch_switches * 1.5
    if math.isfinite(progress_ratio):
        score -= min(progress_ratio, 0.25) * 25.0
    if not closure_pass:
        score += 10000.0

    return {
        "rows": float(len(rows)),
        "active_rows": float(len(active)),
        "progress_ratio": progress_ratio,
        "min_clearance": min_clearance,
        "max_qdd_norm": max(active_qdd_norms) if active_qdd_norms else float("nan"),
        "sat_active": float(sat_active),
        "max_jerk": max_jerk(rows),
        "positive_vdot_ratio": positive_ratio,
        "positive_vdot_count": float(positive_count),
        "vdot_pairs": float(vdot_pairs),
        "branch_switches": float(branch_switches),
        "max_normal_dot": max(normal_dots) if normal_dots else float("nan"),
        "min_metric": min(metrics) if metrics else float("nan"),
        "closure_pass": 1.0 if closure_pass else 0.0,
        "score": score,
    }


def run_launch(
    candidate_name: str,
    scenario: Dict[str, object],
    params_path: Path,
    params: Dict[str, float],
    log_dir: Path,
    domain_id: int,
) -> Optional[Path]:
    launch_args = {
        "params_file": str(params_path),
        "use_rviz": "false",
        "start_fake_proximity": "true",
        "enable_tangent_escape_filter": "false",
        "enable_tangent_escape_rmp": "true",
        "tangent_escape_rmp_leaf_mode": "softmax_gds",
        "tangent_escape_rmp_position_gain": str(params["tangent_escape_rmp_position_gain"]),
        "tangent_escape_rmp_softmax_beta": str(params["tangent_escape_rmp_softmax_beta"]),
        "use_rmpflow_trace_logger": "true",
        "rmpflow_trace_log_rate": "120",
        "rmpflow_trace_log_directory": str(log_dir),
        "rmpflow_trace_console_summary": "false",
    }
    launch_args.update(scenario["launch_args"])  # type: ignore[arg-type]
    arg_string = " ".join(f"{key}:={value}" for key, value in launch_args.items())
    timeout_s = int(scenario["timeout_s"])  # type: ignore[arg-type]
    command = (
        "source /opt/ros/humble/setup.bash && "
        "source install/setup.bash && "
        f"ROS_DOMAIN_ID={domain_id} timeout --signal=SIGINT {timeout_s}s "
        "ros2 launch rb10_rmpflow_rviz rb10_rmpflow_fake_proximity.launch.py "
        f"{arg_string}"
    )
    print(f"RUN {candidate_name}/{scenario['name']} domain={domain_id}", flush=True)
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=str(WS_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s + 20,
    )
    if result.returncode not in (0, 124):
        print(result.stdout[-4000:])
        print(f"WARN launch returned {result.returncode} for {candidate_name}/{scenario['name']}")
    csvs = sorted(log_dir.glob("rmpflow_trace_*.csv"), key=lambda path: path.stat().st_mtime)
    return csvs[-1] if csvs else None


def write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    fieldnames = [
        "candidate",
        "scenario",
        "score",
        "closure_pass",
        "active_rows",
        "progress_ratio",
        "min_clearance",
        "sat_active",
        "max_qdd_norm",
        "max_jerk",
        "positive_vdot_ratio",
        "branch_switches",
        "max_normal_dot",
        "metric",
        "damping",
        "position_gain",
        "max_accel",
        "softmax_beta",
        "csv",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def aggregate(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    by_candidate: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        by_candidate.setdefault(str(row["candidate"]), []).append(row)
    result: List[Dict[str, object]] = []
    for candidate, items in by_candidate.items():
        scores = finite_values(float(item["score"]) for item in items)
        sat = sum(float(item["sat_active"]) for item in items if math.isfinite(float(item["sat_active"])))
        closure = all(float(item["closure_pass"]) >= 0.5 for item in items)
        min_clearances = finite_values(float(item["min_clearance"]) for item in items)
        progress = finite_values(float(item["progress_ratio"]) for item in items)
        jerks = finite_values(float(item["max_jerk"]) for item in items)
        result.append({
            "candidate": candidate,
            "mean_score": sum(scores) / len(scores) if scores else float("nan"),
            "max_score": max(scores) if scores else float("nan"),
            "closure_all_pass": closure,
            "total_sat_active": sat,
            "min_clearance": min(min_clearances) if min_clearances else float("nan"),
            "mean_progress_ratio": sum(progress) / len(progress) if progress else float("nan"),
            "max_jerk": max(jerks) if jerks else float("nan"),
        })
    result.sort(key=lambda item: float(item["mean_score"]))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fake Tangent Escape RMP parameter tuning.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--base-params", default=str(DEFAULT_BASE_PARAMS))
    parser.add_argument("--max-candidates", type=int, default=0, help="Limit number of candidates for quick runs.")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated candidate names to run. Empty means all candidates.",
    )
    parser.add_argument("--start-domain-id", type=int, default=60)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_text = Path(args.base_params).expanduser().read_text(encoding="utf-8")

    candidates = CANDIDATES
    if args.only:
        selected = {name.strip() for name in args.only.split(",") if name.strip()}
        candidates = [candidate for candidate in candidates if str(candidate["name"]) in selected]
        missing = selected - {str(candidate["name"]) for candidate in candidates}
        if missing:
            raise ValueError("Unknown candidates in --only: " + ", ".join(sorted(missing)))
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    result_rows: List[Dict[str, object]] = []
    domain_id = args.start_domain_id
    for candidate in candidates:
        params = candidate["params"]  # type: ignore[assignment]
        candidate_dir = output_dir / str(candidate["name"])
        candidate_dir.mkdir(parents=True, exist_ok=True)
        params_path = candidate_dir / "params.yaml"
        params_path.write_text(update_params(base_text, params), encoding="utf-8")  # type: ignore[arg-type]
        for scenario in SCENARIOS:
            scenario_dir = candidate_dir / str(scenario["name"])
            scenario_dir.mkdir(parents=True, exist_ok=True)
            csv_path = run_launch(
                str(candidate["name"]),
                scenario,
                params_path,
                params,  # type: ignore[arg-type]
                scenario_dir,
                domain_id,
            )
            domain_id += 1
            if csv_path is None:
                result_rows.append({
                    "candidate": candidate["name"],
                    "scenario": scenario["name"],
                    "score": 100000.0,
                    "closure_pass": 0.0,
                    "csv": "",
                })
                continue
            metrics = summarize_rows(load_rows(csv_path), params)  # type: ignore[arg-type]
            row: Dict[str, object] = {
                "candidate": candidate["name"],
                "scenario": scenario["name"],
                "csv": str(csv_path),
                "metric": params["tangent_escape_rmp_metric_scalar"],  # type: ignore[index]
                "damping": params["tangent_escape_rmp_damping_gain"],  # type: ignore[index]
                "position_gain": params["tangent_escape_rmp_position_gain"],  # type: ignore[index]
                "max_accel": params["tangent_escape_rmp_max_accel"],  # type: ignore[index]
                "softmax_beta": params["tangent_escape_rmp_softmax_beta"],  # type: ignore[index]
            }
            row.update(metrics)
            result_rows.append(row)
            print(
                f"  score={fmt(float(metrics['score']))} closure={int(metrics['closure_pass'])} "
                f"active={int(metrics['active_rows'])} sat={int(metrics['sat_active'])} "
                f"jerk={fmt(float(metrics['max_jerk']))} progress={fmt(float(metrics['progress_ratio']))} "
                f"clearance={fmt(float(metrics['min_clearance']))}",
                flush=True,
            )
            time.sleep(0.5)

    summary_csv = output_dir / "tuning_results.csv"
    write_summary_csv(summary_csv, result_rows)
    aggregated = aggregate(result_rows)
    aggregate_csv = output_dir / "tuning_aggregate.csv"
    with aggregate_csv.open("w", newline="") as stream:
        fieldnames = [
            "candidate",
            "mean_score",
            "max_score",
            "closure_all_pass",
            "total_sat_active",
            "min_clearance",
            "mean_progress_ratio",
            "max_jerk",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in aggregated:
            writer.writerow(row)

    print("\nAggregate ranking:")
    for row in aggregated:
        print(
            f"  {row['candidate']}: mean_score={fmt(float(row['mean_score']))}, "
            f"sat={fmt(float(row['total_sat_active']))}, "
            f"min_clearance={fmt(float(row['min_clearance']))}, "
            f"progress={fmt(float(row['mean_progress_ratio']))}, "
            f"max_jerk={fmt(float(row['max_jerk']))}, "
            f"closure={row['closure_all_pass']}"
        )
    print(f"\nSaved: {summary_csv}")
    print(f"Saved: {aggregate_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
