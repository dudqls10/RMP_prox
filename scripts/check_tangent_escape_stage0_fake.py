#!/usr/bin/env python3
import argparse
import csv
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ALL_FAKE_SENSORS = [
    "tof6_1_L",
    "tof6_1_F",
    "tof6_1_R",
    "tof6_1_U",
    "tof_S",
    "tof_E",
    "tof_N",
    "tof_W",
    "tof3_1_S",
    "tof3_1_W",
    "tof3_1_N",
    "tof3_1_E",
    "tof2_1_E",
    "tof2_1_S",
    "tof2_1_W",
    "tof2_1_N",
    "tof2_E",
    "tof2_S",
    "tof2_W",
    "tof2_N",
]

DEFAULT_SENSORS = ALL_FAKE_SENSORS


def parse_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def finite_values(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def shell_command(workspace: Path, command: str) -> List[str]:
    return [
        "bash",
        "-lc",
        (
            "source /opt/ros/humble/setup.bash && "
            f"source {workspace / 'install' / 'setup.bash'} && "
            f"exec {command}"
        ),
    ]


def start_process(
    workspace: Path,
    command: str,
    env: Dict[str, str],
    log_path: Path,
) -> subprocess.Popen:
    handle = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        shell_command(workspace, command),
        cwd=str(workspace),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def stop_process(process: Optional[subprocess.Popen], timeout_s: float = 3.0) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def latest_csv(log_dir: Path) -> Path:
    candidates = sorted(log_dir.glob("rmpflow_trace_*.csv"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No rmpflow_trace CSV was written in {log_dir}")
    return candidates[-1]


def sensor_rows(csv_path: Path) -> Dict[str, List[Dict[str, str]]]:
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = set(reader.fieldnames or [])
        required = {
            "tangent_escape_geometry_active",
            "tangent_escape_geometry_sensor_name",
            "tangent_escape_geometry_sensor_obstacle_dot",
            "tangent_escape_geometry_collision_obstacle_dot",
            "tangent_escape_geometry_bias_sensor_dot",
            "tangent_escape_geometry_jacobian_frobenius_norm",
            "tangent_escape_geometry_obstacle_direction_jacobian_norm",
            "tangent_escape_geometry_tangent_bias_jacobian_norm",
        }
        missing = sorted(required - fieldnames)
        if missing:
            raise ValueError(
                "CSV does not contain Stage-0 geometry columns. Missing: " + ", ".join(missing)
            )
        rows: Dict[str, List[Dict[str, str]]] = {}
        for row in reader:
            if parse_float(row, "tangent_escape_geometry_active") < 0.5:
                continue
            sensor = row.get("tangent_escape_geometry_sensor_name", "")
            if not sensor:
                continue
            rows.setdefault(sensor, []).append(row)
    return rows


def check_sensor(sensor: str, rows: Sequence[Dict[str, str]]) -> Tuple[bool, str]:
    if len(rows) < 5:
        return False, f"{sensor}: only {len(rows)} active geometry rows"

    sensor_dots = finite_values(
        parse_float(row, "tangent_escape_geometry_sensor_obstacle_dot") for row in rows
    )
    collision_dots = finite_values(
        parse_float(row, "tangent_escape_geometry_collision_obstacle_dot") for row in rows
    )
    bias_sensor_dots = finite_values(
        abs(parse_float(row, "tangent_escape_geometry_bias_sensor_dot")) for row in rows
    )
    jacobian_norms = finite_values(
        parse_float(row, "tangent_escape_geometry_jacobian_frobenius_norm") for row in rows
    )
    obstacle_jacobian_norms = finite_values(
        parse_float(row, "tangent_escape_geometry_obstacle_direction_jacobian_norm")
        for row in rows
    )
    tangent_jacobian_norms = finite_values(
        parse_float(row, "tangent_escape_geometry_tangent_bias_jacobian_norm") for row in rows
    )

    failures = []
    if not sensor_dots or median(sensor_dots) < 0.85:
        failures.append(f"sensor_dot median {median(sensor_dots):.3f} < 0.85")
    if not collision_dots or median(collision_dots) > -0.95:
        failures.append(f"collision_dot median {median(collision_dots):.3f} > -0.95")
    if not bias_sensor_dots or median(bias_sensor_dots) > 0.10:
        failures.append(f"|bias_dot_sensor| median {median(bias_sensor_dots):.3f} > 0.10")
    if not jacobian_norms or median(jacobian_norms) <= 1e-4:
        failures.append(f"J frobenius median {median(jacobian_norms):.6f} <= 1e-4")
    if not obstacle_jacobian_norms or median(obstacle_jacobian_norms) <= 1e-5:
        failures.append(f"obstacle J row median {median(obstacle_jacobian_norms):.6f} <= 1e-5")
    if not tangent_jacobian_norms or median(tangent_jacobian_norms) <= 1e-5:
        failures.append(f"tangent J row median {median(tangent_jacobian_norms):.6f} <= 1e-5")

    summary = (
        f"{sensor}: rows={len(rows)} "
        f"sensor_dot_med={median(sensor_dots):.3f} "
        f"collision_dot_med={median(collision_dots):.3f} "
        f"|bias_sensor|_med={median(bias_sensor_dots):.3f} "
        f"J_med={median(jacobian_norms):.4f} "
        f"J_obst_med={median(obstacle_jacobian_norms):.4f} "
        f"J_tan_med={median(tangent_jacobian_norms):.4f}"
    )
    if failures:
        return False, summary + " :: " + "; ".join(failures)
    return True, summary


def wait_for_base(log_path: Path, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"base launch exited early. See {log_path}")
        if log_path.exists() and "C++ RMPflow controller started" in log_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ):
            return
        time.sleep(0.25)
    raise TimeoutError(f"base launch did not become ready within {timeout_s:.1f}s. See {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a headless fake-proximity Stage-0 geometry check."
    )
    parser.add_argument("--workspace", default="~/ros2_ws")
    parser.add_argument("--log-dir", default="")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--sensors", default=",".join(DEFAULT_SENSORS))
    parser.add_argument("--all", action="store_true", help="Check all fake-supported sensors.")
    parser.add_argument("--sensor-duration-s", type=float, default=6.0)
    parser.add_argument("--base-timeout-s", type=float, default=20.0)
    parser.add_argument("--trace-rate-hz", type=float, default=100.0)
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve()
    if not (workspace / "install" / "setup.bash").exists():
        raise SystemExit(f"Workspace is not built or setup file is missing: {workspace}")

    if args.domain_id > 0:
        domain_id = args.domain_id
    else:
        domain_id = 80 + (os.getpid() % 100)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir).expanduser() if args.log_dir else Path(
        tempfile.gettempdir()
    ) / f"rb10_stage0_fake_check_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log_dir = log_dir / "process_logs"
    run_log_dir.mkdir(parents=True, exist_ok=True)

    sensors = ALL_FAKE_SENSORS if args.all else [
        sensor.strip() for sensor in args.sensors.split(",") if sensor.strip()
    ]
    unknown = sorted(set(sensors) - set(ALL_FAKE_SENSORS))
    if unknown:
        raise SystemExit("Unknown sensors: " + ",".join(unknown))

    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(domain_id)

    base_cmd = (
        "ros2 launch rb10_rmpflow_rviz rb10_rmpflow_fake_proximity.launch.py "
        "use_rviz:=false "
        "start_fake_proximity:=false "
        "enable_tangent_escape_rmp:=false "
        "publish_tangent_escape_geometry_debug:=true "
        "proximity_surface_visualization:=false "
        "use_rmpflow_trace_logger:=true "
        f"rmpflow_trace_log_rate:={args.trace_rate_hz:.3f} "
        f"rmpflow_trace_log_directory:={log_dir}"
    )
    base_log = run_log_dir / "base_launch.log"
    base_process: Optional[subprocess.Popen] = None
    sensor_process: Optional[subprocess.Popen] = None
    try:
        print(f"ROS_DOMAIN_ID={domain_id}")
        print(f"log_dir={log_dir}")
        base_process = start_process(workspace, base_cmd, env, base_log)
        wait_for_base(base_log, base_process, args.base_timeout_s)

        for sensor in sensors:
            sensor_log = run_log_dir / f"fake_sensor_{sensor}.log"
            sensor_cmd = (
                "ros2 launch rb10_rmpflow_rviz fake_proximity_scenario.launch.py "
                f"fake_sensor_name:={sensor} "
                "fake_scenario:=approach_retreat "
                "fake_start_range_m:=0.50 "
                "fake_range_m:=0.05 "
                "fake_end_range_m:=0.50 "
                "fake_start_s:=0.0 "
                f"fake_duration_s:={args.sensor_duration_s:.3f} "
                "fake_publish_rate:=50.0"
            )
            print(f"inject {sensor}")
            sensor_process = start_process(workspace, sensor_cmd, env, sensor_log)
            time.sleep(args.sensor_duration_s + 1.0)
            stop_process(sensor_process)
            sensor_process = None
            time.sleep(0.5)
    finally:
        stop_process(sensor_process)
        stop_process(base_process)

    csv_path = latest_csv(log_dir)
    rows_by_sensor = sensor_rows(csv_path)
    print(f"csv={csv_path}")

    ok = True
    for sensor in sensors:
        sensor_ok, message = check_sensor(sensor, rows_by_sensor.get(sensor, []))
        ok = ok and sensor_ok
        print(("PASS " if sensor_ok else "FAIL ") + message)

    if ok:
        print("STAGE0_FAKE_CHECK: PASS")
        return 0
    print("STAGE0_FAKE_CHECK: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
