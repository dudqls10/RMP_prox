#!/usr/bin/env python3
import argparse
import html
import os
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
WS_ROOT = REPO_ROOT.parents[1]
DEFAULT_LOG_ROOT = Path("~/ros2_ws/log/rmpflow_trace").expanduser()

POSITION_GAIN = 16.0
DAMPING_GAIN = 4.0
MAX_ACCEL = 0.6

SCENARIOS: Sequence[Dict[str, object]] = (
    {
        "name": "normal_tof6_1_R",
        "timeout": 17,
        "profile": "normal",
        "args": {
            "fake_scenario": "approach_retreat",
            "fake_sensor_name": "tof6_1_R",
            "fake_range_m": "0.06",
            "fake_start_range_m": "0.50",
            "fake_end_range_m": "0.50",
            "fake_start_s": "1.0",
            "fake_duration_s": "10.0",
            "fake_hold_s": "4.0",
        },
    },
    {
        "name": "duplicate_tof_E",
        "timeout": 17,
        "profile": "normal",
        "expect_duplicate": True,
        "args": {
            "fake_scenario": "approach_retreat",
            "fake_sensor_name": "tof_E",
            "fake_range_m": "0.06",
            "fake_start_range_m": "0.50",
            "fake_end_range_m": "0.50",
            "fake_start_s": "1.0",
            "fake_duration_s": "10.0",
            "fake_hold_s": "4.0",
        },
    },
    {
        "name": "forced_stuck_tof_W",
        "timeout": 18,
        "profile": "forced_stuck",
        "args": {
            "fake_scenario": "approach_retreat",
            "fake_sensor_name": "tof_W",
            "fake_range_m": "0.06",
            "fake_start_range_m": "0.50",
            "fake_end_range_m": "0.50",
            "fake_start_s": "1.0",
            "fake_duration_s": "11.0",
            "fake_hold_s": "6.0",
            "tangent_escape_rmp_stuck_activation_threshold": "0.05",
            "tangent_escape_rmp_stuck_velocity_threshold": "10.0",
            "tangent_escape_rmp_stuck_progress_threshold": "10.0",
            "tangent_escape_rmp_stuck_time_threshold": "0.25",
            "tangent_escape_rmp_stuck_metric_boost": "1.10",
            "tangent_escape_rmp_stuck_accel_boost": "1.02",
            "tangent_escape_rmp_blocked_memory_update_duration": "0.70",
            "tangent_escape_rmp_blocked_memory_progress_threshold": "10.0",
            "tangent_escape_rmp_blocked_memory_clearance_improvement": "10.0",
        },
    },
    {
        "name": "random_sensor_pairs",
        "timeout": 24,
        "profile": "normal",
        "args": {
            "fake_scenario": "random_approach_retreat",
            "fake_active_sensor_names": "all",
            "fake_random_count": "5",
            "fake_random_sensor_count": "2",
            "fake_random_seed": "41",
            "fake_random_allow_repeats": "false",
            "fake_range_m": "0.06",
            "fake_start_range_m": "0.50",
            "fake_end_range_m": "0.50",
            "fake_start_s": "1.0",
            "fake_period_s": "3.2",
            "fake_hold_s": "2.4",
        },
    },
)


def run_command(
    command: str,
    cwd: Path,
    log_path: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["bash", "-lc", command],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    log_path.write_text(result.stdout, encoding="utf-8")
    return result


def latest_csv(directory: Path) -> Path | None:
    paths = sorted(directory.glob("rmpflow_trace_*.csv"), key=lambda path: path.stat().st_mtime)
    return paths[-1] if paths else None


def launch_scenario(
    scenario: Dict[str, object],
    output_dir: Path,
    domain_id: int,
) -> tuple[Path | None, subprocess.CompletedProcess[str]]:
    launch_args: Dict[str, str] = {
        "use_rviz": "false",
        "start_fake_proximity": "true",
        "enable_tangent_escape_filter": "false",
        "enable_tangent_escape_rmp": "true",
        "tangent_escape_rmp_leaf_mode": "stable_hybrid_gds",
        "tangent_escape_rmp_position_gain": str(POSITION_GAIN),
        "tangent_escape_rmp_damping_gain": str(DAMPING_GAIN),
        "tangent_escape_rmp_max_accel": str(MAX_ACCEL),
        "tangent_escape_rmp_supervisor_enabled": "true",
        "publish_tangent_escape_rmp_data": "true",
        "use_rmpflow_trace_logger": "true",
        "rmpflow_trace_log_rate": "100",
        "rmpflow_trace_log_directory": str(output_dir),
        "rmpflow_trace_console_summary": "false",
    }
    launch_args.update(scenario.get("args", {}))  # type: ignore[arg-type]
    rendered_args = " ".join(
        f"{key}:={shlex.quote(value)}" for key, value in launch_args.items()
    )
    timeout_s = int(scenario["timeout"])
    command = (
        "source /opt/ros/humble/setup.bash && "
        "source install/setup.bash && "
        f"ROS_DOMAIN_ID={domain_id} timeout --signal=SIGINT --kill-after=5s {timeout_s}s "
        "ros2 launch rb10_rmpflow_rviz rb10_rmpflow_fake_proximity.launch.py "
        f"{rendered_args}"
    )
    result = run_command(command, WS_ROOT, output_dir / "launch.log", timeout_s + 15)
    return latest_csv(output_dir), result


def validate_csv(
    csv_path: Path,
    scenario: Dict[str, object],
    output_dir: Path,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    report_path = output_dir / "validation.html"
    plot_path = output_dir / "lyapunov.html"
    profile = str(scenario.get("profile", "auto"))
    duplicate_arg = " --expect-duplicate-risk" if scenario.get("expect_duplicate") else ""
    validation_command = (
        f"python3 {shlex.quote(str(REPO_ROOT / 'scripts/check_tangent_escape_rmp_validation.py'))} "
        f"{shlex.quote(str(csv_path))} --save {shlex.quote(str(report_path))} "
        f"--position-gain {POSITION_GAIN} --damping-gain {DAMPING_GAIN} "
        f"--max-accel {MAX_ACCEL} --stage4-profile {shlex.quote(profile)} "
        f"--strict{duplicate_arg}"
    )
    validation = run_command(
        validation_command,
        REPO_ROOT,
        output_dir / "validation.log",
        45,
    )
    plot_command = (
        f"python3 {shlex.quote(str(REPO_ROOT / 'scripts/plot_tangent_escape_lyapunov.py'))} "
        f"{shlex.quote(str(csv_path))} --save {shlex.quote(str(plot_path))} "
        f"--position-gain {POSITION_GAIN} --damping-gain {DAMPING_GAIN} "
        f"--max-accel {MAX_ACCEL}"
    )
    run_command(plot_command, REPO_ROOT, output_dir / "plot.log", 45)
    return validation, report_path, plot_path


def parse_validation(log_path: Path) -> tuple[int, int, int]:
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    statuses = re.findall(r"^\s*\[(PASS|INFO|WARN|FAIL)\]", text, flags=re.MULTILINE)
    return statuses.count("PASS"), statuses.count("WARN"), statuses.count("FAIL")


def write_summary(path: Path, results: Sequence[Dict[str, object]]) -> None:
    failed = sum(1 for result in results if result["status"] == "FAIL")
    warned = sum(1 for result in results if result["status"] == "WARN")
    overall = "FAIL" if failed else ("WARN" if warned else "PASS")
    rows = []
    for result in results:
        scenario_dir = Path(str(result["directory"]))
        relative = scenario_dir.relative_to(path.parent)
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(result['name']))}</td>"
            f"<td><span class=\"status {str(result['status']).lower()}\">"
            f"{html.escape(str(result['status']))}</span></td>"
            f"<td>{result['passes']}</td><td>{result['warnings']}</td><td>{result['failures']}</td>"
            f"<td><a href=\"{html.escape(str(relative / 'validation.html'))}\">validation</a></td>"
            f"<td><a href=\"{html.escape(str(relative / 'lyapunov.html'))}\">energy plot</a></td>"
            f"<td><a href=\"{html.escape(str(relative / 'launch.log'))}\">launch log</a></td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Tangent Escape Stability Suite</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #18202a; }}
h1 {{ font-size: 24px; }} .summary {{ font-size: 20px; font-weight: 700; margin: 14px 0; }}
table {{ width: 100%; border-collapse: collapse; }} th, td {{ padding: 9px; border-bottom: 1px solid #dfe3e8; text-align: left; }}
th {{ background: #f4f6f8; }} .status {{ color: white; padding: 2px 8px; border-radius: 8px; font-weight: 700; }}
.pass {{ background: #2e7d32; }} .warn {{ background: #ef6c00; }} .fail {{ background: #c62828; }}
.note {{ max-width: 960px; color: #536170; line-height: 1.45; }}
</style></head><body>
<h1>Tangent Escape RMP Automatic Validation</h1>
<div class="summary">Overall: <span class="status {overall.lower()}">{overall}</span></div>
<p class="note">PASS certifies the logged fixed-mode GDS identities and scenario expectations.
WARN marks empirical limits such as acceleration saturation, finite-difference energy increase, or
positive hybrid reset energy. It is not a formal proof of the complete switched controller.</p>
<table><thead><tr><th>Scenario</th><th>Status</th><th>Pass</th><th>Warn</th><th>Fail</th><th>Checks</th><th>Plot</th><th>Log</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></body></html>"""
    path.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and run repeated fake-sensor validation for stable_hybrid_gds."
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--start-domain-id", type=int, default=170)
    parser.add_argument("--quick", action="store_true", help="Run only normal and forced-stuck scenarios.")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (args.output_dir or DEFAULT_LOG_ROOT / f"stability_suite_{stamp}").expanduser()
    output_root.mkdir(parents=True, exist_ok=False)

    if not args.skip_build:
        build_command = (
            "source /opt/ros/humble/setup.bash && "
            "colcon build --packages-select rb10_rmpflow_rviz --symlink-install "
            "--cmake-args -DCMAKE_BUILD_TYPE=Release"
        )
        build = run_command(build_command, WS_ROOT, output_root / "build.log", 1200)
        if build.returncode != 0:
            print(f"Build failed. See {output_root / 'build.log'}")
            return 2

    scenarios = [SCENARIOS[0], SCENARIOS[2]] if args.quick else list(SCENARIOS)
    results: List[Dict[str, object]] = []
    for offset, scenario in enumerate(scenarios):
        name = str(scenario["name"])
        scenario_dir = output_root / name
        scenario_dir.mkdir(parents=True)
        print(f"RUN {name} (ROS_DOMAIN_ID={args.start_domain_id + offset})", flush=True)
        csv_path, launch = launch_scenario(
            scenario,
            scenario_dir,
            args.start_domain_id + offset,
        )
        passes = warnings = failures = 0
        status = "FAIL"
        if csv_path is not None:
            validation, _, _ = validate_csv(csv_path, scenario, scenario_dir)
            passes, warnings, failures = parse_validation(scenario_dir / "validation.log")
            if validation.returncode == 0 and failures == 0:
                status = "WARN" if warnings else "PASS"
            elif failures == 0 and validation.returncode not in (0, 2):
                failures = 1
        else:
            failures = 1
        if launch.returncode not in (0, 124, 130, -2) and csv_path is None:
            failures = max(failures, 1)
        results.append({
            "name": name,
            "status": status,
            "passes": passes,
            "warnings": warnings,
            "failures": failures,
            "directory": scenario_dir,
        })
        print(
            f"  {status}: pass={passes} warn={warnings} fail={failures} csv={csv_path}",
            flush=True,
        )

    summary_path = output_root / "index.html"
    write_summary(summary_path, results)
    print(f"Saved suite: {summary_path}")
    return 2 if any(result["status"] == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
