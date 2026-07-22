#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DEFAULT_LOG_DIR = Path("~/ros2_ws/log/rmpflow_trace").expanduser()
GEOMETRY_REQUIRED = [
    "tangent_escape_geometry_active",
    "tangent_escape_geometry_sensor_index",
    "tangent_escape_geometry_sensor_name",
    "tangent_escape_geometry_cp_x",
    "tangent_escape_geometry_cp_y",
    "tangent_escape_geometry_cp_z",
    "tangent_escape_geometry_obstacle_x",
    "tangent_escape_geometry_obstacle_y",
    "tangent_escape_geometry_obstacle_z",
    "tangent_escape_geometry_sensor_normal_x",
    "tangent_escape_geometry_sensor_normal_y",
    "tangent_escape_geometry_sensor_normal_z",
    "tangent_escape_geometry_obstacle_direction_x",
    "tangent_escape_geometry_obstacle_direction_y",
    "tangent_escape_geometry_obstacle_direction_z",
    "tangent_escape_geometry_collision_normal_x",
    "tangent_escape_geometry_collision_normal_y",
    "tangent_escape_geometry_collision_normal_z",
    "tangent_escape_geometry_tangent_bias_x",
    "tangent_escape_geometry_tangent_bias_y",
    "tangent_escape_geometry_tangent_bias_z",
    "tangent_escape_geometry_cp_velocity_x",
    "tangent_escape_geometry_cp_velocity_y",
    "tangent_escape_geometry_cp_velocity_z",
    "tangent_escape_geometry_sensor_obstacle_dot",
    "tangent_escape_geometry_collision_obstacle_dot",
    "tangent_escape_geometry_bias_sensor_dot",
    "tangent_escape_geometry_bias_obstacle_dot",
    "tangent_escape_geometry_jacobian_frobenius_norm",
    "tangent_escape_geometry_sensor_normal_jacobian_norm",
    "tangent_escape_geometry_obstacle_direction_jacobian_norm",
    "tangent_escape_geometry_tangent_bias_jacobian_norm",
    "tangent_escape_geometry_velocity_obstacle_dot",
    "tangent_escape_geometry_velocity_tangent_dot",
]


def parse_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def vector(row: Dict[str, str], prefix: str) -> List[float]:
    return [parse_float(row, f"{prefix}_{axis}") for axis in ("x", "y", "z")]


def norm(values: Sequence[float]) -> float:
    if any(not math.isfinite(value) for value in values):
        return float("nan")
    return math.sqrt(sum(value * value for value in values))


def finite_or_none(value: float) -> Optional[float]:
    return value if math.isfinite(value) else None


def vector_or_none(values: Sequence[float]) -> Optional[List[float]]:
    return list(values) if all(math.isfinite(value) for value in values) else None


def discover_csv(log_dir: Path) -> Path:
    candidates = sorted(log_dir.glob("rmpflow_trace_*.csv"), key=lambda path: path.stat().st_mtime)
    for path in reversed(candidates):
        try:
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames or any(col not in reader.fieldnames for col in GEOMETRY_REQUIRED):
                    continue
                for row in reader:
                    if parse_float(row, "tangent_escape_geometry_active") >= 0.5:
                        return path
        except OSError:
            continue
    raise FileNotFoundError(f"No tangent escape geometry CSV found in {log_dir}")


def read_samples(path: Path) -> List[Dict[str, object]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [column for column in GEOMETRY_REQUIRED if column not in fieldnames]
        if missing:
            raise ValueError(
                "Missing tangent escape geometry columns. Record again with "
                "publish_tangent_escape_geometry_debug:=true. Missing: "
                + ", ".join(missing)
            )
        rows = list(reader)
    first_time = parse_float(rows[0], "time_ros_s") if rows else float("nan")
    samples: List[Dict[str, object]] = []
    for csv_index, row in enumerate(rows):
        geometry_active = parse_float(row, "tangent_escape_geometry_active")
        sensor_index = parse_float(row, "tangent_escape_geometry_sensor_index")
        if (
            not math.isfinite(geometry_active) or
            geometry_active < 0.5 or
            not math.isfinite(sensor_index)
        ):
            continue
        stamp = parse_float(row, "time_ros_s")
        sample = {
            "csvIndex": csv_index,
            "time": finite_or_none(stamp - first_time if math.isfinite(stamp) and math.isfinite(first_time) else stamp),
            "sensorIndex": int(round(sensor_index)),
            "sensorName": row.get("tangent_escape_geometry_sensor_name", ""),
            "sensorParent": row.get("tangent_escape_geometry_sensor_parent_link", ""),
            "clearance": finite_or_none(parse_float(row, "tangent_escape_geometry_clearance")),
            "centerDistance": finite_or_none(parse_float(row, "tangent_escape_geometry_center_distance")),
            "cp": vector_or_none(vector(row, "tangent_escape_geometry_cp")),
            "obstacle": vector_or_none(vector(row, "tangent_escape_geometry_obstacle")),
            "sensorNormal": vector_or_none(vector(row, "tangent_escape_geometry_sensor_normal")),
            "obstacleDirection": vector_or_none(vector(row, "tangent_escape_geometry_obstacle_direction")),
            "collisionNormal": vector_or_none(vector(row, "tangent_escape_geometry_collision_normal")),
            "tangentBias": vector_or_none(vector(row, "tangent_escape_geometry_tangent_bias")),
            "cpVelocity": vector_or_none(vector(row, "tangent_escape_geometry_cp_velocity")),
            "sensorObstacleDot": finite_or_none(parse_float(row, "tangent_escape_geometry_sensor_obstacle_dot")),
            "collisionObstacleDot": finite_or_none(parse_float(row, "tangent_escape_geometry_collision_obstacle_dot")),
            "biasSensorDot": finite_or_none(parse_float(row, "tangent_escape_geometry_bias_sensor_dot")),
            "biasObstacleDot": finite_or_none(parse_float(row, "tangent_escape_geometry_bias_obstacle_dot")),
            "jacobianNorm": finite_or_none(parse_float(row, "tangent_escape_geometry_jacobian_frobenius_norm")),
            "sensorNormalJacobianNorm": finite_or_none(parse_float(row, "tangent_escape_geometry_sensor_normal_jacobian_norm")),
            "obstacleJacobianNorm": finite_or_none(parse_float(row, "tangent_escape_geometry_obstacle_direction_jacobian_norm")),
            "tangentJacobianNorm": finite_or_none(parse_float(row, "tangent_escape_geometry_tangent_bias_jacobian_norm")),
            "velocityObstacleDot": finite_or_none(parse_float(row, "tangent_escape_geometry_velocity_obstacle_dot")),
            "velocityTangentDot": finite_or_none(parse_float(row, "tangent_escape_geometry_velocity_tangent_dot")),
        }
        samples.append(sample)
    if not samples:
        raise ValueError(f"No active tangent escape geometry rows found in {path}")
    return samples


def write_report(path: Path, samples: Sequence[Dict[str, object]], output: Path) -> None:
    payload = {"source": str(path), "samples": samples}
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tangent Escape Geometry Report</title>
<style>
:root {{
  color-scheme: dark;
  --bg: #101316;
  --panel: #171c20;
  --text: #e8ecef;
  --muted: #9aa6af;
  --grid: #303840;
  --cp: #f3f4f6;
  --obs: #ff4d4d;
  --sensor: #2f7cff;
  --obdir: #ff3344;
  --collision: #ff9a2e;
  --bias: #28d66f;
  --velocity: #c084fc;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); }}
main {{ max-width: 1180px; margin: 0 auto; padding: 20px; }}
h1 {{ margin: 0 0 4px; font-size: 22px; }}
.source {{ color: var(--muted); font-size: 13px; word-break: break-all; }}
.toolbar {{ display: flex; gap: 12px; align-items: center; margin: 18px 0; }}
input[type="range"] {{ flex: 1; }}
.grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
.panel {{ background: var(--panel); border: 1px solid #273039; border-radius: 8px; padding: 12px; }}
.wide {{ grid-column: 1 / -1; }}
svg {{ width: 100%; height: 360px; display: block; background: #0c0f12; border-radius: 6px; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 8px 0 0; color: var(--muted); font-size: 13px; }}
.swatch {{ width: 12px; height: 12px; display: inline-block; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }}
.metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }}
.metric {{ background: #0c0f12; border: 1px solid #28313a; border-radius: 6px; padding: 10px; }}
.label {{ color: var(--muted); font-size: 12px; }}
.value {{ font-size: 18px; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
td {{ padding: 6px 8px; border-bottom: 1px solid #273039; }}
td:first-child {{ color: var(--muted); width: 220px; }}
@media (max-width: 800px) {{ .grid, .metrics {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<main>
  <h1>Tangent Escape Geometry Report</h1>
  <div class="source" id="source"></div>
  <div class="toolbar">
    <button id="prevBtn">Prev</button>
    <input id="slider" type="range" min="0" value="0">
    <button id="nextBtn">Next</button>
    <div id="sampleLabel"></div>
  </div>
  <section class="panel wide">
    <div class="metrics">
      <div class="metric"><div class="label">sensor</div><div class="value" id="sensorValue"></div></div>
      <div class="metric"><div class="label">clearance</div><div class="value" id="clearanceValue"></div></div>
      <div class="metric"><div class="label">sensor dot obstacle</div><div class="value" id="sensorDotValue"></div></div>
      <div class="metric"><div class="label">Jacobian norm</div><div class="value" id="jacobianValue"></div></div>
    </div>
  </section>
  <section class="grid">
    <div class="panel">
      <h2>XY</h2>
      <svg id="xyView"></svg>
    </div>
    <div class="panel">
      <h2>XZ</h2>
      <svg id="xzView"></svg>
    </div>
    <div class="panel wide">
      <div class="legend">
        <span><i class="swatch" style="background:var(--cp)"></i>control point</span>
        <span><i class="swatch" style="background:var(--obs)"></i>obstacle</span>
        <span><i class="swatch" style="background:var(--sensor)"></i>sensor normal</span>
        <span><i class="swatch" style="background:var(--obdir)"></i>obstacle direction</span>
        <span><i class="swatch" style="background:var(--collision)"></i>collision normal</span>
        <span><i class="swatch" style="background:var(--bias)"></i>tangent bias</span>
        <span><i class="swatch" style="background:var(--velocity)"></i>CP velocity</span>
      </div>
    </div>
    <div class="panel wide">
      <table id="detailTable"></table>
    </div>
  </section>
</main>
<script>
const reportData = {json.dumps(payload, separators=(",", ":"))};
const slider = document.getElementById("slider");
slider.max = Math.max(reportData.samples.length - 1, 0);
let selectedIndex = 0;

function fmt(value, digits = 3) {{
  return Number.isFinite(value) ? value.toFixed(digits) : "n/a";
}}
function vecFmt(v) {{
  return Array.isArray(v) ? `[${{v.map((x) => fmt(x, 4)).join(", ")}}]` : "n/a";
}}
function sub(a, b) {{ return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }}
function add(a, b) {{ return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }}
function mul(a, s) {{ return [a[0] * s, a[1] * s, a[2] * s]; }}
function normalized(v) {{
  if (!Array.isArray(v)) return null;
  const n = Math.hypot(v[0], v[1], v[2]);
  return n > 1e-9 ? [v[0] / n, v[1] / n, v[2] / n] : null;
}}

function bounds(sample) {{
  const pts = [sample.cp, sample.obstacle].filter(Array.isArray);
  const dirs = [sample.sensorNormal, sample.obstacleDirection, sample.collisionNormal, sample.tangentBias, normalized(sample.cpVelocity)].filter(Array.isArray);
  dirs.forEach((d) => pts.push(add(sample.cp, mul(d, 0.16))));
  const xs = pts.map((p) => p[0]);
  const ys = pts.map((p) => p[1]);
  const zs = pts.map((p) => p[2]);
  return {{x: [Math.min(...xs), Math.max(...xs)], y: [Math.min(...ys), Math.max(...ys)], z: [Math.min(...zs), Math.max(...zs)]}};
}}
function project(p, plane) {{
  return plane === "xy" ? [p[0], p[1]] : [p[0], p[2]];
}}
function drawView(svg, sample, plane) {{
  svg.innerHTML = "";
  if (!Array.isArray(sample.cp) || !Array.isArray(sample.obstacle)) return;
  const b = bounds(sample);
  const minX = b.x[0] - 0.08, maxX = b.x[1] + 0.08;
  const rangeY = plane === "xy" ? b.y : b.z;
  const minY = rangeY[0] - 0.08, maxY = rangeY[1] + 0.08;
  const width = svg.clientWidth || 500, height = svg.clientHeight || 360;
  const sx = width / Math.max(maxX - minX, 1e-6);
  const sy = height / Math.max(maxY - minY, 1e-6);
  const scale = Math.min(sx, sy) * 0.88;
  const cx = 0.5 * (minX + maxX), cy = 0.5 * (minY + maxY);
  const toScreen = (p) => {{
    const q = project(p, plane);
    return [width * 0.5 + (q[0] - cx) * scale, height * 0.5 - (q[1] - cy) * scale];
  }};
  function line(a, b, color, widthPx = 2) {{
    const p = toScreen(a), q = toScreen(b);
    const el = document.createElementNS("http://www.w3.org/2000/svg", "line");
    el.setAttribute("x1", p[0]); el.setAttribute("y1", p[1]);
    el.setAttribute("x2", q[0]); el.setAttribute("y2", q[1]);
    el.setAttribute("stroke", color); el.setAttribute("stroke-width", widthPx);
    svg.appendChild(el);
  }}
  function circle(p, r, color) {{
    const q = toScreen(p);
    const el = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    el.setAttribute("cx", q[0]); el.setAttribute("cy", q[1]); el.setAttribute("r", r);
    el.setAttribute("fill", color); svg.appendChild(el);
  }}
  function arrow(dir, color) {{
    if (!Array.isArray(dir)) return;
    line(sample.cp, add(sample.cp, mul(dir, 0.16)), color, 3);
  }}
  line(sample.cp, sample.obstacle, "#6b7280", 1.5);
  circle(sample.cp, 5, "var(--cp)");
  circle(sample.obstacle, 6, "var(--obs)");
  arrow(sample.sensorNormal, "var(--sensor)");
  arrow(sample.obstacleDirection, "var(--obdir)");
  arrow(sample.collisionNormal, "var(--collision)");
  arrow(sample.tangentBias, "var(--bias)");
  arrow(normalized(sample.cpVelocity), "var(--velocity)");
}}
function update() {{
  selectedIndex = Math.max(0, Math.min(selectedIndex, reportData.samples.length - 1));
  slider.value = selectedIndex;
  const sample = reportData.samples[selectedIndex];
  document.getElementById("source").textContent = reportData.source;
  document.getElementById("sampleLabel").textContent = `${{selectedIndex + 1}} / ${{reportData.samples.length}}`;
  document.getElementById("sensorValue").textContent = `${{sample.sensorName}} (#${{sample.sensorIndex}})`;
  document.getElementById("clearanceValue").textContent = `${{fmt(sample.clearance, 4)}} m`;
  document.getElementById("sensorDotValue").textContent = fmt(sample.sensorObstacleDot, 4);
  document.getElementById("jacobianValue").textContent = fmt(sample.jacobianNorm, 4);
  drawView(document.getElementById("xyView"), sample, "xy");
  drawView(document.getElementById("xzView"), sample, "xz");
  const rows = [
    ["csv index", sample.csvIndex],
    ["time", `${{fmt(sample.time, 3)}} s`],
    ["parent link", sample.sensorParent || "n/a"],
    ["center distance", `${{fmt(sample.centerDistance, 4)}} m`],
    ["cp", vecFmt(sample.cp)],
    ["obstacle", vecFmt(sample.obstacle)],
    ["sensor normal", vecFmt(sample.sensorNormal)],
    ["obstacle direction", vecFmt(sample.obstacleDirection)],
    ["collision normal", vecFmt(sample.collisionNormal)],
    ["tangent bias", vecFmt(sample.tangentBias)],
    ["CP velocity", vecFmt(sample.cpVelocity)],
    ["collision dot obstacle", fmt(sample.collisionObstacleDot, 4)],
    ["bias dot sensor", fmt(sample.biasSensorDot, 4)],
    ["bias dot obstacle", fmt(sample.biasObstacleDot, 4)],
    ["sensor normal J row norm", fmt(sample.sensorNormalJacobianNorm, 4)],
    ["obstacle direction J row norm", fmt(sample.obstacleJacobianNorm, 4)],
    ["tangent bias J row norm", fmt(sample.tangentJacobianNorm, 4)],
    ["velocity dot obstacle direction", fmt(sample.velocityObstacleDot, 4)],
    ["velocity dot tangent bias", fmt(sample.velocityTangentDot, 4)]
  ];
  document.getElementById("detailTable").innerHTML = rows.map(([k, v]) => `<tr><td>${{k}}</td><td>${{v}}</td></tr>`).join("");
}}
slider.addEventListener("input", () => {{ selectedIndex = Number(slider.value); update(); }});
document.getElementById("prevBtn").addEventListener("click", () => {{ selectedIndex -= 1; update(); }});
document.getElementById("nextBtn").addEventListener("click", () => {{ selectedIndex += 1; update(); }});
window.addEventListener("resize", update);
update();
</script>
</body>
</html>
"""
    output.write_text(html)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a Tangent Escape sensor-geometry HTML report."
    )
    parser.add_argument("csv", nargs="?", type=Path, help="Trace CSV. Defaults to latest compatible CSV.")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--save", type=Path, help="Output HTML path.")
    args = parser.parse_args()

    path = args.csv.expanduser() if args.csv else discover_csv(args.log_dir.expanduser())
    samples = read_samples(path)
    output = args.save.expanduser() if args.save else path.with_name(f"{path.stem}_tangent_escape_geometry_report.html")
    write_report(path, samples, output)
    print(f"Input CSV: {path}")
    print(f"Geometry rows: {len(samples)}")
    print(f"Saved report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
