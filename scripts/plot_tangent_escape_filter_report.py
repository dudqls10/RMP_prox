#!/usr/bin/env python3

import argparse
import html
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from plot_tangent_escape_filter_debug import (
    DEFAULT_DATA_DIR,
    active_rows,
    angle_degrees,
    dot,
    goal_points,
    normalize,
    parse_float,
    read_csv_rows,
    require_columns,
    resolve_input_path,
    vector,
    vector_norm,
)


def accel_vector(row: Dict[str, str], prefix: str) -> Tuple[float, float, float]:
    return (
        parse_float(row, f"{prefix}_x_m_s2"),
        parse_float(row, f"{prefix}_y_m_s2"),
        parse_float(row, f"{prefix}_z_m_s2"),
    )


def row_goal(row: Dict[str, str]) -> Optional[Tuple[float, float, float]]:
    if all(key in row for key in ("goal_pose_x", "goal_pose_y", "goal_pose_z")):
        goal = vector(row, "goal_pose")
    else:
        goal = vector(row, "goal")
    if all(math.isfinite(value) for value in goal):
        return goal
    return None


def finite_or_none(value: float) -> Optional[float]:
    return float(value) if math.isfinite(value) else None


def vector_or_none(values: Sequence[float]) -> List[Optional[float]]:
    return [finite_or_none(value) for value in values]


def parse_candidate_payload(row: Dict[str, str]) -> Dict[str, object]:
    raw_payload = row.get("tangent_escape_candidates_json", "")
    if not raw_payload:
        return {
            "active": False,
            "candidate_count": 0,
            "selected_candidate_index": None,
            "weights": {},
            "candidates": [],
        }
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        return {
            "active": False,
            "candidate_count": 0,
            "selected_candidate_index": None,
            "weights": {},
            "candidates": [],
        }
    if not isinstance(payload, dict):
        return {
            "active": False,
            "candidate_count": 0,
            "selected_candidate_index": None,
            "weights": {},
            "candidates": [],
        }
    return payload


def active_row_time(rows: Sequence[Dict[str, str]], row: Dict[str, str], index: int) -> float:
    first_time = parse_float(rows[0], "timestamp_unix") if rows else float("nan")
    stamp = parse_float(row, "timestamp_unix")
    if math.isfinite(stamp) and math.isfinite(first_time):
        return stamp - first_time
    return float(index)


def make_sample(rows: Sequence[Dict[str, str]], row: Dict[str, str], index: int) -> Dict[str, object]:
    normal = vector(row, "tangent_escape_normal")
    tangent = vector(row, "tangent_escape_tangent")
    raw = accel_vector(row, "tangent_escape_raw_cp_accel")
    filtered = accel_vector(row, "tangent_escape_filtered_cp_accel")
    normal_unit = normalize(normal)
    tangent_unit = normalize(tangent)
    raw_dot_normal = dot(raw, normal_unit)
    filtered_dot_normal = dot(filtered, normal_unit)
    raw_dot_tangent = dot(raw, tangent_unit)
    filtered_dot_tangent = dot(filtered, tangent_unit)
    return {
        "activeIndex": index,
        "time": finite_or_none(active_row_time(rows, row, index)),
        "clearance": finite_or_none(parse_float(row, "tangent_escape_clearance")),
        "activation": finite_or_none(parse_float(row, "tangent_escape_activation")),
        "score": finite_or_none(parse_float(row, "tangent_escape_score")),
        "cpIndex": finite_or_none(parse_float(row, "tangent_escape_cp_index")),
        "cp": vector_or_none(vector(row, "tangent_escape_cp")),
        "obstacle": vector_or_none(vector(row, "tangent_escape_obstacle")),
        "goal": vector_or_none(row_goal(row) or (float("nan"), float("nan"), float("nan"))),
        "normal": vector_or_none(normal),
        "tangent": vector_or_none(tangent),
        "raw": vector_or_none(raw),
        "filtered": vector_or_none(filtered),
        "rawDotNormal": finite_or_none(raw_dot_normal),
        "filteredDotNormal": finite_or_none(filtered_dot_normal),
        "rawDotTangent": finite_or_none(raw_dot_tangent),
        "filteredDotTangent": finite_or_none(filtered_dot_tangent),
        "tangentDotNormal": finite_or_none(dot(tangent_unit, normal_unit)),
        "rawFilteredAngleDeg": finite_or_none(angle_degrees(raw, filtered)),
        "rawNorm": finite_or_none(vector_norm(raw)),
        "filteredNorm": finite_or_none(vector_norm(filtered)),
        "candidateData": parse_candidate_payload(row),
    }


def choose_initial_index(samples: Sequence[Dict[str, object]], args: argparse.Namespace) -> int:
    if args.sample_index is not None:
        return max(0, min(int(args.sample_index), len(samples) - 1))
    if args.time is not None:
        return min(
            range(len(samples)),
            key=lambda index: abs(float(samples[index].get("time") or 0.0) - args.time),
        )
    if args.select == "middle":
        return len(samples) // 2
    if args.select == "max-angle":
        return max(
            range(len(samples)),
            key=lambda index: float(samples[index].get("rawFilteredAngleDeg") or -1.0),
        )
    return min(
        range(len(samples)),
        key=lambda index: float(samples[index].get("clearance") or float("inf")),
    )


def html_template() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root {
    color-scheme: light;
    font-family: Inter, Segoe UI, Arial, sans-serif;
    --bg: #f5f6f8;
    --panel: #ffffff;
    --ink: #1e2329;
    --muted: #68707d;
    --grid: #d8dde5;
    --raw: #1f77b4;
    --filtered: #17becf;
    --normal: #d62728;
    --tangent: #2ca02c;
    --candidate: #7d8794;
    --plane: rgba(44, 160, 44, 0.13);
    --obstacle: #ff8c00;
    --goal: #ffd400;
  }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
  }
  header {
    padding: 18px 22px 10px;
  }
  h1 {
    margin: 0 0 4px;
    font-size: 24px;
    font-weight: 700;
  }
  .sub {
    color: var(--muted);
    font-size: 13px;
  }
	  .controls {
	    display: grid;
	    grid-template-columns: auto 1fr auto auto auto;
	    gap: 10px;
	    align-items: center;
	    padding: 10px 22px 16px;
	  }
  .zoom-controls {
    display: grid;
    grid-template-columns: auto minmax(90px, 1fr) auto auto;
    gap: 8px;
    align-items: center;
    margin: 0 0 8px;
    color: var(--muted);
    font-size: 12px;
  }
	  button {
	    border: 1px solid #c9d0da;
	    background: #fff;
	    border-radius: 6px;
	    padding: 7px 10px;
	    font-size: 13px;
	    cursor: pointer;
	  }
  select {
    border: 1px solid #c9d0da;
    background: #fff;
    border-radius: 6px;
    padding: 5px 8px;
    font-size: 12px;
  }
  input[type="range"] {
    width: 100%;
  }
  .layout {
    display: grid;
    grid-template-columns: minmax(520px, 1.35fr) minmax(420px, 0.95fr);
    gap: 14px;
    padding: 0 22px 22px;
  }
  .panel {
    background: var(--panel);
    border: 1px solid #d8dde5;
    border-radius: 8px;
    padding: 12px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  }
  .panel h2 {
    margin: 0 0 8px;
    font-size: 16px;
  }
  svg {
    width: 100%;
    height: auto;
    display: block;
  }
  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 14px;
    margin-top: 8px;
    color: var(--muted);
    font-size: 12px;
  }
  .legend span {
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }
  .swatch {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
    border: 1px solid rgba(0,0,0,0.28);
  }
  .line-swatch {
    width: 18px;
    height: 3px;
    border-radius: 3px;
    display: inline-block;
  }
  .metrics {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
    margin-top: 10px;
  }
  .metric {
    border: 1px solid #e1e5eb;
    border-radius: 6px;
    padding: 8px;
  }
  .metric .label {
    color: var(--muted);
    font-size: 11px;
    margin-bottom: 3px;
  }
  .metric .value {
    font-variant-numeric: tabular-nums;
    font-size: 15px;
  }
  .bars {
    margin-top: 10px;
    display: grid;
    gap: 7px;
  }
  .bar-row {
    display: grid;
    grid-template-columns: 150px 1fr 68px;
    gap: 8px;
    align-items: center;
    font-size: 12px;
  }
  .bar-track {
    position: relative;
    height: 18px;
    background: #eef1f5;
    border-radius: 5px;
    overflow: hidden;
  }
  .bar-zero {
    position: absolute;
    left: 50%;
    top: 0;
    bottom: 0;
    width: 1px;
    background: #9aa3af;
  }
  .bar-fill {
    position: absolute;
    top: 2px;
    bottom: 2px;
    border-radius: 4px;
  }
  .note {
    margin-top: 10px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.45;
  }
  .candidate-section {
    margin-top: 12px;
    border-top: 1px solid #e1e5eb;
    padding-top: 10px;
  }
  .candidate-title {
    display: flex;
    justify-content: space-between;
    gap: 8px;
    align-items: baseline;
    margin-bottom: 7px;
  }
  .candidate-title h3 {
    margin: 0;
    font-size: 14px;
  }
  .candidate-summary {
    color: var(--muted);
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }
  .candidate-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 10px;
    margin-bottom: 7px;
    color: var(--muted);
    font-size: 11px;
  }
  .candidate-legend span {
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .candidate-bars {
    display: grid;
    gap: 4px;
    max-height: 300px;
    overflow-y: auto;
    padding-right: 4px;
  }
  .candidate-row {
    display: grid;
    grid-template-columns: 32px 1fr 52px;
    gap: 7px;
    align-items: center;
    font-size: 11px;
    font-variant-numeric: tabular-nums;
  }
  .candidate-row.selected {
    font-weight: 700;
    color: #1b6c2a;
  }
  .candidate-track {
    position: relative;
    height: 18px;
    background: #eef1f5;
    border-radius: 5px;
    overflow: hidden;
  }
  .candidate-track::after {
    content: "";
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    width: 1px;
    background: #8f99a8;
  }
  .candidate-segment {
    position: absolute;
    top: 2px;
    bottom: 2px;
    min-width: 1px;
  }
  .candidate-selected-band {
    position: absolute;
    inset: 0;
    border: 2px solid rgba(44, 160, 44, 0.45);
    border-radius: 5px;
    pointer-events: none;
  }
</style>
</head>
<body>
<header>
  <h1>Tangent Escape Filter Report</h1>
  <div class="sub">__CSV_NAME__</div>
</header>
<div class="controls">
  <button id="prevBtn">Prev</button>
  <input id="sampleSlider" type="range" min="0" max="0" value="0" step="1">
  <button id="nextBtn">Next</button>
  <div id="sampleLabel" class="sub"></div>
  <div id="sampleTime" class="sub"></div>
</div>
<main class="layout">
	  <section class="panel">
	    <h2>Overall view</h2>
	    <svg id="overallSvg" viewBox="0 0 760 560" role="img" aria-label="Overall projected 3D view">
      <defs>
        <marker id="arrowRaw" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L10,4 L0,8 z" fill="var(--raw)"></path>
        </marker>
        <marker id="arrowFiltered" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L10,4 L0,8 z" fill="var(--filtered)"></path>
        </marker>
        <marker id="arrowNormal" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L10,4 L0,8 z" fill="var(--normal)"></path>
        </marker>
	        <marker id="arrowTangent" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
	          <path d="M0,0 L10,4 L0,8 z" fill="var(--tangent)"></path>
	        </marker>
        <marker id="arrowAxis" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L10,4 L0,8 z" fill="#5c6673"></path>
        </marker>
        <marker id="arrowCandidate" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L10,4 L0,8 z" fill="var(--candidate)"></path>
        </marker>
	      </defs>
    </svg>
	    <div class="legend">
	      <span><i class="line-swatch" style="background:#333"></i>active CP path</span>
	      <span><i class="swatch" style="background:var(--obstacle)"></i>obstacle</span>
      <span><i class="swatch" style="background:var(--goal)"></i>goal</span>
	      <span><i class="swatch" style="background:#fff"></i>selected CP</span>
      <span><i class="line-swatch" style="background:var(--normal)"></i>normal</span>
      <span><i class="line-swatch" style="background:var(--tangent)"></i>selected escape</span>
      <span><i class="line-swatch" style="background:rgba(44,160,44,0.35)"></i>tangent plane</span>
      <span><i class="line-swatch" style="background:var(--raw)"></i>raw accel</span>
	      <span><i class="line-swatch" style="background:var(--filtered)"></i>filtered accel</span>
	      <span><i class="line-swatch" style="background:var(--candidate)"></i>candidate tangent directions</span>
	    </div>
    <div class="note">Click an active CP point/path sample in the overall view to select that moment.</div>
	  </section>
	  <section class="panel">
	    <h2>Selected moment: 3D zoom</h2>
    <div class="zoom-controls">
      <label for="zoomSlider">Zoom radius</label>
      <input id="zoomSlider" type="range" min="0.05" max="0.80" value="0.22" step="0.01">
      <span id="zoomValue"></span>
      <select id="viewSelect" aria-label="3D view direction">
        <option value="iso">iso</option>
        <option value="xy">XY</option>
        <option value="xz">XZ</option>
        <option value="yz">YZ</option>
      </select>
    </div>
	    <svg id="localSvg" viewBox="0 0 560 430" role="img" aria-label="Local 3D zoom view"></svg>
    <div class="metrics">
      <div class="metric"><div class="label">clearance</div><div class="value" id="clearanceValue"></div></div>
      <div class="metric"><div class="label">activation</div><div class="value" id="activationValue"></div></div>
      <div class="metric"><div class="label">raw-filtered angle</div><div class="value" id="angleValue"></div></div>
      <div class="metric"><div class="label">tangent dot normal</div><div class="value" id="orthogonalValue"></div></div>
    </div>
	    <div class="bars" id="componentBars"></div>
    <div class="candidate-section">
      <div class="candidate-title">
        <h3>Candidate score components</h3>
        <div class="candidate-summary" id="candidateSummary"></div>
      </div>
      <div class="candidate-legend">
        <span><i class="line-swatch" style="background:#4c78a8"></i>goal</span>
        <span><i class="line-swatch" style="background:#f58518"></i>nearest clearance</span>
        <span><i class="line-swatch" style="background:#54a24b"></i>aggregate clearance</span>
        <span><i class="line-swatch" style="background:#b279a2"></i>continuity</span>
        <span><i class="line-swatch" style="background:#d62728"></i>duplicate risk</span>
        <span><i class="line-swatch" style="background:#e45756"></i>adjacent block</span>
        <span><i class="line-swatch" style="background:#72b7b2"></i>branch hold</span>
        <span><i class="line-swatch" style="background:#111"></i>total shown at right</span>
      </div>
      <div class="candidate-bars" id="candidateBars"></div>
    </div>
	    <div class="note">
	      The zoom view keeps the original 3D coordinates and only changes the camera projection. Candidate bars update for the selected point and decompose why that tangent direction was selected.
	    </div>
  </section>
</main>
<script>
const reportData = __DATA_JSON__;
let selectedIndex = __INITIAL_INDEX__;
const overallSvg = document.getElementById("overallSvg");
const localSvg = document.getElementById("localSvg");
const slider = document.getElementById("sampleSlider");
const zoomSlider = document.getElementById("zoomSlider");
const zoomValue = document.getElementById("zoomValue");
const viewSelect = document.getElementById("viewSelect");
slider.max = String(Math.max(reportData.samples.length - 1, 0));
slider.value = String(selectedIndex);

function finite(value) {
  return typeof value === "number" && Number.isFinite(value);
}
function fmt(value, digits = 3) {
  return finite(value) ? value.toFixed(digits) : "n/a";
}
function add(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}
function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}
function mul(a, scale) {
  return [a[0] * scale, a[1] * scale, a[2] * scale];
}
function norm(a) {
  return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}
function normalizeJs(a) {
  const n = norm(a);
  if (n <= 1e-12) return [0, 0, 0];
  return [a[0] / n, a[1] / n, a[2] / n];
}
function dotJs(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
function crossJs(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}
function project(p) {
  return [p[0] - 0.52 * p[1], -p[2] + 0.34 * p[1]];
}
function projectView(p, view) {
  if (view === "xy") return [p[0], -p[1]];
  if (view === "xz") return [p[0], -p[2]];
  if (view === "yz") return [p[1], -p[2]];
  return project(p);
}
function starPoints(cx, cy, outer, inner, count = 5) {
  const pts = [];
  for (let i = 0; i < count * 2; i += 1) {
    const angle = -Math.PI / 2 + i * Math.PI / count;
    const r = i % 2 === 0 ? outer : inner;
    pts.push(`${cx + Math.cos(angle) * r},${cy + Math.sin(angle) * r}`);
  }
  return pts.join(" ");
}
function makeSvg(tag, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}
function clearSvg(svg, keepDefs = false) {
  [...svg.children].forEach((child) => {
    if (keepDefs && child.tagName.toLowerCase() === "defs") return;
    child.remove();
  });
}
function validPoint(p) {
  return Array.isArray(p) && p.length === 3 && p.every(finite);
}
function collectOverallPoints(sample) {
  const points = [];
  reportData.samples.forEach((item) => {
    if (validPoint(item.cp)) points.push(item.cp);
    if (validPoint(item.obstacle)) points.push(item.obstacle);
  });
  reportData.goals.forEach((goal) => {
    if (validPoint(goal)) points.push(goal);
  });
  ["normal", "tangent", "raw", "filtered"].forEach((key) => {
    const direction = normalizeJs(sample[key]);
    if (validPoint(sample.cp) && norm(direction) > 0) {
      points.push(add(sample.cp, mul(direction, 0.18)));
    }
  });
  return points;
}
function overallScale(sample) {
  const projected = collectOverallPoints(sample).map(project);
  const xs = projected.map((p) => p[0]);
  const ys = projected.map((p) => p[1]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const width = 760, height = 560, pad = 46;
  const sx = (width - pad * 2) / Math.max(maxX - minX, 1e-6);
  const sy = (height - pad * 2) / Math.max(maxY - minY, 1e-6);
  const scale = Math.min(sx, sy);
  return (point) => {
    const p = project(point);
    return [
      pad + (p[0] - minX) * scale,
      height - pad - (p[1] - minY) * scale,
    ];
  };
}
function drawLine(svg, a, b, attrs = {}) {
  const line = makeSvg("line", {x1: a[0], y1: a[1], x2: b[0], y2: b[1], ...attrs});
  svg.appendChild(line);
  return line;
}
function drawArrow(svg, origin, direction, length, color, markerId, toScreen, width = 3) {
  const unit = normalizeJs(direction);
  if (norm(unit) <= 0) return;
  const end = add(origin, mul(unit, length));
  const a = toScreen(origin);
  const b = toScreen(end);
  drawLine(svg, a, b, {
    stroke: color,
    "stroke-width": width,
    "marker-end": `url(#${markerId})`,
    "stroke-linecap": "round",
  });
}
function drawOverall(sample) {
  clearSvg(overallSvg, true);
  const toScreen = overallScale(sample);
  for (let i = 0; i < reportData.samples.length - 1; i += 1) {
    const a = reportData.samples[i].cp;
    const b = reportData.samples[i + 1].cp;
    if (validPoint(a) && validPoint(b)) {
      drawLine(overallSvg, toScreen(a), toScreen(b), {
        stroke: "#333",
        "stroke-width": 1.7,
        opacity: 0.55,
      });
    }
  }
  reportData.samples.forEach((item, index) => {
    if (!validPoint(item.cp)) return;
    const p = toScreen(item.cp);
    const dot = makeSvg("circle", {
      cx: p[0],
      cy: p[1],
      r: index === selectedIndex ? 6 : 4,
      fill: index === selectedIndex ? "#111" : "#ffffff",
      stroke: "#111",
      "stroke-width": 1.2,
      opacity: index === selectedIndex ? 0.95 : 0.48,
      style: "cursor:pointer",
    });
    dot.addEventListener("click", () => {
      selectedIndex = index;
      render();
    });
    overallSvg.appendChild(dot);
  });
  reportData.samples.forEach((item) => {
    if (!validPoint(item.obstacle)) return;
    const p = toScreen(item.obstacle);
    overallSvg.appendChild(makeSvg("circle", {
      cx: p[0], cy: p[1], r: 4.2, fill: "var(--obstacle)", stroke: "#c76c00", "stroke-width": 1, opacity: 0.72,
    }));
  });
  reportData.goals.forEach((goal) => {
    if (!validPoint(goal)) return;
    const p = toScreen(goal);
    overallSvg.appendChild(makeSvg("polygon", {
      points: starPoints(p[0], p[1], 13, 6),
      fill: "var(--goal)",
      stroke: "#111",
      "stroke-width": 1.5,
    }));
  });
  if (validPoint(sample.obstacle) && validPoint(sample.cp)) {
    drawLine(overallSvg, toScreen(sample.obstacle), toScreen(sample.cp), {
      stroke: "#9aa3af",
      "stroke-width": 1.3,
      opacity: 0.55,
    });
  }
  const cp = toScreen(sample.cp);
  overallSvg.appendChild(makeSvg("circle", {
    cx: cp[0], cy: cp[1], r: 7, fill: "#fff", stroke: "#111", "stroke-width": 2,
  }));
  const obs = toScreen(sample.obstacle);
  overallSvg.appendChild(makeSvg("circle", {
    cx: obs[0], cy: obs[1], r: 7, fill: "var(--obstacle)", stroke: "#c76c00", "stroke-width": 1.5,
  }));
  drawArrow(overallSvg, sample.cp, sample.normal, 0.13, "var(--normal)", "arrowNormal", toScreen);
  drawArrow(overallSvg, sample.cp, sample.tangent, 0.16, "var(--tangent)", "arrowTangent", toScreen);
  drawArrow(overallSvg, sample.cp, sample.raw, 0.16, "var(--raw)", "arrowRaw", toScreen);
  drawArrow(overallSvg, sample.cp, sample.filtered, 0.16, "var(--filtered)", "arrowFiltered", toScreen);
  reportData.samples.forEach((item, index) => {
    if (!validPoint(item.cp)) return;
    const p = toScreen(item.cp);
    const hit = makeSvg("circle", {
      cx: p[0], cy: p[1], r: 10, fill: "transparent", "pointer-events": "all", style: "cursor:pointer",
    });
    hit.addEventListener("click", () => {
      selectedIndex = index;
      render();
    });
    overallSvg.appendChild(hit);
  });
}
function localToScreen(relativePoint, radius, view) {
  const center = [280, 215];
  const scale = 165 / Math.max(radius, 1e-6);
  const p = projectView(relativePoint, view);
  return [center[0] + p[0] * scale, center[1] + p[1] * scale];
}
function drawText(svg, point, text, attrs = {}) {
  const node = makeSvg("text", {x: point[0], y: point[1], "font-size": 12, fill: "#333", ...attrs});
  node.textContent = text;
  svg.appendChild(node);
  return node;
}
function drawLocalLine(svg, aRelative, bRelative, radius, view, attrs = {}) {
  drawLine(svg, localToScreen(aRelative, radius, view), localToScreen(bRelative, radius, view), attrs);
}
function drawLocalArrow3D(svg, originRelative, direction, length, color, label, markerId, radius, view, width = 3) {
  const unit = normalizeJs(direction);
  if (norm(unit) <= 0) return;
  const end = add(originRelative, mul(unit, length));
  const a = localToScreen(originRelative, radius, view);
  const b = localToScreen(end, radius, view);
  drawLine(svg, a, b, {
    stroke: color,
    "stroke-width": width,
    "marker-end": `url(#${markerId})`,
    "stroke-linecap": "round",
  });
  drawText(svg, [b[0] + 5, b[1] - 5], label, {fill: color, "font-weight": 650});
}
function drawLocalCandidateArrow3D(svg, originRelative, direction, length, selected, radius, view) {
  const unit = normalizeJs(direction);
  if (norm(unit) <= 0) return;
  const end = add(originRelative, mul(unit, length));
  const a = localToScreen(originRelative, radius, view);
  const b = localToScreen(end, radius, view);
  drawLine(svg, a, b, {
    stroke: selected ? "var(--tangent)" : "var(--candidate)",
    "stroke-width": selected ? 3.0 : 1.4,
    "marker-end": selected ? "url(#arrowTangent)" : "url(#arrowCandidate)",
    "stroke-linecap": "round",
    opacity: selected ? 0.95 : 0.42,
  });
}
function drawProjectedPoint(svg, relativePoint, radius, view, attrs = {}) {
  const p = localToScreen(relativePoint, radius, view);
  svg.appendChild(makeSvg("circle", {cx: p[0], cy: p[1], ...attrs}));
  return p;
}
function tangentPlaneBasis(normal) {
  const n = normalizeJs(normal);
  if (norm(n) <= 0) return null;
  const ref = Math.abs(n[2]) < 0.9 ? [0, 0, 1] : [0, 1, 0];
  let u = crossJs(ref, n);
  if (norm(u) <= 1e-9) {
    u = crossJs([1, 0, 0], n);
  }
  u = normalizeJs(u);
  const v = normalizeJs(crossJs(n, u));
  if (norm(u) <= 0 || norm(v) <= 0) return null;
  return [u, v];
}
function drawLocalTangentPlane(svg, sample, radius, view) {
  const basis = tangentPlaneBasis(sample.normal);
  if (!basis) return;
  const [u, v] = basis;
  const planeRadius = Math.min(radius * 0.48, 0.16);
  const points = [];
  for (let index = 0; index < 48; index += 1) {
    const angle = (2 * Math.PI * index) / 48;
    const relative = add(
      mul(u, Math.cos(angle) * planeRadius),
      mul(v, Math.sin(angle) * planeRadius),
    );
    const screen = localToScreen(relative, radius, view);
    points.push(`${screen[0]},${screen[1]}`);
  }
  svg.appendChild(makeSvg("polygon", {
    points: points.join(" "),
    fill: "var(--plane)",
    stroke: "rgba(44,160,44,0.55)",
    "stroke-width": 1.2,
    "stroke-dasharray": "4 3",
  }));
}
function activeCandidates(sample) {
  const payload = sample.candidateData || {};
  const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
  return candidates.filter((candidate) => validPoint(candidate.direction));
}
function drawLocal(sample) {
  clearSvg(localSvg, false);
  const defs = overallSvg.querySelector("defs").cloneNode(true);
  localSvg.appendChild(defs);
  const radius = Number(zoomSlider.value);
  const view = viewSelect.value;
  const axisLength = Math.min(radius * 0.45, 0.16);
  const arrowLength = Math.min(radius * 0.55, 0.18);
  const gridRadius = radius;

  localSvg.appendChild(makeSvg("rect", {x: 0, y: 0, width: 560, height: 430, fill: "#fbfcfe"}));
  [-1, -0.5, 0, 0.5, 1].forEach((ratio) => {
    const value = ratio * gridRadius;
    drawLocalLine(localSvg, [-gridRadius, value, 0], [gridRadius, value, 0], radius, view, {
      stroke: "#d8dde5", "stroke-width": value === 0 ? 1.4 : 0.8,
    });
    drawLocalLine(localSvg, [value, -gridRadius, 0], [value, gridRadius, 0], radius, view, {
      stroke: "#d8dde5", "stroke-width": value === 0 ? 1.4 : 0.8,
    });
  });
  drawLocalTangentPlane(localSvg, sample, radius, view);

  for (let i = 0; i < reportData.samples.length - 1; i += 1) {
    const a = sub(reportData.samples[i].cp, sample.cp);
    const b = sub(reportData.samples[i + 1].cp, sample.cp);
    if (norm(a) <= radius * 1.35 || norm(b) <= radius * 1.35) {
      drawLocalLine(localSvg, a, b, radius, view, {
        stroke: "#333", "stroke-width": 1.4, opacity: 0.55,
      });
    }
  }

  reportData.samples.forEach((item, index) => {
    const relObstacle = sub(item.obstacle, sample.cp);
    if (norm(relObstacle) <= radius * 1.25) {
      drawProjectedPoint(localSvg, relObstacle, radius, view, {
        r: index === selectedIndex ? 7 : 4,
        fill: "var(--obstacle)",
        stroke: "#c76c00",
        "stroke-width": 1,
        opacity: index === selectedIndex ? 0.95 : 0.55,
      });
    }
  });

  if (validPoint(sample.goal)) {
    const relGoal = sub(sample.goal, sample.cp);
    if (norm(relGoal) <= radius * 1.45) {
      const gp = localToScreen(relGoal, radius, view);
      localSvg.appendChild(makeSvg("polygon", {
        points: starPoints(gp[0], gp[1], 13, 6), fill: "var(--goal)", stroke: "#111", "stroke-width": 1.4,
      }));
    }
  }

  const obstacle = sub(sample.obstacle, sample.cp);
  drawLocalLine(localSvg, [0, 0, 0], obstacle, radius, view, {
    stroke: "#9aa3af", "stroke-width": 1.2, opacity: 0.75,
  });

  drawLocalArrow3D(localSvg, [0, 0, 0], [1, 0, 0], axisLength, "#5c6673", "x", "arrowAxis", radius, view, 1.8);
  drawLocalArrow3D(localSvg, [0, 0, 0], [0, 1, 0], axisLength, "#5c6673", "y", "arrowAxis", radius, view, 1.8);
  drawLocalArrow3D(localSvg, [0, 0, 0], [0, 0, 1], axisLength, "#5c6673", "z", "arrowAxis", radius, view, 1.8);
  const candidates = activeCandidates(sample);
  candidates.forEach((candidate) => {
    drawLocalCandidateArrow3D(
      localSvg,
      [0, 0, 0],
      candidate.direction,
      arrowLength * 0.78,
      Boolean(candidate.selected),
      radius,
      view,
    );
  });
  if (candidates.length === 0) {
    drawLocalArrow3D(
      localSvg,
      [0, 0, 0],
      sample.tangent,
      arrowLength,
      "var(--tangent)",
      "selected escape",
      "arrowTangent",
      radius,
      view,
    );
  }
  drawLocalArrow3D(localSvg, [0, 0, 0], sample.normal, arrowLength * 0.8, "var(--normal)", "normal", "arrowNormal", radius, view);
  drawLocalArrow3D(localSvg, [0, 0, 0], sample.raw, arrowLength, "var(--raw)", "raw", "arrowRaw", radius, view);
  drawLocalArrow3D(localSvg, [0, 0, 0], sample.filtered, arrowLength, "var(--filtered)", "filtered", "arrowFiltered", radius, view);

  const obs = drawProjectedPoint(localSvg, obstacle, radius, view, {
    r: 9, fill: "var(--obstacle)", stroke: "#c76c00", "stroke-width": 1.4,
  });
  const cp = drawProjectedPoint(localSvg, [0, 0, 0], radius, view, {
    r: 8, fill: "#fff", stroke: "#111", "stroke-width": 2,
  });
  drawText(localSvg, [cp[0] + 8, cp[1] + 18], "active CP", {fill: "#111"});
  drawText(localSvg, [obs[0] + 8, obs[1] - 8], "obstacle", {fill: "var(--obstacle)", "font-weight": 650});
}
function signedBar(label, value, color, maxAbs) {
  const safe = finite(value) ? value : 0;
  const ratio = Math.min(Math.abs(safe) / Math.max(maxAbs, 1e-9), 1);
  const left = safe >= 0 ? 50 : 50 - ratio * 50;
  const width = ratio * 50;
  return `<div class="bar-row">
    <div>${label}</div>
    <div class="bar-track"><div class="bar-zero"></div><div class="bar-fill" style="left:${left}%;width:${width}%;background:${color}"></div></div>
    <div>${fmt(value, 2)}</div>
  </div>`;
}
function updateBars(sample) {
  const values = [
    sample.rawDotTangent,
    sample.filteredDotTangent,
    sample.rawDotNormal,
    sample.filteredDotNormal,
  ].filter(finite);
  const maxAbs = Math.max(...values.map(Math.abs), 1);
  document.getElementById("componentBars").innerHTML = [
    signedBar("raw tangent", sample.rawDotTangent, "var(--raw)", maxAbs),
    signedBar("filtered tangent", sample.filteredDotTangent, "var(--filtered)", maxAbs),
    signedBar("raw normal", sample.rawDotNormal, "var(--raw)", maxAbs),
    signedBar("filtered normal", sample.filteredDotNormal, "var(--filtered)", maxAbs),
  ].join("");
}
function candidateComponentValues(candidate) {
  return [
    ["goal", Number(candidate.weighted_goal || 0), "#4c78a8"],
    ["continuity", Number(candidate.weighted_continuity || 0), "#b279a2"],
    ["duplicate", Number(candidate.weighted_duplicate_risk || 0), "#d62728"],
    ["adjacent", Number(candidate.weighted_adjacent_block || 0), "#e45756"],
    ["branch", Number(candidate.weighted_branch_hold || 0), "#72b7b2"],
    ["up", Number(candidate.weighted_up || 0), "#9aa3af"],
  ];
}
function candidateScale(candidates) {
  const magnitudes = candidates.map((candidate) => {
    const values = candidateComponentValues(candidate).map(([, value]) => value);
    const positive = values.filter((value) => value > 0).reduce((a, b) => a + b, 0);
    const negative = Math.abs(values.filter((value) => value < 0).reduce((a, b) => a + b, 0));
    const total = Math.abs(Number(candidate.total_score || 0));
    return Math.max(positive, negative, total);
  });
  return Math.max(...magnitudes, 1e-6);
}
function candidateSegmentsHtml(candidate, maxMagnitude) {
  let positiveOffset = 0;
  let negativeOffset = 0;
  const scale = 48 / Math.max(maxMagnitude, 1e-9);
  const segments = [];
  candidateComponentValues(candidate).forEach(([, value, color]) => {
    if (!Number.isFinite(value) || Math.abs(value) < 1e-9) return;
    if (value >= 0) {
      const left = 50 + positiveOffset * scale;
      const width = Math.abs(value) * scale;
      segments.push(`<span class="candidate-segment" style="left:${left}%;width:${width}%;background:${color}"></span>`);
      positiveOffset += value;
    } else {
      const nextOffset = negativeOffset + value;
      const left = 50 + nextOffset * scale;
      const width = Math.abs(value) * scale;
      segments.push(`<span class="candidate-segment" style="left:${left}%;width:${width}%;background:${color}"></span>`);
      negativeOffset = nextOffset;
    }
  });
  if (candidate.selected) {
    segments.push(`<span class="candidate-selected-band"></span>`);
  }
  return segments.join("");
}
function updateCandidateScores(sample) {
  const payload = sample.candidateData || {};
  const candidates = activeCandidates(sample);
  const summary = document.getElementById("candidateSummary");
  const bars = document.getElementById("candidateBars");
  if (!payload.active || candidates.length === 0) {
    summary.textContent = "no active candidate data";
    bars.innerHTML = "";
    return;
  }
  const selected = candidates.find((candidate) => candidate.selected);
  const selectedIndex = selected ? selected.index : payload.selected_candidate_index;
  const selectedTotal = selected ? selected.total_score : null;
  const weights = payload.weights || {};
  summary.textContent =
    `selected ${selectedIndex ?? "n/a"}, total ${fmt(selectedTotal, 3)}, ` +
    `N=${candidates.length}, weights g:${fmt(weights.goal, 1)} dup:${fmt(weights.duplicate_risk, 1)} adj:${fmt(weights.adjacent_block, 1)} cont:${fmt(weights.continuity, 1)} hold:${fmt(weights.branch_hold, 1)}`;
  const maxMagnitude = candidateScale(candidates);
  bars.innerHTML = candidates.map((candidate) => {
    const cls = candidate.selected ? "candidate-row selected" : "candidate-row";
    const index = candidate.index ?? "";
    return `<div class="${cls}">
      <div>#${index}</div>
      <div class="candidate-track">${candidateSegmentsHtml(candidate, maxMagnitude)}</div>
      <div>${fmt(candidate.total_score, 2)}</div>
    </div>`;
  }).join("");
}
function render() {
  selectedIndex = Math.max(0, Math.min(selectedIndex, reportData.samples.length - 1));
  slider.value = String(selectedIndex);
  zoomValue.textContent = `${Number(zoomSlider.value).toFixed(2)} m`;
  const sample = reportData.samples[selectedIndex];
  drawOverall(sample);
  drawLocal(sample);
  updateBars(sample);
  updateCandidateScores(sample);
  document.getElementById("sampleLabel").textContent = `active sample ${selectedIndex + 1} / ${reportData.samples.length}`;
  document.getElementById("sampleTime").textContent = `t = ${fmt(sample.time, 2)} s`;
  document.getElementById("clearanceValue").textContent = `${fmt(sample.clearance, 4)} m`;
  document.getElementById("activationValue").textContent = fmt(sample.activation, 3);
  document.getElementById("angleValue").textContent = `${fmt(sample.rawFilteredAngleDeg, 1)} deg`;
  document.getElementById("orthogonalValue").textContent = fmt(sample.tangentDotNormal, 4);
}
slider.addEventListener("input", () => { selectedIndex = Number(slider.value); render(); });
zoomSlider.addEventListener("input", () => { render(); });
viewSelect.addEventListener("change", () => { render(); });
document.getElementById("prevBtn").addEventListener("click", () => { selectedIndex -= 1; render(); });
document.getElementById("nextBtn").addEventListener("click", () => { selectedIndex += 1; render(); });
render();
</script>
</body>
</html>
"""


def write_report(path: Path, samples: Sequence[Dict[str, object]], goals: Sequence[Sequence[float]], initial_index: int, output: Path) -> None:
    payload = {
        "samples": samples,
        "goals": [vector_or_list(goal) for goal in goals],
    }
    rendered = (
        html_template()
        .replace("__TITLE__", html.escape(f"Tangent Escape Filter Report - {path.name}"))
        .replace("__CSV_NAME__", html.escape(path.name))
        .replace("__DATA_JSON__", json.dumps(payload, separators=(",", ":")))
        .replace("__INITIAL_INDEX__", str(initial_index))
    )
    output.write_text(rendered, encoding="utf-8")


def vector_or_list(values: Sequence[float]) -> List[Optional[float]]:
    return [finite_or_none(value) for value in values]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create an interactive HTML report for tangent escape filter samples."
    )
    parser.add_argument("csv_path", nargs="?", help="Recorder CSV path. Defaults to latest CSV.")
    parser.add_argument(
        "--latest-dir",
        default=str(DEFAULT_DATA_DIR),
        help="Directory used when no CSV path is provided.",
    )
    parser.add_argument("--save", help="Output HTML path.")
    parser.add_argument(
        "--select",
        choices=("closest-clearance", "middle", "max-angle"),
        default="closest-clearance",
        help="Default selected sample when --sample-index and --time are not set.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        help="Initial active sample index, zero-based among active tangent escape rows.",
    )
    parser.add_argument(
        "--time",
        type=float,
        help="Initial active sample nearest this relative active-filter time in seconds.",
    )
    parser.add_argument(
        "--max-age",
        type=float,
        default=0.2,
        help="Ignore cached debug samples older than this many seconds.",
    )
    args = parser.parse_args()

    path = resolve_input_path(args.csv_path, args.latest_dir)
    fieldnames, rows = read_csv_rows(path)
    require_columns(fieldnames)
    selected_rows = active_rows(rows, args.max_age)
    if not selected_rows:
        raise ValueError(f"No active tangent escape rows found in: {path}")

    samples = [make_sample(selected_rows, row, index) for index, row in enumerate(selected_rows)]
    initial_index = choose_initial_index(samples, args)
    goals = goal_points(rows)
    output = (
        Path(args.save).expanduser()
        if args.save
        else path.with_name(f"{path.stem}_tangent_escape_report.html")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    write_report(path, samples, goals, initial_index, output)

    print(f"Input CSV: {path}")
    print(f"Active rows: {len(samples)}")
    print(f"Initial active sample index: {initial_index}")
    print(f"Saved report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
