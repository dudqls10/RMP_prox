#!/usr/bin/env python3

import argparse
import html
import os
from pathlib import Path
from typing import List, Tuple

from analyze_rmp_leaf_ablation import (
    LEAF_GROUPS,
    LEAF_LABELS,
    Analysis,
    analyze,
    load_rows,
)


def parse_case(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("case must use LABEL=CSV_PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("case label must not be empty")
    path = Path(os.path.expanduser(raw_path.strip())).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"CSV does not exist: {path}")
    return label, path


def leaf_reduction(result: Analysis, leaf: str) -> float:
    return sum(
        max(value, 0.0)
        for stats in result.leaf_stats[leaf]
        for value in stats.excess_reductions
    )


def leaf_relief_count(result: Analysis, leaf: str) -> int:
    return sum(stats.relief_count for stats in result.leaf_stats[leaf])


def leaf_top_count(result: Analysis, leaf: str) -> int:
    return sum(stats.top_cause_count for stats in result.leaf_stats[leaf])


def leaf_active_rows(result: Analysis, leaf: str) -> int:
    return max((stats.active_rows for stats in result.leaf_stats[leaf]), default=0)


def write_html(cases: List[Tuple[str, Analysis]], output_path: Path) -> None:
    summary_rows = []
    maximum_rates = []
    for label, result in cases:
        saturation_rate = 100.0 * result.saturation_rows / max(result.rows, 1)
        maximum_rates.append(saturation_rate)
        escape_active = leaf_active_rows(result, "tangent_escape")
        escape_top = leaf_top_count(result, "tangent_escape")
        escape_relief = leaf_relief_count(result, "tangent_escape")
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{result.rows}</td>"
            f"<td>{result.duration_s:.3f}</td>"
            f"<td>{result.saturation_rows}</td>"
            f"<td>{saturation_rate:.2f}%</td>"
            f"<td>{result.longest_saturation_s:.3f}</td>"
            f"<td>{result.minimum_clip_cosine:.4f}</td>"
            f"<td>{escape_active}</td>"
            f"<td>{escape_top}</td>"
            f"<td>{escape_relief}</td>"
            "</tr>"
        )

    bar_maximum = max(maximum_rates + [1.0])
    bar_rows = []
    colors = ["#c9362b", "#2979a8", "#40845a", "#805b99"]
    for index, ((label, result), rate) in enumerate(zip(cases, maximum_rates)):
        width = 100.0 * rate / bar_maximum
        bar_rows.append(
            '<div class="bar-row">'
            f'<span class="bar-label">{html.escape(label)}</span>'
            '<span class="bar-track">'
            f'<i style="width:{width:.3f}%;background:{colors[index % len(colors)]}"></i>'
            "</span>"
            f'<strong>{rate:.2f}%</strong>'
            "</div>"
        )

    joint_rows = []
    for joint in range(6):
        cells = [f"<td>J{joint + 1}</td>"]
        for _, result in cases:
            summary = result.joint_summaries[joint]
            top_label = LEAF_LABELS.get(summary.top_leaf, summary.top_leaf)
            cells.append(
                "<td>"
                f"<strong>{summary.max_abs_raw:.3f}</strong>"
                f"<small>{html.escape(top_label)} ({summary.top_leaf_count})</small>"
                "</td>"
            )
        joint_rows.append("<tr>" + "".join(cells) + "</tr>")

    leaf_rows = []
    for leaf in LEAF_GROUPS:
        if not any(leaf_active_rows(result, leaf) for _, result in cases):
            continue
        cells = [f"<td>{html.escape(LEAF_LABELS[leaf])}</td>"]
        for _, result in cases:
            reduction_per_1000_rows = (
                1000.0 * leaf_reduction(result, leaf) / max(result.rows, 1)
            )
            cells.append(
                "<td>"
                f"<strong>{reduction_per_1000_rows:.3f}</strong>"
                f"<small>top {leaf_top_count(result, leaf)}, "
                f"relief {leaf_relief_count(result, leaf)}</small>"
                "</td>"
            )
        leaf_rows.append("<tr>" + "".join(cells) + "</tr>")

    case_headers = "".join(f"<th>{html.escape(label)}</th>" for label, _ in cases)
    source_rows = "".join(
        f"<li><strong>{html.escape(label)}</strong>: {html.escape(str(result.path))}</li>"
        for label, result in cases
    )
    document = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RMP leaf 고정 시나리오 비교</title>
<style>
body {{ margin:0; font-family:Arial,sans-serif; color:#1c252d; background:#fff; }}
header {{ background:#17242d; color:#fff; padding:24px max(24px,calc((100% - 1180px)/2)); }}
main {{ max-width:1180px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 8px; font-size:25px; letter-spacing:0; }}
h2 {{ margin:30px 0 10px; font-size:19px; letter-spacing:0; }}
.meta {{ color:#cbd6dc; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th,td {{ border:1px solid #d8dde2; padding:8px 9px; text-align:right; vertical-align:top; }}
th {{ background:#eef2f4; }}
th:first-child,td:first-child {{ text-align:left; }}
td small {{ display:block; margin-top:4px; color:#65717a; white-space:normal; }}
.bar-row {{ display:grid; grid-template-columns:180px minmax(180px,1fr) 70px; gap:12px; align-items:center; margin:10px 0; }}
.bar-track {{ height:18px; background:#eef2f4; border:1px solid #d8dde2; }}
.bar-track i {{ display:block; height:100%; }}
.bar-row strong {{ text-align:right; }}
.note {{ padding:12px 14px; border-left:4px solid #2979a8; background:#eef6fa; line-height:1.55; }}
.sources {{ color:#57636d; font-size:13px; line-height:1.6; overflow-wrap:anywhere; }}
@media(max-width:760px) {{ main {{ padding:14px; overflow-x:auto; }} .bar-row {{ grid-template-columns:120px minmax(120px,1fr) 60px; }} }}
</style>
</head>
<body>
<header>
  <h1>RMP leaf 고정 시나리오 비교</h1>
  <div class="meta">동일 목표 궤적, 센서 입력과 Escape 활성 조건 비교</div>
</header>
<main>
  <h2>포화 요약</h2>
  <table>
    <thead><tr><th>조건</th><th>샘플</th><th>시간(s)</th><th>포화 행</th><th>포화율</th><th>최장 포화(s)</th><th>최소 방향 보존</th><th>Escape 활성</th><th>Escape top</th><th>Escape relief</th></tr></thead>
    <tbody>{''.join(summary_rows)}</tbody>
  </table>

  <h2>포화 행 비율</h2>
  {''.join(bar_rows)}

  <h2>관절별 최대 raw qdd와 최다 원인 leaf</h2>
  <table>
    <thead><tr><th>관절</th>{case_headers}</tr></thead>
    <tbody>{''.join(joint_rows)}</tbody>
  </table>

  <h2>Leaf 제거 효과</h2>
  <p class="note">굵은 값은 샘플 1000개당 포화 초과량 감소 합계입니다. top은 해당 leaf가 가장 큰 감소를 만든 횟수, relief는 leaf 제거만으로 제한 안쪽에 들어온 관절 표본 수입니다. 이는 같은 상태에서의 국소 반사실 분석이며 leaf별 가속도의 선형 분해가 아닙니다.</p>
  <table>
    <thead><tr><th>Leaf</th>{case_headers}</tr></thead>
    <tbody>{''.join(leaf_rows)}</tbody>
  </table>

  <h2>입력 파일</h2>
  <ul class="sources">{source_rows}</ul>
</main>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two or more RMP leaf-ablation trace CSV files."
    )
    parser.add_argument(
        "--case",
        action="append",
        required=True,
        type=parse_case,
        metavar="LABEL=CSV",
        help="Named trace input; repeat at least twice.",
    )
    parser.add_argument(
        "--output",
        default="~/ros2_ws/log/rmpflow_trace/rmp_leaf_ablation_comparison.html",
        help="HTML output path.",
    )
    args = parser.parse_args()
    if len(args.case) < 2:
        parser.error("provide at least two --case arguments")

    results: List[Tuple[str, Analysis]] = []
    for label, path in args.case:
        _, rows = load_rows(path)
        results.append((label, analyze(path, rows)))

    output_path = Path(os.path.expanduser(args.output)).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_html(results, output_path)
    print(f"Saved comparison: {output_path}")
    for label, result in results:
        rate = 100.0 * result.saturation_rows / max(result.rows, 1)
        print(
            f"  {label}: saturation={result.saturation_rows}/{result.rows} "
            f"({rate:.2f}%), longest={result.longest_saturation_s:.3f}s, "
            f"escape_top={leaf_top_count(result, 'tangent_escape')}, "
            f"escape_relief={leaf_relief_count(result, 'tangent_escape')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
