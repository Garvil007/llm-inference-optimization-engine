"""Renders a ReportData into a standalone HTML document -- same content as the
Markdown renderer (via report/content.py's shared table builders), laid out
with plain inline CSS so it opens correctly from a file:// URL with no
external assets other than the plot PNGs sitting next to it.
"""

from __future__ import annotations

import html

from inference_optimizer.report.content import hardware_table, metrics_table, model_table
from inference_optimizer.report.models import ReportData

_STYLE = (
    "body { font-family: -apple-system, 'Segoe UI', sans-serif; max-width: 900px; "
    "margin: 2rem auto; color: #222; } "
    "table { border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; } "
    "th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; } "
    "th { background: #f0f0f0; } "
    ".flag-error { color: #b00020; } .flag-warning { color: #a86400; } .flag-info { color: #555; } "
    "img { max-width: 100%; margin: 0.5rem 0; }"
)


def _esc(value: object) -> str:
    return html.escape(str(value))


def _table(headers: list[str], rows: list[list[object]]) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(report_data: ReportData) -> str:
    report = report_data.baseline_report
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Baseline report {_esc(report.run_metadata.run_id)}</title>",
        f"<style>{_STYLE}</style></head><body>",
        f"<h1>Baseline report: run {_esc(report.run_metadata.run_id)}</h1>",
        f"<p>Status: <strong>{_esc(report.status)}</strong></p>",
    ]

    if report.status == "unavailable":
        parts.append(f"<p>Unavailable reason: {_esc(report.unavailable_reason)}</p>")
        parts.append("</body></html>")
        return "".join(parts)

    parts.append("<h2>Hardware</h2>")
    parts.append(_table(["Field", "Value"], [[k, v] for k, v in hardware_table(report)]))
    parts.append("<h2>Model &amp; backend</h2>")
    parts.append(_table(["Field", "Value"], [[k, v] for k, v in model_table(report)]))

    metric_rows = metrics_table(report)
    if metric_rows:
        headers = list(metric_rows[0].keys())
        parts.append("<h2>Per-repeat metrics</h2>")
        parts.append(_table(headers, [[row[h] for h in headers] for row in metric_rows]))

    parts.append("<h2>Repeat-to-repeat variability</h2><ul>")
    for stat in report_data.variability:
        if stat.coefficient_of_variation is not None:
            parts.append(
                f"<li>{_esc(stat.metric)}: mean={stat.mean:.2f}, stddev={stat.stddev:.2f}, "
                f"CV={stat.coefficient_of_variation:.3f}</li>"
            )
        else:
            parts.append(
                f"<li>{_esc(stat.metric)}: not enough repeats to compute variability</li>"
            )
    parts.append("</ul>")

    parts.append("<h2>Flags</h2><ul>")
    for flag in report_data.flags:
        parts.append(
            f"<li class='flag-{flag.level}'>[{flag.level.upper()}] "
            f"{_esc(flag.code)}: {_esc(flag.message)}</li>"
        )
    if not report_data.flags:
        parts.append("<li>none</li>")
    parts.append("</ul>")

    parts.append("<h2>Comparisons (internal consistency only)</h2><ul>")
    for note in report_data.comparisons:
        status = "PASS" if note.passed else "FAIL"
        parts.append(f"<li>[{status}] {_esc(note.description)} -- {_esc(note.detail)}</li>")
    parts.append("</ul>")

    if report_data.plot_paths:
        parts.append("<h2>Distribution plots</h2>")
        for name, path in report_data.plot_paths.items():
            parts.append(f"<img src='{_esc(path.name)}' alt='{_esc(name)}'>")

    parts.append("<h2>Validation against raw observations</h2><ul>")
    for result in report_data.validation:
        status = "OK" if result.matches else "MISMATCH"
        parts.append(f"<li>repeat {result.repeat_index}: {status}</li>")
    parts.append("</ul>")

    parts.append("</body></html>")
    return "".join(parts)
