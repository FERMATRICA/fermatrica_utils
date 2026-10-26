"""
Utilities to work with `line_profiler` (`kernprof`) output files: reading `.lprof` dumps and
rendering them as interactive HTML reports
"""


import html
import webbrowser
from pathlib import Path

from fermatrica_utils.flow import fermatrica_utils_error


def _fmt_seconds(t: float) -> str:
    if t >= 1:
        return f"{t:.3f} s"
    if t >= 0.001:
        return f"{t * 1000:.2f} ms"
    return f"{t * 1_000_000:.1f} µs"


def _read_source_lines(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    except OSError:
        return []


def lprof_to_html(lprof_path, out_path=None, open_browser=False) -> str:
    """
    Convert a `line_profiler` `.lprof` file (as produced by `kernprof -l -o`) into a self-contained
    interactive HTML report and write it to disk.

    The report contains a summary of profiled functions sorted by total time and, for every function,
    a per-line breakdown (hits, time, per-hit time, share of function time) with source lines when
    the source file is available.

    :param lprof_path: path to `.lprof` file
    :param out_path: path to write the HTML report to. Defaults to `<lprof_path>.html` next to the input
    :param open_browser: if True, open the report in the default browser
    :return: path of the written HTML report
    """

    try:
        from line_profiler.line_profiler import LineStats
    except ImportError:
        fermatrica_utils_error(
            "`line_profiler` is not installed. Install it (see requirements.txt) to use `lprof_to_html()`"
        )

    try:
        stats = LineStats.from_files(lprof_path)
    except FileNotFoundError:
        fermatrica_utils_error(f"`.lprof` file not found: {lprof_path}")

    unit = stats.unit

    rows = []
    for (filename, firstlineno, func_name), lines in stats.timings.items():
        total = sum(ticks for _, _, ticks in lines) * unit
        rows.append(
            {
                "name": func_name,
                "file": filename,
                "start": firstlineno,
                "total": total,
                "hits": sum(hits for _, hits, _ in lines),
                "lines": sorted(lines, key=lambda t: t[0]),
            }
        )
    rows.sort(key=lambda r: r["total"], reverse=True)

    total_all = sum(r["total"] for r in rows)

    body = ""
    for r in rows:
        pct = (r["total"] / total_all * 100) if total_all else 0.0
        sub = ""
        src = _read_source_lines(r["file"])
        for lineno, hits, ticks in r["lines"]:
            lt = ticks * unit
            lpct = (lt / r["total"] * 100) if r["total"] else 0.0
            text = src[lineno - 1] if lineno <= len(src) else ""
            sub += (
                "<tr>"
                f"<td class='ln'>{lineno}</td>"
                f"<td class='code'><pre>{html.escape(text)}</pre></td>"
                f"<td class='n'>{hits:,}</td>"
                f"<td class='n'>{_fmt_seconds(lt)}</td>"
                f"<td class='n'>{_fmt_seconds(lt / hits) if hits else '-'}</td>"
                f"<td class='n'>{lpct:.1f}</td>"
                "<td class='bar'><div class='fill' style='width:"
                f"{lpct:.1f}%'></div></td>"
                "</tr>"
            )
        body += (
            "<tr class='fn' onclick='toggle(this)'>"
            f"<td>{html.escape(r['name'])}</td>"
            f"<td class='file'>{html.escape(Path(r['file']).name)}:{r['start']}</td>"
            f"<td class='n'>{r['hits']:,}</td>"
            f"<td class='n'>{_fmt_seconds(r['total'])}</td>"
            f"<td class='n'>{pct:.1f}</td>"
            f"<td class='bar'><div class='fill' style='width:{pct * 4:.1f}%'></div></td>"
            "</tr>"
            f"<tr class='lines' style='display:none'><td colspan='6'><table>"
            "<tr><th>Line</th><th>Source</th><th>Hits</th><th>Time</th>"
            "<th>Per Hit</th><th>%</th><th></th></tr>"
            f"{sub}</table></td></tr>"
        )

    html_page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>line profiler</title><style>
body{{font-family:Segoe UI,sans-serif;margin:16px;color:#d4d4d4;background:#1e1e1e}}
h1{{font-size:16px}}p.meta{{margin:0 0 12px;color:#888;font-size:12px}}
table{{border-collapse:collapse;width:100%;margin-bottom:8px}}
th{{text-align:left;font-size:11px;color:#888;border-bottom:1px solid #333;padding:2px 8px}}
td{{padding:2px 8px;font-size:12px;border-bottom:1px solid #222;vertical-align:top}}
tr.fn{{cursor:pointer}}tr.fn:hover{{background:#2a2a2a}}
td.n{{text-align:right}}td.ln{{text-align:right;color:#888;width:40px}}
td.file{{color:#888;font-size:11px}}
pre{{margin:0;font-family:Consolas,monospace;white-space:pre}}
tr.lines>td{{background:#252526}}tr.lines table{{margin:0}}
tr.lines th{{color:#666}}
div.bar{{min-width:120px}}div.fill{{height:12px;background:#0e639c}}
input{{background:#333;color:#d4d4d4;border:1px solid #555;padding:3px 6px;width:260px}}
</style></head><body>
<h1>{html.escape(str(lprof_path))}</h1>
<p class='meta'>{len(rows)} functions, total profiled time {_fmt_seconds(total_all)}</p>
<p><input id='q' oninput='filterFn(this.value)' placeholder='filter functions...'></p>
<table><tr><th>Function</th><th>File</th><th>Hits</th><th>Total</th><th>%</th><th></th></tr>
{body}</table>
<script>
function toggle(r){{var n=r.nextElementSibling;if(n&&n.classList.contains('lines')){{var open=n.style.display!=='none';n.style.display=open?'none':'';r.dataset.open=open?'0':'1';}}}}
function filterFn(v){{v=v.toLowerCase();document.querySelectorAll('tr.fn').forEach(function(r){{
var show=r.cells[0].textContent.toLowerCase().includes(v);r.style.display=show?'':'none';
var n=r.nextElementSibling;if(n&&n.classList.contains('lines')){{n.style.display=(show&&r.dataset.open==='1')?'':'none';}}}});}}
</script></body></html>"""

    if out_path is None:
        out_path = Path(lprof_path).with_suffix(".html")
    out_path = Path(out_path)
    out_path.write_text(html_page, encoding="utf-8")

    if open_browser:
        webbrowser.open(out_path.resolve().as_uri())

    return str(out_path)
