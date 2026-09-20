"""Report writers: machine-readable JSON and a self-contained dark-theme HTML.

The HTML report has no external assets (inline CSS only, no CDN) so it renders
identically offline and can be attached to a CI run or emailed.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from eval_gate.gate import GateVerdict
from eval_gate.runner import SuiteResult


def write_json(result: SuiteResult, path: str | Path, verdict: GateVerdict | None = None) -> Path:
    """Write the suite result (plus optional gate verdict) as JSON. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = result.to_dict()
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    if verdict is not None:
        payload["gate"] = {
            "passed": verdict.passed,
            "exit_code": verdict.exit_code,
            "reasons": verdict.reasons,
            "regressions": verdict.regressions,
        }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


_CSS = """
:root { color-scheme: dark; }
* { box-sizing: border-box; }
body { margin: 0; background: #0d1117; color: #e6edf3;
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 1040px; margin: 0 auto; padding: 32px 24px 80px; }
h1 { font-size: 24px; margin: 0 0 4px; }
.sub { color: #8b949e; margin: 0 0 24px; font-size: 13px; }
.verdict { display: flex; align-items: center; gap: 14px; padding: 18px 22px; border-radius: 12px;
  margin-bottom: 26px; border: 1px solid; }
.verdict.pass { background: #0f2417; border-color: #1f7a3f; }
.verdict.fail { background: #2a1416; border-color: #a03038; }
.verdict .big { font-size: 20px; font-weight: 700; letter-spacing: .3px; }
.verdict.pass .big { color: #3fb950; }
.verdict.fail .big { color: #f85149; }
.verdict ul { margin: 6px 0 0; padding-left: 18px; color: #d0879a; font-size: 13px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 28px; }
.stat { background: #161b22; border: 1px solid #21262d; border-radius: 10px; padding: 14px 16px; }
.stat .n { font-size: 22px; font-weight: 700; }
.stat .l { color: #8b949e; font-size: 12px; text-transform: uppercase; letter-spacing: .5px; }
.case { background: #161b22; border: 1px solid #21262d; border-radius: 10px; margin-bottom: 14px; overflow: hidden; }
.case > summary { cursor: pointer; padding: 14px 18px; display: flex; align-items: center; gap: 12px;
  list-style: none; user-select: none; }
.case > summary::-webkit-details-marker { display: none; }
.badge { font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 20px; letter-spacing: .4px; }
.badge.pass { background: #163a25; color: #3fb950; }
.badge.fail { background: #3a1720; color: #f85149; }
.case .q { flex: 1; font-weight: 600; }
.case .meta { color: #8b949e; font-size: 12px; }
.body { padding: 0 18px 16px; }
.answer { background: #0d1117; border: 1px solid #21262d; border-radius: 8px; padding: 12px 14px;
  white-space: pre-wrap; font-size: 13px; color: #c9d1d9; margin: 6px 0 14px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #21262d; vertical-align: top; }
th { color: #8b949e; font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }
td.s { white-space: nowrap; }
.ok { color: #3fb950; } .no { color: #f85149; }
code { background: #21262d; padding: 1px 5px; border-radius: 4px; font-size: 12px; }
.docs { color: #8b949e; font-size: 12px; margin-top: 6px; }
footer { color: #6e7681; font-size: 12px; margin-top: 40px; }
"""


def _esc(x: Any) -> str:
    return html.escape(str(x))


def render_html(result: SuiteResult, verdict: GateVerdict) -> str:
    """Render a self-contained dark-theme HTML report string."""
    v_class = "pass" if verdict.passed else "fail"
    v_word = "GATE PASSED" if verdict.passed else "GATE FAILED"
    reasons_html = ""
    if verdict.reasons:
        items = "".join(f"<li>{_esc(r)}</li>" for r in verdict.reasons)
        reasons_html = f"<ul>{items}</ul>"

    stats = [
        ("Pass rate", f"{result.pass_rate * 100:.0f}%"),
        ("Threshold", f"{result.threshold * 100:.0f}%"),
        ("Cases", f"{result.n_passed}/{result.n_cases}"),
        ("Checks", f"{result.passed_checks}/{result.total_checks}"),
        ("Provider", _esc(result.provider)),
    ]
    stats_html = "".join(
        f'<div class="stat"><div class="n">{v}</div><div class="l">{l}</div></div>'
        for l, v in stats
    )

    cases_html = []
    for c in result.cases:
        badge = "pass" if c.passed else "fail"
        rows = "".join(
            f'<tr><td class="s"><span class="{"ok" if ck.passed else "no"}">'
            f'{"PASS" if ck.passed else "FAIL"}</span></td>'
            f"<td><code>{_esc(ck.check_type)}</code></td>"
            f"<td>{_esc(ck.message)}</td></tr>"
            for ck in c.checks
        )
        retrieved = ", ".join(_esc(d) for d in c.retrieved_ids) or "none"
        relevant = ", ".join(_esc(d) for d in c.relevant_ids) or "none"
        cases_html.append(
            f"""
    <details class="case" {"open" if not c.passed else ""}>
      <summary>
        <span class="badge {badge}">{"PASS" if c.passed else "FAIL"}</span>
        <span class="q">{_esc(c.question)}</span>
        <span class="meta">{c.n_passed}/{len(c.checks)} checks</span>
      </summary>
      <div class="body">
        <div class="answer">{_esc(c.answer)}</div>
        <div class="docs">retrieved: <code>{retrieved}</code> &nbsp; relevant: <code>{relevant}</code></div>
        <table>
          <thead><tr><th>Result</th><th>Check</th><th>Detail</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </details>"""
        )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>eval-gate report: {_esc(result.name)}</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
  <h1>{_esc(result.name)}</h1>
  <p class="sub">eval-gate evaluation report &middot; generated {generated}</p>
  <div class="verdict {v_class}"><span class="big">{v_word}</span>{reasons_html}</div>
  <div class="grid">{stats_html}</div>
  {"".join(cases_html)}
  <footer>Deterministic offline evaluation. Retrieval computed locally (TF-IDF);
  answers from provider &ldquo;{_esc(result.provider)}&rdquo;. No external assets.</footer>
</div></body></html>"""


def write_html(result: SuiteResult, verdict: GateVerdict, path: str | Path) -> Path:
    """Write the HTML report to *path*. Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(render_html(result, verdict), encoding="utf-8")
    return p
