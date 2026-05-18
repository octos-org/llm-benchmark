#!/usr/bin/env python3
"""Post-process JSONL results into a publishable Markdown report.

Reads every ``results-*.jsonl`` under a results directory and emits:

    results/<run>/summary.md   -- one cell per (model, N) with TTFT/total/relevant
    results/<run>/details.md   -- per-cell breakdown with min/p50/p95/max
    results/<run>/all.csv      -- flat CSV across all trials

Usage:
    python3 scripts/build_report.py results/2026-05-17
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _pct(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _fmt_ms(v: Optional[float]) -> str:
    return f"{round(v)}" if v is not None else "-"


def _cell(trials: List[Dict[str, Any]]) -> str:
    if not trials:
        return "-"
    ok_ttft = [t["ttft_ms"] for t in trials if t.get("ttft_ms") is not None]
    ok_total = [t["total_ms"] for t in trials if t.get("response_started")]
    rel = sum(1 for t in trials if t.get("selected_relevant")) / len(trials)
    errs = [t for t in trials if t.get("error")]
    if not ok_total and errs:
        statuses = sorted({str(t.get("http_status") or "?") for t in errs})
        return f"ERR({'/'.join(statuses)})"
    ttft_med = statistics.median(ok_ttft) if ok_ttft else None
    total_med = statistics.median(ok_total) if ok_total else None
    return f"{_fmt_ms(ttft_med)} / {_fmt_ms(total_med)} ms<br>rel={rel:.0%}"


def main(run_dir: str) -> int:
    root = Path(run_dir)
    if not root.exists():
        print(f"no such directory: {run_dir}", file=sys.stderr)
        return 1
    jsonls = sorted(root.glob("results-*.jsonl"))
    if not jsonls:
        print(f"no results-*.jsonl files under {run_dir}", file=sys.stderr)
        return 1

    all_rows: List[Dict[str, Any]] = []
    for path in jsonls:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                all_rows.append(json.loads(line))

    models: List[str] = list(dict.fromkeys(r["model"] for r in all_rows))
    n_values: List[int] = sorted({r["n_tools"] for r in all_rows})
    by_cell: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in all_rows:
        by_cell.setdefault((r["model"], r["n_tools"]), []).append(r)

    # ---- summary.md (compact cells) --------------------------------------
    header = "| Model | " + " | ".join(f"N={n}" for n in n_values) + " |"
    align = "|" + "|".join(["---"] * (len(n_values) + 1)) + "|"
    lines = ["# Tool-count benchmark — summary", "",
             "Cell format: `TTFT_med / total_med ms` over response-started trials, plus the rate at which the model selected the relevant `get_weather` tool.",
             "",
             header, align]
    for m in models:
        row = [f"`{m}`"] + [_cell(by_cell.get((m, n), [])) for n in n_values]
        lines.append("| " + " | ".join(row) + " |")
    (root / "summary.md").write_text("\n".join(lines) + "\n")

    # ---- details.md (per-cell stats) -------------------------------------
    detail = ["# Tool-count benchmark — per-cell details", ""]
    for m in models:
        detail.append(f"## `{m}`")
        detail.append("")
        detail.append("| N | trials | started | rel-tool | ttft p50 | ttft p95 | total p50 | total p95 | in tok | out tok p50 | errors |")
        detail.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for n in n_values:
            trials = by_cell.get((m, n), [])
            if not trials:
                continue
            ok_ttft = [t["ttft_ms"] for t in trials if t.get("ttft_ms") is not None]
            ok_total = [t["total_ms"] for t in trials if t.get("response_started")]
            in_toks = [t["input_tokens"] for t in trials if t.get("input_tokens")]
            out_toks = [t["output_tokens"] for t in trials if t.get("output_tokens")]
            started = sum(1 for t in trials if t.get("response_started"))
            rel = sum(1 for t in trials if t.get("selected_relevant"))
            errs = [t.get("error", "") for t in trials if t.get("error")]
            err_str = "; ".join(sorted({(e or "")[:60] for e in errs})) or "—"
            detail.append(
                f"| {n} | {len(trials)} | {started} | {rel}/{len(trials)} | "
                f"{_fmt_ms(_pct(ok_ttft,50))} | {_fmt_ms(_pct(ok_ttft,95))} | "
                f"{_fmt_ms(_pct(ok_total,50))} | {_fmt_ms(_pct(ok_total,95))} | "
                f"{in_toks[0] if in_toks else '-'} | "
                f"{round(statistics.median(out_toks)) if out_toks else '-'} | "
                f"{err_str} |"
            )
        detail.append("")
    (root / "details.md").write_text("\n".join(detail) + "\n")

    # ---- all.csv ----------------------------------------------------------
    fields = ["model", "n_tools", "trial", "ttft_ms", "total_ms",
              "emitted_tool_call", "tool_call_name", "selected_relevant",
              "finish_reason", "response_started", "http_status", "error",
              "input_tokens", "output_tokens"]
    with open(root / "all.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in fields})

    print(f"wrote {root/'summary.md'}")
    print(f"wrote {root/'details.md'}")
    print(f"wrote {root/'all.csv'}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: build_report.py <run_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
