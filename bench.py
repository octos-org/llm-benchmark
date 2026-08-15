#!/usr/bin/env python3
"""LLM tool-count benchmark driver.

Sweeps (model, N, trial) tuples, sending each request as a streaming
chat completion with N tool definitions. Records TTFT, total latency,
output tokens, and which tool (if any) the model selected.

Usage examples
--------------

# Run a single provider+model from CLI flags:
DEEPSEEK_API_KEY=... python3 bench.py \\
    --provider deepseek --model deepseek-v4-flash \\
    --n 10 --n 50 --n 100 --trials 3 \\
    --out results/deepseek-v4-flash.jsonl

# Drive a multi-run sweep from a YAML config:
python3 bench.py --config configs/run-2026-05-17.yaml

The YAML config is a list of runs; each run is a dict with the same
fields as the CLI flags. ``out`` may be a path or a directory; when a
directory, file names are derived from the model.

Outputs
-------
For each --out PATH the script writes:
  PATH                 -- JSONL, one trial per line
  PATH.summary.json    -- per-(model, N) aggregate (median, p95, range)

After the full run, a combined CSV (--csv) and a Markdown table (--md)
covering all (model, N) cells are written.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx

# Local imports keep us a flat package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from providers import ProviderConfig, TrialResult, make_provider, run_trial  # noqa: E402
from tools import build_tools  # noqa: E402

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - YAML is only needed for --config.
    yaml = None


# ---------------------------------------------------------------------------
# Pricing table (USD per 1M tokens) -- per public pricing as of May 2026.
# These are intentionally rough; we only use them to print an estimated
# spend so we can honour a hard cap.
# ---------------------------------------------------------------------------

PRICING_USD_PER_M: Dict[str, Tuple[float, float]] = {
    # DeepSeek direct (https://api-docs.deepseek.com/quick_start/pricing)
    "deepseek-v4-flash":               (0.07, 1.10),
    "deepseek-v4-pro":                 (0.55, 2.19),
    "deepseek/deepseek-v4-flash":      (0.07, 1.10),
    "deepseek/deepseek-v4-pro":        (0.55, 2.19),
    # Anthropic via OpenRouter (OR passes through Anthropic list pricing).
    "anthropic/claude-opus-4.7":       (15.0, 75.0),
    "anthropic/claude-opus-4.7-fast":  (15.0, 75.0),
    "anthropic/claude-sonnet-4.6":     (3.0,  15.0),
}


def estimate_cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    in_per_m, out_per_m = PRICING_USD_PER_M.get(model, (0.0, 0.0))
    return (in_tok / 1_000_000.0) * in_per_m + (out_tok / 1_000_000.0) * out_per_m


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def percentile(xs: List[float], pct: float) -> Optional[float]:
    """Inclusive linear-interpolation percentile (matches numpy default)."""
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize(results: List[TrialResult]) -> Dict[str, Any]:
    ok_ttft = [r.ttft_ms for r in results if r.ttft_ms is not None]
    ok_total = [r.total_ms for r in results if r.response_started]
    sel = sum(1 for r in results if r.selected_relevant)
    any_tool = sum(1 for r in results if r.emitted_tool_call)
    started = sum(1 for r in results if r.response_started)
    errors = [r.error for r in results if r.error]
    return {
        "trials": len(results),
        "ttft_median_ms": statistics.median(ok_ttft) if ok_ttft else None,
        "ttft_p95_ms": percentile(ok_ttft, 95) if ok_ttft else None,
        "ttft_min_ms": min(ok_ttft) if ok_ttft else None,
        "ttft_max_ms": max(ok_ttft) if ok_ttft else None,
        "total_median_ms": statistics.median(ok_total) if ok_total else None,
        "total_p95_ms": percentile(ok_total, 95) if ok_total else None,
        "total_min_ms": min(ok_total) if ok_total else None,
        "total_max_ms": max(ok_total) if ok_total else None,
        "relevant_tool_rate": sel / len(results) if results else 0,
        "any_tool_rate": any_tool / len(results) if results else 0,
        "response_start_rate": started / len(results) if results else 0,
        "errors": errors,
        "tool_call_names": sorted({r.tool_call_name for r in results if r.tool_call_name}),
        "finish_reasons": sorted({r.finish_reason for r in results if r.finish_reason}),
    }


# ---------------------------------------------------------------------------
# Run plumbing
# ---------------------------------------------------------------------------


@dataclass
class RunSpec:
    provider: str
    model: str
    n_values: List[int]
    trials: int
    timeout_s: float
    relevant_position: str = "first"
    out: Optional[str] = None


def _resolve_out(out: Optional[str], model: str, default_dir: Path) -> Path:
    if not out:
        slug = model.replace("/", "_").replace(":", "_")
        return default_dir / f"results-{slug}.jsonl"
    p = Path(out)
    if p.is_dir() or out.endswith(os.sep):
        slug = model.replace("/", "_").replace(":", "_")
        return p / f"results-{slug}.jsonl"
    return p


def _execute_run(
    run: RunSpec,
    client: httpx.Client,
    default_dir: Path,
    spend_cap_usd: Optional[float],
    running_spend_usd: float,
) -> Tuple[List[TrialResult], float, Path]:
    """Execute one (provider, model) sweep. Returns (results, spend_delta, out_path)."""
    provider: ProviderConfig = make_provider(run.provider)
    out_path = _resolve_out(run.out, run.model, default_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = out_path.with_suffix(".summary.json")
    all_results: List[TrialResult] = []
    all_summaries: List[Dict[str, Any]] = []
    spend = 0.0

    print(
        f"[bench] >>> provider={provider.name} model={run.model} "
        f"N={run.n_values} trials={run.trials} timeout={run.timeout_s}s",
        flush=True,
    )

    with open(out_path, "w") as fh:
        for n in run.n_values:
            tools = build_tools(n, position=run.relevant_position)
            trial_results: List[TrialResult] = []
            for t in range(run.trials):
                if spend_cap_usd is not None and (running_spend_usd + spend) >= spend_cap_usd:
                    print(
                        f"[bench]   spend cap ${spend_cap_usd:.2f} reached "
                        f"(running ${running_spend_usd + spend:.4f}); skipping remainder",
                        flush=True,
                    )
                    break
                print(
                    f"[bench]   {run.model} N={n} trial={t+1}/{run.trials} ...",
                    end="",
                    flush=True,
                )
                r = run_trial(provider, run.model, tools, run.timeout_s, client)
                r.trial = t
                trial_results.append(r)
                all_results.append(r)
                fh.write(json.dumps(r.to_dict()) + "\n")
                fh.flush()
                if r.input_tokens and r.output_tokens:
                    delta = estimate_cost_usd(run.model, r.input_tokens, r.output_tokens)
                    spend += delta
                status = (
                    "PASS" if r.selected_relevant
                    else ("OTHER-TOOL" if r.emitted_tool_call
                          else ("STARTED" if r.response_started else "NO-START"))
                )
                if r.error:
                    status = f"ERR({(r.error or '')[:60]})"
                ttft_s = f"ttft={round(r.ttft_ms)}" if r.ttft_ms else "ttft=-"
                print(
                    f" {status} {ttft_s} total={round(r.total_ms)}ms "
                    f"tokens={r.input_tokens or '-'}/{r.output_tokens or '-'} "
                    f"tool={r.tool_call_name or '-'}",
                    flush=True,
                )
            summary = summarize(trial_results)
            summary.update({"model": run.model, "n_tools": n})
            all_summaries.append(summary)
            print(
                f"[bench]   => N={n}: ttft_med={summary['ttft_median_ms']} "
                f"total_med={summary['total_median_ms']} "
                f"relevant={summary['relevant_tool_rate']:.2f} "
                f"any-tool={summary['any_tool_rate']:.2f}",
                flush=True,
            )

    with open(summary_path, "w") as fh:
        json.dump(all_summaries, fh, indent=2)
    print(f"[bench] summary -> {summary_path}", flush=True)
    print(f"[bench] estimated spend for {run.model}: ${spend:.4f}", flush=True)
    return all_results, spend, out_path


# ---------------------------------------------------------------------------
# CSV + Markdown emit
# ---------------------------------------------------------------------------


def emit_csv(rows: Iterable[TrialResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model", "n_tools", "trial", "ttft_ms", "total_ms",
        "emitted_tool_call", "tool_call_name", "selected_relevant",
        "finish_reason", "response_started", "http_status", "error",
        "input_tokens", "output_tokens",
    ]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = r.to_dict()
            w.writerow({k: row.get(k) for k in fields})


def emit_markdown_table(
    results: List[TrialResult],
    n_values: List[int],
    models: List[str],
    path: Path,
) -> None:
    """Write a markdown summary table.

    Format: one row per model, one column per N. Each cell is
        TTFT_med / total_med ms (relevant tool rate)
    or  ERR(<status>) when no trial succeeded.
    """
    by_cell: Dict[Tuple[str, int], List[TrialResult]] = {}
    for r in results:
        by_cell.setdefault((r.model, r.n_tools), []).append(r)

    header = "| Model | " + " | ".join(f"N={n}" for n in n_values) + " |"
    align = "|" + "|".join(["---"] * (len(n_values) + 1)) + "|"
    lines = [header, align]
    for m in models:
        cells = []
        for n in n_values:
            trials = by_cell.get((m, n), [])
            if not trials:
                cells.append("-")
                continue
            ok_ttft = [t.ttft_ms for t in trials if t.ttft_ms is not None]
            ok_total = [t.total_ms for t in trials if t.response_started]
            sel = sum(1 for t in trials if t.selected_relevant) / len(trials)
            errs = [t for t in trials if t.error]
            if not ok_total:
                # All trials errored at request-start; show the http status.
                statuses = sorted({t.http_status for t in errs if t.http_status})
                msg = "/".join(str(s) for s in statuses) or "ERR"
                cells.append(f"ERR({msg})")
                continue
            ttft_med = statistics.median(ok_ttft) if ok_ttft else None
            total_med = statistics.median(ok_total)
            ttft_str = f"{round(ttft_med)}" if ttft_med else "-"
            cells.append(
                f"{ttft_str}/{round(total_med)}ms (rel={sel:.0%})"
            )
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI / config loader
# ---------------------------------------------------------------------------


def _parse_cli_run(args: argparse.Namespace) -> List[RunSpec]:
    if not args.model:
        raise SystemExit("at least one --model is required when --config is absent")
    n_values = args.n or [10, 50, 100]
    return [
        RunSpec(
            provider=args.provider,
            model=m,
            n_values=n_values,
            trials=args.trials,
            timeout_s=args.timeout,
            relevant_position=args.relevant_position,
            out=args.out,
        )
        for m in args.model
    ]


def _load_config_runs(path: str) -> Tuple[List[RunSpec], Dict[str, Any]]:
    if yaml is None:
        raise SystemExit("PyYAML is required to use --config; pip install pyyaml")
    with open(path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise SystemExit("config file must be a YAML mapping")
    runs_raw = data.get("runs", [])
    defaults = data.get("defaults", {}) or {}
    runs: List[RunSpec] = []
    for r in runs_raw:
        m = {**defaults, **r}
        models = m.get("models") or [m.get("model")]
        models = [x for x in models if x]
        if not models:
            raise SystemExit(f"config run missing models: {r}")
        for model in models:
            runs.append(
                RunSpec(
                    provider=m["provider"],
                    model=model,
                    n_values=list(m.get("n", [10, 50, 100])),
                    trials=int(m.get("trials", 3)),
                    timeout_s=float(m.get("timeout", 90.0)),
                    relevant_position=m.get("relevant_position", "first"),
                    out=m.get("out"),
                )
            )
    return runs, data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--config", help="YAML config file (overrides --model et al)")
    parser.add_argument(
        "--provider",
        default="openrouter",
        choices=["openrouter", "deepseek", "anthropic", "zhipu", "zai", "autodl", "moonshot"],
    )
    parser.add_argument("--model", action="append", default=None,
                        help="Pass multiple times for multiple models.")
    parser.add_argument("--n", type=int, action="append", default=None,
                        help="Pass multiple times to sweep different N values.")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--relevant-position", default="first",
                        choices=["first", "last", "middle"])
    parser.add_argument("--out", help="Output path (file or directory).")
    parser.add_argument("--results-dir", default="results",
                        help="Default directory for per-model outputs.")
    parser.add_argument("--csv", help="Combined CSV across all runs.")
    parser.add_argument("--md", help="Combined Markdown summary table.")
    parser.add_argument("--spend-cap-usd", type=float, default=None,
                        help="Stop further trials once estimated spend hits this cap.")
    args = parser.parse_args()

    if args.config:
        runs, _cfg = _load_config_runs(args.config)
    else:
        runs = _parse_cli_run(args)

    default_dir = Path(args.results_dir)
    default_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    running_spend = 0.0
    all_trial_results: List[TrialResult] = []

    with httpx.Client(http2=False) as client:
        for run in runs:
            results, spend, _path = _execute_run(
                run,
                client,
                default_dir,
                args.spend_cap_usd,
                running_spend,
            )
            running_spend += spend
            all_trial_results.extend(results)
            print(
                f"[bench] === running spend: ${running_spend:.4f} "
                f"(cap ${args.spend_cap_usd}) ===",
                flush=True,
            )

    # Combined CSV / MD across the whole batch.
    n_values_all = sorted({n for run in runs for n in run.n_values})
    models_all = list(dict.fromkeys(run.model for run in runs))
    csv_path = Path(args.csv) if args.csv else default_dir / "combined.csv"
    md_path = Path(args.md) if args.md else default_dir / "combined.md"
    emit_csv(all_trial_results, csv_path)
    emit_markdown_table(all_trial_results, n_values_all, models_all, md_path)

    print(f"[bench] combined CSV -> {csv_path}", flush=True)
    print(f"[bench] combined MD  -> {md_path}", flush=True)
    print(
        f"[bench] total elapsed: {time.monotonic() - started_at:.1f}s, "
        f"total estimated spend: ${running_spend:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
