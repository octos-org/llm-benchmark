# llm-benchmark

Benchmarks for LLM tool-calling behaviour at scale.

The first experiment in this repo measures **what happens to tool-calling
when you hand a frontier model an absurdly large catalog of tools**. We send
a fixed prompt that requires a specific tool (`get_weather`), padded with
N−1 plausible-but-irrelevant distractors, and record:

- **TTFT** — time from request send to the first non-empty streaming token
- **Total latency** — time to `finish_reason=tool_calls` (OpenAI proto) or `tool_use` stop (Anthropic proto)
- **Output tokens** — from each provider's usage frame
- **Tool selected** — which (if any) tool the model emitted
- **HTTP status / error** — when the request fails before content arrives

## Models tested in the 2026-05-17 run

| Model | Routing |
|---|---|
| `deepseek-v4-flash` | DeepSeek direct (`api.deepseek.com`) |
| `deepseek-v4-pro`   | DeepSeek direct (`api.deepseek.com`) |
| `anthropic/claude-sonnet-4.6` | OpenRouter |
| `anthropic/claude-opus-4.7`   | OpenRouter |

DeepSeek V4 (Flash + Pro) is the current generation as exposed by the
`/v1/models` endpoint at the time of the run; no V3.x variant is reachable
with a fresh key. Anthropic models were routed through OpenRouter because
no direct Anthropic key was available in the run environment; the OpenRouter
model IDs (`anthropic/claude-opus-4.7`, `anthropic/claude-sonnet-4.6`) are
verbatim from `https://openrouter.ai/api/v1/models`.

## Sweep

- `N ∈ {10, 50, 100, 200, 500, 1000}`
- 3 trials per (model, N) for the cheaper models
- 2 trials per (model, N) for Claude Opus (cost control)
- Spend cap: $10 hard

The relevant tool is placed **first** in the list for every run, so we
isolate "model gets confused by N pads" from "model loses recency on a
long tail". (A position sweep is an obvious follow-up.)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set the API keys you need (only the ones you'll actually hit are required):
export DEEPSEEK_API_KEY=sk-...      # direct DeepSeek
export OPENROUTER_API_KEY=sk-or-... # OpenRouter (Claude, GLM, etc.)
export ANTHROPIC_API_KEY=sk-ant-... # direct Anthropic
```

## Run

```bash
# Single provider/model from CLI flags:
python3 bench.py \
    --provider deepseek --model deepseek-v4-flash \
    --n 10 --n 50 --n 100 --n 200 --n 500 --n 1000 \
    --trials 3 --timeout 120 \
    --out results/deepseek-v4-flash.jsonl

# Drive a multi-model sweep from a YAML config:
python3 bench.py --config configs/run-2026-05-17.yaml \
    --spend-cap-usd 10.0 \
    --csv results/all-trials.csv \
    --md results/summary.md
```

Each invocation produces, per model:

- `results/results-<slug>.jsonl` — one JSON object per trial
- `results/results-<slug>.summary.json` — per-(model, N) aggregate (median, p95, range)

Plus a combined `all-trials.csv` and `summary.md` across the full batch.

## Repository layout

```
bench.py                         entry point: CLI flags + --config YAML
providers/base.py                streaming OpenAI + Anthropic protocols
tools/registry.py                pad-tool catalog (99 hand-written + synthetic)
tools/relevant.py                the single get_weather tool
configs/run-2026-05-17.yaml      the sweep config used for the May 2026 run
results/                         JSONL + CSV + MD artefacts from runs
```

## Results: 2026-05-17 run

See [`results/summary.md`](results/summary.md) for the full table.

Headline takeaway: **the four frontier models tested are robust to N up to
1000 padded tools.** None broke at the protocol layer in this sweep; the
cost is paid in input-token volume and end-to-end latency rather than in
tool-selection accuracy.

Each cell shows `TTFT_med / total_med ms (rel=<relevant-tool rate>)`. The
relevant-tool rate is the fraction of trials in which the model selected
`get_weather` (as opposed to no tool or a different tool).

## License

Apache-2.0. See [LICENSE](LICENSE).
