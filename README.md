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

Cell format: `TTFT_med / total_med ms` over response-started trials, plus
the rate at which the model selected the relevant `get_weather` tool.

| Model | N=10 | N=50 | N=100 | N=200 | N=500 | N=1000 |
|---|---|---|---|---|---|---|
| `deepseek-v4-flash` | 439 / 1439 ms<br>rel=100% | 652 / 1657 ms<br>rel=100% | 490 / 1513 ms<br>rel=100% | 635 / 1753 ms<br>rel=100% | 828 / 1776 ms<br>rel=100% | 1008 / 2184 ms<br>rel=100% |
| `deepseek-v4-pro` | 448 / 4124 ms<br>rel=100% | 655 / 4570 ms<br>rel=100% | 503 / 4447 ms<br>rel=100% | 651 / 4864 ms<br>rel=100% | 809 / 4791 ms<br>rel=67% | 1050 / 4995 ms<br>rel=100% |
| `anthropic/claude-sonnet-4.6` | 675 / 1822 ms<br>rel=100% | 1347 / 2046 ms<br>rel=100% | 1997 / 2739 ms<br>rel=100% | 1855 / 2576 ms<br>rel=100% | 2607 / 2613 ms<br>rel=100% | 2804 / 3224 ms<br>rel=100% |
| `anthropic/claude-opus-4.7` | 1535 / 2360 ms<br>rel=100% | 2298 / 2918 ms<br>rel=100% | 1822 / 2798 ms<br>rel=100% | 3914 / 5219 ms<br>rel=100% | 2327 / 3157 ms<br>rel=100% | 5819 / 6595 ms<br>rel=100% |

Per-cell breakdown with p50/p95 and error details lives in
[`results/2026-05-17/details.md`](results/2026-05-17/details.md). Trial-level
CSV is at [`results/2026-05-17/all.csv`](results/2026-05-17/all.csv).

### Breaking points

- **`deepseek-v4-flash`** — clean through `N=1000`. 100% tool selection,
  ~1s TTFT, ~2s total even at 1000 tools and ~78k input tokens.
- **`deepseek-v4-pro`** — clean through `N=200` and again at `N=1000`. At
  `N=500`, one trial hit `RemoteProtocolError: peer closed connection`
  with no tool selected, and one trial timed out at 124s **after**
  emitting the tool name. Pro consistently selected `get_weather` when
  the stream completed, but the server appears flaky at exactly
  `N=500 × ~39k input tokens` — the relevant-tool rate drops to 67%
  purely because of mid-stream disconnects, not because the model chose
  a different tool. N=1000 then succeeds cleanly, so this is not a
  monotonic context-length wall — looks like an episodic mid-stream
  abort on the DeepSeek side at a specific load.
- **`anthropic/claude-sonnet-4.6`** — clean through `N=1000`. 100% tool
  selection, sub-3.3s total even at 86k input tokens.
- **`anthropic/claude-opus-4.7`** — clean through `N=1000`. 100% tool
  selection at every N. Latency does scale with N — TTFT goes from
  ~1.5s at N=10 to ~5.8s at N=1000 (3.8x), and total from 2.4s to
  6.6s (2.8x).

### Headline insight

**The four frontier models tested in May 2026 are all robust to a 1000-tool
padded catalog for tool-selection correctness; the only failures observed
were stochastic mid-stream disconnects on DeepSeek V4 Pro at N=500, not
tool-routing mistakes.**

Cost is paid in input tokens (linear with N — ~78k at N=1000 for the
DeepSeek tools, ~115k for Anthropic's input_schema-based encoding) and in
TTFT inflation (Opus 3.8x, Sonnet 4.2x, DS-Flash 2.3x, DS-Pro 2.3x going
from N=10 to N=1000). Tool-selection accuracy stays at 100% on the
happy path.

### Cost

Total estimated spend: **$8.37 / $10.00 cap** (DeepSeek Flash $0.03 +
DeepSeek Pro $0.20 + Sonnet 4.6 $1.48 + Opus 4.7 $6.65).

## License

Apache-2.0. See [LICENSE](LICENSE).
