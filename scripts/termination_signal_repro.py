"""Direct-API repro for the production loop on `check_workspace_contract`.

Hypothesis under test:

    The production loop happened because `check_workspace_contract` returned
    a 4 KB nested tree with all `present: true` but no top-level
    termination signal (no `all_ready: true`, no "do not re-poll"
    summary). The LLM, lacking a definitive stop signal, kept re-calling
    the tool with identical args. Adding the explicit signal removes the
    loop.

Test design:

    Build the exact conversation shape that failed (assistant has just
    called check_workspace_contract, tool replied, user asked "is it
    done?"). Vary ONE thing across arms: whether the tool result includes
    the new `all_ready` + `summary` fields. Observe whether the LLM's
    NEXT response is a tool re-call (LOOP) or text (DONE).

    Compare kimi-k2.5 (the production model) against claude-opus-4.7
    (frontier reference). Both routed via OpenRouter so the protocol,
    auth, and retry behaviour are identical across arms — the only
    variables are model and tool-result shape.

Outcomes:
    - kimi OLD loops + kimi NEW doesn't  →  hypothesis validated, Fix A works
    - kimi OLD loops + kimi NEW also loops  →  Fix A insufficient
    - kimi OLD doesn't loop  →  can't repro in single-turn; production
                                regime needs multi-turn history to reproduce
    - opus loops too  →  not kimi-specific, points to context shape

Env: OPENROUTER_API_KEY for both arms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


# ---------------------------------------------------------------------------
# The check_workspace_contract tool — OpenAI schema.
# ---------------------------------------------------------------------------

TOOL_OPENAI = {
    "type": "function",
    "function": {
        "name": "check_workspace_contract",
        "description": (
            "Inspect workspace contract state for the current workspace. "
            "Use this to answer whether a slides/site deliverable is "
            "actually ready, which required checks failed, which artifacts "
            "exist, and what revision is currently present. Task state "
            "tells you what happened in execution; workspace state tells "
            "you what is true about the deliverable. The output includes a "
            "top-level `all_ready: bool` — when true, the deliverable is "
            "COMPLETE and you should finish the turn rather than re-poll."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": (
                        "Optional workspace project selector, e.g. "
                        "'slides/my-deck' or 'my-deck'. When omitted, "
                        "returns all workspace contracts under the current "
                        "workspace root."
                    ),
                }
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Tool result variants. OLD = what production saw before Fix A. NEW = what
# production sees after Fix A (PR octos#1360 / 7f2c38cb).
# ---------------------------------------------------------------------------


def _slides_artifacts() -> List[str]:
    return [
        f"skill-output/slides/untitled-deck-8w2ime/output/imgs/slide-{i:02d}.png"
        for i in range(1, 11)
    ] + [
        f"skill-output/slides/untitled-deck-8w2ime/output/imgs_fixed/slide-{i:02d}.png"
        for i in range(1, 9)
    ]


def make_old_result() -> Dict[str, Any]:
    """Pre-Fix-A shape. The exact 4 KB tree the production LLM saw."""
    return {
        "workspace_root": "/Users/cloud/.octos/profiles/dspfac/data/users/slides-1780013669236-8w2ime/workspace",
        "requested_project": "slides/untitled-deck-8w2ime",
        "repo_count": 1,
        "ready_count": 1,
        "contracts": [
            {
                "repo_label": "slides/untitled-deck-8w2ime",
                "kind": "slides",
                "slug": "untitled-deck-8w2ime",
                "ready": True,
                "policy_managed": True,
                "artifacts": [
                    {
                        "name": "deck",
                        "pattern": "skill-output/slides/untitled-deck-8w2ime/output/deck.pptx",
                        "matches": [
                            "skill-output/slides/untitled-deck-8w2ime/output/deck.pptx"
                        ],
                        "present": True,
                    },
                    {
                        "name": "images",
                        "pattern": "skill-output/slides/untitled-deck-8w2ime/output/imgs/slide-*.png",
                        "matches": _slides_artifacts(),
                        "present": True,
                    },
                ],
            }
        ],
    }


def make_new_result() -> Dict[str, Any]:
    """Post-Fix-A shape. Adds the explicit termination signal."""
    base = make_old_result()
    # Re-order so all_ready + summary appear near the top of the JSON.
    return {
        "workspace_root": base["workspace_root"],
        "requested_project": base["requested_project"],
        "all_ready": True,
        "summary": (
            "All 1 contract(s) satisfied — every required artifact is "
            "present and every validator passed. The deliverable is "
            "COMPLETE; do not re-poll, finish the turn."
        ),
        "repo_count": base["repo_count"],
        "ready_count": base["ready_count"],
        "contracts": base["contracts"],
    }


# ---------------------------------------------------------------------------
# Conversation shape — minimal but production-realistic.
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = (
    "You are a slides-generation agent for an internal tool. The user "
    "asked for a deck. You have already produced one. The user is now "
    "asking whether the deliverable is done. Inspect workspace state if "
    "you need to; otherwise respond directly. Do NOT call the same tool "
    "twice with the same arguments if the previous result already "
    "answered the question."
)


def build_messages(tool_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The conversation shape that triggered the loop in production.

    Order: system, user, assistant(tool_call check_workspace_contract),
    tool(result), user("is the deck done?"). The LLM's next response is
    what we measure.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Generate a slides deck about Pu'er tea history, "
                "pure Chinese, 10 slides, puer-woodcut style."
            ),
        },
        {
            "role": "assistant",
            "content": "I'll verify the deliverable state first.",
            "tool_calls": [
                {
                    "id": "call_pre_seed_001",
                    "type": "function",
                    "function": {
                        "name": "check_workspace_contract",
                        "arguments": json.dumps(
                            {"project": "slides/untitled-deck-8w2ime"}
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_pre_seed_001",
            "content": json.dumps(tool_result, indent=2),
        },
        {"role": "user", "content": "is the deck done?"},
    ]


# ---------------------------------------------------------------------------
# OpenRouter call (OpenAI protocol, blocking, single response).
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    arm: str
    trial: int
    elapsed_ms: float
    next_action: str  # "TOOL_CALL" | "TEXT" | "ERROR"
    tool_call_name: Optional[str]
    tool_call_args: Optional[str]
    text_preview: Optional[str]
    output_tokens: Optional[int]
    input_tokens: Optional[int]
    error: Optional[str]


def call_openrouter(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    timeout_s: float,
) -> Outcome:
    body = {
        "model": model,
        "messages": messages,
        "tools": [TOOL_OPENAI],
        "tool_choice": "auto",
        "max_tokens": 512,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/octos-org/llm-benchmark",
        "X-Title": "llm-benchmark-termination-repro",
    }
    start = time.monotonic()
    try:
        resp = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=timeout_s,
        )
        elapsed = (time.monotonic() - start) * 1000.0
        if resp.status_code != 200:
            return Outcome(
                arm="",
                trial=0,
                elapsed_ms=elapsed,
                next_action="ERROR",
                tool_call_name=None,
                tool_call_args=None,
                text_preview=None,
                output_tokens=None,
                input_tokens=None,
                error=f"HTTP {resp.status_code}: {resp.text[:300]}",
            )
        obj = resp.json()
        usage = obj.get("usage") or {}
        choice = (obj.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        tcs = msg.get("tool_calls") or []
        if tcs:
            tc = tcs[0]
            return Outcome(
                arm="",
                trial=0,
                elapsed_ms=elapsed,
                next_action="TOOL_CALL",
                tool_call_name=(tc.get("function") or {}).get("name"),
                tool_call_args=(tc.get("function") or {}).get("arguments"),
                text_preview=(msg.get("content") or "")[:200] or None,
                output_tokens=usage.get("completion_tokens"),
                input_tokens=usage.get("prompt_tokens"),
                error=None,
            )
        return Outcome(
            arm="",
            trial=0,
            elapsed_ms=elapsed,
            next_action="TEXT",
            tool_call_name=None,
            tool_call_args=None,
            text_preview=(msg.get("content") or "")[:200],
            output_tokens=usage.get("completion_tokens"),
            input_tokens=usage.get("prompt_tokens"),
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        return Outcome(
            arm="",
            trial=0,
            elapsed_ms=(time.monotonic() - start) * 1000.0,
            next_action="ERROR",
            tool_call_name=None,
            tool_call_args=None,
            text_preview=None,
            output_tokens=None,
            input_tokens=None,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Run all 4 arms.
# ---------------------------------------------------------------------------

ARMS = [
    ("A: kimi-k2.5 + OLD result", "moonshotai/kimi-k2.5", "old"),
    ("B: kimi-k2.5 + NEW result", "moonshotai/kimi-k2.5", "new"),
    ("C: claude-opus-4.7 + OLD result", "anthropic/claude-opus-4.7", "old"),
    ("D: claude-opus-4.7 + NEW result", "anthropic/claude-opus-4.7", "new"),
]


def run_arm(
    label: str,
    model: str,
    variant: str,
    api_key: str,
    trials: int,
    timeout_s: float,
) -> List[Outcome]:
    tool_result = make_old_result() if variant == "old" else make_new_result()
    messages = build_messages(tool_result)
    results: List[Outcome] = []
    for t in range(1, trials + 1):
        out = call_openrouter(
            api_key=api_key,
            model=model,
            messages=messages,
            timeout_s=timeout_s,
        )
        out.arm = label
        out.trial = t
        results.append(out)
        marker = (
            "LOOP"
            if out.next_action == "TOOL_CALL"
            and out.tool_call_name == "check_workspace_contract"
            else out.next_action
        )
        preview = (out.text_preview or "")[:120].replace("\n", " ")
        print(
            f"  {label} trial {t}/{trials}  {marker:<10}  "
            f"out_tok={out.output_tokens}  "
            f"text={preview!r}"
        )
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument(
        "--out", default="results/2026-05-28/termination-repro.jsonl"
    )
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2

    print(f"Trials per arm: {args.trials}")
    print(f"Timeout: {args.timeout}s")
    print()

    all_outcomes: List[Outcome] = []
    for label, model, variant in ARMS:
        print(f"=== {label} (model={model}) ===")
        outs = run_arm(label, model, variant, api_key, args.trials, args.timeout)
        all_outcomes.extend(outs)
        print()

    # Summary.
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for label, _, _ in ARMS:
        rows = [o for o in all_outcomes if o.arm == label]
        loops = sum(
            1
            for o in rows
            if o.next_action == "TOOL_CALL"
            and o.tool_call_name == "check_workspace_contract"
        )
        texts = sum(1 for o in rows if o.next_action == "TEXT")
        errors = sum(1 for o in rows if o.next_action == "ERROR")
        n = len(rows)
        print(
            f"  {label:<40}  LOOP={loops}/{n}  TEXT={texts}/{n}  ERROR={errors}/{n}"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for o in all_outcomes:
            f.write(
                json.dumps(
                    {
                        "arm": o.arm,
                        "trial": o.trial,
                        "elapsed_ms": o.elapsed_ms,
                        "next_action": o.next_action,
                        "tool_call_name": o.tool_call_name,
                        "tool_call_args": o.tool_call_args,
                        "text_preview": o.text_preview,
                        "output_tokens": o.output_tokens,
                        "input_tokens": o.input_tokens,
                        "error": o.error,
                    }
                )
                + "\n"
            )
    print()
    print(f"raw JSONL -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
