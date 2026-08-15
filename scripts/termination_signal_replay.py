"""Replay the actual failing conversation from mini3 session 8w2ime.

The single-turn termination_signal_repro showed no loop in either kimi or
opus with synthetic 5-message conversations. The next falsification step
(per the user) is to send the EXACT failing conversation up to the user's
trigger message and observe what kimi-k2.5 does.

Cutoff: line 90 in /tmp/8w2ime-session.jsonl —
  user: "是你的 manifest 文件生成问题，图片都存在"
  ("It's a problem with your manifest file generation, the images exist")

This is the message that triggered the loop sequence at 01:27:33.690 UTC.

Hypothesis refinement based on reading the failing turn's iter-2 message
("我注意到工具返回了 'No files found matching pattern: styles/*.toml'，但这
不应该影响 check_workspace_contract"): the LLM looped because the tool it
picked (check_workspace_contract → reports artifact presence) cannot
answer the question the user asked (is the manifest content correct?).
The 4 KB tree said "all present" but the LLM had no way to reconcile that
with "user says manifest is wrong" so it re-polled.

If this replay loops with just the check_workspace_contract tool
available — and stops looping when given the user message PLUS an
additional read_file/list_dir tool — that confirms tool-question
mismatch is the proximate cause, not the contract shape.

Env: OPENROUTER_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import httpx


SYSTEM_PROMPT = (
    "You are a slides-generation agent for an internal tool. The user is "
    "iterating on a deck. Use the tools available to verify state and "
    "make changes.\n\n"
    "## Tool use discipline\n\n"
    "You have generic file-inspection tools — `read_file`, `list_dir`, "
    "`view_image`, `grep`, `glob` — that can answer most questions about "
    "the workspace state by reading the files directly. Use them "
    "aggressively to investigate what's actually there.\n\n"
    "When a tool result does not answer the user's question, do NOT "
    "re-call the same tool with the same arguments — the result will be "
    "identical. Pick a different tool that can answer the specific "
    "question, usually a file-reading tool. If no tool can answer it, "
    "respond with text explaining what's missing."
)


# Just check_workspace_contract — production had 44 but the loop was on this
# one. Smallest tool surface that lets the LLM (a) call it once and (b)
# pick "respond with text" as the alternative.
TOOL_CHECK_CONTRACT = {
    "type": "function",
    "function": {
        "name": "check_workspace_contract",
        "description": (
            "Inspect workspace contract state for the current workspace. "
            "Use this to answer whether a slides/site deliverable is "
            "actually ready, which required checks failed, which artifacts "
            "exist, and what revision is currently present. The output "
            "includes a top-level `all_ready: bool` — when true, the "
            "deliverable is COMPLETE and you should finish the turn rather "
            "than re-poll.\n\n"
            "DOES NOT answer: whether file CONTENT is correct (manifest "
            "layout, script.js logic, image quality, text rendering). This "
            "tool only verifies artifact PRESENCE and validator pass/fail "
            "— it does not read or interpret file contents. If the user "
            "asks about file content (`manifest`, `script`, what's INSIDE "
            "a file, why an image looks wrong, etc.), use `read_file` / "
            "`list_dir` / `view_image` instead. Re-calling this tool will "
            "return the same answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "project": {"type": "string"},
                "only_policy_managed": {"type": "boolean"},
                "only_not_ready": {"type": "boolean"},
            },
        },
    },
}


# Extra tools to give the LLM the option to investigate manifest content.
# Used in the "+ read_file" arm to test whether broader catalog breaks the
# loop by giving the LLM a tool that can actually answer the question.
TOOL_READ_FILE = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}

TOOL_LIST_DIR = {
    "type": "function",
    "function": {
        "name": "list_dir",
        "description": "List a directory in the workspace.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
}


def load_history(path: str, cutoff_line: int) -> List[Dict[str, Any]]:
    """Read mini3 session JSONL and return OpenAI-shape messages up to
    cutoff_line inclusive (1-indexed). Skips the metadata header at
    line 1."""
    messages: List[Dict[str, Any]] = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            if i > cutoff_line:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = obj.get("role")
            if role not in ("user", "assistant", "tool"):
                continue
            # OpenAI shape: role, content, optional tool_calls /
            # tool_call_id. Trim the octos-internal keys.
            m: Dict[str, Any] = {"role": role}
            content = obj.get("content")
            # Assistant messages can have null content if tool_calls is
            # set; OpenRouter requires content to be a string (can be ""),
            # so coerce.
            m["content"] = content if content is not None else ""
            tcs = obj.get("tool_calls")
            if tcs:
                # The session stored tool_calls without an `id` sometimes.
                # OpenRouter requires `id` to be present; synthesize one
                # when missing.
                cleaned: List[Dict[str, Any]] = []
                for idx, tc in enumerate(tcs):
                    tc_id = tc.get("id") or f"call_replay_{i:03d}_{idx}"
                    fn_name = tc.get("name") or (tc.get("function") or {}).get("name")
                    fn_args = tc.get("arguments")
                    if fn_args is None:
                        fn_args = (tc.get("function") or {}).get("arguments", "{}")
                    if not isinstance(fn_args, str):
                        fn_args = json.dumps(fn_args)
                    cleaned.append(
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {"name": fn_name, "arguments": fn_args},
                        }
                    )
                m["tool_calls"] = cleaned
            tc_id = obj.get("tool_call_id")
            if tc_id is not None:
                m["tool_call_id"] = tc_id
            messages.append(m)
    return messages


def repair_tool_call_ids(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Walk messages in order. For every `role:tool` message that lacks a
    `tool_call_id`, attach it to the most recent assistant message's
    last tool_call id. OpenRouter rejects role:tool without a matching
    tool_call_id.
    """
    out: List[Dict[str, Any]] = []
    last_assistant_tool_ids: List[str] = []
    assistant_tool_idx = 0
    for m in messages:
        if m["role"] == "assistant":
            tcs = m.get("tool_calls") or []
            last_assistant_tool_ids = [tc["id"] for tc in tcs]
            assistant_tool_idx = 0
            out.append(m)
            continue
        if m["role"] == "tool":
            if not m.get("tool_call_id"):
                if last_assistant_tool_ids and assistant_tool_idx < len(
                    last_assistant_tool_ids
                ):
                    m = {
                        **m,
                        "tool_call_id": last_assistant_tool_ids[assistant_tool_idx],
                    }
                    assistant_tool_idx += 1
                else:
                    # Skip orphaned tool messages — OpenRouter would reject.
                    continue
        out.append(m)
    return out


def truncate_tool_content(messages: List[Dict[str, Any]], limit: int) -> None:
    """Production session has 4 view_image tool results at ~10 MB each
    (line 36-39 in the JSONL). OpenRouter has a request size limit and
    we don't care about the image bytes for this loop test — truncate
    any tool content larger than `limit` chars."""
    for m in messages:
        if m["role"] == "tool" and isinstance(m.get("content"), str):
            if len(m["content"]) > limit:
                m["content"] = m["content"][:limit] + "...[truncated for replay]"


def call_openrouter(
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    timeout_s: float,
) -> Dict[str, Any]:
    body = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "max_tokens": 512,
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/octos-org/llm-benchmark",
        "X-Title": "llm-benchmark-termination-replay",
    }
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=body,
        timeout=timeout_s,
    )
    return {"status": resp.status_code, "body": resp.json() if resp.headers.get("content-type", "").startswith("application/json") else resp.text}


def categorize(response: Dict[str, Any]) -> Dict[str, Any]:
    if response["status"] != 200:
        return {
            "next_action": "ERROR",
            "tool_call_name": None,
            "tool_call_args": None,
            "text": None,
            "input_tokens": None,
            "output_tokens": None,
            "raw_error": str(response["body"])[:400],
        }
    obj = response["body"]
    usage = obj.get("usage") or {}
    choice = (obj.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    tcs = msg.get("tool_calls") or []
    if tcs:
        tc = tcs[0]
        return {
            "next_action": "TOOL_CALL",
            "tool_call_name": (tc.get("function") or {}).get("name"),
            "tool_call_args": (tc.get("function") or {}).get("arguments"),
            "text": (msg.get("content") or None),
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "raw_error": None,
        }
    return {
        "next_action": "TEXT",
        "tool_call_name": None,
        "tool_call_args": None,
        "text": msg.get("content") or "",
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "raw_error": None,
    }


# Arms differ in which tools the model has access to.
ARMS = [
    ("A: kimi + check_only", "moonshotai/kimi-k2.5", [TOOL_CHECK_CONTRACT]),
    (
        "B: kimi + check + read_file + list_dir",
        "moonshotai/kimi-k2.5",
        [TOOL_CHECK_CONTRACT, TOOL_READ_FILE, TOOL_LIST_DIR],
    ),
    ("C: opus + check_only", "anthropic/claude-opus-4.7", [TOOL_CHECK_CONTRACT]),
    (
        "D: opus + check + read_file + list_dir",
        "anthropic/claude-opus-4.7",
        [TOOL_CHECK_CONTRACT, TOOL_READ_FILE, TOOL_LIST_DIR],
    ),
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jsonl", default="/tmp/8w2ime-session.jsonl")
    p.add_argument("--cutoff-line", type=int, default=90,
                   help="Last JSONL line (1-indexed) to include")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--timeout", type=float, default=90.0)
    p.add_argument("--tool-content-limit", type=int, default=2000,
                   help="Truncate any tool result content longer than this")
    p.add_argument("--out", default="results/2026-05-28/termination-replay.jsonl")
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2

    raw = load_history(args.jsonl, args.cutoff_line)
    raw = repair_tool_call_ids(raw)
    truncate_tool_content(raw, args.tool_content_limit)

    # Prepend system prompt.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + raw
    n_msgs = len(messages)
    last_user = next(
        (m for m in reversed(messages) if m["role"] == "user"), None
    )
    last_user_preview = (last_user or {}).get("content", "")[:120]
    print(f"Loaded {n_msgs} messages (incl system).")
    print(f"Last user message: {last_user_preview!r}")
    print(f"Cutoff line: {args.cutoff_line}")
    print(f"Trials per arm: {args.trials}")
    print()

    outcomes: List[Dict[str, Any]] = []
    for label, model, tools in ARMS:
        print(f"=== {label} ({len(tools)} tools, model={model}) ===")
        for t in range(1, args.trials + 1):
            start = time.monotonic()
            resp = call_openrouter(api_key, model, messages, tools, args.timeout)
            elapsed_ms = (time.monotonic() - start) * 1000.0
            cat = categorize(resp)
            marker = (
                "LOOP"
                if cat["next_action"] == "TOOL_CALL"
                and cat["tool_call_name"] == "check_workspace_contract"
                else cat["next_action"]
            )
            text_preview = ((cat["text"] or "")[:140]).replace("\n", " ")
            print(
                f"  trial {t}/{args.trials}  {marker:<11}  "
                f"in={cat['input_tokens']} out={cat['output_tokens']} "
                f"tool={cat['tool_call_name']!r}  "
                f"text={text_preview!r}"
            )
            outcomes.append(
                {
                    "arm": label,
                    "trial": t,
                    "elapsed_ms": elapsed_ms,
                    **cat,
                }
            )
        print()

    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for label, _, _ in ARMS:
        rows = [o for o in outcomes if o["arm"] == label]
        loops = sum(
            1
            for o in rows
            if o["next_action"] == "TOOL_CALL"
            and o["tool_call_name"] == "check_workspace_contract"
        )
        other_tools = sum(
            1
            for o in rows
            if o["next_action"] == "TOOL_CALL"
            and o["tool_call_name"] != "check_workspace_contract"
        )
        texts = sum(1 for o in rows if o["next_action"] == "TEXT")
        errors = sum(1 for o in rows if o["next_action"] == "ERROR")
        n = len(rows)
        print(
            f"  {label:<48}  LOOP={loops}/{n}  OTHER_TOOL={other_tools}/{n}  "
            f"TEXT={texts}/{n}  ERROR={errors}/{n}"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        for o in outcomes:
            f.write(json.dumps(o) + "\n")
    print()
    print(f"raw JSONL -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
