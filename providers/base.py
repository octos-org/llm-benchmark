"""Provider plumbing: OpenAI + Anthropic streaming protocols.

Three real providers are wired:
    - ``deepseek``    -- https://api.deepseek.com (OpenAI protocol, direct)
    - ``openrouter``  -- https://openrouter.ai/api/v1 (OpenAI protocol)
    - ``anthropic``   -- https://api.anthropic.com (Anthropic protocol)

Each call records TTFT and total latency separately so we can distinguish
"router latency at request start" from "model emits tool call slowly".
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import httpx


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    model: str
    n_tools: int
    trial: int
    ttft_ms: Optional[float]  # ms to first content/tool token; None on no-start
    total_ms: float
    emitted_tool_call: bool
    tool_call_name: Optional[str]
    finish_reason: Optional[str]
    response_started: bool
    error: Optional[str]
    raw_first_event: Optional[str] = None
    # Token-usage book-keeping for cost estimation.
    output_tokens: Optional[int] = None
    input_tokens: Optional[int] = None
    http_status: Optional[int] = None
    selected_relevant: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "n_tools": self.n_tools,
            "trial": self.trial,
            "ttft_ms": self.ttft_ms,
            "total_ms": self.total_ms,
            "emitted_tool_call": self.emitted_tool_call,
            "tool_call_name": self.tool_call_name,
            "selected_relevant": self.selected_relevant,
            "finish_reason": self.finish_reason,
            "response_started": self.response_started,
            "http_status": self.http_status,
            "error": self.error,
            "raw_first_event": self.raw_first_event,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


# ---------------------------------------------------------------------------
# Streaming protocol implementations
# ---------------------------------------------------------------------------


USER_PROMPT = "What's the weather in Tokyo right now? Use the tool to find out."
RELEVANT_TOOL_NAME = "get_weather"


def _run_openai_protocol(
    client: httpx.Client,
    url: str,
    headers: Dict[str, str],
    model: str,
    tools: List[Dict[str, Any]],
    timeout_s: float,
) -> TrialResult:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": USER_PROMPT}],
        "tools": tools,
        "tool_choice": "auto",
        "stream": True,
        "max_tokens": 256,
        # Ask OpenRouter to include usage at the end of the stream; harmless
        # for other OpenAI-protocol providers that ignore the field.
        "stream_options": {"include_usage": True},
    }

    start = time.monotonic()
    ttft: Optional[float] = None
    response_started = False
    tool_call_name: Optional[str] = None
    finish_reason: Optional[str] = None
    raw_first: Optional[str] = None
    error: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    http_status: Optional[int] = None

    try:
        with client.stream(
            "POST", url, headers=headers, json=body, timeout=timeout_s
        ) as resp:
            http_status = resp.status_code
            if resp.status_code != 200:
                err_body = resp.read().decode("utf-8", errors="replace")[:300]
                return TrialResult(
                    model=model,
                    n_tools=len(tools),
                    trial=0,
                    ttft_ms=None,
                    total_ms=(time.monotonic() - start) * 1000.0,
                    emitted_tool_call=False,
                    tool_call_name=None,
                    finish_reason=None,
                    response_started=False,
                    http_status=http_status,
                    error=f"HTTP {resp.status_code}: {err_body}",
                )
            response_started = True
            for line in resp.iter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if raw_first is None:
                    raw_first = payload[:200]
                # Usage frame (OpenRouter / OpenAI when include_usage=True).
                usage = obj.get("usage")
                if isinstance(usage, dict):
                    input_tokens = usage.get("prompt_tokens", input_tokens)
                    output_tokens = usage.get("completion_tokens", output_tokens)
                choices = obj.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})
                if ttft is None and (
                    delta.get("content") or delta.get("tool_calls") or delta.get("role")
                ):
                    ttft = (time.monotonic() - start) * 1000.0
                for tc in delta.get("tool_calls") or []:
                    fn_name = (tc.get("function") or {}).get("name")
                    if fn_name and tool_call_name is None:
                        tool_call_name = fn_name
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr
    except httpx.TimeoutException as exc:
        error = f"timeout: {exc!r}"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    total = (time.monotonic() - start) * 1000.0
    return TrialResult(
        model=model,
        n_tools=len(tools),
        trial=0,
        ttft_ms=ttft,
        total_ms=total,
        emitted_tool_call=tool_call_name is not None,
        tool_call_name=tool_call_name,
        selected_relevant=(tool_call_name == RELEVANT_TOOL_NAME),
        finish_reason=finish_reason,
        response_started=response_started,
        http_status=http_status,
        error=error,
        raw_first_event=raw_first,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _run_anthropic_protocol(
    client: httpx.Client,
    url: str,
    headers: Dict[str, str],
    model: str,
    tools: List[Dict[str, Any]],
    timeout_s: float,
) -> TrialResult:
    # Translate OpenAI-style tools -> Anthropic.
    a_tools = []
    for t in tools:
        f = t["function"]
        a_tools.append(
            {
                "name": f["name"],
                "description": f["description"],
                "input_schema": f["parameters"],
            }
        )

    body = {
        "model": model,
        "messages": [{"role": "user", "content": USER_PROMPT}],
        "tools": a_tools,
        "max_tokens": 256,
        "stream": True,
    }

    start = time.monotonic()
    ttft: Optional[float] = None
    response_started = False
    tool_call_name: Optional[str] = None
    finish_reason: Optional[str] = None
    raw_first: Optional[str] = None
    error: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    http_status: Optional[int] = None

    try:
        with client.stream(
            "POST", url, headers=headers, json=body, timeout=timeout_s
        ) as resp:
            http_status = resp.status_code
            if resp.status_code != 200:
                err_body = resp.read().decode("utf-8", errors="replace")[:300]
                return TrialResult(
                    model=model,
                    n_tools=len(tools),
                    trial=0,
                    ttft_ms=None,
                    total_ms=(time.monotonic() - start) * 1000.0,
                    emitted_tool_call=False,
                    tool_call_name=None,
                    finish_reason=None,
                    response_started=False,
                    http_status=http_status,
                    error=f"HTTP {resp.status_code}: {err_body}",
                )
            response_started = True
            current_event: Optional[str] = None
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    current_event = line[len("event:") :].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if not payload:
                    continue
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                if raw_first is None:
                    raw_first = (current_event or "") + " | " + payload[:200]
                etype = obj.get("type") or current_event
                if etype == "message_start":
                    usage = (obj.get("message") or {}).get("usage") or {}
                    input_tokens = usage.get("input_tokens", input_tokens)
                    output_tokens = usage.get("output_tokens", output_tokens)
                elif etype == "content_block_start":
                    block = obj.get("content_block", {})
                    if ttft is None:
                        ttft = (time.monotonic() - start) * 1000.0
                    if block.get("type") == "tool_use":
                        name = block.get("name")
                        if name and tool_call_name is None:
                            tool_call_name = name
                elif etype == "content_block_delta" and ttft is None:
                    ttft = (time.monotonic() - start) * 1000.0
                elif etype == "message_delta":
                    fr = (obj.get("delta") or {}).get("stop_reason")
                    if fr:
                        finish_reason = fr
                    usage = obj.get("usage") or {}
                    output_tokens = usage.get("output_tokens", output_tokens)
    except httpx.TimeoutException as exc:
        error = f"timeout: {exc!r}"
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    total = (time.monotonic() - start) * 1000.0
    return TrialResult(
        model=model,
        n_tools=len(tools),
        trial=0,
        ttft_ms=ttft,
        total_ms=total,
        emitted_tool_call=tool_call_name is not None,
        tool_call_name=tool_call_name,
        selected_relevant=(tool_call_name == RELEVANT_TOOL_NAME),
        finish_reason=finish_reason,
        response_started=response_started,
        http_status=http_status,
        error=error,
        raw_first_event=raw_first,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# ---------------------------------------------------------------------------
# Provider config + factory
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    name: str
    url: str
    protocol: str  # "openai" | "anthropic"
    headers_fn: Callable[[str], Dict[str, str]]


def _require_env(var: str) -> str:
    val = os.environ.get(var, "")
    if not val:
        raise SystemExit(f"{var} not set in environment")
    return val


def make_provider(provider: str) -> ProviderConfig:
    if provider == "openrouter":
        key = _require_env("OPENROUTER_API_KEY")
        return ProviderConfig(
            name="openrouter",
            url="https://openrouter.ai/api/v1/chat/completions",
            protocol="openai",
            headers_fn=lambda _model: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/octos-org/llm-benchmark",
                "X-Title": "llm-benchmark",
            },
        )
    if provider == "deepseek":
        key = _require_env("DEEPSEEK_API_KEY")
        return ProviderConfig(
            name="deepseek",
            url="https://api.deepseek.com/v1/chat/completions",
            protocol="openai",
            headers_fn=lambda _model: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
    if provider == "anthropic":
        key = _require_env("ANTHROPIC_API_KEY")
        return ProviderConfig(
            name="anthropic",
            url="https://api.anthropic.com/v1/messages",
            protocol="anthropic",
            headers_fn=lambda _model: {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
    if provider == "zhipu":
        key = _require_env("ZHIPU_API_KEY")
        return ProviderConfig(
            name="zhipu",
            url="https://open.bigmodel.cn/api/paas/v4/chat/completions",
            protocol="openai",
            headers_fn=lambda _model: {
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )
    if provider == "zai":
        key = _require_env("ZAI_API_KEY")
        return ProviderConfig(
            name="zai",
            url="https://api.z.ai/api/anthropic/v1/messages",
            protocol="anthropic",
            headers_fn=lambda _model: {
                "x-api-key": key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
    raise SystemExit(f"unknown provider: {provider}")


def run_trial(
    provider: ProviderConfig,
    model: str,
    tools: List[Dict[str, Any]],
    timeout_s: float,
    client: httpx.Client,
) -> TrialResult:
    headers = provider.headers_fn(model)
    if provider.protocol == "openai":
        return _run_openai_protocol(client, provider.url, headers, model, tools, timeout_s)
    return _run_anthropic_protocol(client, provider.url, headers, model, tools, timeout_s)
