"""Tool registry for the llm-benchmark suite.

Public API:
    RELEVANT_TOOL  -- the one tool the model *should* select
    build_tools(n, position) -- assemble a list of N tool dicts (OpenAI format)
"""

from .relevant import RELEVANT_TOOL
from .registry import build_tools, total_available_pads

__all__ = ["RELEVANT_TOOL", "build_tools", "total_available_pads"]
