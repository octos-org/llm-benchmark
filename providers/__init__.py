"""Provider abstraction.

A provider knows how to send a streaming chat completion to its API and
yield (TTFT, total, tool_call_name, finish_reason, error) over the wire.
"""

from .base import ProviderConfig, TrialResult, make_provider, run_trial

__all__ = ["ProviderConfig", "TrialResult", "make_provider", "run_trial"]
