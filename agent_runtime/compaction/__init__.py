"""Context compaction — keep the conversation under the model's window.

Faithful port of Claude Code's services/compact/. Public surface:

    auto_compact_if_needed(ctx, snip_freed)  — Mechanism A (orchestrator)
    compact_conversation(ctx)                — Mechanism B (LLM summary)
    microcompact_messages(ctx)               — Mechanism C (mechanical)
    run_post_compact_cleanup(ctx)            — Mechanism D (cleanup)
    token_count_with_estimation(messages)    — live token size
    calculate_token_warning_state(n, model)  — threshold booleans

The query loop calls auto_compact_if_needed + microcompact_messages at its
§3 COMPACTION extension point. Everything is gated by CONFIG flags
(auto_compact_enabled defaults True; microcompact_enabled defaults False).
"""
from .auto_compact import CompactionOutcome, auto_compact_if_needed
from .budget import (
    calculate_token_warning_state,
    get_context_window_for_model,
)
from .cleanup import run_post_compact_cleanup
from .compact import CompactionError, compact_conversation
from .microcompact import microcompact_messages
from .tokens import (
    get_token_count_from_usage,
    rough_token_estimation,
    token_count_with_estimation,
)

__all__ = [
    "auto_compact_if_needed",
    "CompactionOutcome",
    "compact_conversation",
    "CompactionError",
    "microcompact_messages",
    "run_post_compact_cleanup",
    "token_count_with_estimation",
    "get_token_count_from_usage",
    "rough_token_estimation",
    "calculate_token_warning_state",
    "get_context_window_for_model",
]
