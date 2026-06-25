"""Compaction — token accounting, thresholds, microcompaction, prompt
post-processing. The LLM-summarization path (full auto-compaction) is
validated end-to-end against a live model in
/tmp/e2e_test/compaction_demo.py (can't be faithfully unit-tested — it
needs a real summarizer call)."""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_runtime.config import CONFIG
from agent_runtime.compaction.tokens import (
    get_token_count_from_usage,
    rough_token_estimation,
    token_count_with_estimation,
)
from agent_runtime.compaction.budget import (
    calculate_token_warning_state,
    get_context_window_for_model,
)
from agent_runtime.compaction.microcompact import (
    CLEARED_PLACEHOLDER,
    microcompact_messages,
)
from agent_runtime.compaction.prompt import (
    build_compact_prompt,
    format_compact_summary,
    get_compact_user_summary_message,
)


pytestmark = pytest.mark.asyncio


# ── token accounting ──────────────────────────────────────────────────

async def test_usage_anchor_plus_estimated_tail():
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="ok", usage_metadata={
            "input_tokens": 5000, "output_tokens": 200, "total_tokens": 5200}),
        ToolMessage(content="x" * 3000, tool_call_id="t"),   # ~1000 @ ratio 3
        HumanMessage(content="y" * 400),                      # ~100 @ ratio 4
    ]
    n = token_count_with_estimation(msgs)
    assert 6200 < n < 6400, n        # 5200 + 1000 + 100


async def test_no_usage_estimates_whole_list():
    msgs = [HumanMessage(content="z" * 4000)]
    assert 900 < token_count_with_estimation(msgs) < 1100


async def test_get_token_count_prefers_total():
    assert get_token_count_from_usage(
        {"input_tokens": 10, "output_tokens": 5, "total_tokens": 99}) == 99
    assert get_token_count_from_usage(
        {"input_tokens": 10, "output_tokens": 5}) == 15
    assert get_token_count_from_usage(None) is None
    assert get_token_count_from_usage({}) is None


async def test_tool_message_uses_denser_ratio():
    tool = ToolMessage(content="x" * 30, tool_call_id="t")
    human = HumanMessage(content="x" * 30)
    # JSON ratio (3) yields MORE tokens than text ratio (4) for same length
    assert rough_token_estimation(tool) > rough_token_estimation(human)


# ── thresholds (must match CC's numbers) ──────────────────────────────

async def test_one_m_model_detection():
    assert get_context_window_for_model("claude-opus-4-8[1m]") == 1_000_000
    assert get_context_window_for_model("claude-sonnet-4-5") == 200_000
    assert get_context_window_for_model(None) == 200_000


async def test_threshold_math_matches_claude_code():
    st = calculate_token_warning_state(170_000, "claude-sonnet-4-5")
    assert st.effective_window == 180_000
    assert st.autocompact_threshold == 167_000
    assert st.blocking_limit == 177_000
    assert st.is_above_autocompact is True
    assert st.is_at_blocking_limit is False

    st2 = calculate_token_warning_state(178_000, "claude-sonnet-4-5")
    assert st2.is_at_blocking_limit is True

    st3 = calculate_token_warning_state(100_000, "claude-sonnet-4-5")
    assert st3.is_above_autocompact is False


# ── microcompaction (mechanical) ──────────────────────────────────────

class _Ctx:
    """Minimal RunContext stand-in for microcompaction (it only reads
    .messages, .depth, .compaction_state)."""
    def __init__(self, messages, depth=0):
        self.messages = messages
        self.depth = depth
        self.compaction_state = {}


def _make_tool_round(i: int) -> list:
    """An assistant turn that calls Read, plus its tool result."""
    return [
        AIMessage(content="", tool_calls=[
            {"id": f"call_{i}", "name": "Read", "args": {"file_path": f"/f{i}"}}]),
        ToolMessage(content=f"contents of file {i} " * 50, tool_call_id=f"call_{i}"),
    ]


async def test_microcompact_disabled_by_default():
    msgs = [m for i in range(10) for m in _make_tool_round(i)]
    ctx = _Ctx(msgs)
    assert CONFIG.microcompact_enabled is False
    assert microcompact_messages(ctx) == 0


async def test_microcompact_clears_old_keeps_recent():
    msgs = [m for i in range(10) for m in _make_tool_round(i)]
    ctx = _Ctx(msgs)
    CONFIG.microcompact_enabled = True
    try:
        freed = microcompact_messages(ctx)
        assert freed > 0
        # Last keep_recent (5) tool results preserved; first 5 cleared.
        tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
        cleared = [m for m in tool_msgs if m.content == CLEARED_PLACEHOLDER]
        kept = [m for m in tool_msgs if m.content != CLEARED_PLACEHOLDER]
        assert len(cleared) == 5
        assert len(kept) == 5
        # Idempotent: a second pass clears nothing new (gap gate also blocks,
        # but even without that the placeholders are skipped).
        ctx.compaction_state.pop("last_microcompact_ts", None)
        assert microcompact_messages(ctx) == 0
    finally:
        CONFIG.microcompact_enabled = False


async def test_microcompact_skips_subagents():
    msgs = [m for i in range(10) for m in _make_tool_round(i)]
    ctx = _Ctx(msgs, depth=2)            # subagent
    CONFIG.microcompact_enabled = True
    try:
        assert microcompact_messages(ctx) == 0
    finally:
        CONFIG.microcompact_enabled = False


# ── prompt post-processing ────────────────────────────────────────────

async def test_format_compact_summary_strips_analysis_reframes_summary():
    raw = (
        "<analysis>scratch thoughts here</analysis>\n"
        "<summary>\n1. Did the thing.\n2. Then the other.\n</summary>"
    )
    out = format_compact_summary(raw)
    assert "scratch thoughts" not in out
    assert out.startswith("Summary:\n")
    assert "Did the thing." in out


async def test_build_compact_prompt_inserts_custom_instructions():
    base = build_compact_prompt(None)
    assert "Additional Instructions" not in base
    custom = build_compact_prompt("Focus on the database migration.")
    assert "Additional Instructions:\nFocus on the database migration." in custom
    # Custom block precedes the final REMINDER line.
    assert custom.index("Additional Instructions") < custom.rindex("REMINDER:")


async def test_continuation_framing():
    framed = get_compact_user_summary_message(
        "Summary:\nstuff", transcript_path="/tmp/t.jsonl",
        suppress_follow_up_questions=True)
    assert framed.startswith("This session is being continued")
    assert "Summary:\nstuff" in framed
    assert "/tmp/t.jsonl" in framed
    assert "without asking the user any further questions" in framed
