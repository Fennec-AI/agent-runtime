"""Full conversation compaction — the core LLM summarization (Mechanism B).

Ported from Claude Code's services/compact/compact.ts (compactConversation).

What it does, in order:
  1. Snapshot the pre-compaction token count (for logging/metrics).
  2. Strip image/document blocks to text markers so the summarizer call
     itself can't overflow on media.
  3. Make a SEPARATE, one-shot model call — NOT the agent loop — with
     tools disabled and a large output budget, asking for the 9-section
     summary (prompt.py).
  4. Post-process the summary (strip <analysis>, reframe <summary>).
  5. Frame it in the continuation preamble and REPLACE the conversation
     history with [original system message] + [summary as a user turn].
  6. PTL retry: if the summarizer call itself trips "prompt too long",
     drop the oldest message rounds and retry, up to MAX_PTL_RETRIES.

The replacement is in place: ctx.messages[:] = new — so Session.messages
(the same list object) updates automatically.
"""
from __future__ import annotations

import logging

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from ..config import CONFIG
from ..types import RunContext
from .prompt import (
    COMPACT_SYSTEM_PROMPT,
    build_compact_prompt,
    format_compact_summary,
    get_compact_user_summary_message,
)
from .tokens import token_count_with_estimation


logger = logging.getLogger(__name__)


MAX_PTL_RETRIES = 3
"""How many times to retry the summary call after a 'prompt too long'
error, dropping the oldest rounds each time."""


class CompactionError(Exception):
    """Raised when compaction can't produce a summary (e.g. PTL retries
    exhausted, or the model returned nothing)."""


async def compact_conversation(
    ctx: RunContext,
    *,
    custom_instructions: str | None = None,
    is_auto: bool = True,
    transcript_path: str | None = None,
) -> list[BaseMessage]:
    """Summarize ctx.messages and replace the history with the summary.

    Returns the new (short) message list. Mutates ctx.messages in place.
    Raises CompactionError if no summary could be produced.
    """
    pre_count = token_count_with_estimation(ctx.messages)
    logger.info(
        "compaction: starting (auto=%s, pre_count=%d tokens, %d messages)",
        is_auto, pre_count, len(ctx.messages),
    )

    # Separate the original system message (kept) from the conversation
    # body (summarized). Anthropic allows only ONE system message, so we
    # never stack the compaction system prompt on top of an existing one.
    original_system = next(
        (m for m in ctx.messages if isinstance(m, SystemMessage)), None
    )
    body = [m for m in ctx.messages if not isinstance(m, SystemMessage)]
    body = [_strip_media(m) for m in body]

    summary_text = await _summarize_with_ptl_retry(
        ctx=ctx,
        body=body,
        custom_instructions=custom_instructions,
    )

    formatted = format_compact_summary(summary_text)
    framed = get_compact_user_summary_message(
        formatted,
        transcript_path=transcript_path,
        suppress_follow_up_questions=is_auto,
    )

    # Rebuild history: keep the system message, replace everything else
    # with the framed summary as a single user turn.
    new_messages: list[BaseMessage] = []
    if original_system is not None:
        new_messages.append(original_system)
    new_messages.append(HumanMessage(content=framed))

    ctx.messages[:] = new_messages

    post_count = token_count_with_estimation(ctx.messages)
    logger.info(
        "compaction: done (%d → %d tokens, %d → %d messages)",
        pre_count, post_count, len(body) + (1 if original_system else 0),
        len(new_messages),
    )
    return new_messages


# ── internal ────────────────────────────────────────────────────────────

async def _summarize_with_ptl_retry(
    ctx: RunContext,
    body: list[BaseMessage],
    custom_instructions: str | None,
) -> str:
    """Run the one-shot summarization call, retrying on prompt-too-long by
    dropping the oldest rounds."""
    compact_prompt = build_compact_prompt(custom_instructions)
    # Disable tools and raise the output budget for the summary turn.
    summary_llm = ctx.llm.bind(max_tokens=CONFIG.compact_max_output_tokens)

    working = list(body)
    last_error: Exception | None = None

    for attempt in range(MAX_PTL_RETRIES + 1):
        summary_messages: list[BaseMessage] = [
            SystemMessage(content=COMPACT_SYSTEM_PROMPT),
            *working,
            HumanMessage(content=compact_prompt),
        ]
        try:
            response = await summary_llm.ainvoke(summary_messages)
        except Exception as e:  # noqa: BLE001
            if _is_prompt_too_long(e) and len(working) > 2:
                last_error = e
                drop = max(1, len(working) // 2)
                logger.warning(
                    "compaction: prompt too long (attempt %d), dropping "
                    "oldest %d messages and retrying", attempt + 1, drop,
                )
                working = working[drop:]
                continue
            raise CompactionError(f"summarization call failed: {e}") from e

        text = _extract_text(response.content)
        if text.strip():
            return text
        raise CompactionError("summarizer returned empty content")

    raise CompactionError(
        f"prompt too long after {MAX_PTL_RETRIES} retries: {last_error}"
    )


def _strip_media(message: BaseMessage) -> BaseMessage:
    """Replace image/document blocks with short text markers so the
    summarizer call doesn't carry (or overflow on) media."""
    content = message.content
    if not isinstance(content, list):
        return message

    changed = False
    new_blocks: list = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in ("image", "image_url"):
            new_blocks.append({"type": "text", "text": "[image]"})
            changed = True
        elif isinstance(block, dict) and block.get("type") == "document":
            new_blocks.append({"type": "text", "text": "[document]"})
            changed = True
        else:
            new_blocks.append(block)

    if not changed:
        return message
    return message.model_copy(update={"content": new_blocks})


def _extract_text(content) -> str:
    """Pull plain text out of a (possibly block-list) message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


def _is_prompt_too_long(err: Exception) -> bool:
    """Heuristic: does this exception look like an Anthropic
    prompt-too-long / context-overflow error?"""
    msg = str(err).lower()
    return (
        "prompt is too long" in msg
        or "prompt too long" in msg
        or "exceeds the context window" in msg
        or "maximum context length" in msg
        or ("too many tokens" in msg)
    )
