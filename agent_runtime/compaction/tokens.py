"""Token accounting — how big is the conversation right now?

Ported from Claude Code's utils/tokens.ts. The strategy is a hybrid:

  • For the part of the history the API has already measured, use the
    authoritative `usage_metadata` carried on the most recent AIMessage
    (exact, free — the API told us).
  • For the "tail" appended SINCE that measurement (tool results + the
    pending user turn), the API hasn't counted it yet, so we ESTIMATE
    with a cheap chars/4 heuristic. Claude Code does the same — it has
    no local BPE tokenizer either.

DIVERGENCE FROM CLAUDE CODE (intentional, documented):
  CC's getTokenCountFromUsage sums input + cache_creation + cache_read +
  output because the RAW Anthropic API reports cache tokens as SEPARATE
  fields not included in input_tokens. LangChain NORMALIZES usage so
  `input_tokens` already INCLUDES cache tokens (cache_read/cache_creation
  are sub-details of it). So here we use total_tokens (= input + output)
  and do NOT re-add cache, which would double-count under LangChain.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage


# Heuristic constants (ported from tokens.ts).
DEFAULT_BYTES_PER_TOKEN = 4
"""Rough English-text ratio: ~4 chars per token. CC uses the same."""

JSON_BYTES_PER_TOKEN = 3
"""Tool results are JSON-dense (lots of punctuation/short tokens), so
they pack fewer chars per token. CC uses a tighter ratio for these."""

IMAGE_TOKEN_SIZE = 2_000
"""Flat per-image cost. CC's IMAGE_TOKEN_SIZE. We can't measure image
blocks with a char heuristic, so each counts as a flat 2000."""


def get_token_count_from_usage(usage_metadata: Any) -> int | None:
    """Authoritative token count from a LangChain UsageMetadata dict.

    Returns None if there's no usable usage. Prefers total_tokens; falls
    back to input_tokens + output_tokens. (See module docstring on why we
    don't add cache fields separately under LangChain.)
    """
    if not usage_metadata:
        return None
    # usage_metadata is a TypedDict at runtime → plain dict access.
    total = usage_metadata.get("total_tokens")
    if isinstance(total, int) and total > 0:
        return total
    inp = usage_metadata.get("input_tokens") or 0
    out = usage_metadata.get("output_tokens") or 0
    combined = int(inp) + int(out)
    return combined if combined > 0 else None


def rough_token_estimation(message: BaseMessage) -> int:
    """Cheap chars/N estimate for one message whose tokens the API has
    not measured yet (a tool result, or the pending user turn).

    Mirrors CC's rough_token_estimation: text length / bytes-per-token,
    a flat cost per image block, and a JSON-denser ratio for tool
    results.
    """
    content = message.content
    is_tool = message.type == "tool"
    ratio = JSON_BYTES_PER_TOKEN if is_tool else DEFAULT_BYTES_PER_TOKEN

    # str content — the common case.
    if isinstance(content, str):
        return _chars_to_tokens(len(content), ratio)

    # list-of-blocks content (multimodal / structured). Sum text length,
    # add a flat cost per image block.
    if isinstance(content, list):
        text_len = 0
        tokens = 0
        for block in content:
            if isinstance(block, str):
                text_len += len(block)
            elif isinstance(block, dict):
                btype = block.get("type")
                if btype == "image" or btype == "image_url":
                    tokens += IMAGE_TOKEN_SIZE
                elif btype == "text":
                    text_len += len(block.get("text", ""))
                else:
                    # tool_use / tool_result / thinking blocks etc. —
                    # estimate from their stringified size.
                    text_len += len(str(block))
        tokens += _chars_to_tokens(text_len, ratio)
        return tokens

    # Anything else — stringify and estimate.
    return _chars_to_tokens(len(str(content)), ratio)


def token_count_with_estimation(messages: list[BaseMessage]) -> int:
    """Best estimate of the conversation's current token size.

    Algorithm (CC's tokenCountWithEstimation):
      1. Walk from the END to find the most recent AIMessage carrying
         real usage_metadata. That usage measures EVERYTHING up to and
         including that assistant turn (the API counted the whole prompt
         it was given).
      2. Anchor = that usage count.
      3. Add a rough estimate for every message AFTER the anchor (tool
         results, the next user turn) — the API hasn't seen those yet.
      4. If no usage anywhere (fresh conversation), rough-estimate the
         whole list.
    """
    anchor_index = -1
    anchor_tokens = 0

    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, AIMessage):
            usage = getattr(msg, "usage_metadata", None)
            count = get_token_count_from_usage(usage)
            if count is not None:
                anchor_index = i
                anchor_tokens = count
                break

    if anchor_index == -1:
        # No measured turn yet — estimate everything.
        return sum(rough_token_estimation(m) for m in messages)

    # Anchor covers messages[0..anchor_index]; estimate the tail.
    tail = messages[anchor_index + 1:]
    return anchor_tokens + sum(rough_token_estimation(m) for m in tail)


# ── internal ────────────────────────────────────────────────────────────

def _chars_to_tokens(char_len: int, bytes_per_token: int) -> int:
    if char_len <= 0:
        return 0
    return -(-char_len // bytes_per_token)  # ceil division
