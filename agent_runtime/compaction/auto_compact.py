"""Auto-compaction orchestration (Mechanism A).

Ported from services/compact/autoCompact.ts (autoCompactIfNeeded +
shouldAutoCompact). Runs once per loop iteration, BEFORE the model call.
Decides whether the live token count has crossed the threshold and, if so,
drives the full compaction (Mechanism B) and post-cleanup (Mechanism D).

Two guards beyond the threshold:
  • Circuit breaker — after N consecutive failed compactions, stop trying
    (a persistently-failing summarize would otherwise burn a call every
    turn forever).
  • Blocking limit — if we're past the HARD limit and did NOT compact,
    the prompt genuinely won't fit; signal the caller to refuse the turn.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from ..config import CONFIG
from ..types import RunContext
from .budget import calculate_token_warning_state
from .compact import CompactionError, compact_conversation
from .cleanup import run_post_compact_cleanup
from .tokens import token_count_with_estimation


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CompactionOutcome:
    """Result of an auto-compaction check."""
    was_compacted: bool = False
    blocking_limit: bool = False
    """True when the conversation is past the hard limit AND we did not
    compact — the caller must refuse the turn (prompt won't fit)."""
    token_count: int = 0
    threshold: int = 0


def _model_name(ctx: RunContext) -> str | None:
    return getattr(ctx.llm, "model", None) or getattr(ctx.llm, "model_name", None)


async def auto_compact_if_needed(
    ctx: RunContext,
    snip_freed: int = 0,
    *,
    transcript_path: str | None = None,
) -> CompactionOutcome:
    """Check the threshold and compact if over. Returns a CompactionOutcome.

    `snip_freed` is the token count microcompaction already freed this
    turn — subtracted from the live count so we don't double-trigger.
    """
    model = _model_name(ctx)
    token_count = max(0, token_count_with_estimation(ctx.messages) - snip_freed)
    state = calculate_token_warning_state(token_count, model)

    # ── Disabled / circuit-broken: only the blocking guard still applies ──
    state_dict = ctx.compaction_state
    failures = state_dict.get("consecutive_failures", 0)
    disabled = not CONFIG.auto_compact_enabled
    broken = failures >= CONFIG.max_consecutive_autocompact_failures

    if disabled or broken:
        if disabled is False and broken:
            logger.warning(
                "auto-compaction circuit breaker open (%d consecutive "
                "failures) — not attempting", failures,
            )
        return _maybe_blocking(state)

    # ── Under threshold: nothing to do (blocking guard still checked) ──
    if not state.is_above_autocompact:
        return _maybe_blocking(state)

    # ── Over threshold: compact ──────────────────────────────────────
    logger.info(
        "auto-compaction triggered: %d tokens >= threshold %d",
        token_count, state.autocompact_threshold,
    )
    try:
        await compact_conversation(
            ctx, is_auto=True, transcript_path=transcript_path,
        )
    except CompactionError as e:
        failures += 1
        state_dict["consecutive_failures"] = failures
        logger.warning(
            "auto-compaction failed (%d/%d): %s",
            failures, CONFIG.max_consecutive_autocompact_failures, e,
        )
        # Recompute the blocking guard against the (unchanged) history.
        post_state = calculate_token_warning_state(
            token_count_with_estimation(ctx.messages), model,
        )
        return _maybe_blocking(post_state)

    # Success — reset tracking + run post-compact cleanup.
    state_dict.update({
        "compacted": True,
        "turn_id": str(uuid.uuid4()),
        "turn_counter": 0,
        "consecutive_failures": 0,
    })
    await run_post_compact_cleanup(ctx)
    return CompactionOutcome(
        was_compacted=True,
        token_count=token_count,
        threshold=state.autocompact_threshold,
    )


def _maybe_blocking(state) -> CompactionOutcome:
    """Build a non-compacted outcome, flagging the blocking limit if the
    history is past the hard cap."""
    return CompactionOutcome(
        was_compacted=False,
        blocking_limit=state.is_at_blocking_limit,
        token_count=state.token_count,
        threshold=state.autocompact_threshold,
    )
