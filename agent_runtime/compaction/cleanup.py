"""Post-compaction cleanup (Mechanism D, simplified for a headless port).

Ported from services/compact/postCompactCleanup.ts. In Claude Code this
resets a pile of caches (microcompact state, memoized system prompt, skill
payloads) and is gated on main-thread-vs-subagent because those caches are
module globals.

In our runtime that guard is unnecessary — every agent owns a separate
RunContext, so there are no shared globals to corrupt. The cleanup here is
therefore minimal: reset the microcompaction timer and fire a hook so any
observer (or a future cache layer) can react.
"""
from __future__ import annotations

import logging

from .. import hooks
from ..types import RunContext


logger = logging.getLogger(__name__)


async def run_post_compact_cleanup(ctx: RunContext) -> None:
    """Reset per-agent compaction-adjacent state after a successful
    compaction, and emit a 'post_compact' hook event."""
    # The microcompaction gap timer is meaningless after a full
    # compaction (the old tool results are gone), so clear it.
    ctx.compaction_state.pop("last_microcompact_ts", None)

    # Fire a hook so observers (or future cache invalidation) can react.
    # hooks.fire swallows handler exceptions, so this can't break unwind.
    await hooks.fire(
        "post_compact",
        agent_id=ctx.agent_id,
        message_count=len(ctx.messages),
    )
    logger.debug("post-compact cleanup done for %s", ctx.agent_id)
