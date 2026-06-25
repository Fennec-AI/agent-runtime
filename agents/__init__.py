"""Built-in subagent catalog.

Definitions mirror Claude Code's built-in agents
(src/tools/AgentTool/built-in/) and commands (src/commands/):

  - general-purpose : the default workhorse, inherits all tools
  - Explore         : fast read-only file/code search
  - Plan            : read-only architect that returns an implementation plan
  - security-review : senior security engineer; gathers the branch diff,
                      fans out finder + parallel verifier sub-agents, and
                      returns only high-confidence (>=8) findings. Ported
                      from Claude Code's /security-review command.

Bootstrap imports these and calls register_subagent() for each.
"""
from .explorer import EXPLORE_AGENT
from .general_purpose import GENERAL_PURPOSE_AGENT
from .planner import PLAN_AGENT
from .security_reviewer import SECURITY_REVIEW_AGENT

__all__ = [
    "EXPLORE_AGENT",
    "GENERAL_PURPOSE_AGENT",
    "PLAN_AGENT",
    "SECURITY_REVIEW_AGENT",
]
