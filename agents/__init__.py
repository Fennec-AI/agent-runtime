"""Built-in subagent catalog.

Definitions mirror Claude Code's three core built-in agents
(src/tools/AgentTool/built-in/):

  - general-purpose : the default workhorse, inherits all tools
  - Explore         : fast read-only file/code search
  - Plan            : read-only architect that returns an implementation plan

Bootstrap imports these and calls register_subagent() for each.
"""
from .explorer import EXPLORE_AGENT
from .general_purpose import GENERAL_PURPOSE_AGENT
from .planner import PLAN_AGENT

__all__ = ["EXPLORE_AGENT", "GENERAL_PURPOSE_AGENT", "PLAN_AGENT"]
