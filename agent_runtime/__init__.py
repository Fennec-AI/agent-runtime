"""Agent runtime — minimal, extensible spawning architecture, on LangChain.

Public API:
    Session — main entrance for multi-turn conversations
    AbortController — cancellation primitive
    AGENT_REGISTRY — global registry of running agents

Messages, model, and tools are LangChain-native:
    from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
    from langchain_core.tools import BaseTool, tool
    from langchain_anthropic import ChatAnthropic
"""

from .bash_shells import (
    SHELL_REGISTRY,
    ShellEntry,
    ShellId,
    kill_shell,
    list_running_shells,
    new_shell_id,
    read_new_output,
)
from .cancellation import AbortController, create_child_controller
from .cleanup_registry import cleanup_count, register_cleanup, run_all_cleanups
from . import hooks                                      # noqa: F401 — re-export module
from .hooks import (                                     # noqa: F401 — convenience aliases
    clear_hooks,
    hook_count,
    register_hook,
)
from .permissions import (                               # noqa: F401
    AskCallback,
    PermissionMode,
    PermissionResult,
    evaluate_permission,
)
from .config import CONFIG, Config
from .models import (
    MODEL_REGISTRY,
    ModelProfileNotFoundError,
    get_model,
    has_model,
    list_model_names,
    register_model,
)
from .subagents import (
    SUBAGENT_REGISTRY,
    SubagentDefinition,
    get_subagent,
    list_subagent_types,
    register_subagent,
)
from .tools import (
    BashOutputTool,
    BashTool,
    EditTool,
    GlobTool,
    GrepTool,
    KillShellTool,
    ReadTool,
    SendMessageTool,
    SpawnAgentTool,
    TaskOutputTool,
    TaskStopTool,
    WriteTool,
)
from .registry import (
    AGENT_REGISTRY,
    AgentEntry,
    drain_inbox,
    enqueue_notification,
    get_agent,
    kill_agent,
    list_children,
    list_running_agents,
    new_agent_id,
)
from .session import Session
from .types import AgentId, RunContext, ToolUseId

__all__ = [
    # session
    "Session",
    # cancellation
    "AbortController",
    "create_child_controller",
    # cleanup registry (shutdown handlers)
    "cleanup_count",
    "register_cleanup",
    "run_all_cleanups",
    # hooks (observation + extension)
    "hooks",
    "clear_hooks",
    "hook_count",
    "register_hook",
    # permissions (tool gating)
    "AskCallback",
    "PermissionMode",
    "PermissionResult",
    "evaluate_permission",
    # bash shell registry (persistent background shells)
    "SHELL_REGISTRY",
    "ShellEntry",
    "ShellId",
    "kill_shell",
    "list_running_shells",
    "new_shell_id",
    "read_new_output",
    # config
    "CONFIG",
    "Config",
    # model profiles
    "MODEL_REGISTRY",
    "ModelProfileNotFoundError",
    "get_model",
    "has_model",
    "list_model_names",
    "register_model",
    # subagents
    "SUBAGENT_REGISTRY",
    "SubagentDefinition",
    "get_subagent",
    "list_subagent_types",
    "register_subagent",
    # tools
    "BashOutputTool",
    "BashTool",
    "EditTool",
    "GlobTool",
    "GrepTool",
    "KillShellTool",
    "ReadTool",
    "SendMessageTool",
    "SpawnAgentTool",
    "TaskOutputTool",
    "TaskStopTool",
    "WriteTool",
    # registry
    "AGENT_REGISTRY",
    "AgentEntry",
    "drain_inbox",
    "enqueue_notification",
    "get_agent",
    "kill_agent",
    "list_children",
    "list_running_agents",
    "new_agent_id",
    # types
    "AgentId",
    "RunContext",
    "ToolUseId",
]
