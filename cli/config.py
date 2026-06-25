"""CLI-specific configuration — currently just the tool list.

Model registrations and subagent types live in python_version/bootstrap.py
(the app's composition root), since they're application-wide concerns
shared by every entry point (CLI, future API, tests).
"""
from __future__ import annotations

from langchain_core.tools import BaseTool

from agent_runtime.tools import (
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
    TodoWriteTool,
    WriteTool,
)


def default_tools() -> list[BaseTool]:
    """The standard tool list the CLI registers with the Session.

    Background-agent surface:
      SpawnAgent → TaskStop / SendMessage / TaskOutput
      (model gets the task_id from the spawn ack and uses it on the
      others.)

    Background-shell surface:
      Bash(run_in_background=True) → BashOutput / KillShell
      (same pattern — shell_id flows from spawn to the other tools.)
    """
    return [
        ReadTool(),
        GlobTool(),
        GrepTool(),
        BashTool(),
        BashOutputTool(),
        KillShellTool(),
        EditTool(),
        WriteTool(),
        TodoWriteTool(),
        SpawnAgentTool(),
        TaskStopTool(),
        SendMessageTool(),
        TaskOutputTool(),
    ]
