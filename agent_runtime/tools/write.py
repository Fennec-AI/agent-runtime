"""Write — create or overwrite a file with given content.

Companion to Edit (which does find/replace on an existing file). Write
takes the full new contents and replaces the file wholesale (or creates
it if absent). Mirrors Claude Code's Write tool surface.

Schema:
    file_path: str  — absolute path
    content:   str  — full file contents

Refuses relative paths (same as Read/Edit) — agents in different cwds
would otherwise have ambiguous targets. Errors if the parent directory
doesn't exist (so a typo in the path doesn't silently create files in
the wrong tree).

NOT concurrency-safe — writes mutate the filesystem, and two parallel
writes to the same path race.
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class WriteInput(BaseModel):
    """Arguments for Write."""

    file_path: str = Field(
        description=(
            "Absolute path to the file to write. Must be absolute. "
            "Parent directory must already exist."
        ),
    )
    content: str = Field(
        description="The full contents to write to the file.",
    )


class WriteTool(BaseTool):
    """Create or overwrite a file with the given content."""

    name: str = "Write"
    description: str = (
        "Write a file. If the file exists it is overwritten wholesale; "
        "if it does not exist it is created. The parent directory must "
        "already exist (we do not auto-mkdir, to catch path typos). Use "
        "an absolute path."
    )
    args_schema: type[BaseModel] = WriteInput
    # Gated + edits_files: auto-allowed in acceptEdits mode, prompts in
    # default mode (with a callback), blocked in plan mode.
    metadata: dict[str, Any] = {
        "concurrency_safe": False, "permission": "ask", "edits_files": True,
    }

    async def _arun(self, file_path: str, content: str) -> str:
        if not os.path.isabs(file_path):
            return (
                f"Error: file_path must be absolute, got {file_path!r}."
            )

        path = Path(file_path)
        parent = path.parent
        if not parent.exists():
            return (
                f"Error: parent directory does not exist: {parent}. "
                f"Create it first or fix the path."
            )
        if not parent.is_dir():
            return (
                f"Error: parent path exists but is not a directory: {parent}."
            )
        if path.exists() and path.is_dir():
            return (
                f"Error: path exists and is a directory: {path}. "
                f"Cannot overwrite a directory with file contents."
            )

        # Capture "did we create vs overwrite" BEFORE the write — after
        # the write path.exists() is always True so the post-check is
        # tautological.
        existed = path.exists()

        # Off-thread the write so we don't block the event loop on big
        # files / slow disks. write_text is sync; asyncio.to_thread is
        # the right tool here.
        try:
            n_bytes = await asyncio.to_thread(
                _write_text, path, content,
            )
        except OSError as e:
            return f"Error writing {path}: {e}"

        verb = "Overwrote" if existed else "Created"
        logger.info("Write: %s %s (%d bytes)", verb.lower(), path, n_bytes)
        return f"{verb} {path} ({n_bytes} bytes)"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("WriteTool is async-only; use ainvoke().")


def _write_text(path: Path, content: str) -> int:
    """Sync writer run in a worker thread. Returns bytes written."""
    encoded = content.encode("utf-8")
    path.write_bytes(encoded)
    return len(encoded)
