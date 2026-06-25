"""Read — read a file's contents.

Mirrors Claude Code's `Read` tool:
  - takes an absolute file_path, optional offset (0-indexed line) and limit
  - returns content with line numbers (cat -n style)
  - caps at MAX_LINES_TO_READ to prevent giant payloads back to the model

Read-only and marked concurrency-safe — multiple reads run in parallel.

# TODO — feature parity with Claude Code's FileReadTool, intentionally
#        deferred to keep this minimal:
#
#   [ ] Binary-file detection. Currently we read_text(errors="replace") which
#       produces readable garbage for binaries. Claude Code routes binaries
#       through a separate path (image bytes, PDF parsing, etc.).
#
#   [ ] Token-counting cap. Claude Code measures actual token count and
#       refuses if too large (with a friendly message pointing at offset/
#       limit). We just cap by LINE count — simpler, but a 2000-line file of
#       long lines could still blow context.
#
#   [ ] Image and notebook handling. Claude Code has imageProcessor.ts for
#       images and notebook-specific code paths (.ipynb). We don't.
#
#   [ ] FILE_UNCHANGED_STUB optimization. Claude Code tracks which files
#       have been read this session and returns a stub if the file hasn't
#       changed since the last read. Saves tokens; we re-read every time.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# Cap on how many lines we'll return in a single call. Matches Claude Code's
# MAX_LINES_TO_READ constant; the agent can use offset+limit to paginate.
MAX_LINES_TO_READ = 2000


class ReadInput(BaseModel):
    """Arguments for Read."""
    file_path: str = Field(
        description="The absolute path to the file to read",
    )
    offset: int | None = Field(
        default=None, ge=0,
        description="0-indexed line to start reading from",
    )
    limit: int | None = Field(
        default=None, gt=0,
        description="Maximum number of lines to read",
    )


class ReadTool(BaseTool):
    """Read a file from the local filesystem."""

    name: str = "Read"
    description: str = (
        "Read a file from the local filesystem. Returns content with "
        "1-indexed line numbers in `cat -n` format. Use `offset` and "
        f"`limit` to read a slice of large files (cap: {MAX_LINES_TO_READ} "
        "lines per call). `file_path` must be absolute."
    )
    args_schema: type[BaseModel] = ReadInput
    metadata: dict[str, Any] = {"concurrency_safe": True}

    async def _arun(
        self,
        file_path: str,
        offset: int | None = None,
        limit: int | None = None,
    ) -> str:
        path = Path(file_path)

        # ── Validate ──────────────────────────────────────────────────
        if not path.is_absolute():
            return f"Error: file_path must be absolute (got {file_path!r})."
        if not path.exists():
            return f"Error: file does not exist: {file_path}"
        if not path.is_file():
            return f"Error: not a regular file: {file_path}"

        # ── Read ──────────────────────────────────────────────────────
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except PermissionError as e:
            return f"Error: permission denied: {e}"
        except OSError as e:
            return f"Error reading file: {e}"

        lines = text.splitlines()
        total = len(lines)

        # ── Apply offset / limit ──────────────────────────────────────
        start = offset or 0
        if start >= total and total > 0:
            return f"Error: offset {start} is past end of file ({total} lines)."

        end = min(start + (limit or MAX_LINES_TO_READ), total)
        if (limit is None) and (end - start > MAX_LINES_TO_READ):
            end = start + MAX_LINES_TO_READ

        selected = lines[start:end]

        # ── Format like `cat -n` (1-indexed) ──────────────────────────
        width = len(str(end))                             # pad line numbers
        formatted = "\n".join(
            f"{(start + i + 1):>{width}}\t{line}"
            for i, line in enumerate(selected)
        )

        # Footer note if we truncated
        footer = ""
        if end < total:
            footer = (
                f"\n\n[Showed lines {start + 1}-{end} of {total}. "
                f"Use offset={end} to continue.]"
            )

        return formatted + footer or "(empty file)"

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("ReadTool is async-only; use ainvoke().")
