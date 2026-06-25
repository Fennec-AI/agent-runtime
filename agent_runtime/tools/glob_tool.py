"""Glob — file pattern matching.

Mirrors Claude Code's `Glob` tool:
  - takes a glob pattern, optional search path
  - returns matching file paths, sorted by modification time (newest first)

Read-only and marked concurrency-safe — multiple Glob calls run in
parallel under the dispatcher's safe-batch.

File name uses `glob_tool.py` rather than `glob.py` to avoid shadowing
the stdlib `glob` module.

# TODO — feature parity with Claude Code's GlobTool, intentionally
#        deferred to keep this minimal:
#
#   [ ] Gitignore awareness. Claude Code skips .gitignored files. We
#       walk everything that pathlib.glob returns. Could integrate
#       `pathspec` or shell out to `git ls-files`.
#
#   [ ] Hidden-file handling. We include dotfiles if the pattern matches
#       them. Claude Code is a bit smarter about when to surface them.
#
#   [ ] Symlink loops. pathlib.glob follows symlinks; a cycle could
#       theoretically hang. Real impl should guard.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# Cap on how many paths we'll return. Prevents a runaway `**/*` from
# spewing 100K paths back to the model.
MAX_RESULTS = 100


class GlobInput(BaseModel):
    """Arguments for Glob."""
    pattern: str = Field(
        description=(
            "The glob pattern to match files against, e.g. '**/*.py' or "
            "'src/**/*.ts'. Supports recursive '**' wildcards."
        ),
    )
    path: str | None = Field(
        default=None,
        description=(
            "The directory to search in (absolute path). If omitted, the "
            "current working directory is used."
        ),
    )


class GlobTool(BaseTool):
    """Fast file pattern matching."""

    name: str = "Glob"
    description: str = (
        "Fast file pattern matching tool that works with any codebase size. "
        "Supports glob patterns like '**/*.py' or 'src/**/*.ts'. Returns "
        "matching file paths sorted by modification time (newest first). "
        f"Capped at {MAX_RESULTS} results — use a more specific pattern "
        "or path if you need more."
    )
    args_schema: type[BaseModel] = GlobInput
    metadata: dict[str, Any] = {"concurrency_safe": True}

    async def _arun(
        self,
        pattern: str,
        path: str | None = None,
    ) -> str:
        # ── Resolve search root ───────────────────────────────────────
        root = Path(path) if path else Path.cwd()
        if path and not root.is_absolute():
            return f"Error: path must be absolute (got {path!r})."
        if not root.exists():
            return f"Error: path does not exist: {root}"
        if not root.is_dir():
            return f"Error: not a directory: {root}"

        # ── Run the glob ──────────────────────────────────────────────
        try:
            matches = list(root.glob(pattern))
        except Exception as e:
            return f"Error during glob: {e}"

        # Keep only files (Path.glob also returns directories that match).
        files = [p for p in matches if p.is_file()]
        if not files:
            return f"No files matched {pattern!r} in {root}."

        # ── Sort by modification time, newest first ───────────────────
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # ── Cap + format ──────────────────────────────────────────────
        total = len(files)
        shown = files[:MAX_RESULTS]
        lines = [str(p) for p in shown]
        out = "\n".join(lines)

        if total > MAX_RESULTS:
            out += (
                f"\n\n[Showed {MAX_RESULTS} of {total} matches; use a more "
                "specific pattern or path to narrow.]"
            )
        return out

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("GlobTool is async-only; use ainvoke().")
