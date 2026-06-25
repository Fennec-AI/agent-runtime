"""Grep — content search across files.

Mirrors a useful subset of Claude Code's `Grep` tool (which wraps
ripgrep). Supports the three output modes that matter most:
  - "files_with_matches" (default) — list files that contain a match
  - "content"                       — show each matching line
  - "count"                         — count matches per file

Read-only and marked concurrency-safe.

# TODO — feature parity with Claude Code's GrepTool, intentionally
#        deferred to keep this minimal:
#
#   [ ] Context lines (-A / -B / -C). Claude Code surfaces N lines
#       before/after each match. We just emit the matching line.
#
#   [ ] Multiline mode. Patterns spanning newlines need re.DOTALL +
#       a different chunking strategy. We do one line at a time.
#
#   [ ] Gitignore awareness. We walk everything that matches the file
#       glob; ripgrep skips .gitignored by default.
#
#   [ ] Binary-file detection. We try utf-8 decode with errors='replace';
#       binaries will produce noise. ripgrep skips binaries by default.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


# How many output lines / file paths we'll return at most.
MAX_RESULTS = 200


class GrepInput(BaseModel):
    """Arguments for Grep."""
    pattern: str = Field(
        description="Regular expression pattern to search for in file contents.",
    )
    path: str | None = Field(
        default=None,
        description=(
            "Directory to search in (absolute path). Defaults to the current "
            "working directory."
        ),
    )
    glob: str | None = Field(
        default=None,
        description=(
            'Glob pattern to filter which files to search, e.g. "*.py" or '
            '"src/**/*.ts". Defaults to all files.'
        ),
    )
    output_mode: Literal["content", "files_with_matches", "count"] = Field(
        default="files_with_matches",
        description=(
            'Output format: "files_with_matches" (default) lists files, '
            '"content" shows matching lines, "count" shows per-file counts.'
        ),
    )
    case_insensitive: bool | None = Field(
        default=None,
        description="Case-insensitive matching (like rg -i).",
    )
    head_limit: int | None = Field(
        default=None, gt=0,
        description="Cap output to the first N results.",
    )


class GrepTool(BaseTool):
    """Search file contents with a regex."""

    name: str = "Grep"
    description: str = (
        "Search for a regex pattern in file contents. Walks the given path "
        "(or cwd), optionally filtered by a `glob` pattern. Three output "
        "modes: 'files_with_matches' (default), 'content', 'count'. "
        f"Capped at {MAX_RESULTS} results — use `head_limit` for less."
    )
    args_schema: type[BaseModel] = GrepInput
    metadata: dict[str, Any] = {"concurrency_safe": True}

    async def _arun(
        self,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: Literal["content", "files_with_matches", "count"] = "files_with_matches",
        case_insensitive: bool | None = None,
        head_limit: int | None = None,
    ) -> str:
        # ── Resolve search root ───────────────────────────────────────
        root = Path(path) if path else Path.cwd()
        if path and not root.is_absolute():
            return f"Error: path must be absolute (got {path!r})."
        if not root.exists():
            return f"Error: path does not exist: {root}"
        if not root.is_dir():
            return f"Error: not a directory: {root}"

        # ── Compile pattern ───────────────────────────────────────────
        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: invalid regex {pattern!r}: {e}"

        # ── Pick files to scan ────────────────────────────────────────
        file_glob = glob or "**/*"
        try:
            candidates = [p for p in root.glob(file_glob) if p.is_file()]
        except Exception as e:
            return f"Error during file glob: {e}"

        if not candidates:
            return f"No files matched glob {file_glob!r} in {root}."

        # ── Scan ──────────────────────────────────────────────────────
        # Accumulate as appropriate for the output mode.
        files_with_matches: list[Path] = []
        per_file_counts: dict[Path, int] = {}
        content_lines: list[tuple[Path, int, str]] = []     # (file, lineno, line)

        for fp in candidates:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue                                     # skip unreadable

            matched_in_this_file = False
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    matched_in_this_file = True
                    if output_mode == "content":
                        content_lines.append((fp, lineno, line))
                    elif output_mode == "count":
                        per_file_counts[fp] = per_file_counts.get(fp, 0) + 1
            if matched_in_this_file and output_mode == "files_with_matches":
                files_with_matches.append(fp)

        # ── Format ────────────────────────────────────────────────────
        cap = head_limit if head_limit is not None else MAX_RESULTS

        if output_mode == "files_with_matches":
            if not files_with_matches:
                return "No matches."
            total = len(files_with_matches)
            shown = files_with_matches[:cap]
            out = "\n".join(str(p) for p in shown)
            if total > cap:
                out += f"\n\n[Showed {cap} of {total} matching files.]"
            return out

        if output_mode == "count":
            if not per_file_counts:
                return "No matches."
            items = sorted(per_file_counts.items(), key=lambda kv: kv[1], reverse=True)
            shown = items[:cap]
            out = "\n".join(f"{count:>6}  {fp}" for fp, count in shown)
            if len(items) > cap:
                out += f"\n\n[Showed top {cap} of {len(items)} files.]"
            return out

        # output_mode == "content"
        if not content_lines:
            return "No matches."
        shown = content_lines[:cap]
        out_parts = [f"{fp}:{lineno}: {line}" for fp, lineno, line in shown]
        out = "\n".join(out_parts)
        if len(content_lines) > cap:
            out += f"\n\n[Showed {cap} of {len(content_lines)} matching lines.]"
        return out

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("GrepTool is async-only; use ainvoke().")
