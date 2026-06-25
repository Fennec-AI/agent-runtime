"""Edit — find/replace in a single file.

Mirrors Claude Code's `Edit` tool. Required: file_path, old_string,
new_string. Optional: replace_all (default False).

Semantics:
  - If replace_all=False (default), old_string MUST be unique in the
    file. If it appears 0 or 2+ times, return an error so the model
    can disambiguate by adding more context.
  - If replace_all=True, every occurrence is replaced.
  - If old_string=="" and the file doesn't exist, the file is created
    with new_string as its content (Claude Code's "create new file"
    shorthand).
  - If old_string == new_string, error (no change to make).

NOT concurrency-safe — file mutation. Stays in the unsafe partition.

# TODO — feature parity with Claude Code's FileEditTool, intentionally
#        deferred to keep this minimal:
#
#   [ ] Read-before-edit check. Claude Code verifies the file has been
#       Read at least once in this session before allowing edits — saves
#       the model from blind edits.
#
#   [ ] Atomic write (write to temp + rename). We do a direct write,
#       which is fine on a sane filesystem but not crash-safe.
#
#   [ ] Permission classifier / secret detection. Claude Code refuses
#       edits that would write secrets / tokens. We just write.
#
#   [ ] Encoding preservation. We always write utf-8; original file may
#       have been latin-1 etc. Real impl would detect and round-trip.
#
#   [ ] Newline preservation. Trailing newline handling is best-effort.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class EditInput(BaseModel):
    """Arguments for Edit."""
    file_path: str = Field(
        description="Absolute path to the file to edit.",
    )
    old_string: str = Field(
        description=(
            "Exact text to find and replace. Must be unique in the file "
            "unless replace_all=True. Use empty string + nonexistent file "
            "to create a new file with new_string as its content."
        ),
    )
    new_string: str = Field(
        description="Text to replace old_string with (must differ from old_string).",
    )
    replace_all: bool = Field(
        default=False,
        description="Replace every occurrence (default False = require unique match).",
    )


class EditTool(BaseTool):
    """Find/replace text in a file."""

    name: str = "Edit"
    description: str = (
        "Find and replace text in a file. By default `old_string` must be "
        "unique in the file — add surrounding context if it isn't. Set "
        "`replace_all=True` to replace every occurrence. Empty `old_string` "
        "on a nonexistent file creates the file with `new_string` as content. "
        "Returns the new content's first matching context or an error."
    )
    args_schema: type[BaseModel] = EditInput
    # NOT concurrency-safe — mutates a file. Gated + edits_files: auto-allowed
    # in acceptEdits mode, prompts in default mode, blocked in plan mode.
    metadata: dict[str, Any] = {
        "concurrency_safe": False, "permission": "ask", "edits_files": True,
    }

    async def _arun(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        path = Path(file_path)

        # ── Basic validation ──────────────────────────────────────────
        if not path.is_absolute():
            return f"Error: file_path must be absolute (got {file_path!r})."
        if old_string == new_string:
            return "Error: old_string and new_string are identical — no change to make."

        # ── Special case: create new file when old_string == "" ───────
        if old_string == "":
            if path.exists():
                if path.read_text(encoding="utf-8", errors="replace") != "":
                    return (
                        f"Error: file already exists and is non-empty; "
                        "cannot create with empty old_string."
                    )
                # Empty file → just write new_string
            else:
                # Make parent dirs if needed
                path.parent.mkdir(parents=True, exist_ok=True)
            try:
                path.write_text(new_string, encoding="utf-8")
            except OSError as e:
                return f"Error writing file: {e}"
            return f"Created {path} ({len(new_string)} chars)."

        # ── Normal edit path: file must exist ─────────────────────────
        if not path.exists():
            return f"Error: file does not exist: {file_path}"
        if not path.is_file():
            return f"Error: not a regular file: {file_path}"

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"Error reading file: {e}"

        # ── Verify old_string exists (and uniqueness if needed) ───────
        count = content.count(old_string)
        if count == 0:
            return f"Error: old_string not found in {file_path}."
        if count > 1 and not replace_all:
            return (
                f"Error: old_string appears {count} times in {file_path}. "
                "Add more surrounding context to make it unique, or set "
                "replace_all=True to replace all occurrences."
            )

        # ── Replace ───────────────────────────────────────────────────
        if replace_all:
            new_content = content.replace(old_string, new_string)
        else:
            new_content = content.replace(old_string, new_string, 1)

        # ── Write back ────────────────────────────────────────────────
        try:
            path.write_text(new_content, encoding="utf-8")
        except OSError as e:
            return f"Error writing file: {e}"

        replaced = count if replace_all else 1
        return f"Replaced {replaced} occurrence(s) in {file_path}."

    def _run(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("EditTool is async-only; use ainvoke().")
