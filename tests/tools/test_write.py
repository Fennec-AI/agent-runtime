"""Write tool — create/overwrite + safety rules."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_runtime import WriteTool


pytestmark = pytest.mark.asyncio


async def test_creates_new_file(tmp_path: Path):
    write = WriteTool()
    target = tmp_path / "new.txt"
    msg = await write.ainvoke({
        "file_path": str(target),
        "content": "hello world",
    })
    assert "Created" in msg
    assert "11 bytes" in msg
    assert target.read_text() == "hello world"


async def test_overwrites_existing_file(tmp_path: Path):
    write = WriteTool()
    target = tmp_path / "x.txt"
    target.write_text("old content")

    msg = await write.ainvoke({
        "file_path": str(target),
        "content": "new content",
    })
    assert "Overwrote" in msg
    assert target.read_text() == "new content"


async def test_rejects_relative_path():
    write = WriteTool()
    msg = await write.ainvoke({
        "file_path": "relative/path.txt",
        "content": "x",
    })
    assert "must be absolute" in msg


async def test_rejects_missing_parent(tmp_path: Path):
    write = WriteTool()
    target = tmp_path / "noexist" / "child.txt"
    msg = await write.ainvoke({
        "file_path": str(target),
        "content": "x",
    })
    assert "parent directory does not exist" in msg
    assert not target.exists()


async def test_rejects_directory_target(tmp_path: Path):
    write = WriteTool()
    msg = await write.ainvoke({
        "file_path": str(tmp_path),
        "content": "x",
    })
    assert "is a directory" in msg


async def test_utf8_multibyte_byte_count(tmp_path: Path):
    """`héllo` is 6 bytes in UTF-8, not 5 chars — Write reports bytes."""
    write = WriteTool()
    target = tmp_path / "utf8.txt"
    msg = await write.ainvoke({
        "file_path": str(target),
        "content": "héllo",
    })
    assert "(6 bytes)" in msg
