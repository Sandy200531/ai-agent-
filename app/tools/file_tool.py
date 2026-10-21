"""File tools: read, write, and summarize text files."""

import logging
import re
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


class FilePathInput(BaseModel):
    path: str = Field(description="Path to the text file (relative to project root or absolute).")


class WriteFileInput(BaseModel):
    path: str = Field(description="Path to the text file to write.")
    content: str = Field(description="Text content to write to the file.")


@tool(args_schema=FilePathInput)
def read_file(path: str) -> str:
    """Read the contents of a text file."""
    logger.info("read_file input: %s", path)
    try:
        target = _resolve(path)
        if not target.exists():
            return f"Error: file '{path}' does not exist."
        if target.is_dir():
            return f"Error: '{path}' is a directory, not a file."
        text = target.read_text(encoding="utf-8", errors="replace")
        logger.info("read_file read %d characters", len(text))
        return text if text.strip() else "(file is empty)"
    except OSError as exc:
        return f"Error: could not read file '{path}': {exc}"


@tool(args_schema=WriteFileInput)
def write_file(path: str, content: str) -> str:
    """Write text content to a file, creating directories if needed."""
    logger.info("write_file input: path=%s, %d chars", path, len(content or ""))
    if content is None:
        return "Error: content must not be null."
    try:
        target = _resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} characters to '{target}'."
    except OSError as exc:
        return f"Error: could not write file '{path}': {exc}"


@tool(args_schema=FilePathInput)
def summarize_file(path: str) -> str:
    """Summarize a text file: word/line counts and the first few sentences."""
    logger.info("summarize_file input: %s", path)
    try:
        target = _resolve(path)
        if not target.exists():
            return f"Error: file '{path}' does not exist."
        text = target.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return f"File '{path}' is empty."
        words = len(text.split())
        lines = text.count("\n") + 1
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        preview = " ".join(sentences[:3])
        if len(preview) > 600:
            preview = preview[:600] + "..."
        return (
            f"Summary of '{path}':\n"
            f"- Lines: {lines}, Words: {words}, Characters: {len(text)}\n"
            f"- Preview: {preview}"
        )
    except OSError as exc:
        return f"Error: could not summarize file '{path}': {exc}"
