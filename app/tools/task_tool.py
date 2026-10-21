"""Task / To-Do tools backed by a local JSON file."""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_DEFAULT_FILE = Path(__file__).resolve().parents[2] / "data" / "tasks.json"


def _tasks_file() -> Path:
    # Read at call time so tests can override via environment variable.
    return Path(os.getenv("TASKS_FILE", str(_DEFAULT_FILE)))


def _load_tasks() -> list[dict]:
    path = _tasks_file()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Could not read tasks file %s: %s", path, exc)
        return []


def _save_tasks(tasks: list[dict]) -> None:
    path = _tasks_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")


class CreateTaskInput(BaseModel):
    title: str = Field(description="A short title describing the task.")


class CompleteTaskInput(BaseModel):
    task: str = Field(
        description="The id or the title (exact or partial) of the task to mark as completed."
    )


@tool(args_schema=CreateTaskInput)
def create_task(title: str) -> str:
    """Create a new task in the to-do list."""
    logger.info("create_task input: %s", title)
    if not title or not title.strip():
        return "Error: task title must not be empty."
    tasks = _load_tasks()
    task = {
        "id": uuid.uuid4().hex[:8],
        "title": title.strip(),
        "completed": False,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    tasks.append(task)
    try:
        _save_tasks(tasks)
    except OSError as exc:
        return f"Error: could not save task: {exc}"
    return f"Task created successfully: '{task['title']}' (id: {task['id']})."


@tool
def list_tasks() -> str:
    """List all tasks with their status."""
    logger.info("list_tasks called")
    tasks = _load_tasks()
    if not tasks:
        return "No tasks found."
    lines = []
    for t in tasks:
        status = "done" if t.get("completed") else "pending"
        lines.append(f"- [{status}] {t.get('title', '(untitled)')} (id: {t.get('id', '?')})")
    return "Tasks:\n" + "\n".join(lines)


@tool(args_schema=CompleteTaskInput)
def complete_task(task: str) -> str:
    """Mark a task as completed. Accepts the task id or title."""
    logger.info("complete_task input: %s", task)
    if not task or not task.strip():
        return "Error: please provide the task id or title."
    tasks = _load_tasks()
    needle = task.strip().lower()
    match = None
    for t in tasks:
        if t.get("id", "").lower() == needle or t.get("title", "").lower() == needle:
            match = t
            break
    if match is None:
        for t in tasks:
            if needle in t.get("title", "").lower():
                match = t
                break
    if match is None:
        return f"Error: no task found matching '{task}'."
    if match.get("completed"):
        return f"Task '{match.get('title')}' is already completed."
    match["completed"] = True
    match["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        _save_tasks(tasks)
    except OSError as exc:
        return f"Error: could not save tasks: {exc}"
    return f"Task '{match.get('title')}' marked as completed."
