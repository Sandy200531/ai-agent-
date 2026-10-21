"""Tool registry for the agent."""

from app.tools.calculator import calculator
from app.tools.datetime_tool import days_until_date, get_current_datetime
from app.tools.file_tool import read_file, summarize_file, write_file
from app.tools.search import web_search
from app.tools.task_tool import complete_task, create_task, list_tasks

ALL_TOOLS = [
    calculator,
    web_search,
    get_current_datetime,
    days_until_date,
    create_task,
    list_tasks,
    complete_task,
    read_file,
    write_file,
    summarize_file,
]
