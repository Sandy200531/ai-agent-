"""Date and time tools."""

import logging
from datetime import date, datetime

from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_FORMATS_WITH_YEAR = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%B %d %Y",
    "%b %d, %Y",
    "%b %d %Y",
    "%d.%m.%Y",
]
_FORMATS_WITHOUT_YEAR = [
    "%d/%m",
    "%m/%d",
    "%d-%m",
    "%m-%d",
    "%d %B",
    "%d %b",
    "%B %d",
    "%b %d",
    "%B %d,",
    "%b %d,",
]


@tool
def get_current_datetime() -> str:
    """Return the current date and time (local system clock)."""
    now = datetime.now()
    logger.info("get_current_datetime called")
    return now.strftime("%A, %B %d, %Y %I:%M:%S %p")


class DaysUntilInput(BaseModel):
    target_date: str = Field(
        description=(
            "The target date, e.g. '2026-12-31', '31/12/2026', "
            "'December 31, 2026', or 'December 31'."
        )
    )


def _parse_date(text: str) -> date:
    cleaned = text.strip()
    for fmt in _FORMATS_WITH_YEAR:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    for fmt in _FORMATS_WITHOUT_YEAR:
        try:
            parsed = datetime.strptime(cleaned, fmt)
            candidate = parsed.replace(year=date.today().year).date()
            if candidate < date.today():
                candidate = candidate.replace(year=date.today().year + 1)
            return candidate
        except ValueError:
            continue
    raise ValueError(f"Could not parse date '{text}'. Try formats like '2026-12-31' or 'December 31, 2026'.")


@tool(args_schema=DaysUntilInput)
def days_until_date(target_date: str) -> str:
    """Calculate the number of days from today until the given target date."""
    logger.info("days_until_date input: %s", target_date)
    try:
        target = _parse_date(target_date)
        delta = (target - date.today()).days
        if delta > 0:
            return f"There are {delta} day(s) left until {target.isoformat()}."
        if delta == 0:
            return f"{target.isoformat()} is today."
        return f"{target.isoformat()} was {abs(delta)} day(s) ago."
    except ValueError as exc:
        return f"Error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return f"Error: could not compute days until '{target_date}': {exc}"
