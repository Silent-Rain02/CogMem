"""Conservative date normalization without inventing missing precision."""
from datetime import date, datetime, timedelta
import calendar
import re
from typing import Optional


def normalize_event_time(value: str, reference: Optional[str] = None) -> Optional[str]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().lower()
    # Preserve explicit precision, including common unknown month/day notation.
    text = re.sub(r"-xx(?:-xx)?$", "", text)
    if re.fullmatch(r"\d{4}", text):
        return text if 1 <= int(text) <= 9999 else None
    if re.fullmatch(r"\d{4}-\d{2}", text):
        try:
            date(int(text[:4]), int(text[5:]), 1)
            return text
        except ValueError:
            return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return date.fromisoformat(text).isoformat()
        except ValueError:
            return None
    for pattern, precision in (("%d %B %Y", "day"), ("%B %d %Y", "day"),
                               ("%d %b %Y", "day"), ("%B %Y", "month")):
        try:
            parsed = datetime.strptime(text, pattern)
            return parsed.strftime("%Y-%m" if precision == "month" else "%Y-%m-%d")
        except ValueError:
            pass
    # A relative expression requires a complete, known reference date.
    try:
        anchor = date.fromisoformat(reference or "")
    except (ValueError, TypeError):
        return None
    wrapper = re.fullmatch(r"(.+?)\s*\(relative to (\d{4}-\d{2}-\d{2})\)", text)
    if wrapper:
        if wrapper[2] != anchor.isoformat():
            return None
        text = wrapper[1].strip()
    if text in {"yesterday", "today", "tomorrow"}:
        return (anchor + timedelta(days={"yesterday": -1, "today": 0, "tomorrow": 1}[text])).isoformat()
    if text in {"last year", "this year", "next year"}:
        year = anchor.year + {"last year": -1, "this year": 0, "next year": 1}[text]
        return f"{year:04d}" if 1 <= year <= 9999 else None
    if text in {"last month", "this month", "next month"}:
        offset = {"last month": -1, "this month": 0, "next month": 1}[text]
        year, month = divmod(anchor.year * 12 + anchor.month - 1 + offset, 12)
        return f"{year:04d}-{month + 1:02d}" if 1 <= year <= 9999 else None
    match = re.fullmatch(r"(\d+)\s+(days?|weeks?)\s+ago", text)
    if match:
        days = int(match[1]) * (7 if match[2].startswith("week") else 1)
        try:
            return (anchor - timedelta(days=days)).isoformat()
        except (ValueError, OverflowError):
            return None
    match = re.fullmatch(r"last\s+(" + "|".join(d.lower() for d in calendar.day_name) + r")", text)
    if match:
        weekday = [d.lower() for d in calendar.day_name].index(match[1])
        return (anchor - timedelta(days=(anchor.weekday() - weekday) % 7 or 7)).isoformat()
    # "Recently", "last week", etc. denote intervals or uncertainty, not a day.
    return None


def date_bounds(value):
    """Inclusive calendar-date bounds; coarse precision stays an interval."""
    if not isinstance(value, str):
        return None
    if 'T' in value:
        try:
            point = datetime.fromisoformat(value.replace('Z', '+00:00')).date()
            return point, point
        except ValueError:
            return None
    normalized = normalize_event_time(value)
    if normalized is None:
        return None
    year = int(normalized[:4])
    if len(normalized) == 4:
        return date(year, 1, 1), date(year, 12, 31)
    if len(normalized) == 7:
        month = int(normalized[5:])
        return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])
    point = date.fromisoformat(normalized)
    return point, point
