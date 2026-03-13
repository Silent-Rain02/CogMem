"""Time normalization utilities for parsing relative/absolute time expressions to ISO-8601 format"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Literal, Tuple

Granularity = Literal["year", "month", "day", "time"]

def normalize_time_format(time_str: str) -> Optional[str]:
    """规范化时间字符串为标准ISO-8601格式
    
    处理常见的非标准格式：
    - "1:56 pm on 8 May, 2023" → "2023-05-08T13:56:00"
    - "8 May, 2023" → "2023-05-08"
    - "May 2023" → "2023-05-XX"
    - "2023" → "2023-XX-XX"
    - "2023-04-01XX" → "2023-04-XX" (修正错误格式)
    - "YYYY-XX-XX" → None (字面占位符，无效)
    
    Args:
        time_str: 输入的时间字符串
        
    Returns:
        规范化后的ISO-8601格式字符串，如果无法解析则返回None
    """
    if not time_str or not isinstance(time_str, str):
        return None
    
    time_str = time_str.strip()
    
    # 防止卡死：限制输入长度（时间字符串通常不会超过200字符）
    MAX_INPUT_LENGTH = 200
    if len(time_str) > MAX_INPUT_LENGTH:
        return None
    
    # 检查是否是字面占位符（无效）
    if time_str.upper() in ('YYYY-XX-XX', 'YYYY-MM-XX', 'YYYY-MM-DD', 'YYYY', 'YYYY-MM'):
        return None
    
    # 修正常见错误格式
    # "2023-04-01XX" → "2023-04-XX"
    time_str = re.sub(r'(\d{4})-(\d{2})-\d{2}XX', r'\1-\2-XX', time_str)
    # "2022-04-XE" → "2022-04-XX" (拼写错误)
    time_str = re.sub(r'(\d{4})-(\d{2})-X[A-Z]', r'\1-\2-XX', time_str, flags=re.IGNORECASE)
    
    # 修正缺少秒数的时间格式："2023-05-08T14:56" → "2023-05-08T14:56:00"
    if re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$', time_str):
        return time_str + ":00"
    
    # 修正时间部分用XX的格式："2023-07-08TXX:XX:XX" → "2023-07-08"（只保留日期）
    match = re.match(r'^(\d{4}-\d{2}-\d{2})T[X:]+$', time_str, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # 如果已经是标准格式，直接返回（优化正则表达式，避免复杂嵌套）
    # 简化匹配：先检查基本格式，再检查详细格式
    if re.match(r'^\d{4}(-\d{2}(-\d{2}(T\d{2}:\d{2}:\d{2})?)?)?$', time_str):
        return time_str
    # 检查带XX的格式
    if re.match(r'^\d{4}(-\d{2}(-\d{2}|-XX)|-XX(-XX)?)$', time_str):
        return time_str
    
    # 解析自然语言格式："1:56 pm on 8 May, 2023"
    # Pattern: "HH:MM am/pm on DD Month, YYYY"
    match = re.match(r'(\d{1,2}):(\d{2})\s*(am|pm)\s+on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})', time_str, re.IGNORECASE)
    if match:
        hour, minute, ampm, day, month_name, year = match.groups()
        hour = int(hour)
        if ampm.lower() == 'pm' and hour != 12:
            hour += 12
        elif ampm.lower() == 'am' and hour == 12:
            hour = 0
        
        # 月份名称转数字
        month_map = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        month = month_map.get(month_name.lower())
        if month:
            return f"{year}-{month:02d}-{int(day):02d}T{hour:02d}:{int(minute):02d}:00"
    
    # 解析日期格式："8 May, 2023" 或 "May 8, 2023"
    match = re.match(r'(\d{1,2})\s+(\w+),?\s+(\d{4})', time_str)
    if match:
        day, month_name, year = match.groups()
        month_map = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        month = month_map.get(month_name.lower())
        if month:
            return f"{year}-{month:02d}-{int(day):02d}"
    
    # 解析 "Month YYYY" 格式："May 2023"
    match = re.match(r'(\w+)\s+(\d{4})', time_str)
    if match:
        month_name, year = match.groups()
        month_map = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        month = month_map.get(month_name.lower())
        if month:
            return f"{year}-{month:02d}-XX"
    
    # 如果无法解析，返回None
    return None

def parse_relative_time(
    text: str,
    anchor_date: Optional[str] = None,
    context_start: Optional[str] = None,
    context_end: Optional[str] = None
) -> Tuple[Optional[str], Optional[str], Optional[Granularity], float]:
    """Parse relative time expressions (e.g., "yesterday", "last week", "two months ago")
    
    Args:
        text: Time expression text
        anchor_date: ISO-8601 date string to use as anchor (e.g., question date)
        context_start: ISO-8601 start of context window
        context_end: ISO-8601 end of context window
        
    Returns:
        Tuple of (time_start, time_end, granularity, confidence)
        Returns (None, None, None, 0.0) if parsing fails
    """
    # 防止卡死：限制输入长度
    MAX_INPUT_LENGTH = 200
    if not text or len(text) > MAX_INPUT_LENGTH:
        return None, None, None, 0.0
    
    text_lower = text.lower().strip()
    
    # Determine anchor
    anchor = None
    anchor_source = "unknown"
    if anchor_date:
        try:
            anchor = datetime.fromisoformat(anchor_date.replace('Z', '+00:00'))
            anchor_source = "question_date"
        except Exception:
            pass
    
    if not anchor and context_start:
        try:
            anchor = datetime.fromisoformat(context_start.replace('Z', '+00:00'))
            anchor_source = "context_window"
        except Exception:
            pass
    
    if not anchor:
        anchor = datetime.now()
        anchor_source = "unknown"
    
    # Common relative time patterns
    patterns = [
        # Yesterday/today/tomorrow
        (r'\byesterday\b', lambda a: (a - timedelta(days=1), "day", 0.95)),
        (r'\btoday\b', lambda a: (a, "day", 0.95)),
        (r'\btomorrow\b', lambda a: (a + timedelta(days=1), "day", 0.95)),
        
        # Last/next week/month/year
        (r'\blast\s+week\b', lambda a: (a - timedelta(weeks=1), "day", 0.9)),
        (r'\bnext\s+week\b', lambda a: (a + timedelta(weeks=1), "day", 0.9)),
        (r'\blast\s+month\b', lambda a: (a.replace(day=1) - timedelta(days=1), "month", 0.85)),
        (r'\bnext\s+month\b', lambda a: (a.replace(day=28) + timedelta(days=4), "month", 0.85)),
        (r'\blast\s+year\b', lambda a: (a.replace(month=1, day=1) - timedelta(days=1), "year", 0.8)),
        (r'\bnext\s+year\b', lambda a: (a.replace(month=12, day=31) + timedelta(days=1), "year", 0.8)),
        
        # N days/weeks/months ago/later
        (r'(\d+)\s+days?\s+ago\b', lambda a, n: (a - timedelta(days=int(n)), "day", 0.9)),
        (r'(\d+)\s+days?\s+(?:later|from now)\b', lambda a, n: (a + timedelta(days=int(n)), "day", 0.9)),
        (r'(\d+)\s+weeks?\s+ago\b', lambda a, n: (a - timedelta(weeks=int(n)), "day", 0.85)),
        (r'(\d+)\s+weeks?\s+(?:later|from now)\b', lambda a, n: (a + timedelta(weeks=int(n)), "day", 0.85)),
        (r'(\d+)\s+months?\s+ago\b', lambda a, n: _add_months(a, -int(n)), "month", 0.8),
        (r'(\d+)\s+months?\s+(?:later|from now)\b', lambda a, n: _add_months(a, int(n)), "month", 0.8),
        
        # Day of week
        (r'\b(last\s+)?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b', 
         lambda a, last, day: _get_day_of_week(a, day, last is not None), "day", 0.7),
    ]
    
    for pattern, handler in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            try:
                if callable(handler):
                    if len(match.groups()) > 0 and match.groups()[0].isdigit():
                        # Pattern with number
                        result = handler(anchor, int(match.groups()[0]))
                    elif len(match.groups()) > 1:
                        # Pattern with multiple groups
                        result = handler(anchor, *match.groups())
                    else:
                        result = handler(anchor)
                    
                    if isinstance(result, tuple) and len(result) == 3:
                        dt, gran, conf = result
                    elif isinstance(result, tuple) and len(result) == 2:
                        dt, gran = result
                        conf = 0.8
                    else:
                        dt, gran, conf = result, "day", 0.8
                    
                    time_start = dt.isoformat()
                    time_end = time_start  # Single point for relative times
                    return time_start, time_end, gran, conf
            except Exception:
                continue
    
    return None, None, None, 0.0

def parse_absolute_time(text: str) -> Tuple[Optional[str], Optional[str], Optional[Granularity], float]:
    """Parse absolute time expressions to ISO-8601 format
    
    Args:
        text: Time expression text (e.g., "7 May 2023", "2023-05-07", "May 7, 2023")
        
    Returns:
        Tuple of (time_start, time_end, granularity, confidence)
    """
    # 防止卡死：限制输入长度
    MAX_INPUT_LENGTH = 200
    if not text or len(text) > MAX_INPUT_LENGTH:
        return None, None, None, 0.0
    
    text = text.strip()
    
    # ISO-8601 format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
    iso_pattern = r'(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?)?'
    match = re.match(iso_pattern, text)
    if match:
        year, month, day = match.groups()[:3]
        hour, minute, second = match.groups()[3:6]
        try:
            if hour and minute:
                dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second) if second else 0)
                return dt.isoformat(), dt.isoformat(), "time", 0.95
            else:
                dt = datetime(int(year), int(month), int(day))
                return dt.isoformat(), dt.isoformat(), "day", 0.95
        except ValueError:
            pass
    
    # "7 May 2023" or "May 7, 2023" format
    month_names = {
        'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    # Pattern: "7 May 2023" or "May 7, 2023"
    patterns = [
        r'(\d{1,2})\s+([a-z]+)\s+(\d{4})',  # "7 May 2023"
        r'([a-z]+)\s+(\d{1,2}),?\s+(\d{4})',  # "May 7, 2023" or "May 7 2023"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            groups = match.groups()
            if len(groups) == 3:
                try:
                    if groups[1].isdigit():
                        # First pattern: day month year
                        day, month_str, year = int(groups[0]), groups[1], int(groups[2])
                    else:
                        # Second pattern: month day year
                        month_str, day, year = groups[0], int(groups[1]), int(groups[2])
                    
                    month = month_names.get(month_str.lower())
                    if month:
                        dt = datetime(year, month, day)
                        return dt.isoformat(), dt.isoformat(), "day", 0.9
                except (ValueError, KeyError):
                    pass
    
    # Year only: "2023"
    year_pattern = r'\b(\d{4})\b'
    match = re.search(year_pattern, text)
    if match:
        try:
            year = int(match.group(1))
            if 1900 <= year <= 2100:  # Reasonable range
                dt = datetime(year, 1, 1)
                return dt.isoformat(), datetime(year, 12, 31).isoformat(), "year", 0.7
        except ValueError:
            pass
    
    # Month and year: "May 2023"
    month_year_pattern = r'([a-z]+)\s+(\d{4})'
    match = re.search(month_year_pattern, text.lower())
    if match:
        month_str, year_str = match.groups()
        month = month_names.get(month_str.lower())
        if month:
            try:
                year = int(year_str)
                dt = datetime(year, month, 1)
                # Last day of month
                if month == 12:
                    last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
                else:
                    last_day = datetime(year, month + 1, 1) - timedelta(days=1)
                return dt.isoformat(), last_day.isoformat(), "month", 0.8
            except ValueError:
                pass
    
    return None, None, None, 0.0

def normalize_time(
    text: str,
    anchor_date: Optional[str] = None,
    context_start: Optional[str] = None,
    context_end: Optional[str] = None
) -> Dict[str, any]:
    """Normalize time expression to ISO-8601 format with metadata
    
    Args:
        text: Time expression text
        anchor_date: ISO-8601 date string (e.g., question date)
        context_start: ISO-8601 start of context window
        context_end: ISO-8601 end of context window
        
    Returns:
        Dict with keys: time_start, time_end, granularity, confidence, anchor, original_text
    """
    original_text = text.strip()
    
    # Try absolute time first
    time_start, time_end, granularity, confidence = parse_absolute_time(text)
    anchor = "absolute"
    
    # If absolute fails, try relative
    if time_start is None:
        time_start, time_end, granularity, confidence = parse_relative_time(
            text, anchor_date, context_start, context_end
        )
        anchor = "question_date" if anchor_date else ("context_window" if context_start else "unknown")
    
    return {
        "original_text": original_text,
        "time_start": time_start,
        "time_end": time_end,
        "granularity": granularity,
        "confidence": confidence,
        "anchor": anchor
    }

def _add_months(dt: datetime, months: int) -> Tuple[datetime, Granularity, float]:
    """Add months to a datetime"""
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, [31, 29 if year % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1])
    return datetime(year, month, day), "month", 0.8

def _get_day_of_week(anchor: datetime, day_name: str, last: bool) -> Tuple[datetime, Granularity, float]:
    """Get the most recent or next occurrence of a day of week"""
    days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    target_day = days.index(day_name.lower())
    current_day = anchor.weekday()
    
    if last:
        days_ago = (current_day - target_day) % 7
        if days_ago == 0:
            days_ago = 7  # Last week's occurrence
        result = anchor - timedelta(days=days_ago)
    else:
        days_ahead = (target_day - current_day) % 7
        if days_ahead == 0:
            days_ahead = 7  # Next week's occurrence
        result = anchor + timedelta(days=days_ahead)
    
    return result, "day", 0.7
