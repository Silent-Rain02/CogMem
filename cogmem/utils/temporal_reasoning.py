"""相对时间推理模块：将相对时间表达式转换为绝对日期"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional


class TemporalReasoner:
    """相对时间推理器：处理相对时间表达式"""
    
    # 星期名称映射
    WEEKDAY_MAP = {
        'monday': 0, 'mon': 0,
        'tuesday': 1, 'tue': 1, 'tues': 1,
        'wednesday': 2, 'wed': 2,
        'thursday': 3, 'thu': 3, 'thur': 3, 'thurs': 3,
        'friday': 4, 'fri': 4,
        'saturday': 5, 'sat': 5,
        'sunday': 6, 'sun': 6,
    }
    
    # 月份名称映射
    MONTH_MAP = {
        'january': 1, 'jan': 1,
        'february': 2, 'feb': 2,
        'march': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'may': 5,
        'june': 6, 'jun': 6,
        'july': 7, 'jul': 7,
        'august': 8, 'aug': 8,
        'september': 9, 'sep': 9, 'sept': 9,
        'october': 10, 'oct': 10,
        'november': 11, 'nov': 11,
        'december': 12, 'dec': 12,
    }
    
    def resolve_relative_time(self, expression: str, base_date: datetime) -> Optional[str]:
        """将相对时间表达式转换为绝对日期
        
        Args:
            expression: 相对时间表达式（如"last Friday", "two weeks ago", "two weekends ago", "the week before 25 May 2023"）
            base_date: 基准日期（Session Date）
            
        Returns:
            绝对日期字符串（如"21 May 2023"），如果无法解析则返回None
        """
        if not expression or not base_date:
            return None
        
        expression = expression.lower().strip()
        
        # 1. 处理"the week before X"、"the Friday before X"（更灵活的匹配）
        # 支持格式："the week before 25 May 2023", "the Friday before 15 July, 2023"
        week_before_match = re.search(
            r'the\s+(?:week|weekend|(\w+day))\s+before\s+(\d{1,2})\s+(\w+),?\s+(\d{4})',
            expression
        )
        if week_before_match:
            weekday_name = week_before_match.group(1)
            day = int(week_before_match.group(2))
            month_name = week_before_match.group(3)
            year = int(week_before_match.group(4))
            
            # 解析目标日期
            month = self.MONTH_MAP.get(month_name.lower())
            if not month:
                return None
            
            try:
                target_date = datetime(year, month, day)
            except ValueError:
                return None
            
            if weekday_name:  # 特定星期几（如"the Friday before"）
                weekday = self.WEEKDAY_MAP.get(weekday_name.lower())
                if weekday is not None:
                    resolved = self._find_previous_weekday(target_date, weekday)
                    return self._format_date(resolved)
            elif 'weekend' in expression:  # "the weekend before"
                # 周末是周六，往前找最近的周六
                saturday = self._find_previous_weekday(target_date, 5)
                return self._format_date(saturday)
            else:  # 整周前
                week_start = target_date - timedelta(days=7)
                return self._format_date(week_start)
        
        # 2. 处理"the weekend before X"（独立匹配，更宽松）
        weekend_before_match = re.search(
            r'the\s+weekend\s+before\s+(\d{1,2})\s+(\w+),?\s+(\d{4})',
            expression
        )
        if weekend_before_match:
            day = int(weekend_before_match.group(1))
            month_name = weekend_before_match.group(2)
            year = int(weekend_before_match.group(3))
            
            month = self.MONTH_MAP.get(month_name.lower())
            if not month:
                return None
            
            try:
                target_date = datetime(year, month, day)
            except ValueError:
                return None
            
            # 周末是周六，往前找最近的周六
            saturday = self._find_previous_weekday(target_date, 5)
            return self._format_date(saturday)
        
        # 3. 处理"two weekends ago"、"three weekends ago"等
        # 周末通常指周六，所以"two weekends ago" = 找到最近的周六，再往前推一个周末（7天）
        weekends_ago_match = re.search(r'(\d+)\s+weekends?\s+ago', expression)
        if weekends_ago_match:
            amount = int(weekends_ago_match.group(1))
            # 找到最近的周六（周末的开始）
            current_weekday = base_date.weekday()
            # 计算到上一个周六的天数
            if current_weekday == 5:  # 如果今天就是周六
                # 往前推一周（7天）到上一个周六
                days_to_last_saturday = 7
            elif current_weekday == 6:  # 如果今天是周日
                # 往前推1天到周六
                days_to_last_saturday = 1
            else:  # 周一到周五
                # 计算到上一个周六：如果今天是周一(0)，到上周六是2天；如果是周五(4)，到上周六是6天
                days_to_last_saturday = (current_weekday - 5 + 7) % 7
                if days_to_last_saturday == 0:
                    days_to_last_saturday = 7
            
            last_saturday = base_date - timedelta(days=days_to_last_saturday)
            # 往前推 (amount - 1) 个周末（每个周末7天）
            # 例如："two weekends ago" = last_saturday - 1 week
            target_saturday = last_saturday - timedelta(weeks=amount - 1)
            return self._format_date(target_saturday)
        
        # 4. 处理"last Friday"、"last week"、"last weekend"
        if expression.startswith('last '):
            period = expression.replace('last ', '').strip()
            if period == 'week':
                result = base_date - timedelta(weeks=1)
                return self._format_date(result)
            elif period == 'weekend':
                # 找到最近的周六
                current_weekday = base_date.weekday()
                if current_weekday == 5:  # 如果今天就是周六
                    days_to_last_saturday = 7
                elif current_weekday == 6:  # 如果今天是周日
                    days_to_last_saturday = 1
                else:  # 周一到周五
                    days_to_last_saturday = (current_weekday - 5 + 7) % 7
                    if days_to_last_saturday == 0:
                        days_to_last_saturday = 7
                last_saturday = base_date - timedelta(days=days_to_last_saturday)
                return self._format_date(last_saturday)
            else:
                # 处理"last Friday"、"last Mon"等
                # 移除可能的"day"后缀
                weekday_name = period.replace('day', '').strip()
                if weekday_name.endswith('day'):
                    weekday_name = weekday_name[:-3]
                weekday = self.WEEKDAY_MAP.get(weekday_name.lower())
                if weekday is not None:
                    resolved = self._find_previous_weekday(base_date, weekday)
                    return self._format_date(resolved)
        
        # 5. 处理"two weeks ago"、"three days ago"、"two months ago"等
        ago_match = re.search(r'(\d+)\s+(week|day|month|year)s?\s+ago', expression)
        if ago_match:
            amount = int(ago_match.group(1))
            unit = ago_match.group(2)
            
            if unit == 'week':
                result = base_date - timedelta(weeks=amount)
            elif unit == 'day':
                result = base_date - timedelta(days=amount)
            elif unit == 'month':
                # 更准确的月份计算
                result = self._subtract_months(base_date, amount)
            elif unit == 'year':
                # 更准确的年份计算（考虑闰年）
                try:
                    result = base_date.replace(year=base_date.year - amount)
                except ValueError:
                    # 处理2月29日的情况
                    result = base_date.replace(year=base_date.year - amount, day=28)
            else:
                return None
            
            return self._format_date(result)
        
        # 6. 处理"yesterday"、"today"、"tomorrow"
        if expression == 'yesterday':
            result = base_date - timedelta(days=1)
            return self._format_date(result)
        elif expression == 'today':
            return self._format_date(base_date)
        elif expression == 'tomorrow':
            result = base_date + timedelta(days=1)
            return self._format_date(result)
        
        # 7. 处理"next [weekday]"、"this [weekday]"
        next_match = re.search(r'next\s+(\w+day)', expression)
        if next_match:
            weekday_name = next_match.group(1).replace('day', '').strip()
            weekday = self.WEEKDAY_MAP.get(weekday_name.lower())
            if weekday is not None:
                resolved = self._find_next_weekday(base_date, weekday)
                return self._format_date(resolved)
        
        this_match = re.search(r'this\s+(\w+day)', expression)
        if this_match:
            weekday_name = this_match.group(1).replace('day', '').strip()
            weekday = self.WEEKDAY_MAP.get(weekday_name.lower())
            if weekday is not None:
                # "this Friday" 可能是本周五或下周五，取决于当前日期
                current_weekday = base_date.weekday()
                days_diff = (weekday - current_weekday) % 7
                if days_diff == 0:
                    # 如果今天就是目标星期，返回今天
                    resolved = base_date
                else:
                    resolved = base_date + timedelta(days=days_diff)
                return self._format_date(resolved)
        
        return None
    
    def _find_previous_weekday(self, date: datetime, weekday: int) -> datetime:
        """找到指定日期之前的最近一个指定星期几
        
        Args:
            date: 参考日期
            weekday: 目标星期几（0=Monday, 6=Sunday）
            
        Returns:
            最近的指定星期几的日期
        """
        current_weekday = date.weekday()
        days_diff = (current_weekday - weekday) % 7
        
        if days_diff == 0:
            # 如果当天就是目标星期，往前推一周
            days_diff = 7
        
        return date - timedelta(days=days_diff)
    
    def _find_next_weekday(self, date: datetime, weekday: int) -> datetime:
        """找到指定日期之后的最近一个指定星期几
        
        Args:
            date: 参考日期
            weekday: 目标星期几（0=Monday, 6=Sunday）
            
        Returns:
            最近的指定星期几的日期
        """
        current_weekday = date.weekday()
        days_diff = (weekday - current_weekday) % 7
        
        if days_diff == 0:
            # 如果当天就是目标星期，往后推一周
            days_diff = 7
        
        return date + timedelta(days=days_diff)
    
    def _subtract_months(self, date: datetime, months: int) -> datetime:
        """从日期中减去指定月数（考虑月份天数差异）
        
        Args:
            date: 基准日期
            months: 要减去的月数
            
        Returns:
            减去月份后的日期
        """
        month = date.month - months
        year = date.year
        
        while month <= 0:
            month += 12
            year -= 1
        
        # 处理月份天数差异（如1月31日减去1个月应该是2月28/29日）
        try:
            return date.replace(year=year, month=month)
        except ValueError:
            # 如果目标月份没有该日期（如2月30日），则取该月最后一天
            if month == 2:
                # 检查是否为闰年
                if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
                    last_day = 29
                else:
                    last_day = 28
            elif month in [4, 6, 9, 11]:
                last_day = 30
            else:
                last_day = 31
            
            return date.replace(year=year, month=month, day=last_day)
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """解析日期字符串（支持多种格式）"""
        # 尝试解析 "7 May 2023" 格式
        match = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})', date_str)
        if match:
            day = int(match.group(1))
            month_name = match.group(2)
            year = int(match.group(3))
            
            month = self.MONTH_MAP.get(month_name.lower())
            if month:
                try:
                    return datetime(year, month, day)
                except ValueError:
                    pass
        
        return None
    
    def _parse_session_date(self, session_date_str: str) -> Optional[datetime]:
        """解析Session Date格式（如"1:56 pm on 8 May, 2023"）"""
        # 匹配格式：时间 on 日期
        match = re.search(r'on\s+(\d{1,2})\s+(\w+),?\s+(\d{4})', session_date_str, re.IGNORECASE)
        if match:
            day = int(match.group(1))
            month_name = match.group(2)
            year = int(match.group(3))
            
            month = self.MONTH_MAP.get(month_name.lower())
            if month:
                try:
                    return datetime(year, month, day)
                except ValueError:
                    pass
        
        # 尝试其他格式
        return self._parse_date(session_date_str)
    
    def _format_date(self, date: datetime) -> str:
        """格式化日期为标准格式（如"21 May 2023"）"""
        month_names = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        return f"{date.day} {month_names[date.month - 1]} {date.year}"

