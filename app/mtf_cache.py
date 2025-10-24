#!/usr/bin/env python3
"""
Shared Multi-Timeframe Cache
Used by both bullish and bearish pro traders to avoid duplicate data fetches
"""

import pytz
from typing import Optional
from datetime import datetime, timedelta


class MultiTimeframeCache:
    """
    Caches multi-timeframe data with market-aligned expiry times
    D1: Updates at 5pm NY (4pm CT, 9pm UTC) when daily candle closes
    H4: Updates every 4 hours on the 4H candle close
    H1: Updates every hour on the hour
    """

    def __init__(self):
        self.cache = {
            "D1": {"data": None, "last_update": None},
            "H4": {"data": None, "last_update": None},
            "H1": {"data": None, "last_update": None}
        }

    def is_expired(self, timeframe: str) -> bool:
        """
        Check if cached data has expired based on market times
        D1: Expires after 5pm NY (daily close)
        H4: Expires every 4 hours
        H1: Expires every hour
        """
        cache_entry = self.cache.get(timeframe)
        if not cache_entry or cache_entry["data"] is None or cache_entry["last_update"] is None:
            return True

        now = datetime.now(pytz.UTC)
        last_update = cache_entry["last_update"]

        if timeframe == "D1":
            # Daily candle closes at 5pm NY = 9pm UTC (or 10pm UTC during winter)
            # Check if we've passed the daily close time since last update
            ny_tz = pytz.timezone('America/New_York')
            now_ny = now.astimezone(ny_tz)
            last_update_ny = last_update.astimezone(ny_tz)

            # Daily close is at 5pm NY
            daily_close_hour = 17

            # If last update was before today's 5pm close and now is after, expired
            today_close = now_ny.replace(hour=daily_close_hour, minute=0, second=0, microsecond=0)
            yesterday_close = today_close - timedelta(days=1)

            # If last update was before the most recent close, it's expired
            if last_update_ny < today_close <= now_ny:
                return True
            elif last_update_ny < yesterday_close:
                return True

            return False

        elif timeframe == "H4":
            # H4 expires at actual 4-hour candle boundaries
            # H4 candles close at: 5pm, 9pm, 1am, 5am, 9am, 1pm NY (4pm, 8pm, 12am, 4am, 8am, 12pm CT)
            ny_tz = pytz.timezone('America/New_York')
            now_ny = now.astimezone(ny_tz)
            last_update_ny = last_update.astimezone(ny_tz)

            # H4 candles close at hours: 1, 5, 9, 13, 17, 21 (NY time)
            h4_close_hours = [1, 5, 9, 13, 17, 21]

            # Find the most recent H4 close
            current_hour = now_ny.hour
            most_recent_close_hour = max([h for h in h4_close_hours if h <= current_hour], default=21)

            # If current hour is before the first close of the day, use yesterday's last close
            if current_hour < h4_close_hours[0]:
                most_recent_close = now_ny.replace(hour=h4_close_hours[-1], minute=0, second=0, microsecond=0) - timedelta(days=1)
            else:
                most_recent_close = now_ny.replace(hour=most_recent_close_hour, minute=0, second=0, microsecond=0)

            # If last update was before the most recent close, it's expired
            if last_update_ny < most_recent_close:
                return True

            return False

        elif timeframe == "H1":
            # H1 expires at actual hour boundaries (top of each hour)
            # If we're at 7:45pm and last update was at 6:30pm, we've passed 7:00pm close
            last_update_hour_boundary = last_update.replace(minute=0, second=0, microsecond=0)
            current_hour_boundary = now.replace(minute=0, second=0, microsecond=0)

            # If we've moved to a new hour, the cache is expired
            if current_hour_boundary > last_update_hour_boundary:
                return True

            return False

        return True

    def get(self, timeframe: str):
        """Get cached data if not expired"""
        if self.is_expired(timeframe):
            return None
        return self.cache[timeframe]["data"]

    def set(self, timeframe: str, data):
        """Store data in cache"""
        self.cache[timeframe]["data"] = data
        self.cache[timeframe]["last_update"] = datetime.now(pytz.UTC)

    def get_last_update(self, timeframe: str) -> Optional[datetime]:
        """Get last update time for timeframe"""
        return self.cache[timeframe].get("last_update")

    def get_next_update(self, timeframe: str) -> str:
        """
        Calculate when next update will occur based on market times
        D1: Next 5pm NY daily close
        H4: Next 4-hour boundary
        H1: Next hour boundary
        """
        now = datetime.now(pytz.UTC)
        chicago_tz = pytz.timezone('America/Chicago')
        ny_tz = pytz.timezone('America/New_York')

        if timeframe == "D1":
            # Next daily close is 5pm NY (4pm CT)
            now_ny = now.astimezone(ny_tz)
            today_close = now_ny.replace(hour=17, minute=0, second=0, microsecond=0)

            if now_ny >= today_close:
                # Already passed today's close, next is tomorrow
                next_close = today_close + timedelta(days=1)
            else:
                # Haven't reached today's close yet
                next_close = today_close

            # Convert to Chicago for display
            next_close_ct = next_close.astimezone(chicago_tz)
            time_until = next_close.astimezone(pytz.UTC) - now
            hours = int(time_until.total_seconds() / 3600)
            minutes = int((time_until.total_seconds() % 3600) / 60)

            if hours > 0:
                return f"Next update: {next_close_ct.strftime('%I:%M %p CT')} (in {hours}h {minutes}m)"
            else:
                return f"Next update: {next_close_ct.strftime('%I:%M %p CT')} (in {minutes}m)"

        elif timeframe == "H4":
            # Next H4 candle close at actual market boundaries
            # H4 closes at: 1am, 5am, 9am, 1pm, 5pm, 9pm NY
            now_ny = now.astimezone(ny_tz)
            h4_close_hours = [1, 5, 9, 13, 17, 21]

            # Find next H4 close
            current_hour = now_ny.hour
            next_close_hour = None

            for h in h4_close_hours:
                if h > current_hour:
                    next_close_hour = h
                    break

            if next_close_hour is None:
                # Next close is tomorrow's first close
                next_close = now_ny.replace(hour=h4_close_hours[0], minute=0, second=0, microsecond=0) + timedelta(days=1)
            else:
                next_close = now_ny.replace(hour=next_close_hour, minute=0, second=0, microsecond=0)

            # Convert to Chicago for display
            next_close_ct = next_close.astimezone(chicago_tz)
            time_until = next_close.astimezone(pytz.UTC) - now
            hours = int(time_until.total_seconds() / 3600)
            minutes = int((time_until.total_seconds() % 3600) / 60)

            if hours > 0:
                return f"Next update: {next_close_ct.strftime('%I:%M %p CT')} (in {hours}h {minutes}m)"
            else:
                return f"Next update: {next_close_ct.strftime('%I:%M %p CT')} (in {minutes}m)"

        elif timeframe == "H1":
            # Next H1 candle close at top of next hour
            now_ct = now.astimezone(chicago_tz)

            # Next hour boundary
            if now_ct.minute == 0 and now_ct.second == 0:
                # Exactly on the hour
                next_hour = now_ct + timedelta(hours=1)
            else:
                # Find next hour
                next_hour = now_ct.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

            time_until = next_hour.astimezone(pytz.UTC) - now
            minutes = int(time_until.total_seconds() / 60)

            return f"Next update: {next_hour.strftime('%I:%M %p CT')} (in {minutes}m)"

        return "Unknown"


# Global shared MTF cache instance
# Both bullish and bearish traders use this same instance
mtf_cache = MultiTimeframeCache()
