#!/usr/bin/env python3
"""
Professional Trader Gold System - Educational Setup Tracker
Shows EXACTLY what professional day traders look for, step-by-step
Like having a mentor explaining every detail in real-time
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pytz
from .datafeed import fetch_h1
from .current_price import get_current_xauusd_price

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

# Global MTF cache
mtf_cache = MultiTimeframeCache()

class BullishProTraderGold:
    """
    BULLISH Educational trading system - looks for BUY setups
    Detects breakouts above resistance and bullish rejections
    Professional day trading logic for LONG positions
    """

    def __init__(self):
        self.pair = "XAUUSD"
        self.timeframes = {
            "D1": 200,  # 200 days
            "H4": 500,  # ~83 days of 4H candles (approx 83 days * 6 candles/day)
            "H1": 100   # 100 hours
        }

    async def get_detailed_setup(self) -> Dict[str, Any]:
        """
        Main function: Returns complete setup analysis with educational breakdown
        """
        try:
            # Fetch multi-timeframe data with caching
            d1_data = await self._get_timeframe_data("D1")
            h4_data = await self._get_timeframe_data("H4")
            h1_data = await self._get_timeframe_data("H1")

            if h1_data is None or h1_data.empty:
                return self._error_response("No H1 data available")

            # Get current live price
            current_price = await get_current_xauusd_price()
            if current_price is None:
                current_price = float(h1_data['close'].iloc[-1])

            # Analyze market structure across timeframes (using REAL data)
            daily_analysis = self._analyze_daily_trend(d1_data, current_price)
            h4_levels = self._identify_key_levels_h4(h4_data, current_price)
            h1_setup = self._detect_setup_pattern(h1_data, h4_levels, current_price)

            # Get current candle details
            current_candle = self._get_current_candle_info(h1_data, current_price)

            # Build complete response
            return {
                "status": "success",
                "pair": self.pair,
                "current_price": current_price,
                "setup_status": h1_setup["state"],
                "setup_progress": h1_setup["progress"],
                "pattern_type": h1_setup["pattern_type"],

                # Step-by-step breakdown
                "setup_steps": self._build_setup_steps(h1_setup, current_price, current_candle),

                # Context explanation
                "why_this_setup": {
                    "daily": daily_analysis,
                    "h4": self._explain_h4_structure(h4_levels, current_price),
                    "h1": self._explain_h1_pattern(h1_setup),
                    "session": self._explain_session_context()
                },

                # Live candle tracking
                "live_candle": current_candle,

                # Trade plan (when ready)
                "trade_plan": self._build_trade_plan(h1_setup, h4_levels, current_price),

                # Invalidation conditions
                "invalidation": self._get_invalidation_conditions(h1_setup),

                # Chart data for visualization
                "chart_data": {
                    "h1_candles": self._convert_candles_to_json(h1_data.tail(24)),
                    "key_levels": h4_levels,
                    "current_price": current_price
                },

                "last_update": datetime.now(pytz.UTC).isoformat()
            }

        except Exception as e:
            return self._error_response(f"Error: {str(e)}")

    async def _get_timeframe_data(self, timeframe: str) -> pd.DataFrame:
        """
        Fetch timeframe data with caching
        D1 updates once per day, H4 every 4 hours, H1 every hour
        """
        # Check cache first
        cached_data = mtf_cache.get(timeframe)
        if cached_data is not None:
            return cached_data

        # Fetch fresh data
        data = await fetch_h1(self.pair, timeframe=timeframe)

        # Store in cache
        mtf_cache.set(timeframe, data)

        return data

    def _analyze_daily_trend(self, d1_data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Analyze REAL daily timeframe trend using D1 candles
        Updates once per day when daily candle closes
        """
        if d1_data is None or d1_data.empty:
            return {
                "trend": "UNKNOWN",
                "explanation": "No daily data available",
                "points": ["Unable to fetch daily data"],
                "last_updated": "Never",
                "next_update": "Waiting for data"
            }

        # Calculate 200-day EMA on ACTUAL daily candles
        d1_data['ema_200'] = d1_data['close'].ewm(span=200, adjust=False).mean()

        # Get today's EMA value
        ema_200_current = float(d1_data['ema_200'].iloc[-1])

        # Determine trend
        trend = "BULLISH" if current_price > ema_200_current else "BEARISH"

        # Find recent swing high/low (last 20 days)
        recent_data = d1_data.tail(20)
        recent_high = float(recent_data['high'].max())
        recent_low = float(recent_data['low'].min())

        # Find when these highs/lows occurred
        high_idx = recent_data['high'].idxmax()
        low_idx = recent_data['low'].idxmin()
        days_since_high = len(recent_data) - recent_data.index.get_loc(high_idx) - 1
        days_since_low = len(recent_data) - recent_data.index.get_loc(low_idx) - 1

        # Determine structure
        last_5_highs = d1_data['high'].tail(5)
        last_5_lows = d1_data['low'].tail(5)
        making_higher_highs = last_5_highs.iloc[-1] > last_5_highs.iloc[-3]
        making_lower_lows = last_5_lows.iloc[-1] < last_5_lows.iloc[-3]

        if making_higher_highs:
            structure = "Making higher highs (bullish structure)"
        elif making_lower_lows:
            structure = "Making lower lows (bearish structure)"
        else:
            structure = "Consolidating (no clear direction)"

        # Get update times (convert to Chicago timezone for display)
        chicago_tz = pytz.timezone('America/Chicago')
        last_updated = mtf_cache.get_last_update("D1")
        next_update = mtf_cache.get_next_update("D1")

        # Convert last_updated to Chicago time for display
        last_updated_str = "Just now"
        if last_updated:
            last_updated_ct = last_updated.astimezone(chicago_tz)
            last_updated_str = last_updated_ct.strftime("%b %d, %I:%M %p CT")

        return {
            "trend": trend,
            "explanation": f"Price is {'above' if trend == 'BULLISH' else 'below'} 200-day EMA (${ema_200_current:.2f})",
            "points": [
                f"Trend: {trend} (current: ${current_price:.2f} vs 200 EMA: ${ema_200_current:.2f})",
                f"Recent high: ${recent_high:.2f} ({days_since_high} days ago)",
                f"Recent low: ${recent_low:.2f} ({days_since_low} days ago)",
                f"Structure: {structure}"
            ],
            "last_updated": last_updated_str,
            "next_update": next_update
        }

    def _identify_key_levels_h4(self, h4_data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Identify key support/resistance levels using REAL 4H candles
        Updates every 4 hours when 4H candle closes
        """
        if h4_data is None or h4_data.empty:
            return {
                "key_level": current_price,
                "level_type": "unknown",
                "last_updated": "Never",
                "next_update": "Waiting for data"
            }

        # Use last 100 H4 candles to find key levels (approx 17 days)
        recent_data = h4_data.tail(100)
        highs = recent_data['high']
        lows = recent_data['low']

        # Find significant resistance/support levels
        resistance_candidates = []
        support_candidates = []

        # Find local highs/lows using 4H candles
        for i in range(5, len(highs) - 5):
            # Resistance: local high
            if highs.iloc[i] == highs.iloc[i-5:i+5].max():
                resistance_candidates.append(float(highs.iloc[i]))
            # Support: local low
            if lows.iloc[i] == lows.iloc[i-5:i+5].min():
                support_candidates.append(float(lows.iloc[i]))

        # Get most relevant levels near current price
        resistance = sorted([r for r in resistance_candidates if r > current_price])[:3] if resistance_candidates else []
        support = sorted([s for s in support_candidates if s < current_price], reverse=True)[:3] if support_candidates else []

        # Identify THE key level (closest to price)
        if resistance and support:
            nearest_resistance = resistance[0]
            nearest_support = support[0]
            if abs(current_price - nearest_resistance) < abs(current_price - nearest_support):
                key_level = nearest_resistance
                level_type = "resistance"
            else:
                key_level = nearest_support
                level_type = "support"
        elif resistance:
            key_level = resistance[0]
            level_type = "resistance"
        elif support:
            key_level = support[0]
            level_type = "support"
        else:
            key_level = float(recent_data['close'].iloc[-1])
            level_type = "pivot"

        # Calculate distance to key level
        distance_pips = abs(current_price - key_level)

        # Get update times (convert to Chicago timezone for display)
        chicago_tz = pytz.timezone('America/Chicago')
        last_updated = mtf_cache.get_last_update("H4")
        next_update = mtf_cache.get_next_update("H4")

        # Convert last_updated to Chicago time for display
        last_updated_str = "Just now"
        if last_updated:
            last_updated_ct = last_updated.astimezone(chicago_tz)
            last_updated_str = last_updated_ct.strftime("%b %d, %I:%M %p CT")

        return {
            "key_level": key_level,
            "level_type": level_type,
            "resistance_levels": resistance,
            "support_levels": support,
            "distance_pips": round(distance_pips, 1),
            "last_updated": last_updated_str,
            "next_update": next_update
        }

    def _detect_setup_pattern(self, h1_data: pd.DataFrame, h4_levels: Dict, current_price: float) -> Dict[str, Any]:
        """
        Detect which professional setup pattern is forming
        Returns detailed pattern information
        """
        key_level = h4_levels["key_level"]
        last_candles = h1_data.tail(10)

        # Check for BREAKOUT RETEST pattern
        breakout_setup = self._check_breakout_retest(last_candles, key_level, current_price)
        if breakout_setup["detected"]:
            return breakout_setup

        # Check for DEMAND ZONE pattern
        demand_setup = self._check_demand_zone(last_candles, h4_levels, current_price)
        if demand_setup["detected"]:
            return demand_setup

        # Default: SCANNING
        return {
            "detected": True,
            "pattern_type": "SCANNING",
            "state": "SCANNING",
            "progress": "0/5",
            "key_level": key_level,
            "confirmations": 0,
            "description": "Scanning for professional setups..."
        }

    def _check_breakout_retest(self, candles: pd.DataFrame, key_level: float, current_price: float) -> Dict[str, Any]:
        """
        Check for Breakout Retest pattern:
        1. Price was below resistance
        2. Price broke above resistance (breakout)
        3. Price pulled back to test old resistance (now support)
        4. Looking for rejection + confirmation
        """
        if len(candles) < 5:
            return {"detected": False}

        # Check if we had a breakout in recent candles
        breakout_candle_idx = None
        for i in range(len(candles) - 5, len(candles)):
            candle = candles.iloc[i]
            prev_candle = candles.iloc[i-1] if i > 0 else None

            # Breakout = close moved from below to above key level
            if prev_candle is not None:
                if prev_candle['close'] < key_level and candle['close'] > key_level:
                    breakout_candle_idx = i
                    break

        if breakout_candle_idx is None:
            return {"detected": False}

        # We have a breakout! Now check for retest
        breakout_candle = candles.iloc[breakout_candle_idx]
        candles_after_breakout = candles.iloc[breakout_candle_idx+1:]

        # INVALIDATION CHECK: If any candle after breakout closed significantly below support, setup is dead
        if len(candles_after_breakout) > 0:
            for i, candle in candles_after_breakout.iterrows():
                # If candle closed below support (not just wick, but actual close), setup is invalidated
                if candle['close'] < key_level - 5:  # 5 pips buffer
                    return {"detected": False}  # Setup invalidated

        # Check if price has come back to test the level
        retest_happening = False
        rejection_confirmed = False

        if len(candles_after_breakout) > 0:
            # Retest = price returned close to key level
            for i, candle in candles_after_breakout.iterrows():
                if abs(candle['low'] - key_level) < 10:  # Within 10 pips
                    retest_happening = True

                    # Check for rejection (long lower wick + close above)
                    wick_size = candle['low'] - min(candle['open'], candle['close'])
                    if wick_size > 3 and candle['close'] > key_level + 5:
                        rejection_confirmed = True

        # Determine state
        if rejection_confirmed:
            state = "CONFIRMATION_WAITING"
            progress = "4/5"
            confirmations = 2
        elif retest_happening:
            state = "REJECTION_WAITING"
            progress = "3/5"
            confirmations = 1
        else:
            state = "RETEST_WAITING"
            progress = "2/5"
            confirmations = 1

        return {
            "detected": True,
            "pattern_type": "BREAKOUT_RETEST",
            "direction": "LONG",
            "state": state,
            "progress": progress,
            "key_level": key_level,
            "confirmations": confirmations,
            "breakout_candle": {
                "time": breakout_candle.name.strftime("%I:%M %p") if hasattr(breakout_candle.name, 'strftime') else "Recently",
                "price": float(breakout_candle['close'])
            },
            "retest_candle": {
                "time": "Now" if retest_happening else "Waiting",
                "price": current_price
            } if retest_happening else None,
            "rejection_candle_low": float(candles.tail(1).iloc[0]['low']) if rejection_confirmed else None,
            "expected_entry": current_price
        }

    def _check_demand_zone(self, candles: pd.DataFrame, h4_levels: Dict, current_price: float) -> Dict[str, Any]:
        """
        Check for Demand Zone pattern
        (Simplified for now)
        """
        support_levels = h4_levels.get("support_levels", [])

        if not support_levels:
            return {"detected": False}

        # Check if price is near a support level
        nearest_support = min(support_levels, key=lambda x: abs(x - current_price))

        if abs(current_price - nearest_support) < 10:
            return {
                "detected": True,
                "pattern_type": "DEMAND_ZONE",
                "state": "WATCHING",
                "progress": "2/5",
                "key_level": nearest_support,
                "confirmations": 0,
                "expected_entry": current_price
            }

        return {"detected": False}

    def _get_current_candle_info(self, h1_data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Get details about the currently forming 1H candle
        """
        last_closed_candle = h1_data.iloc[-1]

        # Calculate time remaining in current candle (UTC)
        now_utc = datetime.now(pytz.UTC)
        current_hour_start_utc = now_utc.replace(minute=0, second=0, microsecond=0)
        next_hour_start_utc = current_hour_start_utc + timedelta(hours=1)
        minutes_remaining = int((next_hour_start_utc - now_utc).total_seconds() / 60)

        # Convert to Chicago time for display
        chicago_tz = pytz.timezone('America/Chicago')
        current_hour_start_ct = current_hour_start_utc.astimezone(chicago_tz)
        next_hour_start_ct = next_hour_start_utc.astimezone(chicago_tz)

        # Current candle estimation
        current_candle_open = float(last_closed_candle['close'])
        current_high = max(current_price, current_candle_open)
        current_low = min(current_price, current_candle_open)

        return {
            "timeframe": "1H",
            "open": current_candle_open,
            "high": current_high,
            "low": current_low,
            "current": current_price,
            "time_remaining": minutes_remaining,
            "candle_start": current_hour_start_ct.strftime("%I:%M %p CT"),
            "candle_close_expected": next_hour_start_ct.strftime("%I:%M %p CT")
        }

    def _build_setup_steps(self, setup: Dict, current_price: float, current_candle: Dict) -> List[Dict[str, Any]]:
        """
        Build step-by-step breakdown based on pattern type
        """
        pattern = setup.get("pattern_type", "SCANNING")

        if pattern == "BREAKOUT_RETEST":
            return self._breakout_retest_steps(setup, current_price, current_candle)
        elif pattern == "DEMAND_ZONE":
            return self._demand_zone_steps(setup, current_price)
        else:
            return self._scanning_steps()

    def _breakout_retest_steps(self, setup: Dict, current_price: float, current_candle: Dict) -> List[Dict[str, Any]]:
        """
        Create detailed steps for Breakout Retest pattern
        """
        key_level = setup["key_level"]
        state = setup["state"]
        breakout_candle = setup.get("breakout_candle")
        retest_candle = setup.get("retest_candle")

        steps = []

        # Step 1: Breakout
        steps.append({
            "step": 1,
            "status": "complete",
            "title": f"Price broke above ${key_level:.2f} resistance",
            "details": f"Confirmed at {breakout_candle['time']} (1H candle closed at ${breakout_candle['price']:.2f})",
            "explanation": "This shows strong buying pressure. The level that was resistance is now support."
        })

        # Step 2: Retest/Pullback
        if retest_candle:
            steps.append({
                "step": 2,
                "status": "complete",
                "title": f"Price pulled back to test ${key_level:.2f}",
                "details": f"Retest happening now at ${retest_candle['price']:.2f}",
                "explanation": "Healthy pullback. Professional traders use this to enter at better price."
            })
        else:
            steps.append({
                "step": 2,
                "status": "waiting",
                "title": f"WAITING for pullback to ${key_level:.2f}",
                "details": f"Current: ${current_price:.2f} ({current_price - key_level:.2f} pips above)",
                "explanation": "Waiting for price to dip back down. This creates the 'retest' opportunity."
            })
            return steps

        # Step 3: Rejection Candle
        if state in ["REJECTION_WAITING", "CONFIRMATION_WAITING", "READY_TO_ENTER"]:
            wick_touched = current_candle['low'] <= key_level + 2

            if state == "REJECTION_WAITING":
                # Calculate if early entry conditions are met
                minutes_elapsed = 60 - current_candle['time_remaining']
                price_holding_strong = current_price > key_level + 5
                early_entry_available = wick_touched and minutes_elapsed >= 20 and price_holding_strong

                steps.append({
                    "step": 3,
                    "status": "in_progress",
                    "title": "⏳ WATCHING CURRENT CANDLE for rejection",
                    "details": f"Candle: {current_candle['candle_start']} ({minutes_elapsed}/60 mins complete)",
                    "current_values": {
                        "low": f"${current_candle['low']:.2f}",
                        "current_price": f"${current_price:.2f}",
                        "target_level": f"${key_level:.2f}"
                    },
                    "watching_for": {
                        "wick_requirement": {
                            "status": "✅ Complete" if wick_touched else "⏳ Waiting",
                            "text": f"Wick must touch ${key_level:.2f} (support)",
                            "current": f"Low: ${current_candle['low']:.2f}",
                            "explanation": "The wick shows sellers tried to push lower but failed."
                        },
                        "holding_requirement": {
                            "status": "✅ Complete" if (wick_touched and minutes_elapsed >= 20) else "⏳ Waiting",
                            "text": f"Price holding above ${key_level + 5:.2f} for 20+ minutes",
                            "current": f"Currently: ${current_price:.2f} ({minutes_elapsed} mins elapsed)",
                            "explanation": "Price holding strong shows buyers are in control."
                        },
                        "close_requirement": {
                            "status": "⏳ Watching",
                            "text": f"Candle must close above ${key_level + 5:.2f}",
                            "current": f"Currently: ${current_price:.2f}",
                            "time_left": f"{current_candle['time_remaining']} minutes until close",
                            "explanation": "Close above confirms buyers won the battle."
                        }
                    },
                    "entry_timing": {
                        "early_entry": {
                            "available": early_entry_available,
                            "type": "🟡 EARLY ENTRY (50% Position)",
                            "status": "AVAILABLE NOW" if early_entry_available else "NOT READY",
                            "trigger": f"Wick touched + holding above ${key_level + 5:.2f} for 20+ mins",
                            "entry_price": f"${current_price:.2f} (market order)",
                            "stop_loss": f"${key_level - 5:.2f}",
                            "position_size": "50% of planned trade",
                            "pros": "✓ Catch more of the move\n✓ Better average entry price\n✓ Psychological confidence",
                            "cons": "⚠ Candle could still reverse\n⚠ {0} mins until confirmation".format(current_candle['time_remaining']),
                            "action": "Enter 50% position NOW if confident" if early_entry_available else f"Wait {20 - minutes_elapsed} more minutes"
                        },
                        "confirmation_entry": {
                            "type": "🟢 CONFIRMATION ENTRY (Add 50% More)",
                            "trigger": f"Candle closes above ${key_level + 5:.2f}",
                            "expected_time": current_candle['candle_close_expected'],
                            "time_remaining": f"{current_candle['time_remaining']} minutes",
                            "entry_price": f"~${current_price + 2:.2f}-${current_price + 5:.2f} (estimated)",
                            "position_size": "Add remaining 50%",
                            "pros": "✓ Fully confirmed setup\n✓ High confidence\n✓ Clear invalidation",
                            "cons": "⚠ Slightly worse price\n⚠ Might miss fast moves",
                            "action": "Add second 50% if candle closes strong"
                        },
                        "recommended": "Start with 50% early entry (if available), add 50% on confirmation. This balances catching the move while managing risk."
                    },
                    "explanation": "This candle tells us if support holds. Long wick + bullish close = buyers defending."
                })
            else:
                steps.append({
                    "step": 3,
                    "status": "complete",
                    "title": "✅ Rejection candle confirmed",
                    "details": f"Wick touched ${current_candle['low']:.2f}, closed bullish",
                    "explanation": "Perfect rejection! Buyers defended support. High probability setup forming."
                })

        # Step 4: Confirmation
        if state in ["CONFIRMATION_WAITING", "READY_TO_ENTER"]:
            if state == "CONFIRMATION_WAITING":
                steps.append({
                    "step": 4,
                    "status": "waiting",
                    "title": "WAITING for confirmation candle",
                    "details": f"Next candle starts at {current_candle['candle_close_expected']}",
                    "requirements": [
                        {"text": f"Close above ${key_level + 10:.2f}", "explanation": "Shows momentum continuing"},
                        {"text": "Green/bullish candle", "explanation": "Buyers in control"},
                        {"text": "No large rejection wick", "explanation": "No heavy selling"}
                    ],
                    "explanation": "Second confirmation proves first wasn't a fluke."
                })
            else:
                steps.append({
                    "step": 4,
                    "status": "complete",
                    "title": "✅ Confirmation candle completed",
                    "details": "Bullish continuation confirmed",
                    "explanation": "Setup validated by multiple confirmations."
                })

        # Step 5: Entry
        if state == "READY_TO_ENTER":
            steps.append({
                "step": 5,
                "status": "ready",
                "title": "🎯 READY TO ENTER",
                "entry_options": [
                    {
                        "type": "Option A: Wait for 3rd bullish candle",
                        "trigger": f"When next 1H candle closes above ${current_price:.2f}",
                        "current_count": f"{setup['confirmations']}/3 complete",
                        "pros": "Safest - most confirmation",
                        "cons": "Might miss some entry price"
                    },
                    {
                        "type": "Option B: Enter now (market order)",
                        "trigger": f"Enter at current price: ${current_price:.2f}",
                        "pros": "Catch current price level",
                        "cons": "Less confirmation"
                    }
                ],
                "recommendation": "Option A for conservative, Option B if momentum building",
                "explanation": "Setup complete. Choose entry method based on your style."
            })

        return steps

    def _demand_zone_steps(self, setup: Dict, current_price: float) -> List[Dict[str, Any]]:
        """Steps for demand zone pattern"""
        return [
            {
                "step": 1,
                "status": "in_progress",
                "title": "Price at demand zone",
                "details": f"Support level: ${setup['key_level']:.2f}",
                "explanation": "Waiting for reaction at this level"
            }
        ]

    def _scanning_steps(self) -> List[Dict[str, Any]]:
        """Default scanning state"""
        return [
            {
                "step": 1,
                "status": "waiting",
                "title": "Scanning for professional setups",
                "details": "Checking for breakouts, retests, and key level reactions",
                "explanation": "System is actively looking for high-probability patterns"
            }
        ]

    def _explain_h4_structure(self, h4_levels: Dict, current_price: float) -> Dict[str, Any]:
        """Explain 4H timeframe structure"""
        key_level = h4_levels["key_level"]
        level_type = h4_levels["level_type"]
        distance = h4_levels.get("distance_pips", abs(current_price - key_level))

        return {
            "key_level": key_level,
            "level_type": level_type,
            "points": [
                f"Key Level: ${key_level:.2f} ({level_type})",
                f"This level has been tested multiple times on 4H chart",
                f"Currently {distance:.1f} pips away from key level",
                "Structure shows institutional interest at this price"
            ],
            "last_updated": h4_levels.get("last_updated", "Unknown"),
            "next_update": h4_levels.get("next_update", "Unknown")
        }

    def _explain_h1_pattern(self, setup: Dict) -> Dict[str, Any]:
        """Explain 1H pattern"""
        return {
            "pattern": setup["pattern_type"],
            "points": [
                f"Setup: {setup['pattern_type'].replace('_', ' ').title()}",
                f"Progress: {setup['progress']} confirmations",
                f"State: {setup['state'].replace('_', ' ').title()}",
                "Following professional entry criteria"
            ]
        }

    def _explain_session_context(self) -> Dict[str, Any]:
        """Explain current trading session"""
        now = datetime.now(pytz.timezone('America/Chicago'))
        hour = now.hour

        if 3 <= hour < 11:
            session = "London Session"
            strength = "HIGH VOLUME"
        elif 8 <= hour < 16:
            session = "New York Session"
            strength = "HIGH VOLUME"
        elif 8 <= hour < 11:
            session = "London/NY Overlap"
            strength = "HIGHEST VOLUME"
        else:
            session = "Asian Session"
            strength = "LOW VOLUME"

        return {
            "current_session": session,
            "strength": strength,
            "time": now.strftime("%I:%M %p CT"),
            "explanation": f"{session} - {strength}. " +
                          ("Optimal time for gold trading." if "HIGH" in strength else "Lower probability time.")
        }

    def _build_trade_plan(self, setup: Dict, h4_levels: Dict, current_price: float) -> Dict[str, Any]:
        """Build complete trade plan"""
        state = setup.get("state", "SCANNING")

        if state not in ["CONFIRMATION_WAITING", "READY_TO_ENTER"]:
            return {
                "status": "Not ready yet",
                "message": "Trade plan will appear when setup completes"
            }

        entry = setup["expected_entry"]
        key_level = setup["key_level"]

        # Structure-based SL (below rejection wick)
        rejection_low = setup.get("rejection_candle_low", entry - 10)
        sl = rejection_low - 2  # 2 pips buffer

        # Find TP levels
        resistance_levels = h4_levels.get("resistance_levels", [])
        tp1 = entry + 15 if not resistance_levels else min([r for r in resistance_levels if r > entry], default=entry + 15)
        tp2 = entry + 30 if not resistance_levels else max([r for r in resistance_levels if r > entry], default=entry + 30)

        risk_pips = entry - sl
        reward1_pips = tp1 - entry
        reward2_pips = tp2 - entry

        return {
            "status": "Ready",
            "entry_method": "Market order when step 5 triggers",
            "entry_price": f"${entry:.2f}",
            "stop_loss": {
                "price": f"${sl:.2f}",
                "reason": f"Below rejection wick low (${rejection_low:.2f})",
                "pips": f"{risk_pips:.1f} pips risk",
                "why": "If price goes here, support failed and setup is invalid"
            },
            "take_profit_1": {
                "price": f"${tp1:.2f}",
                "reason": "Next resistance level",
                "rr_ratio": f"{reward1_pips/risk_pips:.1f}:1",
                "action": "Close 50%, move SL to entry",
                "why": "Lock in profit, run rest risk-free"
            },
            "take_profit_2": {
                "price": f"${tp2:.2f}",
                "reason": "Major resistance level",
                "rr_ratio": f"{reward2_pips/risk_pips:.1f}:1",
                "action": "Close remaining 50%",
                "why": "Full profit at high-probability target"
            },
            "position_sizing": {
                "account_risk": "0.5% recommended",
                "risk_pips": f"{risk_pips:.1f}",
                "calculation": f"Position size = (Account × 0.005) / {risk_pips:.1f} pips"
            }
        }

    def _get_invalidation_conditions(self, setup: Dict) -> List[Dict[str, Any]]:
        """
        Return DYNAMIC invalidation conditions based on current setup state
        Conditions update in real-time as the setup progresses
        """
        if not setup.get("detected"):
            return [
                {
                    "condition": "No setup detected yet",
                    "reason": "Still scanning market",
                    "action": "Wait for clear pattern to form",
                    "severity": "INFO"
                }
            ]

        pattern_type = setup.get("pattern_type", "SCANNING")
        state = setup.get("state", "SCANNING")
        key_level = setup.get("key_level", 0)

        # Pattern-specific invalidation conditions
        if pattern_type == "BREAKOUT_RETEST":
            return self._breakout_retest_invalidation(setup, state, key_level)
        elif pattern_type == "DEMAND_ZONE":
            return self._demand_zone_invalidation(setup, state, key_level)
        else:
            return [
                {
                    "condition": "No active setup",
                    "reason": "Scanning for patterns",
                    "action": "Wait for setup to form",
                    "severity": "INFO"
                }
            ]

    def _breakout_retest_invalidation(self, setup: Dict, state: str, key_level: float) -> List[Dict[str, Any]]:
        """Dynamic invalidation for Breakout Retest pattern based on current state"""
        conditions = []

        # State-specific invalidations
        if state == "BREAKOUT_CONFIRMED":
            conditions.append({
                "condition": f"Price closes back below ${key_level:.2f}",
                "reason": "Breakout failed - false breakout",
                "action": "Cancel setup immediately. This was a trap.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "Price doesn't pull back within 4 candles",
                "reason": "Ran away without retest opportunity",
                "action": "Setup expired - wait for new pattern",
                "severity": "HIGH"
            })

        elif state == "RETEST_HAPPENING":
            conditions.append({
                "condition": f"Current candle closes below ${key_level - 5:.2f}",
                "reason": "Retest failed - buyers didn't defend",
                "action": "Cancel setup. Support broken = bearish",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "Candle wick goes below support but closes weak",
                "reason": "Wick is good, but weak close = no buyers",
                "action": "Wait for next candle confirmation",
                "severity": "HIGH"
            })

        elif state == "REJECTION_WAITING":
            conditions.append({
                "condition": f"Candle closes below ${key_level:.2f}",
                "reason": "Failed to reject from support - breakdown",
                "action": "Setup INVALID. Exit if entered early.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "Wick doesn't touch support by candle close",
                "reason": "No actual test of support = fake setup",
                "action": "Cancel. Must see wick touch support.",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": "Price stays below key level for 15+ minutes",
                "reason": "Sitting below support = weak buyers",
                "action": "Probably invalidated, wait for close",
                "severity": "MEDIUM"
            })

        elif state == "CONFIRMATION_WAITING":
            conditions.append({
                "condition": f"Next candle closes below ${key_level:.2f}",
                "reason": "Rejection failed - price broke down after",
                "action": "Setup dead. Do not enter.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "Next candle makes lower low",
                "reason": "Creating bearish structure - trend down",
                "action": "Cancel setup, look for shorts instead",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": "Low volume session starts (Asia)",
                "reason": "Asian session = low liquidity, unreliable",
                "action": "Skip entry or close position",
                "severity": "MEDIUM"
            })

        elif state == "READY_TO_ENTER":
            conditions.append({
                "condition": f"Entry candle closes below ${key_level:.2f}",
                "reason": "Setup collapsed on entry candle",
                "action": "Do NOT enter. Setup failed.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "Strong bearish candle forms",
                "reason": "Momentum shifted bearish",
                "action": "Cancel entry, setup invalid",
                "severity": "HIGH"
            })

        # Universal invalidations (apply to all states)
        conditions.append({
            "condition": f"Price falls ${(key_level * 0.01):.2f}+ below support",
            "reason": "Major support break (1% below key level)",
            "action": "Setup completely dead. Look for new pattern.",
            "severity": "CRITICAL"
        })

        return conditions

    def _demand_zone_invalidation(self, setup: Dict, state: str, key_level: float) -> List[Dict[str, Any]]:
        """Dynamic invalidation for Demand Zone pattern"""
        return [
            {
                "condition": f"Price closes below ${key_level - 10:.2f}",
                "reason": "Demand zone violated",
                "action": "Cancel setup - zone failed",
                "severity": "CRITICAL"
            },
            {
                "condition": "Multiple tests of zone without bounce",
                "reason": "Zone is weak, being absorbed",
                "action": "Setup failing, look elsewhere",
                "severity": "HIGH"
            },
            {
                "condition": "No bullish reaction in zone",
                "reason": "No buyers showing up",
                "action": "Wait or cancel",
                "severity": "MEDIUM"
            }
        ]

    def _convert_candles_to_json(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert pandas DataFrame with Timestamp index to JSON-safe format"""
        candles = []
        for idx, row in df.iterrows():
            candle = {
                "time": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                "open": float(row['open']),
                "high": float(row['high']),
                "low": float(row['low']),
                "close": float(row['close']),
                "volume": int(row.get('volume', 0))
            }
            candles.append(candle)
        return candles

    def _error_response(self, message: str) -> Dict[str, Any]:
        """Return error response"""
        return {
            "status": "error",
            "message": message,
            "setup_status": "ERROR",
            "setup_steps": [],
            "last_update": datetime.now(pytz.UTC).isoformat()
        }


# Convenience function for API
async def get_bullish_pro_trader_analysis() -> Dict[str, Any]:
    """Get BULLISH professional trader analysis for XAUUSD (BUY setups only)"""
    system = BullishProTraderGold()
    return await system.get_detailed_setup()
