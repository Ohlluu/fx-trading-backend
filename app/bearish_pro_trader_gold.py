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
from .mtf_cache import mtf_cache  # Import shared cache

class BearishProTraderGold:
    """
    BEARISH Educational trading system - looks for SELL setups
    Detects breakdowns below support and bearish rejections
    Same professional logic as bullish trader, opposite direction
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
        BEARISH TRADER: Identify key RESISTANCE levels using REAL 4H candles
        Only looks for resistance ABOVE price (potential SHORT zones)
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

        # Find significant resistance levels ONLY (bearish trader looks for SELL opportunities)
        resistance_candidates = []

        # Find local highs using 4H candles
        for i in range(5, len(highs) - 5):
            # Resistance: local high
            if highs.iloc[i] == highs.iloc[i-5:i+5].max():
                resistance_candidates.append(float(highs.iloc[i]))

        # Get resistance levels ABOVE current price (potential SHORT zones)
        resistance = sorted([r for r in resistance_candidates if r > current_price])[:3] if resistance_candidates else []

        # BEARISH TRADER: Always prefer resistance (above price)
        if resistance:
            key_level = resistance[0]  # Closest resistance above price
            level_type = "resistance"
        else:
            # If no resistance above, use recent high as potential resistance
            key_level = float(recent_data['high'].tail(20).max())
            level_type = "resistance"

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
            "resistance_levels": resistance if resistance else [key_level],
            "support_levels": [],  # Bearish trader doesn't track support
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
        last_candles = h1_data.tail(20)  # Increased for FVG detection

        # Check for FAIR VALUE GAP (FVG) - HIGHEST PRIORITY
        fvg_setup = self._check_fvg(last_candles, current_price)
        if fvg_setup["detected"]:
            return fvg_setup

        # Check for BREAKDOWN RETEST pattern
        breakdown_setup = self._check_breakout_retest(last_candles, key_level, current_price)
        if breakdown_setup["detected"]:
            return breakdown_setup

        # Check for SUPPLY ZONE pattern (bearish resistance zone)
        supply_setup = self._check_supply_zone(last_candles, h4_levels, current_price)
        if supply_setup["detected"]:
            return supply_setup

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
        Check for BREAKDOWN Retest pattern (BEARISH):
        1. Price was above support
        2. Price broke BELOW support (breakdown)
        3. Price pulled back UP to test old support (now resistance)
        4. Looking for bearish rejection + confirmation
        """
        if len(candles) < 5:
            return {"detected": False}

        # Check if we had a BREAKDOWN in recent candles
        breakdown_candle_idx = None
        for i in range(len(candles) - 5, len(candles)):
            candle = candles.iloc[i]
            prev_candle = candles.iloc[i-1] if i > 0 else None

            # BREAKDOWN = close moved from ABOVE to BELOW key level
            if prev_candle is not None:
                if prev_candle['close'] > key_level and candle['close'] < key_level:
                    breakdown_candle_idx = i
                    break

        if breakdown_candle_idx is None:
            return {"detected": False}

        # We have a BREAKDOWN! Now check for retest
        breakdown_candle = candles.iloc[breakdown_candle_idx]
        candles_after_breakdown = candles.iloc[breakdown_candle_idx+1:]

        # INVALIDATION CHECK: If any candle after breakdown closed significantly ABOVE resistance, setup is dead
        if len(candles_after_breakdown) > 0:
            for i, candle in candles_after_breakdown.iterrows():
                # If candle closed ABOVE resistance (not just wick, but actual close), setup is invalidated
                if candle['close'] > key_level + 5:  # 5 pips buffer
                    return {"detected": False}  # Setup invalidated

        # Check if price has come back UP to test the level
        retest_happening = False
        rejection_confirmed = False

        if len(candles_after_breakdown) > 0:
            # Retest = price returned UP close to key level (from below)
            for i, candle in candles_after_breakdown.iterrows():
                if abs(candle['high'] - key_level) < 10:  # Within 10 pips (checking HIGH now, not low)
                    retest_happening = True

                    # Check for BEARISH rejection (long UPPER wick + close BELOW)
                    wick_size = candle['high'] - max(candle['open'], candle['close'])
                    if wick_size > 3 and candle['close'] < key_level - 5:
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
            "pattern_type": "BREAKDOWN_RETEST",
            "direction": "SHORT",
            "state": state,
            "progress": progress,
            "key_level": key_level,
            "confirmations": confirmations,
            "breakdown_candle": {
                "time": breakdown_candle.name.strftime("%I:%M %p") if hasattr(breakdown_candle.name, 'strftime') else "Recently",
                "price": float(breakdown_candle['close'])
            },
            "retest_candle": {
                "time": "Now" if retest_happening else "Waiting",
                "price": current_price
            } if retest_happening else None,
            "rejection_candle_high": float(candles.tail(1).iloc[0]['high']) if rejection_confirmed else None,
            "expected_entry": current_price
        }

    def _check_fvg(self, candles: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Check for BEARISH Fair Value Gap (FVG) pattern:

        FVG = 3-candle pattern where there's a GAP (imbalance) in price:
        - Candle 1: Up move (creates the high)
        - Candle 2: BIG bearish move (skips price levels)
        - Candle 3: Continuation down (creates the low)

        If Candle3.high < Candle1.low → There's a GAP!
        The gap zone is unfilled price area where NO trades occurred
        Price is magnetically pulled back UP to "fill" this gap
        """
        if len(candles) < 10:
            return {"detected": False}

        # Scan last 15 candles for FVG patterns
        fvg_zones = []

        for i in range(len(candles) - 3):
            candle1 = candles.iloc[i]
            candle2 = candles.iloc[i + 1]
            candle3 = candles.iloc[i + 2]

            # BEARISH FVG detection:
            # Gap exists if candle3.high < candle1.low (price skipped the middle zone)
            if candle3['high'] < candle1['low']:
                gap_top = candle1['low']
                gap_bottom = candle3['high']
                gap_size = gap_top - gap_bottom

                # Only consider significant gaps (> 5 pips)
                if gap_size > 5:
                    gap_midpoint = (gap_top + gap_bottom) / 2

                    # Check if gap has been filled already
                    filled = False
                    for j in range(i + 3, len(candles)):
                        if candles.iloc[j]['high'] >= gap_top:
                            filled = True
                            break

                    if not filled:
                        fvg_zones.append({
                            "top": float(gap_top),
                            "bottom": float(gap_bottom),
                            "midpoint": float(gap_midpoint),
                            "size": float(gap_size),
                            "candle_index": i,
                            "age_candles": len(candles) - i - 3
                        })

        # No unfilled FVGs found
        if not fvg_zones:
            return {"detected": False}

        # Find nearest FVG above current price (price needs to rise to fill it)
        fvgs_above = [fvg for fvg in fvg_zones if fvg["bottom"] > current_price]

        if not fvgs_above:
            return {"detected": False}

        # Get the nearest FVG (closest to current price)
        nearest_fvg = min(fvgs_above, key=lambda x: abs(current_price - x["midpoint"]))

        # Check if price is approaching the FVG (within 20 pips)
        distance_to_fvg = nearest_fvg["midpoint"] - current_price

        if distance_to_fvg > 20:
            return {"detected": False}  # Too far away

        # Determine state based on proximity and price action
        if current_price >= nearest_fvg["bottom"] and current_price <= nearest_fvg["top"]:
            state = "IN_FVG"
            progress = "3/5"
            confirmations = 2
        elif distance_to_fvg <= 10:
            state = "APPROACHING"
            progress = "2/5"
            confirmations = 1
        else:
            state = "DETECTED"
            progress = "1/5"
            confirmations = 0

        return {
            "detected": True,
            "pattern_type": "FAIR_VALUE_GAP",
            "direction": "SHORT",
            "state": state,
            "progress": progress,
            "key_level": nearest_fvg["midpoint"],
            "confirmations": confirmations,
            "expected_entry": nearest_fvg["midpoint"],
            "fvg_zone": {
                "top": nearest_fvg["top"],
                "bottom": nearest_fvg["bottom"],
                "midpoint": nearest_fvg["midpoint"],
                "size_pips": nearest_fvg["size"],
                "age_candles": nearest_fvg["age_candles"]
            },
            "distance_pips": float(distance_to_fvg)
        }

    def _check_supply_zone(self, candles: pd.DataFrame, h4_levels: Dict, current_price: float) -> Dict[str, Any]:
        """
        Check for SUPPLY ZONE pattern (BEARISH):
        - Price approaching a RESISTANCE level where sellers historically defended
        - Looking for bearish rejection + SHORT opportunity
        """
        resistance_levels = h4_levels.get("resistance_levels", [])

        if not resistance_levels:
            return {"detected": False}

        # Check if price is near a resistance level (within 10 pips)
        nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))

        # SUPPLY ZONE detected if price is within 10 pips of resistance
        if abs(current_price - nearest_resistance) < 10:
            # Check recent candles for bearish confirmation
            last_candle = candles.iloc[-1]

            # Calculate rejection signs
            upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
            body_size = abs(last_candle['close'] - last_candle['open'])
            is_bearish = last_candle['close'] < last_candle['open']

            # Determine state based on confirmation level
            confirmations = 0
            if upper_wick > 3:  # Long upper wick = rejection at resistance
                confirmations += 1
            if is_bearish and body_size > 5:  # Strong bearish candle
                confirmations += 1

            # Determine progress
            if confirmations >= 2:
                state = "REJECTION_CONFIRMED"
                progress = "3/5"
            elif confirmations == 1:
                state = "WATCHING"
                progress = "2/5"
            else:
                state = "AT_ZONE"
                progress = "1/5"

            return {
                "detected": True,
                "pattern_type": "SUPPLY_ZONE",
                "direction": "SHORT",
                "state": state,
                "progress": progress,
                "key_level": nearest_resistance,
                "confirmations": confirmations,
                "expected_entry": current_price,
                "rejection_signs": {
                    "upper_wick": float(upper_wick),
                    "is_bearish": is_bearish,
                    "body_size": float(body_size)
                }
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
        Build step-by-step breakdown based on pattern type (BEARISH)
        """
        pattern = setup.get("pattern_type", "SCANNING")

        if pattern == "FAIR_VALUE_GAP":
            return self._fvg_steps(setup, current_price)
        elif pattern == "BREAKDOWN_RETEST":
            return self._breakdown_retest_steps(setup, current_price, current_candle)
        elif pattern == "SUPPLY_ZONE":
            return self._supply_zone_steps(setup, current_price)
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

    def _breakdown_retest_steps(self, setup: Dict, current_price: float, current_candle: Dict) -> List[Dict[str, Any]]:
        """
        Create detailed steps for BEARISH Breakdown Retest pattern (SHORT direction)
        """
        key_level = setup["key_level"]
        state = setup["state"]
        breakdown_candle = setup.get("breakdown_candle")
        retest_candle = setup.get("retest_candle")

        steps = []

        # Step 1: Breakdown
        steps.append({
            "step": 1,
            "status": "complete",
            "title": f"Price broke BELOW ${key_level:.2f} support",
            "details": f"Confirmed at {breakdown_candle['time']} (1H candle closed at ${breakdown_candle['price']:.2f})",
            "explanation": "This shows strong selling pressure. The level that was support is now resistance."
        })

        # Step 2: Retest/Pullback
        if retest_candle:
            steps.append({
                "step": 2,
                "status": "complete",
                "title": f"Price pulled back UP to test ${key_level:.2f}",
                "details": f"Retest happening now at ${retest_candle['price']:.2f}",
                "explanation": "Healthy pullback. Professional traders use this to enter SHORT at better price."
            })
        else:
            steps.append({
                "step": 2,
                "status": "waiting",
                "title": f"WAITING for pullback UP to ${key_level:.2f}",
                "details": f"Current: ${current_price:.2f} ({key_level - current_price:.2f} pips below)",
                "explanation": "Waiting for price to bounce back up. This creates the 'retest' opportunity."
            })
            return steps

        # Step 3: Rejection Candle
        if state in ["REJECTION_WAITING", "CONFIRMATION_WAITING", "READY_TO_ENTER"]:
            wick_touched = current_candle['high'] >= key_level - 2

            if state == "REJECTION_WAITING":
                # Calculate if early entry conditions are met
                minutes_elapsed = 60 - current_candle['time_remaining']
                price_holding_weak = current_price < key_level - 5
                early_entry_available = wick_touched and minutes_elapsed >= 20 and price_holding_weak

                steps.append({
                    "step": 3,
                    "status": "in_progress",
                    "title": "⏳ WATCHING CURRENT CANDLE for bearish rejection",
                    "details": f"Candle: {current_candle['candle_start']} ({minutes_elapsed}/60 mins complete)",
                    "current_values": {
                        "high": f"${current_candle['high']:.2f}",
                        "current_price": f"${current_price:.2f}",
                        "target_level": f"${key_level:.2f}"
                    },
                    "watching_for": {
                        "wick_requirement": {
                            "status": "✅ Complete" if wick_touched else "⏳ Waiting",
                            "text": f"Wick must touch ${key_level:.2f} (resistance)",
                            "current": f"High: ${current_candle['high']:.2f}",
                            "explanation": "The wick shows buyers tried to push higher but failed."
                        },
                        "holding_requirement": {
                            "status": "✅ Complete" if (wick_touched and minutes_elapsed >= 20) else "⏳ Waiting",
                            "text": f"Price holding BELOW ${key_level - 5:.2f} for 20+ minutes",
                            "current": f"Currently: ${current_price:.2f} ({minutes_elapsed} mins elapsed)",
                            "explanation": "Price staying weak shows sellers are in control."
                        },
                        "close_requirement": {
                            "status": "⏳ Watching",
                            "text": f"Candle must close BELOW ${key_level - 5:.2f}",
                            "current": f"Currently: ${current_price:.2f}",
                            "time_left": f"{current_candle['time_remaining']} minutes until close",
                            "explanation": "Close below confirms sellers won the battle."
                        }
                    },
                    "entry_timing": {
                        "early_entry": {
                            "available": early_entry_available,
                            "type": "🟡 EARLY ENTRY (50% Position)",
                            "status": "AVAILABLE NOW" if early_entry_available else "NOT READY",
                            "trigger": f"Wick touched + holding BELOW ${key_level - 5:.2f} for 20+ mins",
                            "entry_price": f"${current_price:.2f} (market order SHORT)",
                            "stop_loss": f"${key_level + 5:.2f}",
                            "position_size": "50% of planned trade",
                            "pros": "✓ Catch more of the move DOWN\n✓ Better average entry price\n✓ Psychological confidence",
                            "cons": "⚠ Candle could still reverse UP\n⚠ {0} mins until confirmation".format(current_candle['time_remaining']),
                            "action": "Enter 50% SHORT position NOW if confident" if early_entry_available else f"Wait {20 - minutes_elapsed} more minutes"
                        },
                        "confirmation_entry": {
                            "type": "🟢 CONFIRMATION ENTRY (Add 50% More)",
                            "trigger": f"Candle closes BELOW ${key_level - 5:.2f}",
                            "expected_time": current_candle['candle_close_expected'],
                            "time_remaining": f"{current_candle['time_remaining']} minutes",
                            "entry_price": f"~${current_price - 2:.2f}-${current_price - 5:.2f} (estimated)",
                            "position_size": "Add remaining 50% SHORT",
                            "pros": "✓ Fully confirmed setup\n✓ High confidence\n✓ Clear invalidation",
                            "cons": "⚠ Slightly worse price\n⚠ Might miss fast moves",
                            "action": "Add second 50% SHORT if candle closes weak"
                        },
                        "recommended": "Start with 50% early SHORT entry (if available), add 50% on confirmation. This balances catching the move while managing risk."
                    },
                    "explanation": "This candle tells us if resistance holds. Long UPPER wick + bearish close = sellers defending."
                })
            else:
                steps.append({
                    "step": 3,
                    "status": "complete",
                    "title": "✅ Bearish rejection candle confirmed",
                    "details": f"Wick touched ${current_candle['high']:.2f}, closed bearish",
                    "explanation": "Perfect rejection! Sellers defended resistance. High probability SHORT setup forming."
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
                        {"text": f"Close BELOW ${key_level - 10:.2f}", "explanation": "Shows downward momentum continuing"},
                        {"text": "Red/bearish candle", "explanation": "Sellers in control"},
                        {"text": "No large upper wick", "explanation": "No heavy buying"}
                    ],
                    "explanation": "Second confirmation proves first wasn't a fluke."
                })
            else:
                steps.append({
                    "step": 4,
                    "status": "complete",
                    "title": "✅ Confirmation candle completed",
                    "details": "Bearish continuation confirmed",
                    "explanation": "Setup validated by multiple confirmations."
                })

        # Step 5: Entry
        if state == "READY_TO_ENTER":
            steps.append({
                "step": 5,
                "status": "ready",
                "title": "🎯 READY TO ENTER SHORT",
                "entry_options": [
                    {
                        "type": "Option A: Wait for 3rd bearish candle",
                        "trigger": f"When next 1H candle closes BELOW ${current_price:.2f}",
                        "current_count": f"{setup['confirmations']}/3 complete",
                        "pros": "Safest - most confirmation",
                        "cons": "Might miss some entry price"
                    },
                    {
                        "type": "Option B: Enter SHORT now (market order)",
                        "trigger": f"Enter SHORT at current price: ${current_price:.2f}",
                        "pros": "Catch current price level",
                        "cons": "Less confirmation"
                    }
                ],
                "recommendation": "Option A for conservative, Option B if downward momentum building",
                "explanation": "SHORT setup complete. Choose entry method based on your style."
            })

        return steps

    def _supply_zone_steps(self, setup: Dict, current_price: float) -> List[Dict[str, Any]]:
        """
        Create detailed steps for SUPPLY ZONE pattern (BEARISH)
        Price approaching resistance where sellers historically defended
        """
        key_level = setup["key_level"]
        state = setup["state"]
        confirmations = setup.get("confirmations", 0)
        rejection_signs = setup.get("rejection_signs", {})

        steps = []

        # Step 1: Price at supply zone
        steps.append({
            "step": 1,
            "status": "complete",
            "title": f"Price reached supply zone at ${key_level:.2f}",
            "details": f"Current: ${current_price:.2f} (within resistance zone)",
            "explanation": "This is a RESISTANCE level where sellers historically defended. Institutions often place SELL orders here."
        })

        # Step 2: Watching for bearish rejection
        if state in ["AT_ZONE", "WATCHING", "REJECTION_CONFIRMED"]:
            upper_wick = rejection_signs.get("upper_wick", 0)
            is_bearish = rejection_signs.get("is_bearish", False)

            if state == "AT_ZONE":
                steps.append({
                    "step": 2,
                    "status": "in_progress",
                    "title": "⏳ WATCHING for bearish rejection",
                    "details": f"Looking for sellers to defend ${key_level:.2f} resistance",
                    "watching_for": {
                        "upper_wick": {
                            "status": "✅ Detected" if upper_wick > 3 else "⏳ Waiting",
                            "text": "Long upper wick (sellers rejecting higher prices)",
                            "current": f"Upper wick: ${upper_wick:.2f}" if upper_wick > 0 else "No rejection yet"
                        },
                        "bearish_candle": {
                            "status": "✅ Confirmed" if is_bearish else "⏳ Waiting",
                            "text": "Red/bearish candle (close below open)",
                            "current": "Bearish candle" if is_bearish else "Not yet bearish"
                        },
                        "close_requirement": {
                            "status": "⏳ Watching",
                            "text": f"Close BELOW ${key_level - 5:.2f}",
                            "current": f"Currently: ${current_price:.2f}",
                            "explanation": "Close below confirms sellers won"
                        }
                    },
                    "explanation": "Upper wick + bearish close = sellers defending resistance"
                })
            elif state == "WATCHING":
                steps.append({
                    "step": 2,
                    "status": "in_progress",
                    "title": "⚠️ Partial rejection detected",
                    "details": f"1/2 bearish signs confirmed at ${key_level:.2f}",
                    "confirmations": f"{confirmations}/2 rejection signs",
                    "explanation": "Getting closer. Need one more confirmation for high-probability SHORT."
                })
            else:  # REJECTION_CONFIRMED
                steps.append({
                    "step": 2,
                    "status": "complete",
                    "title": "✅ Bearish rejection confirmed",
                    "details": f"Strong selling pressure at ${key_level:.2f}",
                    "confirmations": f"Upper wick: ${upper_wick:.2f}, Bearish candle: Yes",
                    "explanation": "Sellers defended resistance! Setup forming."
                })

        # Step 3: Confirmation candle needed
        if state == "REJECTION_CONFIRMED":
            steps.append({
                "step": 3,
                "status": "waiting",
                "title": "WAITING for confirmation candle",
                "details": "Next 1H candle must close bearish",
                "requirements": [
                    {"text": f"Close BELOW ${key_level - 10:.2f}", "explanation": "Shows downward momentum"},
                    {"text": "Red/bearish candle", "explanation": "Sellers in control"},
                    {"text": "No large lower wick", "explanation": "No strong buying pressure"}
                ],
                "explanation": "Second bearish candle confirms first wasn't a false signal"
            })

        # Step 4: Setup building
        if confirmations >= 2:
            steps.append({
                "step": 4,
                "status": "in_progress",
                "title": "📉 Setup building momentum",
                "details": f"Resistance holding at ${key_level:.2f}, price showing weakness",
                "next": "Wait for 3rd bearish confirmation or enter on momentum",
                "explanation": "Multiple rejections increase probability of successful SHORT"
            })

        # Step 5: Entry ready (if all confirmations)
        if confirmations >= 3:
            steps.append({
                "step": 5,
                "status": "ready",
                "title": "🎯 READY TO ENTER SHORT",
                "entry_options": [
                    {
                        "type": "Option A: Enter SHORT now (market order)",
                        "trigger": f"SELL at current price: ${current_price:.2f}",
                        "stop_loss": f"${key_level + 10:.2f}",
                        "take_profit": f"Target support levels below",
                        "pros": "Catch rejection at resistance",
                        "cons": "Less confirmation"
                    },
                    {
                        "type": "Option B: Wait for breakdown",
                        "trigger": f"Wait for close BELOW ${key_level - 20:.2f}",
                        "pros": "More confirmation, clearer direction",
                        "cons": "Worse entry price"
                    }
                ],
                "recommendation": "Option A for aggressive, Option B for conservative",
                "explanation": "SUPPLY ZONE setup complete. Sellers defending resistance."
            })

        return steps

    def _fvg_steps(self, setup: Dict, current_price: float) -> List[Dict[str, Any]]:
        """
        Create detailed steps for FAIR VALUE GAP (FVG) pattern - BEARISH
        FVG = Price imbalance that acts like a magnet, pulling price UP to fill the gap
        """
        state = setup.get("state", "DETECTED")
        fvg_zone = setup.get("fvg_zone", {})
        distance = setup.get("distance_pips", 0)

        steps = []

        # Step 1: FVG Detected
        steps.append({
            "step": 1,
            "status": "complete" if state in ["APPROACHING", "IN_FVG"] else "in_progress",
            "title": "📊 Fair Value Gap Detected",
            "details": f"FVG Zone: ${fvg_zone.get('bottom', 0):.2f} - ${fvg_zone.get('top', 0):.2f} ({fvg_zone.get('size_pips', 0):.1f} pips)",
            "explanation": f"Unfilled gap from {fvg_zone.get('age_candles', 0)} candles ago. Price magnetically pulled to fill it. Currently {distance:.1f} pips away."
        })

        # Step 2: Price Approaching or Inside FVG
        if state == "APPROACHING":
            steps.append({
                "step": 2,
                "status": "in_progress",
                "title": "⚠️ Price Approaching FVG",
                "details": f"Distance to FVG: {distance:.1f} pips (within 10 pips)",
                "explanation": "Price is rising toward the gap. Watch for bearish rejection reaction."
            })
        elif state == "IN_FVG":
            steps.append({
                "step": 2,
                "status": "complete",
                "title": "✅ Price Inside FVG",
                "details": f"Current: ${current_price:.2f} (inside FVG zone)",
                "explanation": "Price has reached the gap! High probability of bearish rejection."
            })
        else:
            steps.append({
                "step": 2,
                "status": "waiting",
                "title": "Waiting for price to approach FVG",
                "details": f"Still {distance:.1f} pips away (need within 10 pips)",
                "explanation": "Monitoring price action as it moves toward gap."
            })

        # Step 3: Entry Strategy (only when IN_FVG)
        if state == "IN_FVG":
            steps.append({
                "step": 3,
                "status": "ready",
                "title": "🎯 READY TO ENTER",
                "entry_options": [
                    {
                        "type": "Option A: Enter at FVG midpoint (Aggressive)",
                        "trigger": f"Enter NOW at ${fvg_zone.get('midpoint', 0):.2f}",
                        "pros": "Best entry price, highest R:R",
                        "cons": "No confirmation candle"
                    },
                    {
                        "type": "Option B: Wait for bearish confirmation (Conservative)",
                        "trigger": "Wait for 1H bearish candle inside FVG zone",
                        "pros": "Confirmation of sellers stepping in",
                        "cons": "Slightly worse entry price"
                    }
                ],
                "recommendation": "Option A for aggressive (FVG midpoint entry), Option B for conservative (wait for confirmation)",
                "explanation": "FVG filled! Institutions left orders here. High probability of rejection DOWN."
            })
        else:
            steps.append({
                "step": 3,
                "status": "waiting",
                "title": "Entry criteria",
                "details": "Waiting for price to reach FVG zone",
                "explanation": "Entry signal will trigger when price enters the gap."
            })

        return steps

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
        if pattern_type == "FAIR_VALUE_GAP":
            return self._fvg_invalidation(setup, state, key_level)
        elif pattern_type == "BREAKDOWN_RETEST":  # BEARISH breakdown retest
            return self._breakout_retest_invalidation(setup, state, key_level)
        elif pattern_type == "SUPPLY_ZONE":  # BEARISH supply zone (resistance)
            return self._supply_zone_invalidation(setup, state, key_level)
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
        """Dynamic invalidation for BEARISH Breakdown Retest pattern - FLIPPED for SHORT"""
        conditions = []

        # State-specific invalidations (BEARISH - opposite of bullish!)
        if state == "BREAKOUT_CONFIRMED":
            # For bearish: If price closes ABOVE resistance after breakdown, setup fails
            conditions.append({
                "condition": f"Price closes back ABOVE ${key_level:.2f}",
                "reason": "Breakdown failed - false breakdown",
                "action": "Cancel setup immediately. This was a trap.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "Price doesn't pull back UP within 4 candles",
                "reason": "Fell away without retest opportunity",
                "action": "Setup expired - wait for new pattern",
                "severity": "HIGH"
            })

        elif state == "RETEST_HAPPENING":
            # For bearish: If price closes ABOVE resistance, retest failed
            conditions.append({
                "condition": f"Current candle closes ABOVE ${key_level + 5:.2f}",
                "reason": "Retest failed - sellers didn't defend",
                "action": "Cancel setup. Resistance broken = bullish",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "Candle wick goes ABOVE resistance but closes weak",
                "reason": "Wick is good, but weak close = no sellers",
                "action": "Wait for next candle confirmation",
                "severity": "HIGH"
            })

        elif state == "REJECTION_WAITING":
            # For bearish: If price closes ABOVE resistance, rejection failed
            conditions.append({
                "condition": f"Candle closes ABOVE ${key_level:.2f}",
                "reason": "Failed to reject from resistance - breakout",
                "action": "Setup INVALID. Exit if entered early.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "Wick doesn't touch resistance by candle close",
                "reason": "No actual test of resistance = fake setup",
                "action": "Cancel. Must see wick touch resistance.",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": "Price stays ABOVE key level for 15+ minutes",
                "reason": "Sitting above resistance = weak sellers",
                "action": "Probably invalidated, wait for close",
                "severity": "MEDIUM"
            })

        elif state == "CONFIRMATION_WAITING":
            # For bearish: If next candle closes ABOVE resistance, setup dead
            conditions.append({
                "condition": f"Next candle closes ABOVE ${key_level:.2f}",
                "reason": "Rejection failed - price broke UP after",
                "action": "Setup dead. Do not enter.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "Next candle makes higher high",
                "reason": "Creating bullish structure - trend up",
                "action": "Cancel setup, look for longs instead",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": "Low volume session starts (Asia)",
                "reason": "Asian session = low liquidity, unreliable",
                "action": "Skip entry or close position",
                "severity": "MEDIUM"
            })

        elif state == "READY_TO_ENTER":
            # For bearish: If entry candle closes ABOVE resistance, setup failed
            conditions.append({
                "condition": f"Entry candle closes ABOVE ${key_level:.2f}",
                "reason": "Setup collapsed on entry candle",
                "action": "Do NOT enter. Setup failed.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "Strong BULLISH candle forms",
                "reason": "Momentum shifted bullish",
                "action": "Cancel entry, setup invalid",
                "severity": "HIGH"
            })

        # Universal invalidations (BEARISH - price goes UP kills setup)
        conditions.append({
            "condition": f"Price rises ${(key_level * 0.01):.2f}+ ABOVE resistance",
            "reason": "Major resistance break (1% above key level)",
            "action": "Setup completely dead. Look for new pattern.",
            "severity": "CRITICAL"
        })

        return conditions

    def _fvg_invalidation(self, setup: Dict, state: str, key_level: float) -> List[Dict[str, Any]]:
        """Dynamic invalidation for Fair Value Gap (FVG) pattern - BEARISH"""
        fvg_zone = setup.get("fvg_zone", {})
        gap_top = fvg_zone.get("top", key_level + 5)

        conditions = []

        state = setup.get("state", "DETECTED")

        # State-specific invalidations
        if state == "DETECTED":
            conditions.append({
                "condition": f"Price continues DOWN without filling FVG (${gap_top:.2f})",
                "reason": "FVG may not be filled this time",
                "action": "Wait or look for other setups",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": "FVG becomes too old (20+ candles)",
                "reason": "Older gaps less likely to be filled",
                "action": "Consider setup stale",
                "severity": "MEDIUM"
            })

        elif state == "APPROACHING":
            conditions.append({
                "condition": f"Price reverses DOWN before reaching FVG top (${gap_top:.2f})",
                "reason": "Failed to fill the gap completely",
                "action": "Setup invalidated - gap may not fill",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": "Strong bearish momentum away from FVG",
                "reason": "Market rejected the gap fill",
                "action": "Cancel setup, look elsewhere",
                "severity": "MEDIUM"
            })

        elif state == "IN_FVG":
            conditions.append({
                "condition": f"Price closes ABOVE gap top (${gap_top + 5:.2f})",
                "reason": "Broke through FVG without rejecting",
                "action": "Setup FAILED. FVG not respected.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "Strong bullish candles through FVG",
                "reason": "No sellers showing up, buyers in control",
                "action": "Cancel setup immediately",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": "Price consolidates in FVG for 4+ candles",
                "reason": "Weak seller conviction",
                "action": "Setup losing probability",
                "severity": "MEDIUM"
            })

        # Universal FVG invalidations
        conditions.append({
            "condition": "FVG gets filled completely and price continues UP",
            "reason": "Gap filled but sellers didn't step in",
            "action": "Setup failed - no reaction at FVG",
            "severity": "CRITICAL"
        })

        return conditions

    def _supply_zone_invalidation(self, setup: Dict, state: str, key_level: float) -> List[Dict[str, Any]]:
        """Dynamic invalidation for SUPPLY ZONE pattern (BEARISH resistance zone)"""
        conditions = []

        state = setup.get("state", "AT_ZONE")

        # State-specific invalidations
        if state == "AT_ZONE":
            conditions.append({
                "condition": f"Price closes ABOVE ${key_level + 10:.2f}",
                "reason": "Resistance broken - supply zone failed",
                "action": "Cancel setup immediately. Look for LONG instead.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "No bearish reaction within 3 candles",
                "reason": "Sellers not showing up at resistance",
                "action": "Zone may be weak, wait for confirmation",
                "severity": "HIGH"
            })

        elif state == "WATCHING":
            conditions.append({
                "condition": f"Price closes decisively ABOVE ${key_level + 10:.2f}",
                "reason": "Failed resistance - breakout occurring",
                "action": "Setup INVALID. Cancel immediately.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "Bullish candles forming at resistance",
                "reason": "Buyers overwhelming sellers",
                "action": "Setup failing, cancel and reverse bias",
                "severity": "HIGH"
            })

        elif state == "REJECTION_CONFIRMED":
            conditions.append({
                "condition": f"Next candle closes ABOVE ${key_level:.2f}",
                "reason": "Rejection failed - false signal",
                "action": "Do NOT enter. Setup collapsed.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "Strong bullish candle after rejection",
                "reason": "Sellers gave up, buyers took control",
                "action": "Cancel setup, look for longs",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": "Low volume/Asian session starts",
                "reason": "Low liquidity = unreliable moves",
                "action": "Skip entry or reduce size",
                "severity": "MEDIUM"
            })

        # Universal invalidations
        conditions.append({
            "condition": f"Price rises ${(key_level * 0.01):.2f}+ ABOVE resistance",
            "reason": "Major breakout (1% above resistance)",
            "action": "Supply zone completely failed. Look elsewhere.",
            "severity": "CRITICAL"
        })
        conditions.append({
            "condition": "Multiple tests without breakdown",
            "reason": "Resistance being absorbed, zone weakening",
            "action": "Setup probability decreasing, consider canceling",
            "severity": "HIGH"
        })
        conditions.append({
            "condition": "Price consolidates AT resistance for 4+ candles",
            "reason": "No seller conviction, likely to break UP",
            "action": "Setup losing strength, prepare to cancel",
            "severity": "MEDIUM"
        })

        return conditions

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
async def get_bearish_pro_trader_analysis() -> Dict[str, Any]:
    """Get BEARISH professional trader analysis for XAUUSD (SELL setups only)"""
    system = BearishProTraderGold()
    return await system.get_detailed_setup()
