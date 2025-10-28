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
        # Track active OB touches to persist entry signals
        self.active_ob_touch = None  # Will store: {"ob_zone": {...}, "touched_at": timestamp, "entry_valid_until": timestamp}

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
            h1_setup = await self._detect_setup_pattern(h1_data, h4_levels, current_price)

            # Get current candle details
            current_candle = await self._get_current_candle_info(h1_data, current_price)

            # Build complete response
            return {
                "status": "success",
                "pair": self.pair,
                "current_price": current_price,
                "setup_status": h1_setup["state"],
                "setup_progress": h1_setup["progress"],
                "pattern_type": h1_setup["pattern_type"],

                # Step-by-step breakdown
                "setup_steps": self._build_setup_steps(h1_setup, current_price, current_candle, h1_data),

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

    async def _detect_setup_pattern(self, h1_data: pd.DataFrame, h4_levels: Dict, current_price: float) -> Dict[str, Any]:
        """
        Detect which professional setup pattern is forming
        Returns detailed pattern information
        """
        key_level = h4_levels["key_level"]
        last_candles = h1_data.tail(20)  # Increased for FVG detection

        # Get current candle info to check if high touched key zones
        current_candle_info = await self._get_current_candle_info(h1_data, current_price)
        current_candle_high = current_candle_info.get("high")

        # Check for FAIR VALUE GAP (FVG) - HIGHEST PRIORITY
        fvg_setup = self._check_fvg(last_candles, current_price)
        if fvg_setup["detected"]:
            return fvg_setup

        # Check for ORDER BLOCK - SECOND PRIORITY
        ob_setup = self._check_order_block(last_candles, current_price, current_candle_high)
        if ob_setup["detected"]:
            return ob_setup

        # Check for BREAKDOWN RETEST pattern
        breakdown_setup = self._check_breakout_retest(last_candles, key_level, current_price, current_candle_high)
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

    def _check_breakout_retest(self, candles: pd.DataFrame, key_level: float, current_price: float, current_candle_high: float = None) -> Dict[str, Any]:
        """
        Check for BREAKDOWN Retest pattern (BEARISH):
        1. Price was above support
        2. Price broke BELOW support (breakdown)
        3. Price pulled back UP to test old support (now resistance)
        4. Looking for bearish rejection + confirmation

        Args:
            candles: Historical H1 candles
            key_level: The support/resistance level
            current_price: Current market price
            current_candle_high: High of the forming candle (from M5 data)
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

        # First check M5 precision: did current candle's high touch the level?
        candle_touched_level = False
        if current_candle_high is not None:
            distance_to_level = key_level - current_candle_high  # Positive if price below level
            if distance_to_level <= 3 and distance_to_level >= -2:  # Within 3 pips
                candle_touched_level = True
                retest_happening = True

        # Also check closed H1 candles after breakdown
        if len(candles_after_breakdown) > 0:
            # Retest = price returned UP close to key level (from below)
            # MUST actually reach within 3 pips of the level to count as a retest
            for i, candle in candles_after_breakdown.iterrows():
                distance_to_level = key_level - candle['high']  # Positive if price below level

                # Only count as retest if HIGH actually reached within 3 pips of resistance
                if distance_to_level <= 3 and distance_to_level >= -2:  # Within 3 pips below or 2 pips above
                    retest_happening = True

                    # Check for BEARISH rejection (long UPPER wick + close BELOW)
                    wick_size = candle['high'] - max(candle['open'], candle['close'])
                    # wick_size > 0.3 = 3 pips, close < key_level - 0.5 = 5 pips below
                    if wick_size > 0.3 and candle['close'] < key_level - 0.5:
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

    def _check_order_block(self, candles: pd.DataFrame, current_price: float, current_candle_high: float = None) -> Dict[str, Any]:
        """
        Check for BEARISH Order Block pattern:

        Order Block = Last BEARISH candle before a strong DOWNWARD move
        This is where institutions (smart money) placed their SELL orders.
        When price returns UP to this level, those orders get filled = rejection DOWN.

        Args:
            candles: Historical H1 candles
            current_price: Current market price
            current_candle_high: High of the forming candle (from M5 data)
        """
        if len(candles) < 10:
            return {"detected": False}

        order_blocks = []

        # Scan for strong bearish moves and identify their order blocks
        for i in range(5, len(candles) - 2):
            strong_move = False

            # Method 1: Check for 3 consecutive bearish candles
            if (candles.iloc[i]['close'] < candles.iloc[i]['open'] and
                candles.iloc[i+1]['close'] < candles.iloc[i+1]['open'] and
                candles.iloc[i+2]['close'] < candles.iloc[i+2]['open']):

                move_size = candles.iloc[i]['open'] - candles.iloc[i+2]['close']
                if move_size > 15:  # Significant move (15+ pips)
                    strong_move = True

            # Method 2: Check for single large bearish candle
            if not strong_move:
                single_candle_move = candles.iloc[i]['open'] - candles.iloc[i]['close']
                if single_candle_move > 20:  # Large single candle (20+ pips)
                    strong_move = True

            if strong_move:
                # Find the LAST bearish candle BEFORE this move
                for j in range(i-1, max(0, i-5), -1):
                    prev_candle = candles.iloc[j]
                    if prev_candle['close'] < prev_candle['open']:  # Bearish candle
                        ob_top = prev_candle['open']
                        ob_bottom = prev_candle['close']
                        ob_midpoint = (ob_top + ob_bottom) / 2
                        ob_size = ob_top - ob_bottom

                        if ob_size > 3:  # Only significant order blocks (3+ pips)
                            # Check if this OB has been mitigated
                            mitigated = False
                            for k in range(j+1, len(candles)):
                                if candles.iloc[k]['close'] > ob_top:
                                    mitigated = True
                                    break

                            if not mitigated:
                                order_blocks.append({
                                    "top": float(ob_top),
                                    "bottom": float(ob_bottom),
                                    "midpoint": float(ob_midpoint),
                                    "size": float(ob_size),
                                    "candle_index": j,
                                    "age_candles": len(candles) - j - 1
                                })
                        break

        # No unmitigated order blocks found
        if not order_blocks:
            return {"detected": False}

        # Find nearest OB above current price (price needs to rise to test it)
        obs_above = [ob for ob in order_blocks if ob["bottom"] > current_price]

        if not obs_above:
            return {"detected": False}

        # Get the nearest OB (closest to current price)
        nearest_ob = min(obs_above, key=lambda x: abs(current_price - x["midpoint"]))

        # Check if price is approaching the OB (within 20 pips)
        distance_to_ob = nearest_ob["midpoint"] - current_price

        # Check for active OB touch from previous detection
        now_utc = datetime.now(pytz.UTC)
        has_active_touch = False

        if self.active_ob_touch is not None:
            # Check if the active touch is still valid and matches THIS EXACT OB zone
            # Must match within 0.5 pips to prevent different OBs from sharing touch state
            if (now_utc < self.active_ob_touch["entry_valid_until"] and
                abs(self.active_ob_touch["ob_zone"]["top"] - nearest_ob["top"]) < 0.5 and
                abs(self.active_ob_touch["ob_zone"]["bottom"] - nearest_ob["bottom"]) < 0.5):
                has_active_touch = True
            else:
                # Different OB detected - clear the old touch
                self.active_ob_touch = None
                has_active_touch = False

        # Check if current candle's high touched the OB zone
        candle_touched_ob = False
        if current_candle_high is not None:
            # Check if the high of current candle went into or above the OB zone
            if current_candle_high >= nearest_ob["bottom"]:
                candle_touched_ob = True

        # Determine state based on proximity
        if current_price >= nearest_ob["bottom"] and current_price <= nearest_ob["top"]:
            state = "IN_ORDER_BLOCK"
            progress = "3/5"
            confirmations = 2

            # Save this touch - entry valid for 2 hours
            self.active_ob_touch = {
                "ob_zone": nearest_ob,
                "touched_at": now_utc,
                "entry_valid_until": now_utc + timedelta(hours=2)
            }

        elif candle_touched_ob:
            # Current candle's high touched the OB zone (even if price pulled back down)
            state = "IN_ORDER_BLOCK"
            progress = "3/5"
            confirmations = 2

            # Save this touch - entry valid for 2 hours
            self.active_ob_touch = {
                "ob_zone": nearest_ob,
                "touched_at": now_utc,
                "entry_valid_until": now_utc + timedelta(hours=2)
            }

        elif has_active_touch:
            # OB was touched recently - check if price is still reasonably close
            if distance_to_ob <= 30:
                # Price is still close enough (within 30 pips) - entry valid
                state = "IN_ORDER_BLOCK"
                progress = "3/5"
                confirmations = 2
            else:
                # Price moved too far away (>30 pips) - entry no longer valid
                # Clear the active touch and show as detected only
                self.active_ob_touch = None
                state = "DETECTED"
                progress = "1/5"
                confirmations = 0

        elif distance_to_ob <= 10:
            state = "APPROACHING"
            progress = "2/5"
            confirmations = 1
        elif distance_to_ob <= 20:
            state = "DETECTED"
            progress = "1/5"
            confirmations = 0
        else:
            # Too far away, no active setup
            return {"detected": False}

        return {
            "detected": True,
            "pattern_type": "ORDER_BLOCK",
            "direction": "SHORT",
            "state": state,
            "progress": progress,
            "key_level": nearest_ob["midpoint"],
            "confirmations": confirmations,
            "expected_entry": nearest_ob["midpoint"],
            "order_block_zone": {
                "top": nearest_ob["top"],
                "bottom": nearest_ob["bottom"],
                "midpoint": nearest_ob["midpoint"],
                "size_pips": nearest_ob["size"],
                "age_candles": nearest_ob["age_candles"]
            },
            "distance_pips": float(distance_to_ob)
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

    async def _get_current_candle_info(self, h1_data: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Get details about the currently forming 1H candle
        Uses M5 candles to get accurate high/low instead of just estimating
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

        # Current candle open
        current_candle_open = float(last_closed_candle['close'])

        # Get REAL high/low from M5 candles of current hour (including forming candles)
        try:
            from .oanda_feed import get_forming_candles
            m5_data = await get_forming_candles(granularity="M5", count=20)

            if m5_data is not None and not m5_data.empty:
                # Filter M5 candles to only include those from current hour (>= current_hour_start_utc)
                m5_data_utc = m5_data.tz_convert('UTC') if m5_data.index.tz is not None else m5_data.tz_localize('UTC')
                current_hour_m5 = m5_data_utc[m5_data_utc.index >= current_hour_start_utc]

                if len(current_hour_m5) > 0:
                    # Get actual high and low from M5 data of CURRENT hour only
                    actual_high = float(current_hour_m5['high'].max())
                    actual_low = float(current_hour_m5['low'].min())

                    # Include current price in case it's higher/lower than M5 data
                    current_high = max(actual_high, current_price)
                    current_low = min(actual_low, current_price)
                else:
                    # No M5 data for current hour yet, use estimation
                    current_high = max(current_price, current_candle_open)
                    current_low = min(current_price, current_candle_open)
            else:
                # Fallback to estimation if M5 data unavailable
                current_high = max(current_price, current_candle_open)
                current_low = min(current_price, current_candle_open)
        except Exception as e:
            print(f"Error fetching M5 for current candle: {e}")
            # Fallback to estimation
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

    def _build_setup_steps(self, setup: Dict, current_price: float, current_candle: Dict, h1_data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Build step-by-step breakdown based on pattern type (BEARISH)
        """
        pattern = setup.get("pattern_type", "SCANNING")

        if pattern == "FAIR_VALUE_GAP":
            return self._fvg_steps(setup, current_price)
        elif pattern == "ORDER_BLOCK":
            return self._order_block_steps(setup, current_price, h1_data)
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

    def _calculate_ob_sl_tp(self, ob_zone: Dict, current_price: float, h1_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Calculate Stop Loss and Take Profit for Order Block trade (BEARISH/SHORT)

        SL: 5 pips above OB top (OB invalidation level)
        TP: Next support level or 1:2 R:R minimum
        """
        ob_bottom = ob_zone.get('bottom', 0)
        ob_top = ob_zone.get('top', 0)
        ob_midpoint = ob_zone.get('midpoint', 0)

        # Stop Loss: 5 pips above OB top
        stop_loss = ob_top + 5

        # Find next support level from H1 data
        take_profit = None
        if h1_data is not None and not h1_data.empty:
            # Look for swing lows below current price in recent data
            recent_candles = h1_data.tail(50)  # Last 50 hours

            # Find lows below current price
            support_levels = []
            for i in range(1, len(recent_candles) - 1):
                low = recent_candles.iloc[i]['low']
                prev_low = recent_candles.iloc[i-1]['low']
                next_low = recent_candles.iloc[i+1]['low']

                # Swing low: lower than neighbors and below current price
                if low < prev_low and low < next_low and low < current_price:
                    support_levels.append(low)

            if support_levels:
                # Take the nearest support below current price
                take_profit = max(support_levels)

        # If no support found or too close, use 1:2 R:R minimum
        risk = stop_loss - ob_midpoint
        min_tp_1_2 = ob_midpoint - (risk * 2)  # 1:2 R:R from midpoint entry

        if take_profit is None or take_profit > min_tp_1_2:
            take_profit = min_tp_1_2

        return {
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "risk_pips": float(stop_loss - ob_midpoint),
            "reward_pips": float(ob_midpoint - take_profit),
            "risk_reward_ratio": float((ob_midpoint - take_profit) / (stop_loss - ob_midpoint)) if (stop_loss - ob_midpoint) > 0 else 0
        }

    def _order_block_steps(self, setup: Dict, current_price: float, h1_data: pd.DataFrame = None) -> List[Dict[str, Any]]:
        """
        Create detailed steps for ORDER BLOCK pattern - BEARISH
        Order Block = Last bearish candle before strong move = Institutional sell orders
        When price returns UP to this level, those orders get filled = rejection DOWN
        """
        state = setup.get("state", "DETECTED")
        ob_zone = setup.get("order_block_zone", {})
        distance = setup.get("distance_pips", 0)

        steps = []

        # Step 1: Order Block Detected
        steps.append({
            "step": 1,
            "status": "complete" if state in ["APPROACHING", "IN_ORDER_BLOCK"] else "in_progress",
            "title": "📦 Order Block Detected",
            "details": f"OB Zone: ${ob_zone.get('bottom', 0):.2f} - ${ob_zone.get('top', 0):.2f} ({ob_zone.get('size_pips', 0):.1f} pips)",
            "explanation": f"Institutional SELL orders placed {ob_zone.get('age_candles', 0)} candles ago. Currently {distance:.1f} pips away."
        })

        # Step 2: Price Approaching or Inside OB
        if state == "APPROACHING":
            steps.append({
                "step": 2,
                "status": "in_progress",
                "title": "⚠️ Price Approaching Order Block",
                "details": f"Distance to OB: {distance:.1f} pips (within 10 pips)",
                "explanation": "Price is rising toward the institutional order zone. Watch for bearish rejection reaction."
            })
        elif state == "IN_ORDER_BLOCK":
            steps.append({
                "step": 2,
                "status": "complete",
                "title": "✅ Price Touched Order Block",
                "details": f"OB zone was touched (price dropped {distance:.1f} pips from zone)",
                "explanation": "Price reached the institutional order zone and rejected. Entry signal is active."
            })
        else:
            steps.append({
                "step": 2,
                "status": "waiting",
                "title": "Waiting for price to approach OB",
                "details": f"Still {distance:.1f} pips away (need within 10 pips)",
                "explanation": "Monitoring price action as it moves toward order block."
            })

        # Step 3: Entry Strategy (only when IN_ORDER_BLOCK)
        if state == "IN_ORDER_BLOCK":
            # Calculate SL/TP
            sl_tp = self._calculate_ob_sl_tp(ob_zone, current_price, h1_data)

            steps.append({
                "step": 3,
                "status": "ready",
                "title": "🎯 READY TO ENTER",
                "entry_options": [
                    {
                        "type": "Option A: Enter at OB midpoint (Aggressive)",
                        "entry": f"${ob_zone.get('midpoint', 0):.2f}",
                        "stop_loss": f"${sl_tp['stop_loss']:.2f}",
                        "take_profit": f"${sl_tp['take_profit']:.2f}",
                        "risk_pips": f"{sl_tp['risk_pips']:.1f} pips",
                        "reward_pips": f"{sl_tp['reward_pips']:.1f} pips",
                        "risk_reward": f"1:{sl_tp['risk_reward_ratio']:.1f}",
                        "trigger": f"SELL NOW at ${ob_zone.get('midpoint', 0):.2f}",
                        "pros": "Best entry price, highest R:R",
                        "cons": "No confirmation candle",
                        "why_sl": f"SL at ${sl_tp['stop_loss']:.2f} - If price closes above OB zone, setup is invalidated",
                        "why_tp": f"TP at ${sl_tp['take_profit']:.2f} - Next support level or 1:2 R:R minimum"
                    },
                    {
                        "type": "Option B: Wait for bearish confirmation (Conservative)",
                        "entry": f"Wait for close (est. ${ob_zone.get('bottom', 0):.2f})",
                        "stop_loss": f"${sl_tp['stop_loss']:.2f}",
                        "take_profit": f"${sl_tp['take_profit']:.2f}",
                        "risk_pips": f"{sl_tp['risk_pips'] + 5:.1f} pips (slightly more)",
                        "reward_pips": f"{sl_tp['reward_pips'] - 5:.1f} pips (slightly less)",
                        "risk_reward": f"1:{(sl_tp['reward_pips'] - 5) / (sl_tp['risk_pips'] + 5):.1f}",
                        "trigger": "Wait for 1H bearish candle to close inside OB zone",
                        "pros": "Confirmation of sellers stepping in",
                        "cons": "Slightly worse entry price",
                        "why_sl": f"SL at ${sl_tp['stop_loss']:.2f} - If price closes above OB zone, setup is invalidated",
                        "why_tp": f"TP at ${sl_tp['take_profit']:.2f} - Next support level or 1:2 R:R minimum"
                    }
                ],
                "recommendation": "Option A for aggressive (OB midpoint entry), Option B for conservative (wait for confirmation)",
                "explanation": "Order Block activated! Institutions left SELL orders here. High probability of rejection DOWN."
            })
        else:
            steps.append({
                "step": 3,
                "status": "waiting",
                "title": "Entry criteria",
                "details": "Waiting for price to reach order block zone",
                "explanation": "Entry signal will trigger when price enters the OB."
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
        elif pattern_type == "ORDER_BLOCK":
            return self._order_block_invalidation(setup, state, key_level)
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

    def _order_block_invalidation(self, setup: Dict, state: str, key_level: float) -> List[Dict[str, Any]]:
        """Dynamic invalidation for ORDER BLOCK pattern - BEARISH"""
        ob_zone = setup.get("order_block_zone", {})
        ob_top = ob_zone.get("top", key_level + 3)

        conditions = []

        state = setup.get("state", "DETECTED")

        # State-specific invalidations
        if state == "DETECTED":
            conditions.append({
                "condition": f"Price continues DOWN without reaching OB (${ob_top:.2f})",
                "reason": "Order Block may not be tested this time",
                "action": "Wait or look for other setups",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": "Order Block becomes too old (15+ candles)",
                "reason": "Older order blocks less likely to be respected",
                "action": "Consider setup stale",
                "severity": "MEDIUM"
            })

        elif state == "APPROACHING":
            conditions.append({
                "condition": f"Price reverses DOWN before reaching OB top (${ob_top:.2f})",
                "reason": "Failed to test the order block completely",
                "action": "Setup invalidated - OB not tested",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": "Strong bearish momentum away from OB",
                "reason": "Market rejected the pullback",
                "action": "Cancel setup, look elsewhere",
                "severity": "MEDIUM"
            })

        elif state == "IN_OB":
            conditions.append({
                "condition": f"Price closes ABOVE OB top (${ob_top + 3:.2f})",
                "reason": "Broke through order block = mitigated/absorbed",
                "action": "Setup FAILED. Institutional orders absorbed.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "Strong bullish candles through OB",
                "reason": "No sellers showing up, buyers overwhelming",
                "action": "Cancel setup immediately",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": "Price consolidates in OB for 4+ candles",
                "reason": "Weak seller conviction at order block",
                "action": "Setup losing probability",
                "severity": "MEDIUM"
            })

        # Universal ORDER BLOCK invalidations
        conditions.append({
            "condition": "OB gets tested and price continues UP",
            "reason": "Order block tested but sellers didn't step in",
            "action": "Setup failed - OB not respected",
            "severity": "CRITICAL"
        })
        conditions.append({
            "condition": "Previous H4 resistance breaks (strong bullish structure)",
            "reason": "Overall market structure turned bullish",
            "action": "All bearish setups invalid",
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
