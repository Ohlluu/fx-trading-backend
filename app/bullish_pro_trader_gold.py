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
            h1_setup = await self._detect_setup_pattern(h1_data, h4_levels, current_price)

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
        BULLISH TRADER: Identify key SUPPORT levels using REAL 4H candles
        Only looks for support BELOW price (potential BUY zones)
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

        # Find significant support levels ONLY (bullish trader looks for BUY opportunities)
        support_candidates = []

        # Find local lows using 4H candles
        for i in range(5, len(lows) - 5):
            # Support: local low
            if lows.iloc[i] == lows.iloc[i-5:i+5].min():
                support_candidates.append(float(lows.iloc[i]))

        # Get support levels BELOW current price (potential BUY zones)
        support = sorted([s for s in support_candidates if s < current_price], reverse=True)[:3] if support_candidates else []

        # BULLISH TRADER: Always prefer support (below price)
        if support:
            key_level = support[0]  # Closest support below price
            level_type = "support"
        else:
            # If no support below, use recent low as potential support
            key_level = float(recent_data['low'].tail(20).min())
            level_type = "support"

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
            "resistance_levels": [],  # Bullish trader doesn't track resistance
            "support_levels": support if support else [key_level],
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
        last_candles = h1_data.tail(20)  # Need more candles for FVG detection

        # Check for FAIR VALUE GAP (FVG) - HIGHEST PRIORITY
        # FVG = imbalance/gap in price action that price returns to fill
        fvg_setup = self._check_fvg(last_candles, current_price)
        if fvg_setup["detected"]:
            return fvg_setup

        # Check for ORDER BLOCK - SECOND PRIORITY
        # Order Block = last bullish candle before strong move (institutional orders)
        ob_setup = await self._check_order_block(last_candles, current_price)
        if ob_setup["detected"]:
            return ob_setup

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

    def _check_fvg(self, candles: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Check for BULLISH Fair Value Gap (FVG) pattern:

        FVG = 3-candle pattern where there's a GAP (imbalance) in price:
        - Candle 1: Down move (creates the low)
        - Candle 2: BIG bullish move (skips price levels)
        - Candle 3: Continuation up (creates the high)

        If Candle3.low > Candle1.high → There's a GAP!
        The gap zone is unfilled price area where NO trades occurred
        Price is magnetically pulled back to "fill" this gap

        BULLISH FVG = Gap below current price (price returns DOWN to fill it, then bounces UP)
        """
        if len(candles) < 10:
            return {"detected": False}

        # Scan last 15 candles for FVG patterns
        fvg_zones = []

        for i in range(len(candles) - 3):
            candle1 = candles.iloc[i]
            candle2 = candles.iloc[i + 1]
            candle3 = candles.iloc[i + 2]

            # BULLISH FVG detection:
            # Gap exists if candle3.low > candle1.high (price skipped the middle zone)
            if candle3['low'] > candle1['high']:
                gap_top = candle3['low']
                gap_bottom = candle1['high']
                gap_size = gap_top - gap_bottom

                # Only consider significant gaps (> 5 pips)
                if gap_size > 5:
                    gap_midpoint = (gap_top + gap_bottom) / 2

                    # Check if gap has been filled already
                    filled = False
                    for j in range(i + 3, len(candles)):
                        if candles.iloc[j]['low'] <= gap_bottom:
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

        # Find nearest FVG below current price (price needs to drop to fill it)
        fvgs_below = [fvg for fvg in fvg_zones if fvg["top"] < current_price]

        if not fvgs_below:
            return {"detected": False}

        # Get the nearest FVG (closest to current price)
        nearest_fvg = min(fvgs_below, key=lambda x: abs(current_price - x["midpoint"]))

        # Check if price is approaching the FVG (within 20 pips)
        distance_to_fvg = current_price - nearest_fvg["midpoint"]

        if distance_to_fvg > 20:
            return {"detected": False}  # Too far away

        # Determine state based on proximity and price action
        if current_price <= nearest_fvg["top"] and current_price >= nearest_fvg["bottom"]:
            # Price is INSIDE the gap (filling it now!)
            state = "IN_FVG"
            progress = "3/5"
            confirmations = 2
        elif distance_to_fvg <= 10:
            # Price is very close (within 10 pips)
            state = "APPROACHING"
            progress = "2/5"
            confirmations = 1
        else:
            # Price is nearby (10-20 pips away)
            state = "DETECTED"
            progress = "1/5"
            confirmations = 0

        return {
            "detected": True,
            "pattern_type": "FAIR_VALUE_GAP",
            "direction": "LONG",
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

    async def _check_ob_zone_touched_m5(self, ob_top: float, ob_bottom: float, lookback_hours: int = 3) -> bool:
        """
        Check if Order Block zone was touched using 5-minute candles for precision
        Looks back N hours to see if any 5-min candle low touched the OB zone

        Args:
            ob_top: Top of the OB zone
            ob_bottom: Bottom of the OB zone
            lookback_hours: How many hours to check (default 3)

        Returns:
            True if any 5-min candle touched the zone, False otherwise
        """
        try:
            from .datafeed import fetch_h1

            # Fetch 5-minute candles (12 candles per hour * lookback_hours)
            count = 12 * lookback_hours
            m5_data = await fetch_h1("XAUUSD", timeframe="M5")

            if m5_data is None or m5_data.empty:
                return False

            # Check last N candles for touches
            recent_m5 = m5_data.tail(count)

            # Check if any 5-min candle low touched or entered the OB zone
            for _, candle in recent_m5.iterrows():
                if candle['low'] <= ob_top:
                    # Price touched or entered the zone
                    return True

            return False

        except Exception as e:
            print(f"Error checking M5 data: {e}")
            return False

    async def _check_order_block(self, candles: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Check for BULLISH Order Block pattern:

        Order Block = Last BULLISH candle before a strong UPWARD move
        This is where institutions (smart money) placed their BUY orders.
        When price returns to this level, those orders get filled = bounce UP.

        BULLISH ORDER BLOCK:
        1. Find a strong bullish move (e.g., 3+ consecutive bullish candles or big single candle)
        2. Identify the LAST bullish candle BEFORE that move
        3. That candle's body (open to close) is the Order Block zone
        4. When price returns to that zone, expect bounce UP
        """
        if len(candles) < 10:
            return {"detected": False}

        order_blocks = []

        # Scan for strong bullish moves and identify their order blocks
        for i in range(5, len(candles) - 2):
            # Check if there's a strong bullish move starting at candle i
            strong_move = False

            # Method 1: Check for 3 consecutive bullish candles
            if (candles.iloc[i]['close'] > candles.iloc[i]['open'] and
                candles.iloc[i+1]['close'] > candles.iloc[i+1]['open'] and
                candles.iloc[i+2]['close'] > candles.iloc[i+2]['open']):

                # Calculate total move size
                move_size = candles.iloc[i+2]['close'] - candles.iloc[i]['open']
                if move_size > 15:  # Significant move (15+ pips)
                    strong_move = True

            # Method 2: Check for single large bullish candle
            if not strong_move:
                single_candle_move = candles.iloc[i]['close'] - candles.iloc[i]['open']
                if single_candle_move > 20:  # Large single candle (20+ pips)
                    strong_move = True

            if strong_move:
                # Find the LAST bullish candle BEFORE this move
                # Look back from candle (i-1)
                for j in range(i-1, max(0, i-5), -1):
                    prev_candle = candles.iloc[j]
                    if prev_candle['close'] > prev_candle['open']:  # Bullish candle
                        # This is the Order Block!
                        ob_top = prev_candle['close']
                        ob_bottom = prev_candle['open']
                        ob_midpoint = (ob_top + ob_bottom) / 2
                        ob_size = ob_top - ob_bottom

                        if ob_size > 3:  # Only significant order blocks (3+ pips)
                            # Check if this OB has been mitigated (price returned and closed below it)
                            mitigated = False
                            for k in range(j+1, len(candles)):
                                if candles.iloc[k]['close'] < ob_bottom:
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
                        break  # Found the order block for this move

        # No order blocks found
        if not order_blocks:
            return {"detected": False}

        # Find order blocks BELOW current price (price needs to drop to reach them)
        obs_below = [ob for ob in order_blocks if ob["top"] < current_price]

        if not obs_below:
            return {"detected": False}

        # Get the nearest order block (closest to current price)
        nearest_ob = min(obs_below, key=lambda x: abs(current_price - x["midpoint"]))

        # Check if price is approaching the OB (within 20 pips)
        distance_to_ob = current_price - nearest_ob["midpoint"]

        if distance_to_ob > 20:
            return {"detected": False}  # Too far away

        # Check 5-minute data for precise touches (only if not already inside the zone)
        m5_touched = False
        if current_price > nearest_ob["top"]:
            # Price is above the zone on H1 data, but check M5 for precise touches
            m5_touched = await self._check_ob_zone_touched_m5(nearest_ob["top"], nearest_ob["bottom"], lookback_hours=3)

        # Determine state based on proximity and M5 data
        if current_price <= nearest_ob["top"] and current_price >= nearest_ob["bottom"]:
            # Price is INSIDE the order block (filling orders now!)
            state = "IN_ORDER_BLOCK"
            progress = "3/5"
            confirmations = 2
        elif m5_touched:
            # M5 data shows zone was touched (upgrade state)
            state = "IN_ORDER_BLOCK"
            progress = "3/5"
            confirmations = 2
        elif distance_to_ob <= 10:
            # Price is very close (within 10 pips)
            state = "APPROACHING"
            progress = "2/5"
            confirmations = 1
        else:
            # Price is nearby (10-20 pips away)
            state = "DETECTED"
            progress = "1/5"
            confirmations = 0

        return {
            "detected": True,
            "pattern_type": "ORDER_BLOCK",
            "direction": "LONG",
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

        if pattern == "FAIR_VALUE_GAP":
            return self._fvg_steps(setup, current_price)
        elif pattern == "ORDER_BLOCK":
            return self._order_block_steps(setup, current_price)
        elif pattern == "BREAKOUT_RETEST":
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

    def _fvg_steps(self, setup: Dict, current_price: float) -> List[Dict[str, Any]]:
        """
        Create detailed steps for FAIR VALUE GAP (FVG) pattern
        FVG = Price imbalance that acts like a magnet, pulling price back to fill the gap
        """
        state = setup.get("state", "DETECTED")
        fvg_zone = setup.get("fvg_zone", {})
        distance = setup.get("distance_pips", 0)

        steps = []

        # Step 1: FVG Detected
        steps.append({
            "step": 1,
            "status": "complete",
            "title": f"Fair Value Gap detected at ${fvg_zone.get('midpoint', 0):.2f}",
            "details": f"Gap size: {fvg_zone.get('size_pips', 0):.1f} pips | Age: {fvg_zone.get('age_candles', 0)} candles ago",
            "zone": {
                "top": f"${fvg_zone.get('top', 0):.2f}",
                "midpoint": f"${fvg_zone.get('midpoint', 0):.2f}",
                "bottom": f"${fvg_zone.get('bottom', 0):.2f}"
            },
            "explanation": "FVG is an IMBALANCE in price where no trades occurred. Price is magnetically pulled back to fill this gap."
        })

        # Step 2: Price Approaching or In FVG
        if state == "IN_FVG":
            steps.append({
                "step": 2,
                "status": "in_progress",
                "title": "🎯 Price INSIDE the Fair Value Gap!",
                "details": f"Current: ${current_price:.2f} | Gap: ${fvg_zone.get('bottom', 0):.2f}-${fvg_zone.get('top', 0):.2f}",
                "watching_for": {
                    "bullish_reaction": {
                        "text": "Bullish candle forming (buyers stepping in)",
                        "explanation": "Gap is being filled, buyers should defend here"
                    },
                    "wick_rejection": {
                        "text": "Lower wick rejection (sellers fail)",
                        "explanation": "Long lower wick = buyers absorbing selling pressure"
                    },
                    "volume_increase": {
                        "text": "Volume spike (institutions active)",
                        "explanation": "High volume = smart money filling orders at this level"
                    }
                },
                "explanation": "Price reached the FVG! This is where buyers typically step in. Watch for bullish rejection."
            })
        elif state == "APPROACHING":
            steps.append({
                "step": 2,
                "status": "in_progress",
                "title": f"⏳ Price approaching FVG ({distance:.1f} pips away)",
                "details": f"Current: ${current_price:.2f} → Target: ${fvg_zone.get('midpoint', 0):.2f}",
                "explanation": "Price is dropping toward the gap. High probability it will fill the FVG before reversing UP."
            })
        else:  # DETECTED
            steps.append({
                "step": 2,
                "status": "waiting",
                "title": f"FVG detected {distance:.1f} pips below price",
                "details": f"Waiting for price to drop from ${current_price:.2f} to ${fvg_zone.get('midpoint', 0):.2f}",
                "explanation": "FVG acts like a magnet. Price often retraces to fill gaps before continuing."
            })

        # Step 3: Entry Setup
        if state == "IN_FVG":
            steps.append({
                "step": 3,
                "status": "ready",
                "title": "🎯 READY TO ENTER (FVG Fill Play)",
                "entry_options": [
                    {
                        "type": "Option A: Enter at FVG midpoint",
                        "trigger": f"BUY at ${fvg_zone.get('midpoint', 0):.2f} (limit order)",
                        "stop_loss": f"${fvg_zone.get('bottom', 0) - 5:.2f} (5 pips below gap)",
                        "take_profit": f"Recent highs or ${current_price + 20:.2f}+",
                        "pros": "Best price, high probability bounce",
                        "cons": "Might not fill completely"
                    },
                    {
                        "type": "Option B: Wait for bullish confirmation",
                        "trigger": "Enter after bullish rejection candle closes",
                        "stop_loss": f"${fvg_zone.get('bottom', 0) - 5:.2f}",
                        "pros": "Confirmation of buyers stepping in",
                        "cons": "Slightly worse entry price"
                    }
                ],
                "recommendation": "Option A for aggressive (FVG midpoint entry), Option B for conservative (wait for confirmation)",
                "explanation": "FVG filled! Institutions left orders here. High probability of bounce UP."
            })

        return steps

    def _order_block_steps(self, setup: Dict, current_price: float) -> List[Dict[str, Any]]:
        """
        Create detailed steps for ORDER BLOCK pattern
        Order Block = Last bullish candle before strong move = Institutional buy orders
        When price returns to this level, those orders get filled = bounce UP
        """
        state = setup.get("state", "DETECTED")
        ob_zone = setup.get("order_block_zone", {})
        distance = setup.get("distance_pips", 0)

        steps = []

        # Step 1: Order Block Detected
        steps.append({
            "step": 1,
            "status": "complete" if state in ["APPROACHING", "IN_OB"] else "in_progress",
            "title": "📦 Order Block Detected",
            "details": f"OB Zone: ${ob_zone.get('bottom', 0):.2f} - ${ob_zone.get('top', 0):.2f} ({ob_zone.get('size_pips', 0):.1f} pips)",
            "explanation": f"Institutional BUY orders placed {ob_zone.get('age_candles', 0)} candles ago. Currently {distance:.1f} pips away."
        })

        # Step 2: Price Approaching or Inside OB
        if state == "APPROACHING":
            steps.append({
                "step": 2,
                "status": "in_progress",
                "title": "⚠️ Price Approaching Order Block",
                "details": f"Distance to OB: {distance:.1f} pips (within 10 pips)",
                "explanation": "Price is dropping toward the institutional order zone. Watch for bounce reaction."
            })
        elif state == "IN_OB":
            steps.append({
                "step": 2,
                "status": "complete",
                "title": "✅ Price Inside Order Block",
                "details": f"Current: ${current_price:.2f} (inside OB zone)",
                "explanation": "Price has reached the institutional order zone. High probability of bounce."
            })
        else:
            steps.append({
                "step": 2,
                "status": "waiting",
                "title": "Waiting for price to approach OB",
                "details": f"Still {distance:.1f} pips away (need within 10 pips)",
                "explanation": "Monitoring price action as it moves toward order block."
            })

        # Step 3: Entry Strategy (only when IN_OB)
        if state == "IN_OB":
            steps.append({
                "step": 3,
                "status": "ready",
                "title": "🎯 READY TO ENTER",
                "entry_options": [
                    {
                        "type": "Option A: Enter at OB midpoint (Aggressive)",
                        "trigger": f"Enter NOW at ${ob_zone.get('midpoint', 0):.2f}",
                        "pros": "Best entry price, highest R:R",
                        "cons": "No confirmation candle"
                    },
                    {
                        "type": "Option B: Wait for bullish confirmation (Conservative)",
                        "trigger": "Wait for 1H bullish candle inside OB zone",
                        "pros": "Confirmation of buyers stepping in",
                        "cons": "Slightly worse entry price"
                    }
                ],
                "recommendation": "Option A for aggressive (OB midpoint entry), Option B for conservative (wait for confirmation)",
                "explanation": "Order Block activated! Institutions left BUY orders here. High probability of bounce UP."
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
        elif pattern_type == "BREAKOUT_RETEST":
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

    def _fvg_invalidation(self, setup: Dict, state: str, key_level: float) -> List[Dict[str, Any]]:
        """Dynamic invalidation for Fair Value Gap (FVG) pattern"""
        fvg_zone = setup.get("fvg_zone", {})
        gap_bottom = fvg_zone.get("bottom", key_level - 5)

        conditions = []

        state = setup.get("state", "DETECTED")

        # State-specific invalidations
        if state == "DETECTED":
            conditions.append({
                "condition": f"Price continues UP without filling FVG (${gap_bottom:.2f})",
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
                "condition": f"Price reverses UP before reaching FVG bottom (${gap_bottom:.2f})",
                "reason": "Failed to fill the gap completely",
                "action": "Setup invalidated - gap may not fill",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": "Strong bullish momentum away from FVG",
                "reason": "Market rejected the gap fill",
                "action": "Cancel setup, look elsewhere",
                "severity": "MEDIUM"
            })

        elif state == "IN_FVG":
            conditions.append({
                "condition": f"Price closes BELOW gap bottom (${gap_bottom - 5:.2f})",
                "reason": "Broke through FVG without bouncing",
                "action": "Setup FAILED. FVG not respected.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "Strong bearish candles through FVG",
                "reason": "No buyers showing up, sellers in control",
                "action": "Cancel setup immediately",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": "Price consolidates in FVG for 4+ candles",
                "reason": "Weak buyer conviction",
                "action": "Setup losing probability",
                "severity": "MEDIUM"
            })

        # Universal FVG invalidations
        conditions.append({
            "condition": "FVG gets filled completely and price continues DOWN",
            "reason": "Gap filled but buyers didn't step in",
            "action": "Setup failed - no reaction at FVG",
            "severity": "CRITICAL"
        })

        return conditions

    def _order_block_invalidation(self, setup: Dict, state: str, key_level: float) -> List[Dict[str, Any]]:
        """Dynamic invalidation for ORDER BLOCK pattern"""
        ob_zone = setup.get("order_block", {})
        ob_bottom = ob_zone.get("bottom", key_level - 3)

        conditions = []

        state = setup.get("state", "DETECTED")

        # State-specific invalidations
        if state == "DETECTED":
            conditions.append({
                "condition": f"Price continues UP without reaching OB (${ob_bottom:.2f})",
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
                "condition": f"Price reverses UP before reaching OB bottom (${ob_bottom:.2f})",
                "reason": "Failed to test the order block completely",
                "action": "Setup invalidated - OB not tested",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": "Strong bullish momentum away from OB",
                "reason": "Market rejected the pullback",
                "action": "Cancel setup, look elsewhere",
                "severity": "MEDIUM"
            })

        elif state == "IN_OB":
            conditions.append({
                "condition": f"Price closes BELOW OB bottom (${ob_bottom - 3:.2f})",
                "reason": "Broke through order block = mitigated/absorbed",
                "action": "Setup FAILED. Institutional orders absorbed.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "Strong bearish candles through OB",
                "reason": "No buyers showing up, sellers overwhelming",
                "action": "Cancel setup immediately",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": "Price consolidates in OB for 4+ candles",
                "reason": "Weak buyer conviction at order block",
                "action": "Setup losing probability",
                "severity": "MEDIUM"
            })

        # Universal ORDER BLOCK invalidations
        conditions.append({
            "condition": "OB gets tested and price continues DOWN",
            "reason": "Order block tested but buyers didn't step in",
            "action": "Setup failed - OB not respected",
            "severity": "CRITICAL"
        })
        conditions.append({
            "condition": "Previous H4 support breaks (strong bearish structure)",
            "reason": "Overall market structure turned bearish",
            "action": "All bullish setups invalid",
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
