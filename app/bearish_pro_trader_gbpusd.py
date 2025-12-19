#!/usr/bin/env python3
"""
Professional Trader GBP/USD System - Educational Setup Tracker
Shows EXACTLY what professional day traders look for, step-by-step
Like having a mentor explaining every detail in real-time
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import pytz
import json
import os
from .datafeed import fetch_h1
from .oanda_feed import get_current_price
from .mtf_cache import mtf_cache  # Import shared cache

class BearishProTraderGBPUSD:
    """
    BEARISH Educational trading system - looks for SELL setups
    Detects breakdowns below support and bearish rejections
    Same professional logic as bullish trader, opposite direction
    """

    def __init__(self):
        self.pair = "GBPUSD"
        self.pip_size = 0.0001  # GBP/USD: 1 pip = 0.0001 (4th decimal, same as EUR/USD)
        self.timeframes = {
            "D1": 200,  # 200 days
            "H4": 500,  # ~83 days of 4H candles (approx 83 days * 6 candles/day)
            "H1": 100   # 100 hours
        }
        # Track active OB touches to persist entry signals
        self.active_ob_touch = None  # Will store: {"ob_zone": {...}, "touched_at": timestamp, "entry_valid_until": timestamp}

        # STABILITY TRACKER: Patterns must be stable for 15 minutes before counting
        self.pattern_tracker = {
            "liquidity_grab": None,  # {"first_seen": timestamp, "data": {...}}
            "fvg": None,
            "order_block": None,
            "breakout_retest": None,
            "supply_zone": None,
            "last_entry_state": None,  # Track last entry state to prevent flickering
            "entry_state_since": None  # When current state started
        }

        # ACTIVE TRADE TRACKER: Lock in setup when 5+ confluence triggers
        # Once a trade is active, stop re-validating confluences and just monitor TP/SL
        # PERSISTED to file system so it survives across API calls
        self.active_trade_file = "/tmp/active_trade_bearish_gbpusd.json"
        self.active_trade = None  # {"setup": {...}, "trade_plan": {...}, "triggered_at": timestamp, "entry": price}

    def _to_pips(self, price_diff: float) -> float:
        """Convert price difference to pips (e.g., 0.0005 -> 5.0 pips)"""
        return abs(price_diff) / self.pip_size

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
            current_price = await get_current_price("GBP_USD")
            if current_price is None:
                current_price = float(h1_data['close'].iloc[-1])

            # LOAD ACTIVE TRADE FROM FILE (persists across API calls)
            self._load_active_trade()

            # CHECK IF ACTIVE TRADE EXISTS FIRST
            # If we have an active trade, monitor it instead of re-scanning
            if self.active_trade is not None:
                # Get current candle info FIRST
                current_candle = await self._get_current_candle_info(h1_data, current_price)
                trade_status = self._monitor_active_trade(current_price, current_candle)

                # If trade is still active, return trade monitoring data
                if trade_status["is_active"]:
                    return trade_status["response"]
                else:
                    # Trade finished (TP or SL hit), clear it and resume scanning
                    self.active_trade = None
                    self._delete_active_trade()

            # Analyze market structure across timeframes (using REAL data)
            daily_analysis = self._analyze_daily_trend(d1_data, current_price)
            h4_levels = self._identify_key_levels_h4(h4_data, current_price)
            h1_setup = await self._detect_setup_pattern(h1_data, h4_data, h4_levels, current_price)

            # CHECK IF SETUP TRIGGERED (5+ confluence)
            # If yes, lock it in as active trade
            # IMPORTANT: Only create if no active trade exists (don't overwrite locked trades)
            if h1_setup.get("total_score", 0) >= 5 and self.active_trade is None:
                trade_plan = self._build_trade_plan(h1_setup, h4_levels, current_price)
                if trade_plan.get("status") == "Ready":
                    # Save as active trade (in memory AND file)
                    self.active_trade = {
                        "setup": h1_setup,
                        "trade_plan": trade_plan,
                        "triggered_at": datetime.now(pytz.UTC).isoformat(),
                        "entry": float(trade_plan["entry_price"].replace("$", "").replace(",", "")),
                        "h4_levels": h4_levels,
                        "daily_analysis": daily_analysis
                    }
                    # Persist to file so it survives across API calls
                    self._save_active_trade()

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

                # Confluence data (NEW)
                "confluences": h1_setup.get("confluences", []),
                "total_score": h1_setup.get("total_score", 0),
                "confidence": h1_setup.get("confidence", None),
                "structure": h1_setup.get("structure", {"structure_type": "NEUTRAL", "score": 0}),

                # GATED SCORING (Professional categorization)
                "has_location": h1_setup.get("has_location", False),
                "has_trigger": h1_setup.get("has_trigger", False),
                "has_confirmation": h1_setup.get("has_confirmation", False),
                "grade": h1_setup.get("grade", "NO_TRADE"),
                "tradable": h1_setup.get("tradable", False),

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
        # Build pair-specific cache key
        cache_key = f"{self.pair}_{timeframe}"

        # Check cache first
        cached_data = mtf_cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # Fetch fresh data
        data = await fetch_h1(self.pair, timeframe=timeframe)

        # Store in cache with pair-specific key
        mtf_cache.set(cache_key, data)

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
        last_updated = mtf_cache.get_last_update(f"{self.pair}_D1")
        next_update = mtf_cache.get_next_update("D1")

        # Convert last_updated to Chicago time for display
        last_updated_str = "Just now"
        if last_updated:
            last_updated_ct = last_updated.astimezone(chicago_tz)
            last_updated_str = last_updated_ct.strftime("%b %d, %I:%M %p CT")

        return {
            "trend": trend,
            "explanation": f"Price is {'above' if trend == 'BULLISH' else 'below'} 200-day EMA ({ema_200_current:.5f})",
            "points": [
                f"Trend: {trend} (current: {current_price:.5f} vs 200 EMA: {ema_200_current:.5f})",
                f"Recent high: {recent_high:.5f} ({days_since_high} days ago)",
                f"Recent low: {recent_low:.5f} ({days_since_low} days ago)",
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

        # Find significant resistance AND support levels
        resistance_candidates = []
        support_candidates = []

        # Find local highs (resistance) and local lows (support) using 4H candles
        for i in range(5, len(highs) - 5):
            # Resistance: local high
            if highs.iloc[i] == highs.iloc[i-5:i+5].max():
                resistance_candidates.append(float(highs.iloc[i]))

            # Support: local low
            if lows.iloc[i] == lows.iloc[i-5:i+5].min():
                support_candidates.append(float(lows.iloc[i]))

        # Add psychological levels (round numbers: $3,900, $3,950, $4,000, etc.)
        # These are important because traders place orders at round numbers
        psychological_levels_resistance = []
        psychological_levels_support = []
        price_range_start = int((current_price - 0.0080) / 0.0050) * 0.0050  # Start 80 pips below
        price_range_end = int((current_price + 0.0080) / 0.0050) * 0.0050    # End 80 pips above

        for level_int in range(int(price_range_start * 10000), int(price_range_end * 10000) + 50, 50):
            level = level_int / 10000
            # Resistance: above current price
            if level > current_price and (level - current_price) <= 0.0080:
                psychological_levels_resistance.append(float(level))
            # Support: below current price
            elif level < current_price and (current_price - level) <= 0.0080:
                psychological_levels_support.append(float(level))

        # Combine swing levels + psychological levels
        all_resistance_candidates = resistance_candidates + psychological_levels_resistance
        all_support_candidates = support_candidates + psychological_levels_support

        # Remove duplicates and get resistance levels ABOVE current price (potential SHORT zones)
        resistance = sorted(list(set([r for r in all_resistance_candidates if r > current_price])))[:5] if all_resistance_candidates else []

        # Remove duplicates and get support levels BELOW current price (potential TP zones for shorts)
        support = sorted(list(set([s for s in all_support_candidates if s < current_price])), reverse=True)[:5] if all_support_candidates else []

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
        last_updated = mtf_cache.get_last_update(f"{self.pair}_H4")
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
            "support_levels": support,  # FIX: Track support for TP targets
            "distance_pips": round(distance_pips, 1),
            "last_updated": last_updated_str,
            "next_update": next_update
        }

    def _check_bos_choch(self, candles: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Check for Break of Structure (BOS) or Change of Character (CHoCH)

        BOS = Price breaks previous swing high/low in SAME direction (trend continuation)
        CHoCH = Price breaks previous swing high/low in OPPOSITE direction (trend reversal)

        For BEARISH trader:
        - Bearish BOS = Close below previous swing low (confirms downtrend)
        - Bullish CHoCH = Close above previous lower high (signals potential reversal UP)

        Returns:
            - structure_type: "BEARISH_BOS", "BULLISH_CHOCH", or "NEUTRAL"
            - score: +2 for BOS, -3 for CHoCH against trend
            - swing_level: The level that was broken
        """
        if len(candles) < 20:
            return {"structure_type": "NEUTRAL", "score": 0, "swing_level": None}

        recent_candles = candles.tail(20)

        # Identify swing highs and lows (local extremes)
        swing_highs = []
        swing_lows = []

        for i in range(5, len(recent_candles) - 5):
            candle = recent_candles.iloc[i]

            # Swing High: high is higher than 5 candles before and after
            if candle['high'] == recent_candles.iloc[i-5:i+6]['high'].max():
                swing_highs.append({
                    "level": float(candle['high']),
                    "index": i,
                    "time": candle.name
                })

            # Swing Low: low is lower than 5 candles before and after
            if candle['low'] == recent_candles.iloc[i-5:i+6]['low'].min():
                swing_lows.append({
                    "level": float(candle['low']),
                    "index": i,
                    "time": candle.name
                })

        if not swing_highs and not swing_lows:
            return {"structure_type": "NEUTRAL", "score": 0, "swing_level": None}

        # Check for recent BOS or CHoCH (last 3 candles)
        last_3_candles = recent_candles.tail(3)

        # Check for BEARISH BOS (close below previous swing low)
        if swing_lows:
            latest_swing_low = swing_lows[-1]["level"]
            for _, candle in last_3_candles.iterrows():
                if candle['close'] < latest_swing_low:
                    return {
                        "structure_type": "BEARISH_BOS",
                        "score": 2,
                        "swing_level": latest_swing_low,
                        "description": f"Bearish BOS detected! Price closed below swing low at {latest_swing_low:.5f}"
                    }

        # Check for BULLISH CHoCH (close above previous lower high)
        if len(swing_highs) >= 2:
            # Check if we're in downtrend (lower highs)
            recent_highs = swing_highs[-2:]
            if recent_highs[-1]["level"] < recent_highs[-2]["level"]:
                # We have lower highs (downtrend)
                # Now check if price broke above the most recent lower high
                latest_lower_high = recent_highs[-1]["level"]
                for _, candle in last_3_candles.iterrows():
                    if candle['close'] > latest_lower_high:
                        return {
                            "structure_type": "BULLISH_CHOCH",
                            "score": -3,  # Negative score = filter out bearish setups
                            "swing_level": latest_lower_high,
                            "description": f"⚠️ Bullish CHoCH! Price broke above lower high at {latest_lower_high:.5f} - Trend may be reversing"
                        }

        return {"structure_type": "NEUTRAL", "score": 0, "swing_level": None}

    def _check_liquidity_grab(self, candles: pd.DataFrame, current_price: float, h4_levels: Dict, current_candle_high: float = None) -> Dict[str, Any]:
        """
        Check for Liquidity Grab / Stop Hunt pattern (REAL-TIME DETECTION)

        Liquidity Grab = Price spikes beyond key level to trigger stops, then reverses sharply
        - Very high probability setup when combined with OB/FVG
        - Shows institutions filling large orders after sweeping retail stops

        For BEARISH trader, looking for:
        - Price spikes ABOVE resistance/swing high (grabs buy stops)
        - Large wick (20+ pips)
        - Immediate rejection back below the level
        - Entry on the reversal

        Args:
            candles: Historical H1 candles
            current_price: Current market price
            h4_levels: H4 support/resistance levels
            current_candle_high: High of the CURRENT forming candle (real-time detection)

        Returns:
            - detected: True if liquidity grab confirmed
            - score: +5 points (current candle), +4 points (last closed), +3 points (2 candles ago)
            - grab_level: The level where stops were grabbed
            - grab_high: Highest point of the grab
        """
        if len(candles) < 10:
            return {"detected": False, "score": 0}

        recent_candles = candles.tail(10)

        # Identify recent swing highs (local highs in last 10 candles)
        swing_highs = []
        for i in range(2, len(recent_candles) - 2):
            candle = recent_candles.iloc[i]
            window = recent_candles.iloc[max(0, i-2):min(len(recent_candles), i+3)]
            if candle['high'] == window['high'].max():
                swing_highs.append({
                    "level": float(candle['high']),
                    "index": i
                })

        if not swing_highs:
            return {"detected": False, "score": 0}

        # PRIORITY 1: Check CURRENT forming candle for liquidity grab (REAL-TIME)
        # Professional traders see this happening NOW and enter immediately
        if current_candle_high is not None:
            for swing_high in swing_highs:
                swing_level = swing_high["level"]

                # Check if current candle spiked above swing high
                if current_candle_high > swing_level:
                    # Measure current wick size
                    wick_size = current_candle_high - current_price
                    pips_above_swing = current_candle_high - swing_level

                    # Convert to pips for EUR/USD
                    wick_size_pips = self._to_pips(wick_size)
                    pips_above_swing_pips = self._to_pips(pips_above_swing)

                    # Liquidity grab requirements (real-time):
                    # 1. Wick extends 5+ pips above swing level
                    # 2. Strong rejection (wick 8+ pips)
                    # 3. Price currently back below swing level
                    if pips_above_swing_pips >= 5.0 and wick_size_pips >= 8.0 and current_price < swing_level:
                        # LIVE GRAB SCORE (4 points) - same as bullish to prevent single-pattern trades
                        return {
                            "detected": True,
                            "score": 4,
                            "grab_level": swing_level,
                            "grab_high": current_candle_high,
                            "rejection_size": round(wick_size_pips, 1),
                            "pips_above": round(pips_above_swing_pips, 1),
                            "description": f"🔥 LIVE Liquidity Grab at {swing_level:.5f}! Price spiking to {current_candle_high:.5f} ({pips_above_swing_pips:.1f} pips above) rejecting {wick_size_pips:.1f} pips DOWN NOW!"
                        }

            # Also check H4 resistance levels for current candle grabs
            resistance_levels = h4_levels.get("resistance_levels", [])
            for resistance in resistance_levels:
                if abs(resistance - current_price) < 0.0030:  # Within 30 pips for EUR/USD
                    if current_candle_high > resistance:
                        wick_size = current_candle_high - current_price
                        pips_above = current_candle_high - resistance

                        # Convert to pips
                        wick_size_pips = self._to_pips(wick_size)
                        pips_above_pips = self._to_pips(pips_above)

                        if pips_above_pips >= 5.0 and wick_size_pips >= 8.0 and current_price < resistance:
                            return {
                                "detected": True,
                                "score": 4,
                                "grab_level": resistance,
                                "grab_high": current_candle_high,
                                "rejection_size": round(wick_size_pips, 1),
                                "pips_above": round(pips_above_pips, 1),
                                "description": f"🔥 LIVE Liquidity Grab at H4 resistance {resistance:.5f}! Price spiking to {current_candle_high:.5f} rejecting {wick_size_pips:.1f} pips DOWN NOW!"
                            }

        # PRIORITY 2: Check last 2 CLOSED candles for liquidity grab pattern
        # Score based on recency: 4 points if last candle, 3 points if 2 candles ago
        last_2 = recent_candles.tail(2)

        for swing_high in swing_highs:
            swing_level = swing_high["level"]

            for candle_position, (idx, candle) in enumerate(last_2.iterrows()):
                # Check if this candle spiked above the swing high
                if candle['high'] > swing_level:
                    # Measure wick size
                    wick_size = candle['high'] - candle['close']

                    # Liquidity grab requirements:
                    # 1. Wick extends 5+ pips above swing level
                    # 2. Strong rejection (wick 8+ pips)
                    # 3. Candle closed back below swing level

                    pips_above_swing = candle['high'] - swing_level

                    # Convert to pips
                    wick_size_pips = self._to_pips(wick_size)
                    pips_above_swing_pips = self._to_pips(pips_above_swing)

                    # Professional thresholds: 5+ pips spike, 8+ pips rejection (institutional moves)
                    if pips_above_swing_pips >= 5.0 and wick_size_pips >= 8.0 and candle['close'] < swing_level:
                        # Score based on recency
                        # candle_position: 0 = 2 candles ago, 1 = last candle
                        score = 4 if candle_position == 1 else 3
                        freshness = "FRESH (last candle)" if candle_position == 1 else "RECENT (2 candles ago)"

                        # Liquidity grab confirmed!
                        return {
                            "detected": True,
                            "score": score,
                            "grab_level": swing_level,
                            "grab_high": float(candle['high']),
                            "rejection_size": round(wick_size_pips, 1),
                            "pips_above": round(pips_above_swing_pips, 1),
                            "description": f"🔥 Liquidity Grab at {swing_level:.5f}! Price spiked to {candle['high']:.5f} ({pips_above_swing_pips:.1f} pips above) then rejected {wick_size_pips:.1f} pips DOWN - {freshness}"
                        }

        # Also check H4 resistance levels for liquidity grabs (last 2 candles only)
        resistance_levels = h4_levels.get("resistance_levels", [])
        for resistance in resistance_levels:
            if abs(resistance - current_price) < 0.0030:  # Within 30 pips for EUR/USD
                for candle_position, (idx, candle) in enumerate(last_2.iterrows()):
                    if candle['high'] > resistance:
                        wick_size = candle['high'] - candle['close']
                        pips_above = candle['high'] - resistance

                        # Convert to pips
                        wick_size_pips = self._to_pips(wick_size)
                        pips_above_pips = self._to_pips(pips_above)

                        # Professional thresholds: 5+ pips spike, 8+ pips rejection
                        if pips_above_pips >= 5.0 and wick_size_pips >= 8.0 and candle['close'] < resistance:
                            # Score based on recency
                            score = 4 if candle_position == 1 else 3
                            freshness = "FRESH (last candle)" if candle_position == 1 else "RECENT (2 candles ago)"

                            return {
                                "detected": True,
                                "score": score,
                                "grab_level": resistance,
                                "grab_high": float(candle['high']),
                                "rejection_size": round(wick_size_pips, 1),
                                "pips_above": round(pips_above_pips, 1),
                                "description": f"🔥 Liquidity Grab at H4 resistance {resistance:.5f}! Price spiked to {candle['high']:.5f} then rejected {wick_size_pips:.1f} pips - {freshness}"
                            }

        return {"detected": False, "score": 0}

    async def _detect_setup_pattern(self, h1_data: pd.DataFrame, h4_data: pd.DataFrame, h4_levels: Dict, current_price: float) -> Dict[str, Any]:
        """
        Detect which professional setup patterns are forming

        NEW PROFESSIONAL APPROACH:
        - Checks ALL patterns (not just first match)
        - Calculates confluence score
        - Filters by trend (BOS/CHoCH)
        - Prioritizes liquidity grabs
        - Returns combined analysis
        """
        key_level = h4_levels["key_level"]
        last_candles = h1_data.tail(20)  # Need more candles for FVG detection

        # Get current candle info to check if high touched key zones
        current_candle_info = await self._get_current_candle_info(h1_data, current_price)
        current_candle_high = current_candle_info.get("high")

        # STEP 1: Check trend structure (BOS/CHoCH) - FILTER FIRST
        structure = self._check_bos_choch(last_candles, current_price)

        # If bullish CHoCH detected, filter out bearish setups
        if structure["structure_type"] == "BULLISH_CHOCH":
            return {
                "detected": True,
                "pattern_type": "FILTERED_OUT",
                "state": "BULLISH_CHOCH_DETECTED",
                "progress": "0/5",
                "structure": structure,
                "description": "⚠️ Bullish CHoCH detected - Filtering out bearish setups until trend confirmation",
                "confluences": [],
                "total_score": structure["score"]
            }

        # STEP 2: Check ALL patterns (don't stop at first match)
        # PROFESSIONAL: Use only closed candles, EXCEPT Liquidity Grab (real-time detection)
        fvg_setup = self._check_fvg(last_candles, current_price)
        ob_setup = self._check_order_block(last_candles, current_price)
        breakdown_setup = self._check_breakout_retest(last_candles, key_level, current_price)
        supply_setup = self._check_supply_zone(last_candles, h4_levels, current_price)
        liquidity_grab = self._check_liquidity_grab(last_candles, current_price, h4_levels, current_candle_high)  # Uses forming candle for real-time spike detection

        # STEP 3: Calculate confluence score
        confluences_raw = []


        # Add structure score with VALIDATION
        # For BEARISH_BOS: Only count if price is still BELOW the BOS level
        if structure["structure_type"] == "BEARISH_BOS":
            bos_level = structure.get("swing_level", 0)
            current_candle_close = last_candles['close'].iloc[-1]

            # BOS is INVALID if price closed back above the BOS level
            bos_invalidated = current_candle_close > bos_level

            if not bos_invalidated:
                # BOS is still valid - price holding below
                confluences_raw.append({
                    "type": "BEARISH_BOS",
                    "score": 2,
                    "description": structure["description"]
                })
            # else: BOS invalidated - don't add to confluences

        # Add liquidity grab (highest priority)
        # NO stability check needed - based on completed historical candles
        if liquidity_grab["detected"]:
            confluences_raw.append({
                "type": "LIQUIDITY_GRAB",
                "score": liquidity_grab["score"],  # Use actual score (4 or 3 based on recency)
                "description": liquidity_grab["description"],
                "grab_high": liquidity_grab.get("grab_high"),  # Pass actual high for SL calculation
                "grab_level": liquidity_grab.get("grab_level")  # Pass grab level
            })

        # Add FVG
        # Only give points when price is ACTUALLY IN or REJECTED FROM the FVG zone
        # Don't give points just because FVG exists somewhere above price
        if fvg_setup["detected"]:
            state = fvg_setup.get("state", "")
            strong_rejection = fvg_setup.get("strong_rejection", False)

            # Only count if price is IN the FVG or has been strongly rejected from it
            # Don't give points when just "DETECTED" or "APPROACHING"
            if state == "IN_FVG" or strong_rejection:
                confluences_raw.append({
                    "type": "FVG",
                    "score": 3,
                    "description": f"FVG at {fvg_setup.get('fvg_zone', {}).get('midpoint', 0):.5f}"
                })
            # else: FVG detected but price not there yet - don't give points

        # Add Order Block
        if ob_setup["detected"]:
            # CRITICAL: Validate that OB zone is still holding (proper invalidation)
            ob_zone = ob_setup.get('order_block_zone', {})
            ob_bottom = ob_zone.get('bottom', 0)
            ob_top = ob_zone.get('top', 0)

            # Get current candle high to check if OB was violated
            current_candle_high = last_candles['high'].iloc[-1] if len(last_candles) > 0 else current_price

            # For BEARISH OB: Invalidate if price closed ABOVE the OB top (not just touched)
            # Allow wicks into the zone, but not full candle closes above it
            ob_violated = current_price > ob_top or (current_candle_high > ob_top * 1.002)  # 0.2% buffer for wicks

            # Pattern shows immediately when detected (professional methodology)
            # Only check if OB is still valid (not violated)
            if not ob_violated:
                confluences_raw.append({
                    "type": "ORDER_BLOCK",
                    "score": 3,
                    "description": f"Order Block at {ob_zone.get('midpoint', 0):.5f} ({ob_bottom:.5f}-{ob_top:.5f})"
                })

        # Add Breakdown Retest
        # Only give points when retest is ACTUALLY HAPPENING, not just detected
        if breakdown_setup["detected"]:
            state = breakdown_setup.get("state", "")

            # Only count if retest is happening or rejection confirmed
            # Don't give points during "RETEST_WAITING" - that's too early
            if state in ["REJECTION_WAITING", "CONFIRMATION_WAITING"]:
                confluences_raw.append({
                    "type": "BREAKDOWN_RETEST",
                    "score": 2,
                    "description": f"Breakdown Retest at {breakdown_setup.get('key_level', 0):.5f}"
                })
            # else: Breakdown detected but retest not happening yet - don't give points

        # Add Supply Zone
        # Only give points when zone is ACTUALLY TOUCHED and REJECTED (closed candle confirmation)
        if supply_setup["detected"]:
            # Check if we have actual touch + rejection confirmation
            strong_rejection = supply_setup.get("strong_rejection", False)

            # Only give confluence points when zone shows strong rejection
            if strong_rejection:
                confluences_raw.append({
                    "type": "SUPPLY_ZONE",
                    "score": 2,
                    "description": f"Supply Zone at {supply_setup.get('key_level', 0):.5f}"
                })
            # else: Zone detected but no strong rejection yet - don't give points

        # STEP 3.5: DEDUPLICATE to prevent double-counting correlated signals
        # Example: BOS + BREAKOUT/BREAKDOWN_RETEST from same structural break should not both count
        confluences, total_score = self._dedupe_confluences(confluences_raw)

        # STEP 4: Determine if we have enough confluence to enter
        if total_score < 5:
            # Not enough confluence
            # Add GATED SCORING for LOW_CONFLUENCE
            gated_score = self._grade_setup(confluences)
            return {
                "detected": True,
                "pattern_type": "LOW_CONFLUENCE",
                "state": "SCANNING",
                "progress": "0/5",
                "confluences": confluences,
                "total_score": total_score,
                "structure": structure,
                "description": f"⚠️ Low confluence (Score: {total_score}/5 minimum) - Need more confirmation",
                "has_location": gated_score["has_location"],
                "has_trigger": gated_score["has_trigger"],
                "has_confirmation": gated_score["has_confirmation"],
                "grade": gated_score["grade"],
                "tradable": gated_score["tradable"]
            }

        # STEP 4.5: SMART VALIDATION - Check price action, distance, and timing
        validation_warnings = []
        confidence_penalty = 0

        # Get current candle data
        last_candle = last_candles.iloc[-1]
        candle_open = last_candle['open']
        candle_high = last_candles['high'].iloc[-1]
        candle_low = last_candles['low'].iloc[-1]
        candle_range = candle_high - candle_low

        # 1. PRICE ACTION CHECK: Is candle bullish or bearish?
        candle_is_bullish = current_price > candle_open  # For bearish setups, we want bearish candles
        price_from_low = current_price - candle_low
        price_from_high = candle_high - current_price
        candle_position = (current_price - candle_low) / candle_range if candle_range > 0 else 0.5

        if candle_is_bullish:
            validation_warnings.append("⚠️ Bullish candle (price above open)")
            confidence_penalty += 2

        if candle_position > 0.7:  # In top 30% of candle (bad for bearish)
            validation_warnings.append("📈 Price at candle highs (weak position)")
            confidence_penalty += 2
        elif candle_position > 0.5:  # In upper half
            validation_warnings.append("⚠️ Price in upper half of candle")
            confidence_penalty += 1

        # 2. CONFLUENCE DISTANCE CHECK: Are patterns close to current price?
        distant_patterns = []

        # Check Order Block distance
        if ob_setup["detected"]:
            ob_zone = ob_setup.get('order_block_zone', {})
            ob_bottom = ob_zone.get('bottom', 0)
            ob_distance = abs(current_price - ob_bottom)
            if ob_distance > 20:  # More than 20 pips away
                distant_patterns.append(f"Order Block ({ob_distance:.1f} pips away)")
                confidence_penalty += 1

        # Check FVG distance
        if fvg_setup["detected"]:
            fvg_zone = fvg_setup.get('fvg_zone', {})
            fvg_mid = fvg_zone.get('midpoint', current_price)
            fvg_distance = abs(current_price - fvg_mid)
            if fvg_distance > 15:  # More than 15 pips away
                distant_patterns.append(f"FVG ({fvg_distance:.1f} pips away)")
                confidence_penalty += 1

        if distant_patterns:
            validation_warnings.append(f"📍 Distant confluences: {', '.join(distant_patterns)}")

        # 3. MOMENTUM CHECK: Last 3 candles direction
        if len(last_candles) >= 3:
            last_3_closes = last_candles['close'].iloc[-3:].values
            bullish_momentum = sum(last_3_closes[i] > last_3_closes[i-1] for i in range(1, len(last_3_closes)))
            if bullish_momentum >= 2:  # 2+ bullish candles (bad for bearish setup)
                validation_warnings.append("📈 Bullish momentum (last 3 candles)")
                confidence_penalty += 2

        # 4. ADJUST CONFIDENCE based on validation
        adjusted_score = total_score - confidence_penalty

        # Determine final confidence with validation
        if validation_warnings:
            if adjusted_score < 5:
                # Too many warnings - don't show as ready entry
                # Add GATED SCORING for WAITING_CONFIRMATION
                gated_score = self._grade_setup(confluences)
                return {
                    "detected": True,
                    "pattern_type": "CONFLUENCE_DETECTED",
                    "state": "WAITING_CONFIRMATION",
                    "progress": "3/5",
                    "key_level": key_level,  # FIX: Add key_level so trade plan can use it
                    "confluences": confluences,
                    "total_score": total_score,
                    "adjusted_score": adjusted_score,
                    "structure": structure,
                    "validation_warnings": validation_warnings,
                    "candle_position": round(candle_position * 100, 1),
                    "description": f"⏳ Setup detected ({total_score} points) but price action weak - WAIT FOR CONFIRMATION",
                    "has_location": gated_score["has_location"],
                    "has_trigger": gated_score["has_trigger"],
                    "has_confirmation": gated_score["has_confirmation"],
                    "grade": gated_score["grade"],
                    "tradable": gated_score["tradable"]
                }

        # STEP 5: We have confluence! Return the primary pattern
        # Priority: Liquidity Grab > FVG > OB > Breakdown > Supply
        primary_setup = None
        if liquidity_grab["detected"] and fvg_setup["detected"]:
            primary_setup = fvg_setup
            primary_setup["pattern_type"] = "FVG_CONFLUENCE"
        elif liquidity_grab["detected"] and ob_setup["detected"]:
            primary_setup = ob_setup
            primary_setup["pattern_type"] = "OB_CONFLUENCE"
        elif fvg_setup["detected"]:
            primary_setup = fvg_setup
        elif ob_setup["detected"]:
            primary_setup = ob_setup
        elif breakdown_setup["detected"]:
            primary_setup = breakdown_setup
        elif supply_setup["detected"]:
            primary_setup = supply_setup

        if primary_setup:
            # Enhance with confluence data
            primary_setup["confluences"] = confluences
            primary_setup["total_score"] = total_score
            primary_setup["adjusted_score"] = adjusted_score if validation_warnings else total_score
            primary_setup["structure"] = structure
            primary_setup["validation_warnings"] = validation_warnings
            primary_setup["candle_position"] = round(candle_position * 100, 1)

            # Add GATED SCORING
            gated_score = self._grade_setup(confluences)
            primary_setup["has_location"] = gated_score["has_location"]
            primary_setup["has_trigger"] = gated_score["has_trigger"]
            primary_setup["has_confirmation"] = gated_score["has_confirmation"]
            primary_setup["grade"] = gated_score["grade"]
            primary_setup["tradable"] = gated_score["tradable"]

            # ENTRY STATE SYSTEM: Determine entry timing (state machine)
            entry_state_data = self._determine_entry_state(
                confluences=confluences,
                grading=gated_score,
                current_price=current_price,
                h1_data=last_candles,
                structure=structure
            )

            # Add STEP 4: TRADE-WORTHINESS GATING (NEW)
            trade_worthiness = self._assess_trade_worthiness(
                entry_state_data=entry_state_data,
                confluences=confluences,
                h4_data=h4_data,
                current_price=current_price,
                grading=gated_score,
                direction="BEARISH"
            )

            # Add state machine fields to response
            primary_setup["setup_state"] = entry_state_data["setup_state"]
            primary_setup["entry_ready"] = entry_state_data["entry_ready"]
            primary_setup["entry_window"] = entry_state_data["entry_window"]
            primary_setup["distance_to_invalidation"] = entry_state_data["distance_to_invalidation"]
            primary_setup["distance_to_zone"] = entry_state_data["distance_to_zone"]
            primary_setup["trigger_time"] = entry_state_data["trigger_time"]
            primary_setup["trigger_type"] = entry_state_data["trigger_type"]
            primary_setup["trigger_validated"] = entry_state_data["trigger_validated"]
            primary_setup["trigger_validation_reason"] = entry_state_data["trigger_validation_reason"]
            primary_setup["zone_time"] = entry_state_data["zone_time"]
            primary_setup["zone_price"] = entry_state_data["zone_price"]
            primary_setup["invalidation_price"] = entry_state_data["invalidation_price"]
            primary_setup["atr"] = entry_state_data["atr"]
            primary_setup["state_reason"] = entry_state_data["reason"]
            # Bias fields (deterministic)
            primary_setup["direction"] = entry_state_data["direction"]
            primary_setup["bias_source"] = entry_state_data["bias_source"]
            primary_setup["bias_conflict"] = entry_state_data["bias_conflict"]
            primary_setup["conflicting_structure"] = entry_state_data["conflicting_structure"]

            # Step 4: Trade-Worthiness fields (NEW)
            primary_setup["trade_worthy"] = trade_worthiness["trade_worthy"]
            primary_setup["trade_decision"] = trade_worthiness["trade_decision"]
            primary_setup["decision_reason"] = trade_worthiness["decision_reason"]
            primary_setup["entry_price"] = trade_worthiness["entry_price"]
            primary_setup["stop_price"] = trade_worthiness["stop_price"]
            primary_setup["targets"] = trade_worthiness["targets"]
            primary_setup["tp_source"] = trade_worthiness["tp_source"]
            primary_setup["rr"] = trade_worthiness["rr"]
            primary_setup["risk_distance"] = trade_worthiness["risk_distance"]
            primary_setup["reward_distance_tp1"] = trade_worthiness["reward_distance_tp1"]
            primary_setup["reward_distance_tp2"] = trade_worthiness["reward_distance_tp2"]
            primary_setup["stop_atr_multiple"] = trade_worthiness["stop_atr_multiple"]
            primary_setup["gates"] = trade_worthiness["gates"]

            # Confidence hierarchy (ENTRY_READY > TRIGGERED > SETUP > WATCH > NO_TRADE/LATE)
            if entry_state_data["setup_state"] == "ENTRY_READY":
                primary_setup["confidence"] = "⭐️⭐️⭐️ EXTREME"  # Most urgent
            elif entry_state_data["setup_state"] == "TRIGGERED":
                primary_setup["confidence"] = "⭐️⭐️ HIGH"  # Trigger validated, checking risk
            elif entry_state_data["setup_state"] == "SETUP":
                primary_setup["confidence"] = "⭐️ MODERATE"  # Good location, waiting for trigger
            elif entry_state_data["setup_state"] == "WATCH":
                primary_setup["confidence"] = "⏳ WATCH (monitoring)"  # Concerns present
            else:  # NO_TRADE, LATE
                primary_setup["confidence"] = "❌ NOT TRADABLE"  # No entry or too late

            return primary_setup

        # Default: SCANNING
        # Add GATED SCORING for SCANNING state
        gated_score = self._grade_setup(confluences)

        # Add entry state for SCANNING too
        entry_state_data = self._determine_entry_state(
            confluences=confluences,
            grading=gated_score,
            current_price=current_price,
            h1_data=last_candles,
            structure=structure
        )

        # Add STEP 4: TRADE-WORTHINESS GATING for SCANNING too
        trade_worthiness = self._assess_trade_worthiness(
            entry_state_data=entry_state_data,
            confluences=confluences,
            h4_data=h4_data,
            current_price=current_price,
            grading=gated_score,
            direction="BEARISH"
        )

        return {
            "detected": True,
            "pattern_type": "SCANNING",
            "state": "SCANNING",
            "progress": "0/5",
            "key_level": key_level,
            "confluences": confluences,
            "total_score": total_score,
            "structure": structure,
            "description": "Scanning for professional setups...",
            "has_location": gated_score["has_location"],
            "has_trigger": gated_score["has_trigger"],
            "has_confirmation": gated_score["has_confirmation"],
            "grade": gated_score["grade"],
            "tradable": gated_score["tradable"],
            # Add state machine fields
            "setup_state": entry_state_data["setup_state"],
            "entry_ready": entry_state_data["entry_ready"],
            "entry_window": entry_state_data["entry_window"],
            "distance_to_invalidation": entry_state_data["distance_to_invalidation"],
            "distance_to_zone": entry_state_data["distance_to_zone"],
            "trigger_time": entry_state_data["trigger_time"],
            "trigger_type": entry_state_data["trigger_type"],
            "trigger_validated": entry_state_data["trigger_validated"],
            "trigger_validation_reason": entry_state_data["trigger_validation_reason"],
            "zone_time": entry_state_data["zone_time"],
            "zone_price": entry_state_data["zone_price"],
            "invalidation_price": entry_state_data["invalidation_price"],
            "atr": entry_state_data["atr"],
            "state_reason": entry_state_data["reason"],
            # Bias fields (deterministic)
            "direction": entry_state_data["direction"],
            "bias_source": entry_state_data["bias_source"],
            "bias_conflict": entry_state_data["bias_conflict"],
            "conflicting_structure": entry_state_data["conflicting_structure"],
            # Step 4: Trade-Worthiness fields (NEW)
            "trade_worthy": trade_worthiness["trade_worthy"],
            "trade_decision": trade_worthiness["trade_decision"],
            "decision_reason": trade_worthiness["decision_reason"],
            "entry_price": trade_worthiness["entry_price"],
            "stop_price": trade_worthiness["stop_price"],
            "targets": trade_worthiness["targets"],
            "tp_source": trade_worthiness["tp_source"],
            "rr": trade_worthiness["rr"],
            "risk_distance": trade_worthiness["risk_distance"],
            "reward_distance_tp1": trade_worthiness["reward_distance_tp1"],
            "reward_distance_tp2": trade_worthiness["reward_distance_tp2"],
            "stop_atr_multiple": trade_worthiness["stop_atr_multiple"],
            "gates": trade_worthiness["gates"]
        }

    def _check_pattern_stability(self, pattern_name: str, pattern_data: Dict, stability_minutes: int = 10) -> bool:
        """
        Check if a pattern has been stable for minimum time (default 10 minutes)

        Args:
            pattern_name: Name of pattern (liquidity_grab, fvg, order_block, etc.)
            pattern_data: Current pattern data
            stability_minutes: Minimum minutes pattern must be present

        Returns:
            True if pattern is stable (present for >= stability_minutes), False otherwise
        """
        if not pattern_data or not pattern_data.get("detected"):
            # Pattern not detected - clear tracker
            self.pattern_tracker[pattern_name] = None
            return False

        now_utc = datetime.now(pytz.UTC)

        # Check if we're tracking this pattern
        tracked = self.pattern_tracker[pattern_name]

        if tracked is None:
            # First time seeing this pattern - start tracking
            self.pattern_tracker[pattern_name] = {
                "first_seen": now_utc,
                "data": pattern_data
            }
            return False  # Not stable yet (just appeared)

        # Pattern already being tracked - check if it's changed significantly
        # Simple check: if key level changed by > 2 pips, reset tracker
        old_level = tracked["data"].get("key_level", 0)
        new_level = pattern_data.get("key_level", 0)

        if abs(old_level - new_level) > 2:  # Pattern moved significantly
            # Reset tracker with new data
            self.pattern_tracker[pattern_name] = {
                "first_seen": now_utc,
                "data": pattern_data
            }
            return False

        # Pattern stable - check if enough time has passed
        time_stable = (now_utc - tracked["first_seen"]).total_seconds() / 60  # minutes

        return time_stable >= stability_minutes

    def _dedupe_confluences(self, confluences: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Deduplicate correlated confluences to prevent double-counting.

        Problem: Multiple detections can fire from same market event:
        - BOS + BREAKOUT_RETEST (same structural break)
        - ORDER_BLOCK + DEMAND_ZONE (same price level)

        Solution: Group by correlation and take strongest per group.

        Correlation Groups:
        - STRUCTURE_SHIFT: BULLISH_BOS, BEARISH_BOS, BULLISH_CHOCH, BEARISH_CHOCH
        - BREAKOUT_EVENT: BREAKOUT_RETEST, BREAKDOWN_RETEST
        - LIQUIDITY_EVENT: LIQUIDITY_GRAB
        - ZONE_LOCATION: ORDER_BLOCK, SUPPLY_ZONE, DEMAND_ZONE, FVG

        Returns:
            (deduplicated_confluences, total_score)
        """
        from collections import defaultdict

        # Group confluences by correlation group
        grouped = defaultdict(list)

        for conf in confluences:
            conf_type = conf.get("type", "")

            # Assign to correlation group
            if conf_type in ["BULLISH_BOS", "BEARISH_BOS", "BULLISH_CHOCH", "BEARISH_CHOCH"]:
                group = "STRUCTURE_SHIFT"
            elif conf_type in ["BREAKOUT_RETEST", "BREAKDOWN_RETEST"]:
                group = "BREAKOUT_EVENT"
            elif conf_type == "LIQUIDITY_GRAB":
                group = "LIQUIDITY_EVENT"
            elif conf_type in ["ORDER_BLOCK", "SUPPLY_ZONE", "DEMAND_ZONE", "FVG"]:
                group = "ZONE_LOCATION"
            else:
                group = "OTHER"

            # Add group field to confluence
            conf["group"] = group
            grouped[group].append(conf)

        # Select strongest confluence per group (max score)
        deduplicated = []
        total_score = 0

        for group, group_confs in grouped.items():
            # Sort by score descending
            sorted_confs = sorted(group_confs, key=lambda c: c.get("score", 0), reverse=True)

            # Take strongest one
            strongest = sorted_confs[0]
            strongest["counted_for_score"] = True
            deduplicated.append(strongest)
            total_score += strongest.get("score", 0)

            # Mark others as not counted
            for conf in sorted_confs[1:]:
                conf["counted_for_score"] = False
                deduplicated.append(conf)

        return deduplicated, total_score


    def _grade_setup(self, confluences: List[Dict]) -> Dict:
        """
        Grade setup using GATED SCORING system (BEARISH VERSION)

        Professional traders categorize signals:
        - LOCATION: Where to enter (ORDER_BLOCK, SUPPLY_ZONE)
        - TRIGGER: Why enter now (LIQUIDITY_GRAB, BREAKOUT_RETEST)
        - CONFIRMATION: Is it moving? (BEARISH_BOS, BEARISH_CHOCH)

        Grades:
        - NO_TRADE: No location identified
        - WAIT: Only location (no trigger/confirmation)
        - B: Location + confirmation (but no trigger)
        - TRADE: Location + trigger (ready to trade)
        - A+: All three categories present (best setup)

        Returns:
            Dict with has_location, has_trigger, has_confirmation, grade, tradable
        """
        has_location = False
        has_trigger = False
        has_confirmation = False

        # Define categories (BEARISH VERSION: SUPPLY_ZONE instead of DEMAND_ZONE)
        location_types = ["ORDER_BLOCK", "SUPPLY_ZONE"]
        trigger_types = ["LIQUIDITY_GRAB", "BREAKOUT_RETEST"]
        confirmation_types = ["BEARISH_BOS", "BEARISH_CHOCH"]

        # Categorize confluences
        for conf in confluences:
            conf_type = conf.get("type", "")
            if conf_type in location_types:
                has_location = True
            if conf_type in trigger_types:
                has_trigger = True
            if conf_type in confirmation_types:
                has_confirmation = True

        # Determine grade using gated logic
        if not has_location:
            grade = "NO_TRADE"
        elif has_location and has_trigger and has_confirmation:
            grade = "A+"
        elif has_location and has_trigger:
            grade = "TRADE"
        elif has_location and has_confirmation:
            grade = "B"
        else:
            grade = "WAIT"

        return {
            "has_location": has_location,
            "has_trigger": has_trigger,
            "has_confirmation": has_confirmation,
            "grade": grade,
            "tradable": grade in ["TRADE", "A+", "A", "B"]
        }

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range (ATR)"""
        if data is None or len(data) < period:
            # Default fallback for GBP/USD: 20 pips * pip_size
            # GBP/USD pip_size = 0.0001 (quotes like 1.2650)
            return 20.0 * self.pip_size  # 20 pips = 0.0020 for GBP/USD

        high = data['high']
        low = data['low']
        close = data['close']

        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]

        return float(atr) if not pd.isna(atr) else 20.0 * self.pip_size

    def _get_zone_from_confluences(self, confluences: List[Dict], current_price: float) -> Dict:
        """
        Extract zone information from confluences
        Returns zone with bounds if available, otherwise midpoint
        """
        # Find ZONE_LOCATION confluences that counted
        zone_confs = [c for c in confluences
                     if c.get("group") == "ZONE_LOCATION"
                     and c.get("counted_for_score") == True]

        if not zone_confs:
            # Fallback: find any zone (even if not counted)
            zone_confs = [c for c in confluences if c.get("group") == "ZONE_LOCATION"]

        if not zone_confs:
            # No zone found - use current price as fallback
            return {
                "top": current_price,
                "bottom": current_price,
                "midpoint": current_price
            }

        # Use first zone (or closest to price if multiple)
        zone = zone_confs[0]

        # Extract bounds with fallback
        if "top" in zone and "bottom" in zone:
            top = zone["top"]
            bottom = zone["bottom"]
            midpoint = (top + bottom) / 2
        elif "price" in zone:
            # Single price - create tiny band around it
            price = zone["price"]
            buffer = 0.5 * self.pip_size  # 0.5 pips buffer
            top = price + buffer
            bottom = price - buffer
            midpoint = price
        else:
            # Fallback to current price
            buffer = 0.5 * self.pip_size
            top = current_price + buffer
            bottom = current_price - buffer
            midpoint = current_price

        return {
            "top": top,
            "bottom": bottom,
            "midpoint": midpoint
        }

    def _calculate_invalidation_level(self, zone: Dict, direction: str = "BEARISH") -> float:
        """Calculate stop loss / invalidation level"""
        if direction == "BULLISH":
            # For bullish setups, stop goes below zone
            return zone["bottom"]
        else:
            # For bearish setups, stop goes above zone
            return zone["top"]

    def _get_trigger_info(self, confluences: List[Dict]) -> Dict:
        """
        Extract trigger information (type and timestamp)
        Priority: detected_at (immediate) > pattern_tracker (stable) > None (conservative)
        """
        # Find trigger confluences
        trigger_types = ["LIQUIDITY_GRAB", "BREAKDOWN_RETEST"]
        triggers = [c for c in confluences if c.get("type") in trigger_types]

        if not triggers:
            return {
                "has_trigger": False,
                "trigger_type": None,
                "trigger_time": None,
                "trigger_age_hours": None
            }

        # If multiple triggers, pick LIQUIDITY_GRAB first (higher priority)
        liquidity_grabs = [t for t in triggers if t.get("type") == "LIQUIDITY_GRAB"]
        breakdown_retests = [t for t in triggers if t.get("type") == "BREAKDOWN_RETEST"]

        if liquidity_grabs:
            trigger = liquidity_grabs[0]
        elif breakdown_retests:
            trigger = breakdown_retests[0]
        else:
            trigger = triggers[0]

        # Try to get real timestamp (3-tier priority)
        trigger_time = None
        trigger_age_hours = None
        trigger_type = trigger.get("type")

        # Priority 1: Check for detected_at (immediate triggers)
        if "detected_at" in trigger:
            trigger_time = trigger["detected_at"]
            detected_dt = datetime.fromisoformat(trigger_time.replace('Z', '+00:00'))
            age_seconds = (datetime.now(pytz.UTC) - detected_dt).total_seconds()
            trigger_age_hours = age_seconds / 3600

        # Priority 2: Check pattern_tracker for stable patterns
        elif trigger_type in self.pattern_tracker and self.pattern_tracker[trigger_type]:
            tracked = self.pattern_tracker[trigger_type]
            if "first_seen" in tracked:
                first_seen = tracked["first_seen"]
                age_seconds = (datetime.now(pytz.UTC) - first_seen).total_seconds()
                trigger_age_hours = age_seconds / 3600
                trigger_time = first_seen.isoformat()

        # Priority 3: No timestamp available - mark as None (conservative)
        # This will block ENTRY_READY until we have a real timestamp

        return {
            "has_trigger": True,
            "trigger_type": trigger_type,
            "trigger_time": trigger_time,
            "trigger_age_hours": trigger_age_hours
        }

    def _determine_entry_state(self, confluences: List[Dict], grading: Dict,
                               current_price: float, h1_data: pd.DataFrame,
                               structure: Dict) -> Dict:
        """
        Determine setup state and entry readiness (BEARISH TRADER)

        States:
        - NO_TRADE: No valid location
        - WATCH: Location exists but not tradeable quality yet
        - SETUP: Strong location, waiting for trigger
        - TRIGGERED: Trigger detected and validated (near zone + fresh)
        - ENTRY_READY: Trigger + acceptable risk (actionable entry)
        - LATE: Trigger but too far from zone/invalidation

        Returns dict with setup_state, entry_ready, entry_window, distances, etc.
        """
        has_location = grading["has_location"]
        has_trigger = grading["has_trigger"]
        has_confirmation = grading["has_confirmation"]

        # Calculate ATR for EUR/USD
        atr = self._calculate_atr(h1_data, period=14)

        # EUR/USD-specific thresholds (FX pairs: more lenient than Gold)
        zone_distance_max = 0.25 * atr  # 25% of ATR
        risk_distance_max = 0.60 * atr  # 60% of ATR

        # Extract zone information
        zone = self._get_zone_from_confluences(confluences, current_price)

        # Calculate distance to zone
        if current_price >= zone["bottom"] and current_price <= zone["top"]:
            # Price is inside zone
            distance_to_zone = 0.0
            zone_proximity = "GOOD"
        else:
            # Price outside zone - measure to nearest boundary
            if current_price < zone["bottom"]:
                distance_to_zone = zone["bottom"] - current_price
            else:
                distance_to_zone = current_price - zone["top"]

            # Convert to pips for readability
            distance_to_zone = self._to_pips(distance_to_zone)

            # Classify proximity
            if distance_to_zone <= self._to_pips(zone_distance_max * 0.5):
                zone_proximity = "GOOD"
            elif distance_to_zone <= self._to_pips(zone_distance_max):
                zone_proximity = "OK"
            else:
                zone_proximity = "LATE"

        # Calculate distance to invalidation
        invalidation_level = self._calculate_invalidation_level(zone, direction="BEARISH")
        distance_to_invalidation = abs(current_price - invalidation_level)
        distance_to_invalidation_pips = self._to_pips(distance_to_invalidation)

        # Classify risk proximity
        if distance_to_invalidation <= risk_distance_max * 0.6:
            risk_proximity = "GOOD"
        elif distance_to_invalidation <= risk_distance_max:
            risk_proximity = "OK"
        else:
            risk_proximity = "LATE"

        # Final entry_window (worst of zone and risk)
        proximity_rank = {"GOOD": 0, "OK": 1, "LATE": 2}
        if proximity_rank[zone_proximity] >= proximity_rank[risk_proximity]:
            entry_window = zone_proximity
        else:
            entry_window = risk_proximity

        # Get trigger info
        trigger_info = self._get_trigger_info(confluences)
        trigger_age_hours = trigger_info["trigger_age_hours"]
        trigger_type = trigger_info["trigger_type"]

        # Determine bias conflict (deterministic)
        direction = "BEARISH"
        bias_conflict = False
        bias_source = "STRUCTURE_SHIFT"
        conflicting_structure = None

        if structure.get("structure_type") in ["BULLISH_CHOCH", "BULLISH_BOS"]:
            bias_conflict = True
            conflicting_structure = structure.get("structure_type")

        # Check trigger freshness with None handling
        if trigger_info["has_trigger"]:
            if trigger_type == "LIQUIDITY_GRAB":
                freshness_max = 2.0  # 2 hours
            else:  # BREAKDOWN_RETEST
                freshness_max = 4.0  # 4 hours

            # None handling: if timestamp missing, mark as not fresh
            if trigger_age_hours is None:
                is_fresh = False
                freshness_reason = "Trigger time unknown; skipping freshness gate"
            elif trigger_age_hours <= freshness_max:
                is_fresh = True
                freshness_reason = f"Trigger fresh ({trigger_age_hours:.1f}h old, max {freshness_max}h)"
            else:
                is_fresh = False
                freshness_reason = f"Trigger stale ({trigger_age_hours:.1f}h old, max {freshness_max}h)"
        else:
            is_fresh = False
            freshness_reason = "No trigger detected"

        # Validate trigger: fresh + near zone
        trigger_validated = False
        trigger_validation_reason = ""

        if not has_trigger:
            trigger_validated = False
            trigger_validation_reason = "No trigger detected"
        elif not is_fresh:
            trigger_validated = False
            trigger_validation_reason = freshness_reason
        elif zone_proximity == "LATE":
            trigger_validated = False
            trigger_validation_reason = f"Trigger too far from zone ({distance_to_zone:.1f} pips away)"
        else:
            trigger_validated = True
            trigger_validation_reason = f"Fresh {trigger_type} at zone ({zone_proximity} proximity)"

        # STATE DETERMINATION LOGIC (Clear separation)
        # NO_TRADE: No location
        if not has_location:
            setup_state = "NO_TRADE"
            entry_ready = False
            reason = "No valid location identified"

        # WATCH: Location exists but bias conflict
        elif bias_conflict:
            setup_state = "WATCH"
            entry_ready = False
            reason = f"Location present but {conflicting_structure} conflicts with {direction} bias"

        # SETUP: Location valid, waiting for trigger
        elif has_location and not has_trigger:
            setup_state = "SETUP"
            entry_ready = False
            reason = f"Strong location at ${zone['midpoint']:.5f}; waiting for trigger (liquidity grab or breakdown retest)"

        # LATE: Trigger exists but validation failed
        elif has_trigger and not trigger_validated:
            setup_state = "LATE"
            entry_ready = False
            reason = f"Trigger detected but invalid: {trigger_validation_reason}"

        # TRIGGERED: Trigger validated, checking risk
        elif has_trigger and trigger_validated:
            if risk_proximity in ["GOOD", "OK"]:
                setup_state = "ENTRY_READY"  # Final go signal
                entry_ready = True
                reason = f"{trigger_type} at zone ${zone['midpoint']:.5f} - entry actionable now ({entry_window} window)"
            else:
                setup_state = "LATE"  # Trigger good but risk bad
                entry_ready = False
                reason = f"Trigger validated but risk too high ({distance_to_invalidation_pips:.1f} pips to stop, {risk_proximity} proximity)"

        # Fallback
        else:
            setup_state = "WATCH"
            entry_ready = False
            reason = "Monitoring setup development"

        return {
            "setup_state": setup_state,
            "entry_ready": entry_ready,
            "entry_window": entry_window,
            "distance_to_invalidation": round(distance_to_invalidation_pips, 1),
            "distance_to_zone": round(distance_to_zone, 1),
            "trigger_time": trigger_info["trigger_time"],
            "trigger_type": trigger_type,
            "trigger_validated": trigger_validated,
            "trigger_validation_reason": trigger_validation_reason,
            "zone_time": None,  # TODO: Track when zone formed
            "zone_price": round(zone["midpoint"], 5),
            "invalidation_price": round(invalidation_level, 5),
            "atr": round(atr, 5),
            "reason": reason,
            # Bias fields (deterministic)
            "direction": direction,
            "bias_source": bias_source,
            "bias_conflict": bias_conflict,
            "conflicting_structure": conflicting_structure
        }

    def _find_targets(self, current_price: float, confluences: List[Dict],
                     h4_data: pd.DataFrame, direction: str) -> List[Dict]:
        """
        Find target levels for Step 4 trade worthiness (BEARISH GBP/USD)

        Returns ranked list of levels (nearest first) that serve as both:
        - Potential take-profit targets
        - Obstacles for feasibility checks

        Order: opposing zones → swing extremes → ATR projection fallback
        """
        levels = []

        # 1. Find opposing zones from confluences
        # For shorts, find DEMAND zones below current price
        opposing_types = ["DEMAND_ZONE", "ORDER_BLOCK"]

        for conf in confluences:
            if conf.get("type") not in opposing_types:
                continue

            # Extract level from zone
            # For bearish: use top of demand zone (first support)
            if "level" in conf:
                level_price = conf["level"]
            elif "bottom" in conf and "top" in conf:
                level_price = conf["top"]  # Top of demand zone = first support hit
            else:
                continue

            # Only consider levels below current price (bearish targets down)
            if level_price >= current_price:
                continue

            distance = abs(current_price - level_price)
            levels.append({
                "price": level_price,
                "distance": self._to_pips(distance),  # Convert to pips for GBP/USD
                "source": "STRUCTURE",
                "type": conf.get("type", "UNKNOWN")
            })

        # 2. Find swing extremes from H4 data
        # For bearish: find swing lows (support levels below price)
        if h4_data is not None and not h4_data.empty and len(h4_data) >= 5:
            for i in range(2, len(h4_data) - 2):
                current_low = h4_data.iloc[i]['low']

                # Swing low: lower than 2 candles before and after
                is_swing_low = (
                    current_low < h4_data.iloc[i-1]['low'] and
                    current_low < h4_data.iloc[i-2]['low'] and
                    current_low < h4_data.iloc[i+1]['low'] and
                    current_low < h4_data.iloc[i+2]['low']
                )

                if is_swing_low and current_low < current_price:
                    distance = abs(current_price - current_low)
                    levels.append({
                        "price": current_low,
                        "distance": self._to_pips(distance),
                        "source": "SWING_LOW",
                        "type": "H4_SWING"
                    })

        # Sort by distance (nearest first)
        levels.sort(key=lambda x: x["distance"])

        # Deduplicate within 3 pips for GBP/USD
        deduped_levels = []
        for level in levels:
            is_duplicate = False
            for existing in deduped_levels:
                if abs(self._to_pips(level["price"] - existing["price"])) <= 3.0:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduped_levels.append(level)

        # Return max 3 targets
        return deduped_levels[:3]

    def _assess_trade_worthiness(self, entry_state_data: Dict, confluences: List[Dict],
                                 h4_data: pd.DataFrame, current_price: float,
                                 grading: Dict, direction: str) -> Dict:
        """
        Step 4: Trade-worthiness gating (R:R + stop sanity + target feasibility) (BEARISH GBP/USD)

        Evaluates economic quality of ENTRY_READY setups:
        - Gate 1: Stop Sanity (ATR-normalized stop width)
        - Gate 2: Minimum R:R (economic justification)
        - Gate 3: Target Feasibility (path-to-profit check)

        Returns independent trade_worthy decision alongside Step 3 timing.
        """
        # Only assess when ENTRY_READY or TRIGGERED
        if entry_state_data["setup_state"] not in ["ENTRY_READY", "TRIGGERED"]:
            return {
                "trade_worthy": False,
                "trade_decision": "NOT_ASSESSED",
                "decision_reason": f"Not assessing worthiness in {entry_state_data['setup_state']} state",
                "entry_price": None,
                "stop_price": None,
                "targets": {"tp1": None, "tp2": None},
                "tp_source": None,
                "rr": {"to_tp1": None, "to_tp2": None},
                "risk_distance": None,
                "reward_distance_tp1": None,
                "reward_distance_tp2": None,
                "stop_atr_multiple": None,
                "gates": {
                    "stop_sanity_passed": False,
                    "rr_passed": False,
                    "feasibility_passed": False,
                    "feasibility_details": None
                }
            }

        # Use current_price for entry (CRITICAL: reality check, not zone midpoint)
        entry_price = current_price
        stop_price = entry_state_data["invalidation_price"]
        atr = entry_state_data["atr"]
        risk_distance = abs(entry_price - stop_price)
        risk_distance_pips = self._to_pips(risk_distance)

        # Gate 1: Stop Sanity (GBP/USD: <= 0.50 ATR)
        stop_atr_multiple = risk_distance / atr if atr > 0 else 999
        max_stop_atr = 0.50  # GBP/USD threshold
        stop_sanity_passed = stop_atr_multiple <= max_stop_atr

        # Find structural targets
        structural_targets = self._find_targets(current_price, confluences, h4_data, direction)

        # ATR projection fallback (STRICT requirements)
        # Only if: grade="TRADE"/"A+" AND stop <= 0.425 ATR AND entry_window="GOOD"
        can_use_atr_projection = (
            grading["grade"] in ["TRADE", "A+"] and
            stop_atr_multiple <= (max_stop_atr * 0.85) and  # 0.425 for GBP/USD
            entry_state_data["entry_window"] == "GOOD"
        )

        # Build targets
        tp1 = None
        tp2 = None
        tp_source = None

        if len(structural_targets) >= 2:
            # Use structural targets
            tp1 = structural_targets[0]["price"]
            tp2 = structural_targets[1]["price"]
            tp_source = "STRUCTURE"
        elif len(structural_targets) == 1:
            # One structural target
            tp1 = structural_targets[0]["price"]
            # Try ATR projection for TP2 if eligible
            if can_use_atr_projection:
                tp2 = entry_price - (risk_distance * 2.5)
                tp_source = "STRUCTURE_ATR_HYBRID"
            else:
                tp2 = None
                tp_source = "STRUCTURE"
        elif can_use_atr_projection:
            # No structural targets but ATR projection eligible
            tp1 = entry_price - (risk_distance * 1.5)
            tp2 = entry_price - (risk_distance * 2.5)
            tp_source = "ATR_PROJECTION"
        else:
            # No targets available
            tp1 = None
            tp2 = None
            tp_source = "NONE"

        # Calculate R:R
        reward_distance_tp1 = abs(entry_price - tp1) if tp1 is not None else None
        reward_distance_tp2 = abs(entry_price - tp2) if tp2 is not None else None
        reward_distance_tp1_pips = self._to_pips(reward_distance_tp1) if reward_distance_tp1 is not None else None
        reward_distance_tp2_pips = self._to_pips(reward_distance_tp2) if reward_distance_tp2 is not None else None

        rr = {}
        if reward_distance_tp1 is not None and risk_distance > 0:
            rr["to_tp1"] = round(reward_distance_tp1 / risk_distance, 2)
        else:
            rr["to_tp1"] = None

        if reward_distance_tp2 is not None and risk_distance > 0:
            rr["to_tp2"] = round(reward_distance_tp2 / risk_distance, 2)
        else:
            rr["to_tp2"] = None

        # Gate 2: Minimum R:R (GBP/USD: >= 1.6)
        min_rr_required = 1.6
        max_rr = max([r for r in rr.values() if r is not None], default=0)
        rr_passed = max_rr >= min_rr_required

        # Gate 3: Target Feasibility
        # Check if any obstacle <= max(0.15*ATR, 0.30*stop_distance)
        atr_pips = self._to_pips(atr)
        feasibility_threshold = max(0.15 * atr_pips, 0.30 * risk_distance_pips)

        # All levels (structural targets) are potential obstacles
        obstacles = structural_targets

        # Find closest obstacle distance
        closest_obstacle = min([obs["distance"] for obs in obstacles], default=999999)

        feasibility_passed = closest_obstacle > feasibility_threshold
        feasibility_details = {
            "closest_obstacle_distance": round(closest_obstacle, 1) if closest_obstacle < 999999 else None,
            "threshold": round(feasibility_threshold, 1),
            "passed": feasibility_passed
        }

        # Final decision: TAKE if all gates pass, PASS otherwise
        all_gates_passed = stop_sanity_passed and rr_passed and feasibility_passed

        if all_gates_passed:
            trade_decision = "TAKE"
            decision_reason = f"All gates passed: stop={stop_atr_multiple:.2f} ATR, R:R={max_rr:.1f}, clear path"
        else:
            trade_decision = "PASS"
            failures = []
            if not stop_sanity_passed:
                failures.append(f"stop too wide ({stop_atr_multiple:.2f} > {max_stop_atr} ATR)")
            if not rr_passed:
                failures.append(f"R:R insufficient ({max_rr:.1f} < {min_rr_required})")
            if not feasibility_passed:
                failures.append(f"obstacle too close ({closest_obstacle:.1f} pips)")
            decision_reason = "Gates failed: " + "; ".join(failures)

        return {
            "trade_worthy": all_gates_passed,
            "trade_decision": trade_decision,
            "decision_reason": decision_reason,
            "entry_price": round(entry_price, 5),
            "stop_price": round(stop_price, 5),
            "targets": {
                "tp1": round(tp1, 5) if tp1 is not None else None,
                "tp2": round(tp2, 5) if tp2 is not None else None
            },
            "tp_source": tp_source,
            "rr": rr,
            "risk_distance": round(risk_distance_pips, 1),
            "reward_distance_tp1": round(reward_distance_tp1_pips, 1) if reward_distance_tp1_pips is not None else None,
            "reward_distance_tp2": round(reward_distance_tp2_pips, 1) if reward_distance_tp2_pips is not None else None,
            "stop_atr_multiple": round(stop_atr_multiple, 2),
            "gates": {
                "stop_sanity_passed": stop_sanity_passed,
                "rr_passed": rr_passed,
                "feasibility_passed": feasibility_passed,
                "feasibility_details": feasibility_details
            }
        }

    def _check_breakout_retest(self, candles: pd.DataFrame, key_level: float, current_price: float) -> Dict[str, Any]:
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
                if candle['close'] > key_level + 0.0005:  # 5 pips buffer
                    return {"detected": False}  # Setup invalidated

        # PROFESSIONAL RULE: Only check CLOSED H1 candles after breakdown
        retest_happening = False
        rejection_confirmed = False

        # Check only CLOSED H1 candles after breakdown (no forming candles)
        if len(candles_after_breakdown) > 0:
            # Retest = price returned UP close to key level (from below)
            # MUST actually reach within 3 pips of the level to count as a retest
            for i, candle in candles_after_breakdown.iterrows():
                distance_to_level = key_level - candle['high']  # Positive if price below level

                # Only count as retest if HIGH actually reached within 3 pips of resistance
                if distance_to_level <= 0.0003 and distance_to_level >= -0.0002:  # Within 3 pips below or 2 pips above
                    retest_happening = True

                    # Check for BEARISH rejection (long UPPER wick + close BELOW)
                    wick_size = candle['high'] - max(candle['open'], candle['close'])
                    # wick_size > 0.00003 = 3 pips, close < key_level - 0.00005 = 5 pips below
                    if wick_size > 0.00003 and candle['close'] < key_level - 0.00005:
                        rejection_confirmed = True

        # INVALIDATION CHECKS: Only apply if NO confirmation yet
        # Once retest confirmed by closed candles, keep pattern visible
        if not rejection_confirmed and not retest_happening:
            # Time-based expiration (4+ candles without retest)
            if len(candles_after_breakdown) >= 4:
                return {"detected": False}  # Setup expired - took too long

            # Price ran away without retest (20+ pips BELOW breakdown level)
            if current_price < key_level - 0.0020:  # 20 pips = 0.0020 for EUR/USD
                return {"detected": False}  # Setup ran away - retest opportunity expired

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
            "expected_entry": current_price,
            "rejection_confirmed": rejection_confirmed  # Conservative confirmation flag
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

        Args:
            candles: Historical H1 candles (closed only)
            current_price: Current market price (for distance calculation)
        
        PROFESSIONAL RULE: Only uses CLOSED H1 candles for confirmation
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

        # Calculate distance for state determination
        distance_to_fvg = nearest_fvg["midpoint"] - current_price

        # PROFESSIONAL RULE: Only check CLOSED H1 candles, not forming candles
        # Check if any recent CLOSED candle touched the FVG zone
        candle_touched_fvg = False
        strong_rejection = False

        # Check last 3 closed H1 candles for interaction with FVG zone
        recent_candles = candles.tail(3)
        for idx, candle in recent_candles.iterrows():
            # Check if candle's HIGH reached the FVG zone
            if candle['high'] >= nearest_fvg["bottom"]:
                candle_touched_fvg = True

                # Check for STRONG REJECTION (bearish):
                # High touched FVG + closed significantly below (0.0008 = 8 pips for EUR/USD)
                wick_size = candle['high'] - max(candle['open'], candle['close'])
                rejection_distance = candle['high'] - candle['close']

                # Strong rejection = wick into zone + close 8+ pips below the touch
                if wick_size > 0.0003 and rejection_distance >= 0.0008:
                    strong_rejection = True
                    break

        # Only check proximity if no strong rejection yet
        # Once closed candle confirms rejection, keep FVG visible regardless of distance
        if not strong_rejection and not candle_touched_fvg:
            if distance_to_fvg > 0.0020:  # 20 pips = 0.0020 for EUR/USD
                return {"detected": False}  # Too far away and no confirmation yet

        # Determine state based on CLOSED candle price action only
        last_close = float(candles.iloc[-1]['close'])
        if candle_touched_fvg and strong_rejection:
            state = "IN_FVG"  # Touched and rejected - ready for entry
            progress = "3/5"
            confirmations = 2
        elif candle_touched_fvg:
            state = "IN_FVG"  # In the zone but no strong rejection yet
            progress = "2/5"
            confirmations = 1
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
            "key_level": nearest_fvg["top"],  # BEARISH: Use top (price fills gap from above)
            "confirmations": confirmations,
            "expected_entry": nearest_fvg["top"],  # Entry at top of FVG
            "fvg_zone": {
                "top": nearest_fvg["top"],
                "bottom": nearest_fvg["bottom"],
                "midpoint": nearest_fvg["midpoint"],
                "size_pips": nearest_fvg["size"],
                "age_candles": nearest_fvg["age_candles"]
            },
            "distance_pips": float(distance_to_fvg),
            "strong_rejection": strong_rejection  # Conservative confirmation flag
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

        # PROFESSIONAL: Check last 3 CLOSED H1 candles for touches to the OB zone
        candle_touched_ob = False
        strong_rejection = False

        if len(candles) >= 3:
            recent_candles = candles.tail(3)

            for idx, candle in recent_candles.iterrows():
                # Check if candle high touched the OB zone (price went into or above the bottom)
                if candle['high'] >= nearest_ob["bottom"]:
                    candle_touched_ob = True

                    # Check for STRONG REJECTION: candle touched OB and closed 8+ pips below the high
                    rejection_distance = candle['high'] - candle['close']
                    if rejection_distance >= 0.0008:  # 8+ pips = 0.0008 for EUR/USD
                        strong_rejection = True
                        break

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
            "distance_pips": float(distance_to_ob),
            "strong_rejection": strong_rejection  # Conservative confirmation flag
        }

    def _check_supply_zone(self, candles: pd.DataFrame, h4_levels: Dict, current_price: float) -> Dict[str, Any]:
        """
        Check for SUPPLY ZONE pattern (BEARISH):
        - Price approaching H4 RESISTANCE level where sellers historically defended
        - Looking for bearish rejection + SHORT opportunity

        Similar to Order Block but at H4 key levels

        PROFESSIONAL RULE: Only uses CLOSED H1 candles for confirmation

        Args:
            candles: Historical H1 candles (closed only)
            h4_levels: H4 support/resistance levels
            current_price: Current market price (for distance calculation)
        """
        resistance_levels = h4_levels.get("resistance_levels", [])

        if not resistance_levels:
            return {"detected": False}

        # Find nearest resistance level
        nearest_resistance = min(resistance_levels, key=lambda x: abs(x - current_price))

        # Check price distance for state determination
        distance_to_resistance = nearest_resistance - current_price

        # PROFESSIONAL RULE: Only check CLOSED H1 candles for zone touches and rejections
        candle_touched_resistance = False
        strong_rejection = False

        # Check last 3 closed H1 candles for interaction with supply zone
        if len(candles) >= 3:
            recent_candles = candles.tail(3)

            for idx, candle in recent_candles.iterrows():
                # Check if candle high touched resistance (within 3 pips = 0.0003 for EUR/USD)
                if candle['high'] >= nearest_resistance - 0.0003:
                    candle_touched_resistance = True

                    # Check if it had strong rejection (8+ pips = 0.0008 from high to close)
                    candle_rejection = candle['high'] - candle['close']
                    if candle_rejection >= 0.0008:
                        strong_rejection = True
                        break

        # Only check proximity if no strong rejection yet
        # Once closed candle confirms rejection, keep zone visible regardless of distance
        if not strong_rejection:
            if distance_to_resistance > 0.0008 or distance_to_resistance < -0.0005:
                return {"detected": False}  # Too far away and no confirmation yet

        # Determine state based on price action
        if candle_touched_resistance:
            state = "AT_ZONE"
            progress = "2/5"
            confirmations = 1
        elif distance_to_resistance <= 5:
            state = "APPROACHING"
            progress = "1/5"
            confirmations = 0
        else:
            state = "DETECTED"
            progress = "1/5"
            confirmations = 0

        return {
            "detected": True,
            "pattern_type": "SUPPLY_ZONE",
            "direction": "SHORT",
            "state": state,
            "progress": progress,
            "key_level": nearest_resistance,
            "confirmations": confirmations,
            "expected_entry": nearest_resistance,
            "distance_pips": float(distance_to_resistance),
            "strong_rejection": strong_rejection  # Conservative confirmation flag
        }

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
            m5_data = await get_forming_candles(instrument="GBP_USD", granularity="M5", count=20)

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
        elif pattern == "CONFLUENCE_DETECTED":
            return self._confluence_detected_steps(setup, current_price)
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
            "title": f"Price broke above {key_level:.5f} resistance",
            "details": f"Confirmed at {breakout_candle['time']} (1H candle closed at {breakout_candle['price']:.5f})",
            "explanation": "This shows strong buying pressure. The level that was resistance is now support."
        })

        # Step 2: Retest/Pullback
        if retest_candle:
            steps.append({
                "step": 2,
                "status": "complete",
                "title": f"Price pulled back to test {key_level:.5f}",
                "details": f"Retest happening now at {retest_candle['price']:.5f}",
                "explanation": "Healthy pullback. Professional traders use this to enter at better price."
            })
        else:
            steps.append({
                "step": 2,
                "status": "waiting",
                "title": f"WAITING for pullback to {key_level:.5f}",
                "details": f"Current: {current_price:.5f} ({current_price - key_level:.5f} pips above)",
                "explanation": "Waiting for price to dip back down. This creates the 'retest' opportunity."
            })
            return steps

        # Step 3: Rejection Candle
        if state in ["REJECTION_WAITING", "CONFIRMATION_WAITING", "READY_TO_ENTER"]:
            wick_touched = current_candle['low'] <= key_level + 0.0002

            if state == "REJECTION_WAITING":
                # Calculate if early entry conditions are met
                minutes_elapsed = 60 - current_candle['time_remaining']
                price_holding_strong = current_price > key_level + 0.0005
                early_entry_available = wick_touched and minutes_elapsed >= 20 and price_holding_strong

                steps.append({
                    "step": 3,
                    "status": "in_progress",
                    "title": "⏳ WATCHING CURRENT CANDLE for rejection",
                    "details": f"Candle: {current_candle['candle_start']} ({minutes_elapsed}/60 mins complete)",
                    "current_values": {
                        "low": f"{current_candle['low']:.5f}",
                        "current_price": f"{current_price:.5f}",
                        "target_level": f"{key_level:.5f}"
                    },
                    "watching_for": {
                        "wick_requirement": {
                            "status": "✅ Complete" if wick_touched else "⏳ Waiting",
                            "text": f"Wick must touch {key_level:.5f} (support)",
                            "current": f"Low: {current_candle['low']:.5f}",
                            "explanation": "The wick shows sellers tried to push lower but failed."
                        },
                        "holding_requirement": {
                            "status": "✅ Complete" if (wick_touched and minutes_elapsed >= 20) else "⏳ Waiting",
                            "text": f"Price holding above {key_level + 0.0005:.5f} for 20+ minutes",
                            "current": f"Currently: {current_price:.5f} ({minutes_elapsed} mins elapsed)",
                            "explanation": "Price holding strong shows buyers are in control."
                        },
                        "close_requirement": {
                            "status": "⏳ Watching",
                            "text": f"Candle must close above {key_level + 0.0005:.5f}",
                            "current": f"Currently: {current_price:.5f}",
                            "time_left": f"{current_candle['time_remaining']} minutes until close",
                            "explanation": "Close above confirms buyers won the battle."
                        }
                    },
                    "entry_timing": {
                        "early_entry": {
                            "available": early_entry_available,
                            "type": "🟡 EARLY ENTRY (50% Position)",
                            "status": "AVAILABLE NOW" if early_entry_available else "NOT READY",
                            "trigger": f"Wick touched + holding above {key_level + 0.0005:.5f} for 20+ mins",
                            "entry_price": f"{current_price:.5f} (market order)",
                            "stop_loss": f"{key_level - 0.0005:.5f}",
                            "position_size": "50% of planned trade",
                            "pros": "✓ Catch more of the move\n✓ Better average entry price\n✓ Psychological confidence",
                            "cons": "⚠ Candle could still reverse\n⚠ {0} mins until confirmation".format(current_candle['time_remaining']),
                            "action": "Enter 50% position NOW if confident" if early_entry_available else f"Wait {20 - minutes_elapsed} more minutes"
                        },
                        "confirmation_entry": {
                            "type": "🟢 CONFIRMATION ENTRY (Add 50% More)",
                            "trigger": f"Candle closes above {key_level + 0.0005:.5f}",
                            "expected_time": current_candle['candle_close_expected'],
                            "time_remaining": f"{current_candle['time_remaining']} minutes",
                            "entry_price": f"~{current_price + 0.0002:.5f}-{current_price + 0.0005:.5f} (estimated)",
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
                    "details": f"Wick touched {current_candle['low']:.5f}, closed bullish",
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
                        {"text": f"Close above {key_level + 0.0010:.5f}", "explanation": "Shows momentum continuing"},
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
                        "trigger": f"When next 1H candle closes above {current_price:.5f}",
                        "current_count": f"{setup['confirmations']}/3 complete",
                        "pros": "Safest - most confirmation",
                        "cons": "Might miss some entry price"
                    },
                    {
                        "type": "Option B: Enter now (market order)",
                        "trigger": f"Enter at current price: {current_price:.5f}",
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
            "title": f"Price broke BELOW {key_level:.5f} support",
            "details": f"Confirmed at {breakdown_candle['time']} (1H candle closed at {breakdown_candle['price']:.5f})",
            "explanation": "This shows strong selling pressure. The level that was support is now resistance."
        })

        # Step 2: Retest/Pullback
        if retest_candle:
            steps.append({
                "step": 2,
                "status": "complete",
                "title": f"Price pulled back UP to test {key_level:.5f}",
                "details": f"Retest happening now at {retest_candle['price']:.5f}",
                "explanation": "Healthy pullback. Professional traders use this to enter SHORT at better price."
            })
        else:
            steps.append({
                "step": 2,
                "status": "waiting",
                "title": f"WAITING for pullback UP to {key_level:.5f}",
                "details": f"Current: {current_price:.5f} ({key_level - current_price:.5f} pips below)",
                "explanation": "Waiting for price to bounce back up. This creates the 'retest' opportunity."
            })
            return steps

        # Step 3: Rejection Candle
        if state in ["REJECTION_WAITING", "CONFIRMATION_WAITING", "READY_TO_ENTER"]:
            wick_touched = current_candle['high'] >= key_level - 0.0002

            if state == "REJECTION_WAITING":
                # Calculate if early entry conditions are met
                minutes_elapsed = 60 - current_candle['time_remaining']
                price_holding_weak = current_price < key_level - 0.0005
                early_entry_available = wick_touched and minutes_elapsed >= 20 and price_holding_weak

                steps.append({
                    "step": 3,
                    "status": "in_progress",
                    "title": "⏳ WATCHING CURRENT CANDLE for bearish rejection",
                    "details": f"Candle: {current_candle['candle_start']} ({minutes_elapsed}/60 mins complete)",
                    "current_values": {
                        "high": f"{current_candle['high']:.5f}",
                        "current_price": f"{current_price:.5f}",
                        "target_level": f"{key_level:.5f}"
                    },
                    "watching_for": {
                        "wick_requirement": {
                            "status": "✅ Complete" if wick_touched else "⏳ Waiting",
                            "text": f"Wick must touch {key_level:.5f} (resistance)",
                            "current": f"High: {current_candle['high']:.5f}",
                            "explanation": "The wick shows buyers tried to push higher but failed."
                        },
                        "holding_requirement": {
                            "status": "✅ Complete" if (wick_touched and minutes_elapsed >= 20) else "⏳ Waiting",
                            "text": f"Price holding BELOW {key_level - 0.0005:.5f} for 20+ minutes",
                            "current": f"Currently: {current_price:.5f} ({minutes_elapsed} mins elapsed)",
                            "explanation": "Price staying weak shows sellers are in control."
                        },
                        "close_requirement": {
                            "status": "⏳ Watching",
                            "text": f"Candle must close BELOW {key_level - 0.0005:.5f}",
                            "current": f"Currently: {current_price:.5f}",
                            "time_left": f"{current_candle['time_remaining']} minutes until close",
                            "explanation": "Close below confirms sellers won the battle."
                        }
                    },
                    "entry_timing": {
                        "early_entry": {
                            "available": early_entry_available,
                            "type": "🟡 EARLY ENTRY (50% Position)",
                            "status": "AVAILABLE NOW" if early_entry_available else "NOT READY",
                            "trigger": f"Wick touched + holding BELOW {key_level - 0.0005:.5f} for 20+ mins",
                            "entry_price": f"{current_price:.5f} (market order SHORT)",
                            "stop_loss": f"{key_level + 0.0005:.5f}",
                            "position_size": "50% of planned trade",
                            "pros": "✓ Catch more of the move DOWN\n✓ Better average entry price\n✓ Psychological confidence",
                            "cons": "⚠ Candle could still reverse UP\n⚠ {0} mins until confirmation".format(current_candle['time_remaining']),
                            "action": "Enter 50% SHORT position NOW if confident" if early_entry_available else f"Wait {20 - minutes_elapsed} more minutes"
                        },
                        "confirmation_entry": {
                            "type": "🟢 CONFIRMATION ENTRY (Add 50% More)",
                            "trigger": f"Candle closes BELOW {key_level - 0.0005:.5f}",
                            "expected_time": current_candle['candle_close_expected'],
                            "time_remaining": f"{current_candle['time_remaining']} minutes",
                            "entry_price": f"~{current_price - 0.0002:.5f}-{current_price - 0.0005:.5f} (estimated)",
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
                    "details": f"Wick touched {current_candle['high']:.5f}, closed bearish",
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
                        {"text": f"Close BELOW {key_level - 0.0010:.5f}", "explanation": "Shows downward momentum continuing"},
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
            # Calculate SL/TP
            stop_loss = key_level + 0.0005  # 5 pips above resistance
            risk_pips = abs(current_price - stop_loss) / 0.0001
            take_profit = current_price - (risk_pips * 2 * 0.0001)  # 1:2 R:R minimum
            reward_pips = abs(current_price - take_profit) / 0.0001

            steps.append({
                "step": 5,
                "status": "ready",
                "title": "🎯 READY TO ENTER SHORT",
                "entry_options": [
                    {
                        "type": "Option A: Enter at retest level (Aggressive)",
                        "entry": f"{key_level:.5f}",
                        "stop_loss": f"{stop_loss:.5f}",
                        "take_profit": f"{take_profit:.5f}",
                        "risk_pips": f"{risk_pips:.1f} pips",
                        "reward_pips": f"{reward_pips:.1f} pips",
                        "risk_reward": f"1:{reward_pips/risk_pips:.1f}",
                        "trigger": f"Enter SHORT NOW at {current_price:.5f}",
                        "pros": "Best entry price, catch full move",
                        "cons": "No confirmation candle yet",
                        "why_sl": f"SL at {stop_loss:.5f} - If price closes above resistance, retest failed",
                        "why_tp": f"TP at {take_profit:.5f} - Next support level or 1:2 R:R minimum"
                    },
                    {
                        "type": "Option B: Wait for rejection confirmation (Conservative)",
                        "entry": f"Wait for rejection (est. {current_price - 0.0005:.5f})",
                        "stop_loss": f"{stop_loss:.5f}",
                        "take_profit": f"{take_profit:.5f}",
                        "risk_pips": f"{risk_pips + 5:.1f} pips (slightly more)",
                        "reward_pips": f"{reward_pips - 5:.1f} pips (slightly less)",
                        "risk_reward": f"1:{(reward_pips - 5)/(risk_pips + 5):.1f}",
                        "trigger": "Wait for: 1H bearish rejection candle (wick touches resistance + closes 5+ pips below)",
                        "confirmed": setup.get("rejection_confirmed", False),
                        "pros": "Confirmation of sellers defending resistance",
                        "cons": "Slightly worse entry price",
                        "why_sl": f"SL at {stop_loss:.5f} - If price closes above resistance, retest failed",
                        "why_tp": f"TP at {take_profit:.5f} - Next support level or 1:2 R:R minimum"
                    }
                ],
                "recommendation": "Option A for aggressive (retest entry), Option B for conservative (wait for rejection)",
                "explanation": "Breakdown Retest complete! Resistance should hold. High probability SHORT setup."
            })

        return steps

    def _supply_zone_steps(self, setup: Dict, current_price: float) -> List[Dict[str, Any]]:
        """
        Create detailed steps for SUPPLY ZONE pattern (BEARISH)
        H4 resistance level where sellers historically defended
        """
        key_level = setup["key_level"]
        state = setup["state"]
        distance_pips = setup.get("distance_pips", 0)
        strong_rejection = setup.get("strong_rejection", False)

        steps = []

        # Step 1: Supply Zone Detected
        if state == "DETECTED":
            steps.append({
                "step": 1,
                "status": "complete",
                "title": "🏔️ Supply Zone Detected (H4 Resistance)",
                "details": f"Resistance at {key_level:.5f} - Sellers defend this level",
                "explanation": f"H4 resistance zone tested multiple times. Currently {distance_pips:.1f} pips away. Institutions place SELL orders here."
            })

            steps.append({
                "step": 2,
                "status": "in_progress",
                "title": "📍 Price Action",
                "details": f"Price {current_price:.5f} is {distance_pips:.1f} pips below resistance",
                "explanation": "Waiting for price to approach resistance zone (within 10 pips)"
            })

        # Step 2: Approaching
        elif state == "APPROACHING":
            steps.append({
                "step": 1,
                "status": "complete",
                "title": "🏔️ Supply Zone Detected (H4 Resistance)",
                "details": f"Resistance at {key_level:.5f}",
                "explanation": "H4 resistance zone where sellers historically defended"
            })

            steps.append({
                "step": 2,
                "status": "in_progress",
                "title": "📍 Price Approaching",
                "details": f"Price {current_price:.5f} is {distance_pips:.1f} pips from resistance",
                "explanation": "Price getting close to resistance. Prepare for potential rejection."
            })

        # Step 3: At Zone (Entry Options)
        elif state == "AT_ZONE":
            steps.append({
                "step": 1,
                "status": "complete",
                "title": "🏔️ Supply Zone Detected (H4 Resistance)",
                "details": f"Resistance at {key_level:.5f}",
                "explanation": "H4 resistance where sellers historically defended"
            })

            steps.append({
                "step": 2,
                "status": "complete",
                "title": "✅ Price Reached Resistance",
                "details": f"Price touched resistance zone (current: {current_price:.5f}, {distance_pips:.1f} pips from level)",
                "explanation": "Price at key resistance level. Entry signal is active."
            })

            # Calculate SL/TP
            stop_loss = key_level + 0.0005  # 5 pips above resistance
            risk_pips = abs(current_price - stop_loss) / 0.0001
            reward_pips = risk_pips * 2  # 1:2 minimum R:R
            take_profit = current_price - (reward_pips * 0.0001)

            # Calculate for aggressive entry at resistance level
            aggressive_risk_pips = abs(key_level - stop_loss) / 0.0001
            aggressive_reward_pips = aggressive_risk_pips * 2
            aggressive_take_profit = key_level - (aggressive_reward_pips * 0.0001)

            steps.append({
                "step": 3,
                "status": "ready",
                "title": "🎯 READY TO ENTER",
                "entry_options": [
                    {
                        "type": "Option A: Enter at resistance level (Aggressive)",
                        "entry": f"{key_level:.5f}",
                        "stop_loss": f"{stop_loss:.5f}",
                        "take_profit": f"{aggressive_take_profit:.5f}",
                        "risk_pips": f"{aggressive_risk_pips:.1f} pips",
                        "reward_pips": f"{aggressive_reward_pips:.1f} pips",
                        "risk_reward": f"1:{aggressive_reward_pips/aggressive_risk_pips:.1f}",
                        "trigger": f"Enter NOW at {key_level:.5f}",
                        "pros": "Best entry price, highest R:R",
                        "cons": "No confirmation of rejection",
                        "why_sl": f"SL at {stop_loss:.5f} - If price closes above resistance, setup invalidated",
                        "why_tp": f"TP at {aggressive_take_profit:.5f} - Next support level or 1:2 R:R minimum"
                    },
                    {
                        "type": "Option B: Wait for bearish rejection (Conservative)",
                        "entry": f"Wait for rejection (est. {current_price:.5f})",
                        "stop_loss": f"{stop_loss:.5f}",
                        "take_profit": f"{take_profit:.5f}",
                        "risk_pips": f"{risk_pips:.1f} pips (slightly more)",
                        "reward_pips": f"{reward_pips:.1f} pips (slightly less)",
                        "risk_reward": f"1:{reward_pips/risk_pips:.1f}",
                        "trigger": "Wait for 15+ pip bearish rejection from resistance",
                        "confirmed": strong_rejection,
                        "pros": "Confirmation of sellers stepping in",
                        "cons": "Slightly worse entry price",
                        "why_sl": f"SL at {stop_loss:.5f} - If price closes above resistance, setup invalidated",
                        "why_tp": f"TP at {take_profit:.5f} - Next support level or 1:2 R:R minimum"
                    }
                ],
                "recommendation": "Option A for aggressive (resistance level entry), Option B for conservative (wait for rejection)",
                "explanation": "Supply Zone activated! Institutions left SELL orders at this resistance. High probability of rejection DOWN."
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
            "details": f"FVG Zone: {fvg_zone.get('bottom', 0):.5f} - {fvg_zone.get('top', 0):.5f} ({fvg_zone.get('size_pips', 0):.1f} pips)",
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
                "details": f"Current: {current_price:.5f} (inside FVG zone)",
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
                        "trigger": f"Enter NOW at {fvg_zone.get('midpoint', 0):.5f}",
                        "pros": "Best entry price, highest R:R",
                        "cons": "No confirmation candle"
                    },
                    {
                        "type": "Option B: Wait for bearish confirmation (Conservative)",
                        "trigger": "Wait for: (1) 1H bearish candle close inside FVG OR (2) Strong rejection (15+ pip drop from FVG)",
                        "confirmed": setup.get("strong_rejection", False),
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
        stop_loss = ob_top + 0.0005

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
            "details": f"OB Zone: {ob_zone.get('bottom', 0):.5f} - {ob_zone.get('top', 0):.5f} ({ob_zone.get('size_pips', 0):.1f} pips)",
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
                        "entry": f"{ob_zone.get('midpoint', 0):.5f}",
                        "stop_loss": f"{sl_tp['stop_loss']:.5f}",
                        "take_profit": f"{sl_tp['take_profit']:.5f}",
                        "risk_pips": f"{sl_tp['risk_pips']:.1f} pips",
                        "reward_pips": f"{sl_tp['reward_pips']:.1f} pips",
                        "risk_reward": f"1:{sl_tp['risk_reward_ratio']:.1f}",
                        "trigger": f"SELL NOW at {ob_zone.get('midpoint', 0):.5f}",
                        "pros": "Best entry price, highest R:R",
                        "cons": "No confirmation candle",
                        "why_sl": f"SL at {sl_tp['stop_loss']:.5f} - If price closes above OB zone, setup is invalidated",
                        "why_tp": f"TP at {sl_tp['take_profit']:.5f} - Next support level or 1:2 R:R minimum"
                    },
                    {
                        "type": "Option B: Wait for bearish confirmation (Conservative)",
                        "entry": f"Wait for close (est. {ob_zone.get('bottom', 0):.5f})",
                        "stop_loss": f"{sl_tp['stop_loss']:.5f}",
                        "take_profit": f"{sl_tp['take_profit']:.5f}",
                        "risk_pips": f"{sl_tp['risk_pips'] + 5:.1f} pips (slightly more)",
                        "reward_pips": f"{sl_tp['reward_pips'] - 5:.1f} pips (slightly less)",
                        "risk_reward": f"1:{(sl_tp['reward_pips'] - 5) / (sl_tp['risk_pips'] + 5):.1f}",
                        "trigger": "Wait for: (1) 1H bearish candle close inside OB OR (2) Strong rejection (15+ pip drop from OB)",
                        "confirmed": setup.get("strong_rejection", False),
                        "pros": "Confirmation of sellers stepping in",
                        "cons": "Slightly worse entry price",
                        "why_sl": f"SL at {sl_tp['stop_loss']:.5f} - If price closes above OB zone, setup is invalidated",
                        "why_tp": f"TP at {sl_tp['take_profit']:.5f} - Next support level or 1:2 R:R minimum"
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

    def _confluence_detected_steps(self, setup: Dict, current_price: float) -> List[Dict[str, Any]]:
        """Steps for CONFLUENCE_DETECTED pattern (BEARISH)"""
        confluences = setup.get("confluences", [])
        total_score = setup.get("total_score", 0)
        key_level = setup.get("key_level")
        state = setup.get("state", "WAITING_CONFIRMATION")

        steps = []

        # Step 1: Show all confluences detected
        confluence_list = []
        for conf in confluences:
            confluence_list.append(f"• {conf['type'].replace('_', ' ').title()}: {conf['score']} points")

        steps.append({
            "step": 1,
            "status": "complete",
            "title": f"✅ {total_score} Confluence Points Detected",
            "details": "\n".join(confluence_list),
            "explanation": f"Multiple patterns aligned at {key_level:.5f} resistance. Professional setup forming."
        })

        # Step 2: Dynamic entry status based on price location (BEARISH)
        entry_zone = key_level - 0.0002
        distance_to_entry = current_price - entry_zone  # Positive = above entry, negative = below entry

        if state == "WAITING_CONFIRMATION":
            # Check if price is at/near entry zone (within 2 pips)
            if abs(distance_to_entry) <= 0.0002:
                # PRICE AT ENTRY ZONE!
                steps.append({
                    "step": 2,
                    "status": "complete",
                    "title": "🎯 PRICE AT ENTRY ZONE - READY TO ENTER!",
                    "details": f"Current: {current_price:.5f} | Entry: {entry_zone:.5f} ({abs(distance_to_entry):.1f} pips away)",
                    "explanation": "Price reached entry zone! Enter now with market/limit order."
                })
            elif distance_to_entry < -0.0002:
                # Price below entry - waiting for bounce up
                steps.append({
                    "step": 2,
                    "status": "in_progress",
                    "title": "⏳ Waiting for bounce to entry zone",
                    "details": f"Price: {current_price:.5f} | Entry: {entry_zone:.5f} ({abs(distance_to_entry):.1f} pips below)",
                    "explanation": "Place limit order at entry or wait for price to bounce up."
                })
            else:
                # Price above entry - already through zone
                steps.append({
                    "step": 2,
                    "status": "in_progress",
                    "title": "⚠️ Price above entry zone",
                    "details": f"Price: {current_price:.5f} | Entry: {entry_zone:.5f} ({distance_to_entry:.1f} pips above)",
                    "explanation": "Price moved above entry. Wait for price to come back down or monitor for invalidation."
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
                f"Key Level: {key_level:.5f} ({level_type})",
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
        """Build complete trade plan with entry, SL, TP based on confluence patterns"""
        total_score = setup.get("total_score", 0)

        # Only provide trade plan if we have sufficient confluence (5+ points)
        if total_score < 5:
            return {
                "status": "Not ready yet",
                "message": f"Need 5+ confluence points (currently {total_score}/5)"
            }

        # Determine entry, SL, TP based on detected confluences
        confluences = setup.get("confluences", [])
        key_level = setup.get("key_level", 0)

        # Find key levels from confluences
        liquidity_grab = next((c for c in confluences if c["type"] == "LIQUIDITY_GRAB"), None)
        supply_zone = next((c for c in confluences if c["type"] == "SUPPLY_ZONE"), None)
        order_block = next((c for c in confluences if c["type"] == "ORDER_BLOCK"), None)
        fvg = next((c for c in confluences if c["type"] == "FVG"), None)
        breakdown_retest = next((c for c in confluences if c["type"] == "BREAKDOWN_RETEST"), None)

        # Get actual liquidity grab high (not from description parsing!)
        liquidity_grab_high = None
        if liquidity_grab:
            liquidity_grab_high = liquidity_grab.get("grab_high")

        # ENTRY LOGIC: Professional methodology (BEARISH)
        # Entry is at H4 resistance (key_level) or order block/FVG zone
        # NOT at the liquidity grab level itself
        if liquidity_grab or order_block:
            # Enter at H4 resistance/order block (where institutions will sell)
            entry = key_level - 0.0002  # 2 pips below H4 resistance
            entry_reason = f"Enter 2 pips below H4 resistance {key_level:.5f} (order block/liquidity grab confirmed)"
        elif supply_zone:
            # Enter at supply zone
            entry = key_level - 0.0002
            entry_reason = f"Enter 2 pips below {key_level:.5f} supply zone"
        else:
            # Default: enter at key level
            entry = current_price if current_price < key_level else key_level - 0.0002
            entry_reason = f"Enter at current price {current_price:.5f}"

        # STOP LOSS: Professional methodology (BEARISH)
        # OPTION 3: Use closest confluence to entry for SL
        # All patterns use 6 pip buffer (0.0006 for EUR/USD)
        resistance_levels = h4_levels.get("resistance_levels", [])

        confluence_levels = []

        # Collect all confluences with their levels and 6 pip buffers
        if liquidity_grab and liquidity_grab_high:
            confluence_levels.append({
                "name": "Liquidity Grab",
                "level": liquidity_grab_high,
                "sl": liquidity_grab_high + 0.0006,  # Changed from 0.0005 to 0.0006, ABOVE for bearish
                "reason": f"Above liquidity grab high ({liquidity_grab_high:.5f}) - stops already swept"
            })

        if supply_zone:
            confluence_levels.append({
                "name": "Supply Zone",
                "level": key_level,
                "sl": key_level + 0.0006,  # ABOVE for bearish
                "reason": f"Above supply zone ({key_level:.5f}) with 6 pip buffer"
            })

        if order_block:
            ob_desc = order_block.get("description", "")
            import re
            match = re.search(r'\$(\d+\.\d+)-\$(\d+\.\d+)', ob_desc)
            if match:
                ob_top = float(match.group(2))  # Use top for bearish
                confluence_levels.append({
                    "name": "Order Block",
                    "level": ob_top,
                    "sl": ob_top + 0.0006,  # ABOVE for bearish
                    "reason": f"Above Order Block top ({ob_top:.5f}) with 6 pip buffer"
                })

        if fvg:
            fvg_level = setup.get("fvg_zone", {}).get("top") if "fvg_zone" in setup else None  # Use top for bearish
            if fvg_level:
                confluence_levels.append({
                    "name": "FVG",
                    "level": fvg_level,
                    "sl": fvg_level + 0.0006,  # ABOVE for bearish
                    "reason": f"Above FVG top ({fvg_level:.5f}) with 6 pip buffer"
                })

        if breakdown_retest:
            breakdown_level = breakdown_retest.get("key_level") if isinstance(breakdown_retest, dict) else key_level
            if breakdown_level:
                confluence_levels.append({
                    "name": "Breakdown Retest",
                    "level": breakdown_level,
                    "sl": breakdown_level + 0.0006,  # ABOVE for bearish
                    "reason": f"Above breakdown level ({breakdown_level:.5f}) with 6 pip buffer"
                })

        # Find closest confluence to entry
        if confluence_levels:
            for conf in confluence_levels:
                conf["distance"] = abs(entry - conf["level"])

            closest = min(confluence_levels, key=lambda x: x["distance"])
            sl = closest["sl"]
            sl_reason = closest["reason"]
        else:
            # Fallback: 6 pips above entry (changed from 15)
            sl = entry + 0.0006
            sl_reason = "6 pip buffer above entry"

        # TAKE PROFIT: Professional methodology - target opposing liquidity pools (BEARISH)
        # Look for support levels (swing lows, psychological levels) below entry
        support_levels = h4_levels.get("support_levels", [])

        # Calculate minimum TP based on R:R ratios (aim for 2:1 and 3:1)
        risk_pips = sl - entry
        min_tp1_distance = risk_pips * 2  # 2:1 R:R minimum
        min_tp2_distance = risk_pips * 3  # 3:1 R:R minimum

        # Find support within reasonable range (80 pips for EUR/USD)
        nearby_support = [s for s in support_levels if s < entry and s > entry - 0.0080]

        # Filter support that meets minimum R:R requirements
        valid_tp1_levels = [s for s in nearby_support if (entry - s) >= min_tp1_distance]
        valid_tp2_levels = [s for s in nearby_support if (entry - s) >= min_tp2_distance]

        if len(valid_tp1_levels) >= 1 and len(valid_tp2_levels) >= 1:
            # Use structure-based support levels - place 5 pips BEFORE (above) to ensure fill
            support_1 = valid_tp1_levels[0]
            support_2 = valid_tp2_levels[0] if valid_tp2_levels[0] != support_1 else (valid_tp2_levels[1] if len(valid_tp2_levels) > 1 else support_1 - 0.0015)
            tp1 = support_1 + 0.0005  # 5 pips before (above) support (1 pip = 0.0001 for EUR/USD)
            tp2 = support_2 + 0.0005  # 5 pips before (above) support
            tp1_reason = f"5 pips before H4 support {support_1:.5f} (at {tp1:.5f})"
            tp2_reason = f"5 pips before H4 support {support_2:.5f} (at {tp2:.5f})"
        elif len(valid_tp1_levels) >= 1:
            # One valid support found
            support_1 = valid_tp1_levels[0]
            tp1 = support_1 + 0.0005  # 5 pips before (above) support
            tp2 = entry - min_tp2_distance  # Use R:R based target for TP2
            tp1_reason = f"5 pips before H4 support {support_1:.5f} (at {tp1:.5f})"
            tp2_reason = f"Extended target (3:1 R:R) at {tp2:.5f}"
        else:
            # No structure found - use R:R based targets (2:1 and 3:1 minimum)
            tp1 = entry - min_tp1_distance
            tp2 = entry - min_tp2_distance
            tp1_reason = f"Target (2:1 R:R) at {tp1:.5f}"
            tp2_reason = f"Extended target (3:1 R:R) at {tp2:.5f}"

        # Calculate R:R ratios
        # For EUR/USD: 1 pip = 0.0001, so multiply dollar difference by 10000
        risk_pips = (sl - entry) * 10000
        reward1_pips = (entry - tp1) * 10000
        reward2_pips = (entry - tp2) * 10000
        rr1 = reward1_pips / risk_pips if risk_pips > 0 else 0
        rr2 = reward2_pips / risk_pips if risk_pips > 0 else 0

        # Build confluence reasoning
        confluence_reasons = []
        for c in confluences:
            confluence_reasons.append(f"• {c['type'].replace('_', ' ').title()}: {c['score']} points")

        return {
            "status": "Ready",
            "entry_method": "Limit order" if entry > current_price else "Market order",
            "entry_price": f"{entry:.5f}",
            "entry_reasoning": entry_reason,
            "stop_loss": {
                "price": f"{sl:.5f}",
                "reason": sl_reason,
                "pips": f"{risk_pips:.1f} pips risk",
                "why": "If price goes here, setup invalidated - liquidity already grabbed, further upside means failure"
            },
            "take_profit_1": {
                "price": f"{tp1:.5f}",
                "reason": tp1_reason,
                "pips": f"-{reward1_pips:.1f} pips",
                "rr_ratio": f"{rr1:.2f}:1",
                "action": "Close 50%, move SL to breakeven",
                "why": "Lock in profit, secure risk-free trade"
            },
            "take_profit_2": {
                "price": f"{tp2:.5f}",
                "reason": tp2_reason,
                "pips": f"-{reward2_pips:.1f} pips",
                "rr_ratio": f"{rr2:.2f}:1",
                "action": "Close remaining 50%",
                "why": "Full profit target reached"
            },
            "confluence_summary": {
                "total_points": total_score,
                "reasons": confluence_reasons,
                "why_trade": f"High-probability setup with {total_score}/5 confluence points. " +
                            ("Liquidity already grabbed above, stops swept. " if liquidity_grab else "") +
                            ("Strong supply zone holding. " if supply_zone else "") +
                            ("Institutional selling visible. " if order_block else "")
            },
            "position_sizing": {
                "account_risk": "0.5% recommended",
                "risk_pips": f"{risk_pips:.1f}",
                "calculation": f"Position size = (Account × 0.005) / {risk_pips:.1f} pips"
            }
        }

    def _monitor_active_trade(self, current_price: float, current_candle: Dict = None) -> Dict[str, Any]:
        """
        Monitor an active SHORT trade's progress toward TP/SL
        Don't re-validate confluences - just track price vs targets
        """
        if self.active_trade is None:
            return {"is_active": False}

        # TIME-BASED EXIT: After 12 hours, close setup and look for new one
        triggered_at_str = self.active_trade.get("triggered_at")
        if triggered_at_str:
            triggered_at = datetime.fromisoformat(triggered_at_str)
            now_utc = datetime.now(pytz.UTC)
            hours_elapsed = (now_utc - triggered_at).total_seconds() / 3600

            if hours_elapsed >= 12:
                # Clear active trade - resume scanning for new setups
                self.active_trade = None
                self._save_active_trade()  # Clear from file
                return {"is_active": False}

        trade_plan = self.active_trade["trade_plan"]
        setup = self.active_trade["setup"]
        entry = self.active_trade["entry"]
        h4_levels = self.active_trade.get("h4_levels", {})
        daily_analysis = self.active_trade.get("daily_analysis", {})

        # Use full current candle data if provided
        if current_candle is None:
            current_candle = {
                "timeframe": "1H",
                "current": current_price,
                "entry": entry,
                "pnl_pips": round(entry - current_price, 1) if entry else 0,
                "pnl_percent": 0
            }

        # Extract TP and SL levels
        tp1_str = trade_plan.get("take_profit_1", {}).get("price", "$0")
        tp2_str = trade_plan.get("take_profit_2", {}).get("price", "$0")
        sl_str = trade_plan.get("stop_loss", {}).get("price", "$0")

        tp1 = float(tp1_str.replace("$", "").replace(",", ""))
        tp2 = float(tp2_str.replace("$", "").replace(",", ""))
        sl = float(sl_str.replace("$", "").replace(",", ""))

        # Check if TP or SL hit (SHORT: TP is below entry, SL is above)
        # For SHORT trades: Check LOW for TP (price touched below), HIGH for SL (price touched above)
        trade_result = None
        candle_high = current_candle.get('high', current_price)
        candle_low = current_candle.get('low', current_price)

        if candle_low <= tp2:
            trade_result = "TP2_HIT"
        elif candle_low <= tp1:
            trade_result = "TP1_HIT"
        elif candle_high >= sl:
            trade_result = "SL_HIT"

        # Calculate current P&L (SHORT: profit when price goes down)
        pnl_pips = entry - current_price
        pnl_percent = (pnl_pips / entry) * 100 if entry > 0 else 0

        # Build trade progress status
        if trade_result == "SL_HIT":
            status = "TRADE_STOPPED_OUT"
            status_message = f"❌ Stop Loss Hit at {sl:.5f}"
            is_active = False  # Trade finished
        elif trade_result == "TP2_HIT":
            status = "TRADE_COMPLETED"
            status_message = f"✅ TP2 Hit! Full profit taken at {tp2:.5f}"
            is_active = False  # Trade finished
        elif trade_result == "TP1_HIT":
            status = "TRADE_ACTIVE_TP1_HIT"
            status_message = f"✅ TP1 Hit! Waiting for TP2 at {tp2:.5f}"
            is_active = True  # Still watching for TP2
        else:
            # Trade still active, no TP/SL hit yet
            status = "TRADE_ACTIVE"

            # Show progress based on price location
            # For EUR/USD: 1 pip = 0.0001, so multiply dollar difference by 10000
            distance_to_tp1 = (current_price - tp1) * 10000  # SHORT: distance is current - tp1
            distance_to_sl = (sl - current_price) * 10000

            if pnl_pips >= 0:
                if distance_to_tp1 <= 50:  # Within 50 pips = approaching
                    status_message = f"🎯 Approaching TP1 ({tp1:.5f}) - {distance_to_tp1:.1f} pips away"
                else:
                    status_message = f"📈 In Profit +{pnl_pips:.1f} pips | Target: TP1 {tp1:.5f}"
            else:
                if abs(pnl_pips) <= 3:
                    status_message = f"⏸️ At Entry {entry:.5f} ({pnl_pips:+.1f} pips)"
                else:
                    status_message = f"📉 Pullback {pnl_pips:.1f} pips | SL: {sl:.5f} ({distance_to_sl:.1f} pips away)"

            is_active = True

        # Build response with locked-in trade data
        response = {
            "status": "success",
            "pair": self.pair,
            "current_price": current_price,
            "setup_status": status,
            "setup_progress": "MONITORING",
            "pattern_type": "ACTIVE_TRADE",

            # Keep original confluence data (locked in)
            "confluences": setup.get("confluences", []),
            "total_score": setup.get("total_score", 0),
            "confidence": setup.get("confidence", None),
            "structure": setup.get("structure", {"structure_type": "NEUTRAL", "score": 0}),

            # Trade monitoring steps
            "setup_steps": self._build_active_trade_steps(current_price, entry, tp1, tp2, sl, pnl_pips, trade_result),

            # Context (doesn't change)
            "why_this_setup": {
                "daily": daily_analysis,
                "h4": self._explain_h4_structure(h4_levels, current_price),
                "h1": {
                    "pattern": "ACTIVE_TRADE",
                    "points": [
                        f"Trade Status: {status}",
                        f"Entry: {entry:.5f}",
                        f"Current P&L: {pnl_pips:+.1f} pips ({pnl_percent:+.2f}%)",
                        status_message
                    ]
                },
                "session": self._explain_session_context()
            },

            # Show current candle (full data with high/low/open + P&L)
            "live_candle": {
                **current_candle,  # Include all candle data (high, low, open, time_remaining, etc.)
                "entry": entry,
                "pnl_pips": round(pnl_pips, 1),
                "pnl_percent": round(pnl_percent, 2)
            },

            # Trade plan (locked in)
            "trade_plan": trade_plan,

            # Invalidation = SL level
            "invalidation": [
                {
                    "condition": f"Stop Loss at {sl:.5f}",
                    "reason": f"Trade invalidated if price hits {sl:.5f}",
                    "action": f"Exit trade - {abs(entry - sl):.1f} pip loss",
                    "severity": "CRITICAL"
                }
            ],

            # Chart data would need to be fetched separately if needed
            "chart_data": {
                "h1_candles": [],  # Skip for now
                "key_levels": h4_levels,
                "current_price": current_price
            },

            "last_update": datetime.now(pytz.UTC).isoformat()
        }

        return {"is_active": is_active, "response": response}

    def _build_active_trade_steps(self, current_price: float, entry: float, tp1: float, tp2: float, sl: float, pnl_pips: float, trade_result: str) -> List[Dict[str, Any]]:
        """Build steps display for active SHORT trade monitoring"""
        steps = []

        # Step 1: Entry confirmation
        steps.append({
            "step": 1,
            "status": "complete",
            "title": f"✅ Entered SHORT at {entry:.5f}",
            "details": f"Trade active with {abs(entry - sl) * 10000:.1f} pip stop loss",
            "explanation": "Position opened based on 5+ confluence points"
        })

        # Step 2: TP1 status (SHORT: TP is below entry)
        # For EUR/USD: 1 pip = 0.0001, so multiply dollar difference by 10000
        distance_to_tp1 = (current_price - tp1) * 10000  # For SHORT positions
        if trade_result == "TP2_HIT" or trade_result == "TP1_HIT":
            tp1_status = "complete"
            tp1_title = f"✅ TP1 Hit at {tp1:.5f}"
        elif distance_to_tp1 <= 50:  # Within 50 pips = approaching
            tp1_status = "in_progress"
            tp1_title = f"🎯 Approaching TP1 - {distance_to_tp1:.1f} pips away"
        else:
            tp1_status = "pending"
            tp1_title = f"⏳ Target TP1 at {tp1:.5f} ({distance_to_tp1:.1f} pips)"

        steps.append({
            "step": 2,
            "status": tp1_status,
            "title": tp1_title,
            "details": f"Close 50% position, move SL to breakeven",
            "explanation": "Lock in profit and make trade risk-free"
        })

        # Step 3: TP2 status
        # For EUR/USD: 1 pip = 0.0001, so multiply dollar difference by 10000
        distance_to_tp2 = (current_price - tp2) * 10000  # For SHORT positions
        if trade_result == "TP2_HIT":
            tp2_status = "complete"
            tp2_title = f"✅ TP2 Hit at {tp2:.5f} - Trade Complete!"
        elif trade_result == "TP1_HIT":
            tp2_status = "in_progress"
            tp2_title = f"🎯 Targeting TP2 at {tp2:.5f} ({distance_to_tp2:.1f} pips)"
        else:
            tp2_status = "pending"
            tp2_title = f"⏳ Final Target TP2 at {tp2:.5f} ({distance_to_tp2:.1f} pips)"

        steps.append({
            "step": 3,
            "status": tp2_status,
            "title": tp2_title,
            "details": f"Close remaining 50% position",
            "explanation": "Full profit target - trade complete"
        })

        # Step 4: Current P&L
        if pnl_pips >= 0:
            pnl_emoji = "📈"
            pnl_status = "complete"
        else:
            pnl_emoji = "📉"
            pnl_status = "pending"

        steps.append({
            "step": 4,
            "status": pnl_status,
            "title": f"{pnl_emoji} Current P&L: {pnl_pips:+.1f} pips",
            "details": f"Price: {current_price:.5f} | Entry: {entry:.5f}",
            "explanation": "Live profit/loss tracking"
        })

        return steps

    def _load_active_trade(self):
        """Load active trade from file if exists (persists across API calls)"""
        if os.path.exists(self.active_trade_file):
            try:
                with open(self.active_trade_file, 'r') as f:
                    self.active_trade = json.load(f)
            except Exception as e:
                # If file is corrupted, delete it and start fresh
                print(f"Error loading active trade: {e}")
                self.active_trade = None
                if os.path.exists(self.active_trade_file):
                    os.remove(self.active_trade_file)
        else:
            self.active_trade = None

    def _save_active_trade(self):
        """Save active trade to file (persists across API calls)"""
        if self.active_trade is not None:
            try:
                with open(self.active_trade_file, 'w') as f:
                    json.dump(self.active_trade, f, indent=2)
            except Exception as e:
                print(f"Error saving active trade: {e}")

    def _delete_active_trade(self):
        """Delete active trade file when trade exits"""
        self.active_trade = None
        if os.path.exists(self.active_trade_file):
            try:
                os.remove(self.active_trade_file)
            except Exception as e:
                print(f"Error deleting active trade: {e}")

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
                "condition": f"Price closes back ABOVE {key_level:.5f}",
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
                "condition": f"Current candle closes ABOVE {key_level + 0.0005:.5f}",
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

        elif state == "RETEST_WAITING":
            # Calculate dynamic distance threshold (20 pips below breakdown for BEARISH)
            runaway_threshold = key_level - 20

            conditions.append({
                "condition": f"Price moves below {runaway_threshold:.5f} without pullback",
                "reason": "Setup ran away - strong momentum without retest (20+ pips below breakdown)",
                "action": "Cancel setup. Look for new entry or pattern.",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": f"Price closes back ABOVE {key_level:.5f}",
                "reason": "Breakdown failed - false breakdown",
                "action": "Cancel setup immediately. Breakdown reversed.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "4+ hours passed without retest",
                "reason": "Setup expired - retest took too long",
                "action": "Move on to new patterns. Opportunity missed.",
                "severity": "MEDIUM"
            })

        elif state == "REJECTION_WAITING":
            # For bearish: If price closes ABOVE resistance, rejection failed
            conditions.append({
                "condition": f"Candle closes ABOVE {key_level:.5f}",
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
                "condition": f"Next candle closes ABOVE {key_level:.5f}",
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
                "condition": f"Entry candle closes ABOVE {key_level:.5f}",
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
            "condition": f"Price rises {(key_level * 0.01):.5f}+ ABOVE resistance",
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
                "condition": f"Price continues DOWN without filling FVG ({gap_top:.5f})",
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
                "condition": f"Price reverses DOWN before reaching FVG top ({gap_top:.5f})",
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
                "condition": f"Price closes ABOVE gap top ({gap_top + 0.0005:.5f})",
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
                "condition": f"Price continues DOWN without reaching OB ({ob_top:.5f})",
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
                "condition": f"Price reverses DOWN before reaching OB top ({ob_top:.5f})",
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
                "condition": f"Price closes ABOVE OB top ({ob_top + 0.0003:.5f})",
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
                "condition": f"Price closes ABOVE {key_level + 0.0010:.5f}",
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
                "condition": f"Price closes decisively ABOVE {key_level + 0.0010:.5f}",
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
                "condition": f"Next candle closes ABOVE {key_level:.5f}",
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
            "condition": f"Price rises {(key_level * 0.01):.5f}+ ABOVE resistance",
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
    """Get BEARISH professional trader analysis for GBP/USD (SELL setups only)"""
    system = BearishProTraderGBPUSD()
    return await system.get_detailed_setup()
