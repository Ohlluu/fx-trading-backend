#!/usr/bin/env python3
"""
Professional Trader EUR/USD System - Educational Setup Tracker
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

class BullishProTraderEURUSD:
    """
    BULLISH Educational trading system - looks for BUY setups
    Detects breakouts above resistance and bullish rejections
    Professional day trading logic for LONG positions
    """

    def __init__(self):
        self.pair = "EURUSD"
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
            "demand_zone": None,
            "last_entry_state": None,  # Track last entry state to prevent flickering
            "entry_state_since": None  # When current state started
        }

        # ACTIVE TRADE TRACKER: Lock in setup when 5+ confluence triggers
        # Once a trade is active, stop re-validating confluences and just monitor TP/SL
        # PERSISTED to file system so it survives across API calls
        self.active_trade_file = "/tmp/active_trade_bullish_eurusd.json"
        self.active_trade = None  # {"setup": {...}, "trade_plan": {...}, "triggered_at": timestamp, "entry": price}

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
            current_price = await get_current_price("EUR_USD")
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
            h1_setup = await self._detect_setup_pattern(h1_data, h4_levels, current_price)

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
        last_updated = mtf_cache.get_last_update("D1")
        next_update = mtf_cache.get_next_update("D1")

        # Convert last_updated to Chicago time for display
        last_updated_str = "Just now"
        if last_updated:
            last_updated_ct = last_updated.astimezone(chicago_tz)
            last_updated_str = last_updated_ct.strftime("%b %d, %I:%M %p CT")

        return {
            "trend": trend,
            "explanation": f"Price is {'above' if trend == 'BULLISH' else 'below'} 200-day EMA (${ema_200_current:.5f})",
            "points": [
                f"Trend: {trend} (current: ${current_price:.5f} vs 200 EMA: ${ema_200_current:.5f})",
                f"Recent high: ${recent_high:.5f} ({days_since_high} days ago)",
                f"Recent low: ${recent_low:.5f} ({days_since_low} days ago)",
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

        # Find significant support AND resistance levels
        support_candidates = []
        resistance_candidates = []

        # Find local lows (support) and local highs (resistance) using 4H candles
        for i in range(5, len(lows) - 5):
            # Support: local low
            if lows.iloc[i] == lows.iloc[i-5:i+5].min():
                support_candidates.append(float(lows.iloc[i]))

            # Resistance: local high
            if highs.iloc[i] == highs.iloc[i-5:i+5].max():
                resistance_candidates.append(float(highs.iloc[i]))

        # Add psychological levels (round numbers: $3,900, $3,950, $4,000, etc.)
        # These are important because traders place orders at round numbers
        psychological_levels_support = []
        psychological_levels_resistance = []
        price_range_start = round((current_price - 0.0050) * 10000) / 10000  # Start 50 pips below
        price_range_end = round((current_price + 0.0050) * 10000) / 10000    # End 50 pips above

        # For EUR/USD, use 0.0050 increments (50 pips)
        level = price_range_start
        while level <= price_range_end:
            # Support: below current price
            if level < current_price and (current_price - level) <= 0.0050:
                psychological_levels_support.append(float(level))
            # Resistance: above current price
            elif level > current_price and (level - current_price) <= 0.0050:
                psychological_levels_resistance.append(float(level))
            level = round((level + 0.0050) * 10000) / 10000

        # Combine swing levels + psychological levels
        all_support_candidates = support_candidates + psychological_levels_support
        all_resistance_candidates = resistance_candidates + psychological_levels_resistance

        # Remove duplicates and get support levels BELOW current price (potential BUY zones)
        support = sorted(list(set([s for s in all_support_candidates if s < current_price])), reverse=True)[:5] if all_support_candidates else []

        # Remove duplicates and get resistance levels ABOVE current price (potential TP zones)
        resistance = sorted(list(set([r for r in all_resistance_candidates if r > current_price])))[:5] if all_resistance_candidates else []

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
            "resistance_levels": resistance,  # FIX: Track resistance for TP targets
            "support_levels": support if support else [key_level],
            "distance_pips": round(distance_pips, 1),
            "last_updated": last_updated_str,
            "next_update": next_update
        }

    def _check_bos_choch(self, candles: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Check for Break of Structure (BOS) or Change of Character (CHoCH)

        BOS = Price breaks previous swing high/low in SAME direction (trend continuation)
        CHoCH = Price breaks previous swing high/low in OPPOSITE direction (trend reversal)

        For BULLISH trader:
        - Bullish BOS = Close above previous swing high (confirms uptrend)
        - Bearish CHoCH = Close below previous higher low (signals potential reversal DOWN)

        Returns:
            - structure_type: "BULLISH_BOS", "BEARISH_CHOCH", or "NEUTRAL"
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

        # Check for BULLISH BOS (close above previous swing high)
        if swing_highs:
            latest_swing_high = swing_highs[-1]["level"]
            for _, candle in last_3_candles.iterrows():
                if candle['close'] > latest_swing_high:
                    return {
                        "structure_type": "BULLISH_BOS",
                        "score": 2,
                        "swing_level": latest_swing_high,
                        "description": f"Bullish BOS detected! Price closed above swing high at ${latest_swing_high:.5f}"
                    }

        # Check for BEARISH CHoCH (close below previous higher low)
        if len(swing_lows) >= 2:
            # Check if we're in uptrend (higher lows)
            recent_lows = swing_lows[-2:]
            if recent_lows[-1]["level"] > recent_lows[-2]["level"]:
                # We have higher lows (uptrend)
                # Now check if price broke below the most recent higher low
                latest_higher_low = recent_lows[-1]["level"]
                for _, candle in last_3_candles.iterrows():
                    if candle['close'] < latest_higher_low:
                        return {
                            "structure_type": "BEARISH_CHOCH",
                            "score": -3,  # Negative score = filter out bullish setups
                            "swing_level": latest_higher_low,
                            "description": f"⚠️ Bearish CHoCH! Price broke below higher low at ${latest_higher_low:.5f} - Trend may be reversing"
                        }

        return {"structure_type": "NEUTRAL", "score": 0, "swing_level": None}

    def _check_liquidity_grab(self, candles: pd.DataFrame, current_price: float, h4_levels: Dict, current_candle_low: float = None) -> Dict[str, Any]:
        """
        Check for Liquidity Grab / Stop Hunt pattern (REAL-TIME DETECTION)

        Liquidity Grab = Price spikes beyond key level to trigger stops, then reverses sharply
        - Very high probability setup when combined with OB/FVG
        - Shows institutions filling large orders after sweeping retail stops

        For BULLISH trader, looking for:
        - Price spikes BELOW support/swing low (grabs sell stops)
        - Large wick (20+ pips)
        - Immediate rejection back above the level
        - Entry on the reversal

        Args:
            candles: Historical H1 candles
            current_price: Current market price
            h4_levels: H4 support/resistance levels
            current_candle_low: Low of the CURRENT forming candle (real-time detection)

        Returns:
            - detected: True if liquidity grab confirmed
            - score: +5 points (current candle), +4 points (last closed), +3 points (2 candles ago)
            - grab_level: The level where stops were grabbed
            - grab_low: Lowest point of the grab
        """
        if len(candles) < 10:
            return {"detected": False, "score": 0}

        recent_candles = candles.tail(10)

        # Identify recent swing lows (local lows in last 10 candles)
        swing_lows = []
        for i in range(2, len(recent_candles) - 2):
            candle = recent_candles.iloc[i]
            window = recent_candles.iloc[max(0, i-2):min(len(recent_candles), i+3)]
            if candle['low'] == window['low'].min():
                swing_lows.append({
                    "level": float(candle['low']),
                    "index": i
                })

        if not swing_lows:
            return {"detected": False, "score": 0}

        # PRIORITY 1: Check CURRENT forming candle for liquidity grab (REAL-TIME)
        # Professional traders see this happening NOW and enter immediately
        if current_candle_low is not None:
            for swing_low in swing_lows:
                swing_level = swing_low["level"]

                # Check if current candle spiked below swing low
                if current_candle_low < swing_level:
                    # Measure current wick size
                    wick_size = current_price - current_candle_low
                    pips_below_swing = swing_level - current_candle_low

                    # Liquidity grab requirements (real-time):
                    # 1. Wick extends 5+ pips below swing level
                    # 2. Strong rejection (wick 8+ pips)
                    # 3. Price currently back above swing level
                    if pips_below_swing >= 5.0 and wick_size >= 8.0 and current_price > swing_level:
                        # LIVE GRAB SCORE (4 points) - prevents single-pattern trades
                        return {
                            "detected": True,
                            "score": 4,
                            "grab_level": swing_level,
                            "grab_low": current_candle_low,
                            "rejection_size": round(wick_size, 1),
                            "pips_below": round(pips_below_swing, 1),
                            "description": f"🔥 LIVE Liquidity Grab at ${swing_level:.5f}! Price spiking to ${current_candle_low:.5f} ({pips_below_swing:.1f} pips below) rejecting {wick_size:.1f} pips UP NOW!"
                        }

            # Also check H4 support levels for current candle grabs
            support_levels = h4_levels.get("support_levels", [])
            for support in support_levels:
                if abs(support - current_price) < 30:
                    if current_candle_low < support:
                        wick_size = current_price - current_candle_low
                        pips_below = support - current_candle_low

                        if pips_below >= 5.0 and wick_size >= 8.0 and current_price > support:
                            return {
                                "detected": True,
                                "score": 4,
                                "grab_level": support,
                                "grab_low": current_candle_low,
                                "rejection_size": round(wick_size, 1),
                                "pips_below": round(pips_below, 1),
                                "description": f"🔥 LIVE Liquidity Grab at H4 support ${support:.5f}! Price spiking to ${current_candle_low:.5f} rejecting {wick_size:.1f} pips UP NOW!"
                            }

        # PRIORITY 2: Check last 2 CLOSED candles for liquidity grab pattern
        # Score based on recency: 4 points if last candle, 3 points if 2 candles ago
        last_2 = recent_candles.tail(2)

        for swing_low in swing_lows:
            swing_level = swing_low["level"]

            for candle_position, (idx, candle) in enumerate(last_2.iterrows()):
                # Check if this candle spiked below the swing low
                if candle['low'] < swing_level:
                    # Measure wick size
                    wick_size = candle['close'] - candle['low']

                    # Liquidity grab requirements:
                    # 1. Wick extends 5+ pips below swing level
                    # 2. Strong rejection (wick 8+ pips)
                    # 3. Candle closed back above swing level

                    pips_below_swing = swing_level - candle['low']

                    # Professional thresholds: 5+ pips spike, 8+ pips rejection (institutional moves)
                    if pips_below_swing >= 5.0 and wick_size >= 8.0 and candle['close'] > swing_level:
                        # Score based on recency
                        # candle_position: 0 = 2 candles ago, 1 = last candle
                        score = 4 if candle_position == 1 else 3
                        freshness = "FRESH (last candle)" if candle_position == 1 else "RECENT (2 candles ago)"

                        # Liquidity grab confirmed!
                        return {
                            "detected": True,
                            "score": score,
                            "grab_level": swing_level,
                            "grab_low": float(candle['low']),
                            "rejection_size": round(wick_size, 1),
                            "pips_below": round(pips_below_swing, 1),
                            "description": f"🔥 Liquidity Grab at ${swing_level:.5f}! Price spiked to ${candle['low']:.5f} ({pips_below_swing:.1f} pips below) then rejected {wick_size:.1f} pips UP - {freshness}"
                        }

        # Also check H4 support levels for liquidity grabs (last 2 candles only)
        support_levels = h4_levels.get("support_levels", [])
        for support in support_levels:
            if abs(support - current_price) < 30:  # Only check nearby levels
                for candle_position, (idx, candle) in enumerate(last_2.iterrows()):
                    if candle['low'] < support:
                        wick_size = candle['close'] - candle['low']
                        pips_below = support - candle['low']

                        # Professional thresholds: 5+ pips spike, 8+ pips rejection
                        if pips_below >= 5.0 and wick_size >= 8.0 and candle['close'] > support:
                            # Score based on recency
                            score = 4 if candle_position == 1 else 3
                            freshness = "FRESH (last candle)" if candle_position == 1 else "RECENT (2 candles ago)"

                            return {
                                "detected": True,
                                "score": score,
                                "grab_level": support,
                                "grab_low": float(candle['low']),
                                "rejection_size": round(wick_size, 1),
                                "pips_below": round(pips_below, 1),
                                "description": f"🔥 Liquidity Grab at H4 support ${support:.5f}! Price spiked to ${candle['low']:.5f} then rejected {wick_size:.1f} pips - {freshness}"
                            }

        return {"detected": False, "score": 0}

    async def _detect_setup_pattern(self, h1_data: pd.DataFrame, h4_levels: Dict, current_price: float) -> Dict[str, Any]:
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

        # Get current candle info to check if low touched key zones
        current_candle_info = await self._get_current_candle_info(h1_data, current_price)
        current_candle_low = current_candle_info.get("low")

        # STEP 1: Check trend structure (BOS/CHoCH) - FILTER FIRST
        structure = self._check_bos_choch(last_candles, current_price)

        # If bearish CHoCH detected, filter out bullish setups
        if structure["structure_type"] == "BEARISH_CHOCH":
            return {
                "detected": True,
                "pattern_type": "FILTERED_OUT",
                "state": "BEARISH_CHOCH_DETECTED",
                "progress": "0/5",
                "structure": structure,
                "description": "⚠️ Bearish CHoCH detected - Filtering out bullish setups until trend confirmation",
                "confluences": [],
                "total_score": structure["score"]
            }

        # STEP 2: Check ALL patterns (don't stop at first match)
        # PROFESSIONAL: Use only closed candles, EXCEPT Liquidity Grab (real-time detection)
        fvg_setup = self._check_fvg(last_candles, current_price)
        ob_setup = await self._check_order_block(last_candles, current_price)
        breakout_setup = self._check_breakout_retest(last_candles, key_level, current_price)
        demand_setup = self._check_demand_zone(last_candles, h4_levels, current_price)
        liquidity_grab = self._check_liquidity_grab(last_candles, current_price, h4_levels, current_candle_low)  # Uses forming candle for real-time spike detection

        # STEP 3: Calculate confluence score
        confluences = []
        total_score = 0

        # Add structure score with VALIDATION
        # For BULLISH_BOS: Only count if price is still ABOVE the BOS level
        if structure["structure_type"] == "BULLISH_BOS":
            bos_level = structure.get("swing_level", 0)
            current_candle_close = last_candles['close'].iloc[-1]

            # BOS is INVALID if price closed back below the BOS level
            bos_invalidated = current_candle_close < bos_level

            if not bos_invalidated:
                # BOS is still valid - price holding above
                confluences.append({
                    "type": "BULLISH_BOS",
                    "score": 2,
                    "description": structure["description"]
                })
                total_score += structure["score"]
            # else: BOS invalidated - don't add to confluences
        else:
            # Other structure types (CHOCH, etc.)
            total_score += structure["score"]

        # Add liquidity grab (highest priority)
        # NO stability check needed - based on completed historical candles
        if liquidity_grab["detected"]:
            confluences.append({
                "type": "LIQUIDITY_GRAB",
                "score": liquidity_grab["score"],  # Use actual score (4 or 3 based on recency)
                "description": liquidity_grab["description"],
                "grab_low": liquidity_grab.get("grab_low"),  # Pass actual low for SL calculation
                "grab_level": liquidity_grab.get("grab_level")  # Pass grab level
            })
            total_score += liquidity_grab["score"]

        # Add FVG
        # Only give points when price is ACTUALLY IN or REJECTED FROM the FVG zone
        # Don't give points just because FVG exists somewhere below price
        if fvg_setup["detected"]:
            state = fvg_setup.get("state", "")
            strong_rejection = fvg_setup.get("strong_rejection", False)

            # Only count if price is IN the FVG or has been strongly rejected from it
            # Don't give points when just "DETECTED" or "APPROACHING"
            if state == "IN_FVG" or strong_rejection:
                confluences.append({
                    "type": "FVG",
                    "score": 3,
                    "description": f"FVG at ${fvg_setup.get('fvg_zone', {}).get('midpoint', 0):.5f}"
                })
                total_score += 3
            # else: FVG detected but price not there yet - don't give points

        # Add Order Block
        if ob_setup["detected"]:
            # CRITICAL: Validate that OB zone is still holding (proper invalidation)
            ob_zone = ob_setup.get('order_block_zone', {})
            ob_bottom = ob_zone.get('bottom', 0)
            ob_top = ob_zone.get('top', 0)

            # Get current candle low to check if OB was violated
            current_candle_low = last_candles['low'].iloc[-1] if len(last_candles) > 0 else current_price

            # For BULLISH OB: Invalidate if price closed BELOW the OB bottom (not just touched)
            # Allow wicks into the zone, but not full candle closes below it
            ob_violated = current_price < ob_bottom or (current_candle_low < ob_bottom * 0.998)  # 0.2% buffer for wicks

            # Pattern shows immediately when detected (professional methodology)
            # Only check if OB is still valid (not violated)
            if not ob_violated:
                confluences.append({
                    "type": "ORDER_BLOCK",
                    "score": 3,
                    "description": f"Order Block at ${ob_zone.get('midpoint', 0):.5f} (${ob_bottom:.5f}-${ob_top:.5f})"
                })
                total_score += 3

        # Add Breakout Retest
        # Only give points when retest is ACTUALLY HAPPENING, not just detected
        if breakout_setup["detected"]:
            state = breakout_setup.get("state", "")

            # Only count if retest is happening or rejection confirmed
            # Don't give points during "RETEST_WAITING" - that's too early
            if state in ["REJECTION_WAITING", "CONFIRMATION_WAITING"]:
                confluences.append({
                    "type": "BREAKOUT_RETEST",
                    "score": 2,
                    "description": f"Breakout Retest at ${breakout_setup.get('key_level', 0):.5f}"
                })
                total_score += 2
            # else: Breakout detected but retest not happening yet - don't give points

        # Add Demand Zone
        # Only give points when zone is ACTUALLY TOUCHED and REJECTED (closed candle confirmation)
        if demand_setup["detected"]:
            # Check if we have actual touch + rejection confirmation
            strong_rejection = demand_setup.get("strong_rejection", False)

            # Only give confluence points when zone shows strong rejection
            if strong_rejection:
                confluences.append({
                    "type": "DEMAND_ZONE",
                    "score": 2,
                    "description": f"Demand Zone at ${demand_setup.get('key_level', 0):.5f} (Strong Rejection: {strong_rejection})"
                })
                total_score += 2
            # else: Zone detected but no strong rejection yet - don't give points

        # STEP 4: Determine if we have enough confluence to enter
        if total_score < 5:
            # Not enough confluence
            return {
                "detected": True,
                "pattern_type": "LOW_CONFLUENCE",
                "state": "SCANNING",
                "progress": "0/5",
                "confluences": confluences,
                "total_score": total_score,
                "structure": structure,
                "description": f"⚠️ Low confluence (Score: {total_score}/5 minimum) - Need more confirmation"
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
        candle_is_bearish = current_price < candle_open
        price_from_low = current_price - candle_low
        price_from_high = candle_high - current_price
        candle_position = (current_price - candle_low) / candle_range if candle_range > 0 else 0.5

        if candle_is_bearish:
            validation_warnings.append("⚠️ Bearish candle (price below open)")
            confidence_penalty += 2

        if candle_position < 0.3:  # In bottom 30% of candle
            validation_warnings.append("📉 Price at candle lows (weak position)")
            confidence_penalty += 2
        elif candle_position < 0.5:  # In lower half
            validation_warnings.append("⚠️ Price in lower half of candle")
            confidence_penalty += 1

        # 2. CONFLUENCE DISTANCE CHECK: Are patterns close to current price?
        distant_patterns = []

        # Check Order Block distance
        if ob_setup["detected"]:
            ob_zone = ob_setup.get('order_block_zone', {})
            ob_top = ob_zone.get('top', 0)
            ob_distance = abs(current_price - ob_top)
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
            bearish_momentum = sum(last_3_closes[i] < last_3_closes[i-1] for i in range(1, len(last_3_closes)))
            if bearish_momentum >= 2:  # 2+ bearish candles
                validation_warnings.append("📉 Bearish momentum (last 3 candles)")
                confidence_penalty += 2

        # 4. ADJUST CONFIDENCE based on validation
        adjusted_score = total_score - confidence_penalty

        # Determine final confidence with validation
        if validation_warnings:
            if adjusted_score < 5:
                # Too many warnings - don't show as ready entry
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
                    "description": f"⏳ Setup detected ({total_score} points) but price action weak - WAIT FOR CONFIRMATION"
                }

        # STEP 5: We have confluence! Return the primary pattern
        # Priority: Liquidity Grab > FVG > OB > Breakout > Demand
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
        elif breakout_setup["detected"]:
            primary_setup = breakout_setup
        elif demand_setup["detected"]:
            primary_setup = demand_setup

        if primary_setup:
            # Enhance with confluence data
            primary_setup["confluences"] = confluences
            primary_setup["total_score"] = total_score
            primary_setup["adjusted_score"] = adjusted_score if validation_warnings else total_score
            primary_setup["structure"] = structure
            primary_setup["validation_warnings"] = validation_warnings
            primary_setup["candle_position"] = round(candle_position * 100, 1)

            # ENTRY STATE SYSTEM: Replace confidence with clear entry states
            entry_state_data = self._determine_entry_state(
                total_score=total_score,
                adjusted_score=adjusted_score,
                validation_warnings=validation_warnings,
                candle_position=candle_position,
                candle_is_bearish=candle_is_bearish
            )

            primary_setup["entry_state"] = entry_state_data["entry_state"]
            primary_setup["entry_signal"] = entry_state_data["entry_signal"]
            primary_setup["state_description"] = entry_state_data["state_description"]
            primary_setup["time_in_state_minutes"] = entry_state_data.get("time_in_state_minutes", 0)

            # Keep confidence for backward compatibility but make it match entry state
            if entry_state_data["entry_state"] == "🚀 STRONG ENTRY":
                primary_setup["confidence"] = "⭐️⭐️⭐️ EXTREME"
            elif entry_state_data["entry_state"] == "✅ READY TO ENTER":
                primary_setup["confidence"] = "⭐️⭐️ HIGH" if adjusted_score >= 7 else "⭐️ MODERATE"
            elif entry_state_data["entry_state"] == "⏳ SETUP FORMING":
                primary_setup["confidence"] = "⏳ FORMING (wait for entry)"
            else:  # SCANNING
                primary_setup["confidence"] = "🔍 SCANNING"

            return primary_setup

        # Default: SCANNING
        return {
            "detected": True,
            "pattern_type": "SCANNING",
            "state": "SCANNING",
            "progress": "0/5",
            "key_level": key_level,
            "confluences": confluences,
            "total_score": total_score,
            "structure": structure,
            "description": "Scanning for professional setups..."
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

    def _determine_entry_state(self, total_score: int, adjusted_score: int, validation_warnings: List[str],
                               candle_position: float, candle_is_bearish: bool) -> Dict[str, Any]:
        """
        Determine clear entry state based on score and price action

        States:
        - 🔍 SCANNING: < 5 points, looking for patterns
        - ⏳ SETUP FORMING: 5-6 points, patterns detected but not ready
        - ✅ READY TO ENTER: 7+ points, bullish candle, good position
        - 🚀 STRONG ENTRY: 10+ points, perfect conditions

        Returns:
            Dict with entry_state, entry_signal, state_description
        """
        now_utc = datetime.now(pytz.UTC)

        # Determine base state from score
        if adjusted_score < 5:
            new_state = "🔍 SCANNING"
            entry_signal = False
            description = f"Looking for setups (need {5 - adjusted_score} more points)"

        elif adjusted_score >= 5 and adjusted_score < 7:
            # Check price action for entry readiness
            if candle_is_bearish or candle_position < 40:
                new_state = "⏳ SETUP FORMING"
                entry_signal = False
                description = f"Setup detected ({adjusted_score} points) - waiting for bullish confirmation"
            else:
                # Borderline - moderate confidence with good price action
                new_state = "✅ READY TO ENTER"
                entry_signal = True
                description = f"Entry conditions met ({adjusted_score} points, moderate confidence)"

        elif adjusted_score >= 7 and adjusted_score < 10:
            # Check price action for entry readiness
            if candle_is_bearish or candle_position < 30:
                new_state = "⏳ SETUP FORMING"
                entry_signal = False
                description = f"Strong setup ({adjusted_score} points) - waiting for price confirmation"
            else:
                new_state = "✅ READY TO ENTER"
                entry_signal = True
                description = f"High confidence entry ({adjusted_score} points, good price action)"

        else:  # 10+ points
            # Check price action
            if candle_is_bearish or candle_position < 30:
                new_state = "⏳ SETUP FORMING"
                entry_signal = False
                description = f"Extreme setup ({adjusted_score} points) - waiting for bullish candle"
            else:
                new_state = "🚀 STRONG ENTRY"
                entry_signal = True
                description = f"STRONG ENTRY SIGNAL ({adjusted_score} points, excellent conditions)"

        # STABILITY CHECK: Prevent state flickering
        # Once in READY or STRONG state, stay there for minimum 5 minutes unless score drops significantly
        if self.pattern_tracker["last_entry_state"] in ["✅ READY TO ENTER", "🚀 STRONG ENTRY"]:
            if self.pattern_tracker["entry_state_since"]:
                time_in_state = (now_utc - self.pattern_tracker["entry_state_since"]).total_seconds() / 60

                # If we've been in entry state < 5 minutes and new state is not entry, stay in entry state
                # UNLESS score dropped below 5 (setup actually failed)
                if time_in_state < 5 and new_state in ["⏳ SETUP FORMING"] and adjusted_score >= 5:
                    # Keep previous entry state (prevents flickering)
                    return {
                        "entry_state": self.pattern_tracker["last_entry_state"],
                        "entry_signal": True,
                        "state_description": description + " (entry active)",
                        "time_in_state_minutes": round(time_in_state, 1)
                    }

        # Update state tracking
        if new_state != self.pattern_tracker["last_entry_state"]:
            self.pattern_tracker["last_entry_state"] = new_state
            self.pattern_tracker["entry_state_since"] = now_utc

        time_in_state = 0
        if self.pattern_tracker["entry_state_since"]:
            time_in_state = (now_utc - self.pattern_tracker["entry_state_since"]).total_seconds() / 60

        return {
            "entry_state": new_state,
            "entry_signal": entry_signal,
            "state_description": description,
            "time_in_state_minutes": round(time_in_state, 1)
        }

    def _check_breakout_retest(self, candles: pd.DataFrame, key_level: float, current_price: float) -> Dict[str, Any]:
        """
        Check for Breakout Retest pattern:
        1. Price was below resistance
        2. Price broke above resistance (breakout)
        3. Price pulled back to test old resistance (now support)
        4. Looking for rejection + confirmation

        Args:
            candles: Historical H1 candles
            key_level: The support/resistance level
            current_price: Current market price
            current_candle_low: Low of the forming candle (from M5 data)
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
                if candle['close'] < key_level - 0.0005:  # 5 pips buffer
                    return {"detected": False}  # Setup invalidated

        # PROFESSIONAL RULE: Only check CLOSED H1 candles after breakout
        retest_happening = False
        rejection_confirmed = False

        # Check only CLOSED H1 candles after breakout (no forming candles)
        if len(candles_after_breakout) > 0:
            # Retest = price returned DOWN close to key level (from above)
            # MUST actually reach within 3 pips of the level to count as a retest
            for i, candle in candles_after_breakout.iterrows():
                distance_to_level = candle['low'] - key_level  # Positive if price above level

                # Only count as retest if LOW actually reached within 3 pips of support
                if distance_to_level <= 0.0003 and distance_to_level >= -0.0002:  # Within 3 pips above or 2 pips below
                    retest_happening = True

                    # Check for BULLISH rejection (long LOWER wick + close ABOVE)
                    wick_size = min(candle['open'], candle['close']) - candle['low']
                    # wick_size > 0.00003 = 3 pips, close > key_level + 0.00005 = 5 pips above
                    if wick_size > 0.00003 and candle['close'] > key_level + 0.00005:
                        rejection_confirmed = True

        # INVALIDATION CHECKS: Only apply if NO confirmation yet
        # Once retest confirmed by closed candles, keep pattern visible
        if not rejection_confirmed and not retest_happening:
            # Time-based expiration (4+ candles without retest)
            if len(candles_after_breakout) >= 4:
                return {"detected": False}  # Setup expired - took too long

            # Price ran away without retest (20+ pips above breakout level)
            if current_price > key_level + 0.0020:  # 20 pips = 0.0020 for EUR/USD
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
            "expected_entry": current_price,
            "rejection_confirmed": rejection_confirmed  # Conservative confirmation flag
        }

    def _check_demand_zone(self, candles: pd.DataFrame, h4_levels: Dict, current_price: float) -> Dict[str, Any]:
        """
        Check for DEMAND ZONE pattern (BULLISH):
        - Price approaching H4 SUPPORT level where buyers historically defended
        - Looking for bullish rejection + LONG opportunity

        Similar to Order Block but at H4 key levels

        PROFESSIONAL RULE: Only uses CLOSED H1 candles for confirmation

        Args:
            candles: Historical H1 candles (closed only)
            h4_levels: H4 support/resistance levels
            current_price: Current market price (for distance calculation)
        """
        support_levels = h4_levels.get("support_levels", [])

        if not support_levels:
            return {"detected": False}

        # Find nearest support level
        nearest_support = min(support_levels, key=lambda x: abs(x - current_price))

        # Check price distance for state determination
        distance_to_support = current_price - nearest_support

        # PROFESSIONAL RULE: Only check CLOSED H1 candles for zone touches and rejections
        candle_touched_support = False
        strong_rejection = False

        # Check last 3 closed H1 candles for interaction with demand zone
        if len(candles) >= 3:
            recent_candles = candles.tail(3)

            for idx, candle in recent_candles.iterrows():
                # Check if candle low touched support (within 3 pips = 0.0003 for EUR/USD)
                if candle['low'] <= nearest_support + 0.0003:
                    candle_touched_support = True

                    # Check if it had strong rejection (8+ pips = 0.0008 from low to close)
                    candle_rejection = candle['close'] - candle['low']
                    if candle_rejection >= 0.0008:
                        strong_rejection = True
                        break

        # Only check proximity if no strong rejection yet
        # Once closed candle confirms rejection, keep zone visible regardless of distance
        if not strong_rejection:
            if distance_to_support > 0.0008 or distance_to_support < -0.0005:
                return {"detected": False}  # Too far away and no confirmation yet

        # Determine state based on price action
        if candle_touched_support:
            state = "AT_ZONE"
            progress = "2/5"
            confirmations = 1
        elif distance_to_support <= 5:
            state = "APPROACHING"
            progress = "1/5"
            confirmations = 0
        else:
            state = "DETECTED"
            progress = "1/5"
            confirmations = 0

        return {
            "detected": True,
            "pattern_type": "DEMAND_ZONE",
            "direction": "LONG",
            "state": state,
            "progress": progress,
            "key_level": nearest_support,
            "confirmations": confirmations,
            "expected_entry": nearest_support,
            "distance_pips": float(distance_to_support),
            "strong_rejection": strong_rejection  # Conservative confirmation flag
        }

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

        Args:
            candles: Historical H1 candles
            current_price: Current market price
            current_candle_low: Low of the forming candle (from M5 data)
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

        # Calculate distance for state determination
        distance_to_fvg = current_price - nearest_fvg["midpoint"]

        # PROFESSIONAL RULE: Only check CLOSED H1 candles, not forming candles
        # Check if any recent CLOSED candle touched the FVG zone
        candle_touched_fvg = False
        strong_rejection = False

        # Check last 3 closed H1 candles for interaction with FVG zone
        recent_candles = candles.tail(3)
        for idx, candle in recent_candles.iterrows():
            # Check if candle's LOW reached the FVG zone
            if candle['low'] <= nearest_fvg["top"]:
                candle_touched_fvg = True

                # Check for STRONG REJECTION (bullish):
                # Low touched FVG + closed significantly above (0.0008 = 8 pips for EUR/USD)
                wick_size = min(candle['open'], candle['close']) - candle['low']
                rejection_distance = candle['close'] - candle['low']

                # Strong rejection = wick into zone + close 8+ pips above the touch
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
            "direction": "LONG",
            "state": state,
            "progress": progress,
            "key_level": nearest_fvg["bottom"],  # BULLISH: Use bottom (price fills gap from below)
            "confirmations": confirmations,
            "expected_entry": nearest_fvg["bottom"],  # Entry at bottom of FVG
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
            m5_data = await fetch_h1("EURUSD", timeframe="M5")

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

    async def _check_order_block(self, candles: pd.DataFrame, current_price: float, current_candle_low: float = None) -> Dict[str, Any]:
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

        Args:
            candles: Historical H1 candles
            current_price: Current market price
            current_candle_low: Low of the forming candle (from M5 data)
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

        # Check 5-minute data for precise touches (only if not already inside the zone)
        m5_touched = False
        if current_price > nearest_ob["top"]:
            # Price is above the zone on H1 data, but check M5 for precise touches
            m5_touched = await self._check_ob_zone_touched_m5(nearest_ob["top"], nearest_ob["bottom"], lookback_hours=3)

        # PROFESSIONAL: Check last 3 CLOSED H1 candles for touches to the OB zone
        candle_touched_ob = False
        strong_rejection = False

        if len(candles) >= 3:
            recent_candles = candles.tail(3)

            for idx, candle in recent_candles.iterrows():
                # Check if candle low touched the OB zone (price went into or below the top)
                if candle['low'] <= nearest_ob["top"]:
                    candle_touched_ob = True

                    # Check for STRONG REJECTION: candle touched OB and closed 8+ pips above the low
                    rejection_distance = candle['close'] - candle['low']
                    if rejection_distance >= 0.0008:  # 8+ pips = 0.0008 for EUR/USD
                        strong_rejection = True
                        break

        # Determine state based on proximity and M5 data
        if current_price <= nearest_ob["top"] and current_price >= nearest_ob["bottom"]:
            # Price is INSIDE the order block (filling orders now!)
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
            # Current candle's low touched the OB zone (even if price bounced back up)
            state = "IN_ORDER_BLOCK"
            progress = "3/5"
            confirmations = 2

            # Save this touch - entry valid for 2 hours
            self.active_ob_touch = {
                "ob_zone": nearest_ob,
                "touched_at": now_utc,
                "entry_valid_until": now_utc + timedelta(hours=2)
            }

        elif m5_touched:
            # M5 data shows zone was touched (upgrade state)
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
            # Price is very close (within 10 pips)
            state = "APPROACHING"
            progress = "2/5"
            confirmations = 1
        elif distance_to_ob <= 20:
            # Price is nearby (10-20 pips away)
            state = "DETECTED"
            progress = "1/5"
            confirmations = 0
        else:
            # Too far away, no active setup
            return {"detected": False}

        return {
            "detected": True,
            "pattern_type": "ORDER_BLOCK",
            "direction": "LONG",
            "state": state,
            "progress": progress,
            "key_level": nearest_ob["midpoint"],
            "confirmations": confirmations,
            "expected_entry": nearest_ob["midpoint"],
            "strong_rejection": strong_rejection,  # Conservative confirmation flag
            "order_block_zone": {
                "top": nearest_ob["top"],
                "bottom": nearest_ob["bottom"],
                "midpoint": nearest_ob["midpoint"],
                "size_pips": nearest_ob["size"],
                "age_candles": nearest_ob["age_candles"]
            },
            "distance_pips": float(distance_to_ob)
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
            m5_data = await get_forming_candles(instrument="EUR_USD", granularity="M5", count=20)

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
        Build step-by-step breakdown based on pattern type
        """
        pattern = setup.get("pattern_type", "SCANNING")

        if pattern == "FAIR_VALUE_GAP":
            return self._fvg_steps(setup, current_price)
        elif pattern == "ORDER_BLOCK":
            return self._order_block_steps(setup, current_price, h1_data)
        elif pattern == "BREAKOUT_RETEST":
            return self._breakout_retest_steps(setup, current_price, current_candle)
        elif pattern == "DEMAND_ZONE":
            return self._demand_zone_steps(setup, current_price)
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
            "title": f"Price broke above ${key_level:.5f} resistance",
            "details": f"Confirmed at {breakout_candle['time']} (1H candle closed at ${breakout_candle['price']:.5f})",
            "explanation": "This shows strong buying pressure. The level that was resistance is now support."
        })

        # Step 2: Retest/Pullback
        if retest_candle:
            steps.append({
                "step": 2,
                "status": "complete",
                "title": f"Price pulled back to test ${key_level:.5f}",
                "details": f"Retest happening now at ${retest_candle['price']:.5f}",
                "explanation": "Healthy pullback. Professional traders use this to enter at better price."
            })
        else:
            steps.append({
                "step": 2,
                "status": "waiting",
                "title": f"WAITING for pullback to ${key_level:.5f}",
                "details": f"Current: ${current_price:.5f} ({current_price - key_level:.5f} pips above)",
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
                        "low": f"${current_candle['low']:.5f}",
                        "current_price": f"${current_price:.5f}",
                        "target_level": f"${key_level:.5f}"
                    },
                    "watching_for": {
                        "wick_requirement": {
                            "status": "✅ Complete" if wick_touched else "⏳ Waiting",
                            "text": f"Wick must touch ${key_level:.5f} (support)",
                            "current": f"Low: ${current_candle['low']:.5f}",
                            "explanation": "The wick shows sellers tried to push lower but failed."
                        },
                        "holding_requirement": {
                            "status": "✅ Complete" if (wick_touched and minutes_elapsed >= 20) else "⏳ Waiting",
                            "text": f"Price holding above ${key_level + 5:.5f} for 20+ minutes",
                            "current": f"Currently: ${current_price:.5f} ({minutes_elapsed} mins elapsed)",
                            "explanation": "Price holding strong shows buyers are in control."
                        },
                        "close_requirement": {
                            "status": "⏳ Watching",
                            "text": f"Candle must close above ${key_level + 5:.5f}",
                            "current": f"Currently: ${current_price:.5f}",
                            "time_left": f"{current_candle['time_remaining']} minutes until close",
                            "explanation": "Close above confirms buyers won the battle."
                        }
                    },
                    "entry_timing": {
                        "early_entry": {
                            "available": early_entry_available,
                            "type": "🟡 EARLY ENTRY (50% Position)",
                            "status": "AVAILABLE NOW" if early_entry_available else "NOT READY",
                            "trigger": f"Wick touched + holding above ${key_level + 0.0005:.5f} for 20+ mins",
                            "entry_price": f"${current_price:.5f} (market order)",
                            "stop_loss": f"${key_level - 0.0005:.5f}",
                            "position_size": "50% of planned trade",
                            "pros": "✓ Catch more of the move\n✓ Better average entry price\n✓ Psychological confidence",
                            "cons": "⚠ Candle could still reverse\n⚠ {0} mins until confirmation".format(current_candle['time_remaining']),
                            "action": "Enter 50% position NOW if confident" if early_entry_available else f"Wait {20 - minutes_elapsed} more minutes"
                        },
                        "confirmation_entry": {
                            "type": "🟢 CONFIRMATION ENTRY (Add 50% More)",
                            "trigger": f"Candle closes above ${key_level + 0.0005:.5f}",
                            "expected_time": current_candle['candle_close_expected'],
                            "time_remaining": f"{current_candle['time_remaining']} minutes",
                            "entry_price": f"~${current_price + 0.0002:.5f}-${current_price + 0.0005:.5f} (estimated)",
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
                    "details": f"Wick touched ${current_candle['low']:.5f}, closed bullish",
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
                        {"text": f"Close above ${key_level + 10:.5f}", "explanation": "Shows momentum continuing"},
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
            # Calculate SL/TP
            stop_loss = key_level - 0.0005  # 5 pips below support
            risk_pips = abs(current_price - stop_loss) * 10000
            take_profit = current_price + (risk_pips * 2 * 0.0001)  # 1:2 R:R minimum
            reward_pips = abs(take_profit - current_price) * 10000

            steps.append({
                "step": 5,
                "status": "ready",
                "title": "🎯 READY TO ENTER",
                "entry_options": [
                    {
                        "type": "Option A: Enter at retest level (Aggressive)",
                        "entry": f"${key_level:.5f}",
                        "stop_loss": f"${stop_loss:.5f}",
                        "take_profit": f"${take_profit:.5f}",
                        "risk_pips": f"{risk_pips:.1f} pips",
                        "reward_pips": f"{reward_pips:.1f} pips",
                        "risk_reward": f"1:{reward_pips/risk_pips:.1f}",
                        "trigger": f"Enter NOW at ${current_price:.5f}",
                        "pros": "Best entry price, catch full move",
                        "cons": "No confirmation candle yet",
                        "why_sl": f"SL at ${stop_loss:.5f} - If price closes below support, retest failed",
                        "why_tp": f"TP at ${take_profit:.5f} - Next resistance level or 1:2 R:R minimum"
                    },
                    {
                        "type": "Option B: Wait for rejection confirmation (Conservative)",
                        "entry": f"Wait for rejection (est. ${current_price + 0.0005:.5f})",
                        "stop_loss": f"${stop_loss:.5f}",
                        "take_profit": f"${take_profit:.5f}",
                        "risk_pips": f"{risk_pips + 5:.1f} pips (slightly more)",
                        "reward_pips": f"{reward_pips - 5:.1f} pips (slightly less)",
                        "risk_reward": f"1:{(reward_pips - 5)/(risk_pips + 5):.1f}",
                        "trigger": "Wait for: 1H bullish rejection candle (wick touches support + closes 5+ pips above)",
                        "confirmed": setup.get("rejection_confirmed", False),
                        "pros": "Confirmation of buyers defending support",
                        "cons": "Slightly worse entry price",
                        "why_sl": f"SL at ${stop_loss:.5f} - If price closes below support, retest failed",
                        "why_tp": f"TP at ${take_profit:.5f} - Next resistance level or 1:2 R:R minimum"
                    }
                ],
                "recommendation": "Option A for aggressive (retest entry), Option B for conservative (wait for rejection)",
                "explanation": "Breakout Retest complete! Support should hold. High probability setup."
            })

        return steps

    def _demand_zone_steps(self, setup: Dict, current_price: float) -> List[Dict[str, Any]]:
        """
        Create detailed steps for DEMAND ZONE pattern (BULLISH)
        H4 Support level where buyers historically defended
        """
        key_level = setup["key_level"]
        state = setup["state"]
        distance_pips = setup.get("distance_pips", 0)

        steps = []

        # Step 1: Demand Zone Detected
        steps.append({
            "step": 1,
            "status": "complete",
            "title": f"🛡️ Demand Zone at ${key_level:.5f}",
            "details": f"H4 Support level | Current price: ${current_price:.5f} ({distance_pips:.1f} pips away)",
            "explanation": "This is a key H4 support level where buyers historically defended. Institutions often place BUY orders here."
        })

        # Step 2: Price Action
        if state == "AT_ZONE":
            steps.append({
                "step": 2,
                "status": "ready",
                "title": "🎯 READY TO ENTER",
                "details": f"Price touched support at ${key_level:.5f}",
                "explanation": "Price reached the demand zone. Looking for bullish reaction."
            })

            # Calculate SL/TP
            stop_loss = key_level - 0.0005  # 5 pips below support
            risk_pips = abs(current_price - stop_loss) * 10000
            take_profit = current_price + (risk_pips * 2 * 0.0001)  # 1:2 R:R minimum
            reward_pips = abs(take_profit - current_price) * 10000

            # Step 3: Entry Options
            steps.append({
                "step": 3,
                "status": "ready",
                "title": "🎯 Entry Options",
                "entry_options": [
                    {
                        "type": "Option A: Enter at support level (Aggressive)",
                        "entry": f"${key_level:.5f}",
                        "stop_loss": f"${stop_loss:.5f}",
                        "take_profit": f"${take_profit:.5f}",
                        "risk_pips": f"{risk_pips:.1f} pips",
                        "reward_pips": f"{reward_pips:.1f} pips",
                        "risk_reward": f"1:{reward_pips/risk_pips:.1f}",
                        "trigger": f"Enter NOW at ${current_price:.5f}",
                        "pros": "Best entry price, catch full move",
                        "cons": "No confirmation yet",
                        "why_sl": f"SL at ${stop_loss:.5f} - If price closes below support, level is broken",
                        "why_tp": f"TP at ${take_profit:.5f} - Next resistance or 1:2 R:R minimum"
                    },
                    {
                        "type": "Option B: Wait for bullish rejection (Conservative)",
                        "entry": f"Wait for rejection (est. ${current_price + 5:.5f})",
                        "stop_loss": f"${stop_loss:.5f}",
                        "take_profit": f"${take_profit:.5f}",
                        "risk_pips": f"{risk_pips + 5:.1f} pips (slightly more)",
                        "reward_pips": f"{reward_pips - 5:.1f} pips (slightly less)",
                        "risk_reward": f"1:{(reward_pips - 5)/(risk_pips + 5):.1f}",
                        "trigger": "Wait for: Strong rejection (15+ pip bounce from support)",
                        "confirmed": setup.get("strong_rejection", False),
                        "pros": "Confirmation of buyers defending support",
                        "cons": "Slightly worse entry price",
                        "why_sl": f"SL at ${stop_loss:.5f} - If price closes below support, level is broken",
                        "why_tp": f"TP at ${take_profit:.5f} - Next resistance or 1:2 R:R minimum"
                    }
                ],
                "recommendation": "Option A for aggressive (support entry), Option B for conservative (wait for rejection)",
                "explanation": "Demand Zone activated! Buyers should defend this H4 support. High probability bounce UP."
            })
        elif state == "APPROACHING":
            steps.append({
                "step": 2,
                "status": "in_progress",
                "title": f"⏳ Price approaching demand zone ({distance_pips:.1f} pips away)",
                "details": f"Current: ${current_price:.5f} → Target: ${key_level:.5f}",
                "explanation": "Price is dropping toward H4 support. Prepare for potential bullish reaction."
            })
        else:  # DETECTED
            steps.append({
                "step": 2,
                "status": "waiting",
                "title": f"Demand zone detected {distance_pips:.1f} pips below",
                "details": f"Waiting for price to drop from ${current_price:.5f} to ${key_level:.5f}",
                "explanation": "H4 support below. Price may drop to this level before bouncing."
            })

        return steps

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
            "title": f"Fair Value Gap detected at ${fvg_zone.get('midpoint', 0):.5f}",
            "details": f"Gap size: {fvg_zone.get('size_pips', 0):.1f} pips | Age: {fvg_zone.get('age_candles', 0)} candles ago",
            "zone": {
                "top": f"${fvg_zone.get('top', 0):.5f}",
                "midpoint": f"${fvg_zone.get('midpoint', 0):.5f}",
                "bottom": f"${fvg_zone.get('bottom', 0):.5f}"
            },
            "explanation": "FVG is an IMBALANCE in price where no trades occurred. Price is magnetically pulled back to fill this gap."
        })

        # Step 2: Price Approaching or In FVG
        if state == "IN_FVG":
            steps.append({
                "step": 2,
                "status": "in_progress",
                "title": "🎯 Price INSIDE the Fair Value Gap!",
                "details": f"Current: ${current_price:.5f} | Gap: ${fvg_zone.get('bottom', 0):.5f}-${fvg_zone.get('top', 0):.5f}",
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
                "details": f"Current: ${current_price:.5f} → Target: ${fvg_zone.get('midpoint', 0):.5f}",
                "explanation": "Price is dropping toward the gap. High probability it will fill the FVG before reversing UP."
            })
        else:  # DETECTED
            steps.append({
                "step": 2,
                "status": "waiting",
                "title": f"FVG detected {distance:.1f} pips below price",
                "details": f"Waiting for price to drop from ${current_price:.5f} to ${fvg_zone.get('midpoint', 0):.5f}",
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
                        "trigger": f"BUY at ${fvg_zone.get('midpoint', 0):.5f} (limit order)",
                        "stop_loss": f"${fvg_zone.get('bottom', 0) - 0.0005:.5f} (5 pips below gap)",
                        "take_profit": f"Recent highs or ${current_price + 0.0020:.5f}+",
                        "pros": "Best price, high probability bounce",
                        "cons": "Might not fill completely"
                    },
                    {
                        "type": "Option B: Wait for bullish confirmation (Conservative)",
                        "trigger": "Wait for: (1) 1H bullish candle close inside FVG OR (2) Strong rejection (15+ pip bounce from FVG)",
                        "confirmed": setup.get("strong_rejection", False),
                        "stop_loss": f"${fvg_zone.get('bottom', 0) - 0.0005:.5f}",
                        "pros": "Confirmation of buyers stepping in",
                        "cons": "Slightly worse entry price"
                    }
                ],
                "recommendation": "Option A for aggressive (FVG midpoint entry), Option B for conservative (wait for confirmation)",
                "explanation": "FVG filled! Institutions left orders here. High probability of bounce UP."
            })

        return steps

    def _calculate_ob_sl_tp(self, ob_zone: Dict, current_price: float, h1_data: pd.DataFrame = None) -> Dict[str, float]:
        """
        Calculate Stop Loss and Take Profit for Order Block trade (BULLISH)

        SL: 5 pips below OB bottom (OB invalidation level)
        TP: Next resistance level or 1:2 R:R minimum
        """
        ob_bottom = ob_zone.get('bottom', 0)
        ob_top = ob_zone.get('top', 0)
        ob_midpoint = ob_zone.get('midpoint', 0)

        # Stop Loss: 5 pips below OB bottom
        stop_loss = ob_bottom - 5

        # Find next resistance level from H1 data
        take_profit = None
        if h1_data is not None and not h1_data.empty:
            # Look for swing highs above current price in recent data
            recent_candles = h1_data.tail(50)  # Last 50 hours

            # Find highs above current price
            resistance_levels = []
            for i in range(1, len(recent_candles) - 1):
                high = recent_candles.iloc[i]['high']
                prev_high = recent_candles.iloc[i-1]['high']
                next_high = recent_candles.iloc[i+1]['high']

                # Swing high: higher than neighbors and above current price
                if high > prev_high and high > next_high and high > current_price:
                    resistance_levels.append(high)

            if resistance_levels:
                # Take the nearest resistance above current price
                take_profit = min(resistance_levels)

        # If no resistance found or too close, use 1:2 R:R minimum
        risk = ob_midpoint - stop_loss
        min_tp_1_2 = ob_midpoint + (risk * 2)  # 1:2 R:R from midpoint entry

        if take_profit is None or take_profit < min_tp_1_2:
            take_profit = min_tp_1_2

        return {
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
            "risk_pips": float(ob_midpoint - stop_loss),
            "reward_pips": float(take_profit - ob_midpoint),
            "risk_reward_ratio": float((take_profit - ob_midpoint) / (ob_midpoint - stop_loss)) if (ob_midpoint - stop_loss) > 0 else 0
        }

    def _order_block_steps(self, setup: Dict, current_price: float, h1_data: pd.DataFrame = None) -> List[Dict[str, Any]]:
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
            "status": "complete" if state in ["APPROACHING", "IN_ORDER_BLOCK"] else "in_progress",
            "title": "📦 Order Block Detected",
            "details": f"OB Zone: ${ob_zone.get('bottom', 0):.5f} - ${ob_zone.get('top', 0):.5f} ({ob_zone.get('size_pips', 0):.1f} pips)",
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
        elif state == "IN_ORDER_BLOCK":
            steps.append({
                "step": 2,
                "status": "complete",
                "title": "✅ Price Touched Order Block",
                "details": f"OB zone was touched (price bounced {distance:.1f} pips from zone)",
                "explanation": "Price reached the institutional order zone and bounced. Entry signal is active."
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
                        "entry": f"${ob_zone.get('midpoint', 0):.5f}",
                        "stop_loss": f"${sl_tp['stop_loss']:.5f}",
                        "take_profit": f"${sl_tp['take_profit']:.5f}",
                        "risk_pips": f"{sl_tp['risk_pips']:.1f} pips",
                        "reward_pips": f"{sl_tp['reward_pips']:.1f} pips",
                        "risk_reward": f"1:{sl_tp['risk_reward_ratio']:.1f}",
                        "trigger": f"Enter NOW at ${ob_zone.get('midpoint', 0):.5f}",
                        "pros": "Best entry price, highest R:R",
                        "cons": "No confirmation candle",
                        "why_sl": f"SL at ${sl_tp['stop_loss']:.5f} - If price closes below OB zone, setup is invalidated",
                        "why_tp": f"TP at ${sl_tp['take_profit']:.5f} - Next resistance level or 1:2 R:R minimum"
                    },
                    {
                        "type": "Option B: Wait for bullish confirmation (Conservative)",
                        "entry": f"Wait for close (est. ${ob_zone.get('top', 0):.5f})",
                        "stop_loss": f"${sl_tp['stop_loss']:.5f}",
                        "take_profit": f"${sl_tp['take_profit']:.5f}",
                        "risk_pips": f"{sl_tp['risk_pips'] + 5:.1f} pips (slightly more)",
                        "reward_pips": f"{sl_tp['reward_pips'] - 5:.1f} pips (slightly less)",
                        "risk_reward": f"1:{(sl_tp['reward_pips'] - 5) / (sl_tp['risk_pips'] + 5):.1f}",
                        "trigger": "Wait for: (1) 1H bullish candle close inside OB OR (2) Strong rejection (15+ pip bounce from OB)",
                        "confirmed": setup.get("strong_rejection", False),
                        "pros": "Confirmation of buyers stepping in",
                        "cons": "Slightly worse entry price",
                        "why_sl": f"SL at ${sl_tp['stop_loss']:.5f} - If price closes below OB zone, setup is invalidated",
                        "why_tp": f"TP at ${sl_tp['take_profit']:.5f} - Next resistance level or 1:2 R:R minimum"
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

    def _confluence_detected_steps(self, setup: Dict, current_price: float) -> List[Dict[str, Any]]:
        """Steps for CONFLUENCE_DETECTED pattern"""
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
            "explanation": f"Multiple patterns aligned at ${key_level:.5f} support. Professional setup forming."
        })

        # Step 2: Dynamic entry status based on price location
        entry_zone = key_level + 2
        distance_to_entry = current_price - entry_zone

        if state == "WAITING_CONFIRMATION":
            # Check if price is at/near entry zone (within 2 pips)
            if abs(distance_to_entry) <= 2:
                # PRICE AT ENTRY ZONE!
                steps.append({
                    "step": 2,
                    "status": "complete",
                    "title": "🎯 PRICE AT ENTRY ZONE - READY TO ENTER!",
                    "details": f"Current: ${current_price:.5f} | Entry: ${entry_zone:.5f} ({abs(distance_to_entry):.1f} pips away)",
                    "explanation": "Price reached entry zone! Enter now with market/limit order."
                })
            elif distance_to_entry > 2:
                # Price above entry - waiting for pullback
                steps.append({
                    "step": 2,
                    "status": "in_progress",
                    "title": "⏳ Waiting for pullback to entry zone",
                    "details": f"Price: ${current_price:.5f} | Entry: ${entry_zone:.5f} ({distance_to_entry:.1f} pips above)",
                    "explanation": "Place limit order at entry or wait for price to pull back."
                })
            else:
                # Price below entry - already through zone
                steps.append({
                    "step": 2,
                    "status": "in_progress",
                    "title": "⚠️ Price below entry zone",
                    "details": f"Price: ${current_price:.5f} | Entry: ${entry_zone:.5f} ({abs(distance_to_entry):.1f} pips below)",
                    "explanation": "Price moved below entry. Wait for price to come back up or monitor for invalidation."
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
                f"Key Level: ${key_level:.5f} ({level_type})",
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
                          ("Optimal time for EUR/USD trading." if "HIGH" in strength else "Lower probability time.")
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
        demand_zone = next((c for c in confluences if c["type"] == "DEMAND_ZONE"), None)
        order_block = next((c for c in confluences if c["type"] == "ORDER_BLOCK"), None)
        fvg = next((c for c in confluences if c["type"] == "FVG"), None)
        breakout_retest = next((c for c in confluences if c["type"] == "BREAKOUT_RETEST"), None)

        # Get actual liquidity grab low (not from description parsing!)
        liquidity_grab_low = None
        if liquidity_grab:
            liquidity_grab_low = liquidity_grab.get("grab_low")

        # ENTRY LOGIC: Professional methodology
        # Entry is at H4 support (key_level) or order block/FVG zone
        # NOT at the liquidity grab level itself
        if liquidity_grab or order_block:
            # Enter at H4 support/order block (where institutions will buy)
            entry = key_level + 0.0002  # 2 pips above H4 support
            entry_reason = f"Enter 2 pips above H4 support ${key_level:.5f} (order block/liquidity grab confirmed)"
        elif demand_zone:
            # Enter at demand zone
            entry = key_level + 0.0002
            entry_reason = f"Enter 2 pips above ${key_level:.5f} demand zone"
        else:
            # Default: enter at key level
            entry = current_price if current_price > key_level else key_level + 0.0002
            entry_reason = f"Enter at current price ${current_price:.5f}"

        # STOP LOSS: Professional methodology
        # OPTION 3: Use closest confluence to entry for SL
        # All patterns use 6 pip buffer (0.0006 for EUR/USD)
        support_levels = h4_levels.get("support_levels", [])

        confluence_levels = []

        # Collect all confluences with their levels and 6 pip buffers
        if liquidity_grab and liquidity_grab_low:
            confluence_levels.append({
                "name": "Liquidity Grab",
                "level": liquidity_grab_low,
                "sl": liquidity_grab_low - 0.0006,  # Changed from 0.0005 to 0.0006
                "reason": f"Below liquidity grab low (${liquidity_grab_low:.5f}) - stops already swept"
            })

        if demand_zone:
            confluence_levels.append({
                "name": "Demand Zone",
                "level": key_level,
                "sl": key_level - 0.0006,
                "reason": f"Below demand zone (${key_level:.5f}) with 6 pip buffer"
            })

        if order_block:
            ob_desc = order_block.get("description", "")
            import re
            match = re.search(r'\$(\d+\.\d+)-\$(\d+\.\d+)', ob_desc)
            if match:
                ob_bottom = float(match.group(1))
                confluence_levels.append({
                    "name": "Order Block",
                    "level": ob_bottom,
                    "sl": ob_bottom - 0.0006,
                    "reason": f"Below Order Block bottom (${ob_bottom:.5f}) with 6 pip buffer"
                })

        if fvg:
            fvg_level = setup.get("fvg_zone", {}).get("bottom") if "fvg_zone" in setup else None
            if fvg_level:
                confluence_levels.append({
                    "name": "FVG",
                    "level": fvg_level,
                    "sl": fvg_level - 0.0006,
                    "reason": f"Below FVG bottom (${fvg_level:.5f}) with 6 pip buffer"
                })

        if breakout_retest:
            breakout_level = breakout_retest.get("key_level") if isinstance(breakout_retest, dict) else key_level
            if breakout_level:
                confluence_levels.append({
                    "name": "Breakout Retest",
                    "level": breakout_level,
                    "sl": breakout_level - 0.0006,
                    "reason": f"Below breakout level (${breakout_level:.5f}) with 6 pip buffer"
                })

        # Find closest confluence to entry
        if confluence_levels:
            for conf in confluence_levels:
                conf["distance"] = abs(entry - conf["level"])

            closest = min(confluence_levels, key=lambda x: x["distance"])
            sl = closest["sl"]
            sl_reason = closest["reason"]
        else:
            # Fallback: 6 pips below entry (changed from 15)
            sl = entry - 0.0006
            sl_reason = "6 pip buffer below entry"

        # TAKE PROFIT: Professional methodology - target opposing liquidity pools
        # Look for resistance levels (swing highs, psychological levels) above entry
        resistance_levels = h4_levels.get("resistance_levels", [])

        # Calculate minimum TP based on R:R ratios (aim for 2:1 and 3:1)
        risk_pips = entry - sl
        min_tp1_distance = risk_pips * 2  # 2:1 R:R minimum
        min_tp2_distance = risk_pips * 3  # 3:1 R:R minimum

        # Find resistance within reasonable range (100 pips for EUR/USD = 0.0100)
        nearby_resistance = [r for r in resistance_levels if r > entry and r < entry + 0.0100]

        # Filter resistance that meets minimum R:R requirements
        valid_tp1_levels = [r for r in nearby_resistance if (r - entry) >= min_tp1_distance]
        valid_tp2_levels = [r for r in nearby_resistance if (r - entry) >= min_tp2_distance]

        if len(valid_tp1_levels) >= 1 and len(valid_tp2_levels) >= 1:
            # Use structure-based resistance levels - place 5 pips BEFORE to ensure fill
            resistance_1 = valid_tp1_levels[0]
            resistance_2 = valid_tp2_levels[0] if valid_tp2_levels[0] != resistance_1 else (valid_tp2_levels[1] if len(valid_tp2_levels) > 1 else resistance_1 + 0.0015)
            tp1 = resistance_1 - 0.0005  # 5 pips before resistance
            tp2 = resistance_2 - 0.0005  # 5 pips before resistance
            tp1_reason = f"5 pips before H4 resistance ${resistance_1:.5f} (at ${tp1:.5f})"
            tp2_reason = f"5 pips before H4 resistance ${resistance_2:.5f} (at ${tp2:.5f})"
        elif len(valid_tp1_levels) >= 1:
            # One valid resistance found
            resistance_1 = valid_tp1_levels[0]
            tp1 = resistance_1 - 0.0005  # 5 pips before resistance
            tp2 = entry + min_tp2_distance  # Use R:R based target for TP2
            tp1_reason = f"5 pips before H4 resistance ${resistance_1:.5f} (at ${tp1:.5f})"
            tp2_reason = f"Extended target (3:1 R:R) at ${tp2:.5f}"
        else:
            # No structure found - use R:R based targets (2:1 and 3:1 minimum)
            tp1 = entry + min_tp1_distance
            tp2 = entry + min_tp2_distance
            tp1_reason = f"Target (2:1 R:R) at ${tp1:.5f}"
            tp2_reason = f"Extended target (3:1 R:R) at ${tp2:.5f}"

        # Calculate R:R ratios
        # For EUR/USD: 1 pip = 0.0001, so multiply dollar difference by 10000
        risk_pips = (entry - sl) * 10000
        reward1_pips = (tp1 - entry) * 10000
        reward2_pips = (tp2 - entry) * 10000
        rr1 = reward1_pips / risk_pips if risk_pips > 0 else 0
        rr2 = reward2_pips / risk_pips if risk_pips > 0 else 0

        # Build confluence reasoning
        confluence_reasons = []
        for c in confluences:
            confluence_reasons.append(f"• {c['type'].replace('_', ' ').title()}: {c['score']} points")

        return {
            "status": "Ready",
            "entry_method": "Limit order" if entry < current_price else "Market order",
            "entry_price": f"${entry:.5f}",
            "entry_reasoning": entry_reason,
            "stop_loss": {
                "price": f"${sl:.5f}",
                "reason": sl_reason,
                "pips": f"{risk_pips:.1f} pips risk",
                "why": "If price goes here, setup invalidated - liquidity already grabbed, further downside means failure"
            },
            "take_profit_1": {
                "price": f"${tp1:.5f}",
                "reason": tp1_reason,
                "pips": f"+{reward1_pips:.1f} pips",
                "rr_ratio": f"{rr1:.5f}:1",
                "action": "Close 50%, move SL to breakeven",
                "why": "Lock in profit, secure risk-free trade"
            },
            "take_profit_2": {
                "price": f"${tp2:.5f}",
                "reason": tp2_reason,
                "pips": f"+{reward2_pips:.1f} pips",
                "rr_ratio": f"{rr2:.5f}:1",
                "action": "Close remaining 50%",
                "why": "Full profit target reached"
            },
            "confluence_summary": {
                "total_points": total_score,
                "reasons": confluence_reasons,
                "why_trade": f"High-probability setup with {total_score}/5 confluence points. " +
                            ("Liquidity already grabbed below, stops swept. " if liquidity_grab else "") +
                            ("Strong demand zone holding. " if demand_zone else "") +
                            ("Institutional buying visible. " if order_block else "")
            },
            "position_sizing": {
                "account_risk": "0.5% recommended",
                "risk_pips": f"{risk_pips:.1f}",
                "calculation": f"Position size = (Account × 0.005) / {risk_pips:.1f} pips"
            }
        }

    def _monitor_active_trade(self, current_price: float, current_candle: Dict = None) -> Dict[str, Any]:
        """
        Monitor an active trade's progress toward TP/SL
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

        # Check if TP or SL hit
        # For LONG trades: Check HIGH for TP (price touched above), LOW for SL (price touched below)
        trade_result = None
        candle_high = current_candle.get('high', current_price)
        candle_low = current_candle.get('low', current_price)

        if candle_high >= tp2:
            trade_result = "TP2_HIT"
        elif candle_high >= tp1:
            trade_result = "TP1_HIT"
        elif candle_low <= sl:
            trade_result = "SL_HIT"

        # Calculate current P&L
        pnl_pips = current_price - entry
        pnl_percent = (pnl_pips / entry) * 100 if entry > 0 else 0

        # Build trade progress status
        if trade_result == "SL_HIT":
            status = "TRADE_STOPPED_OUT"
            status_message = f"❌ Stop Loss Hit at ${sl:.5f}"
            is_active = False  # Trade finished
        elif trade_result == "TP2_HIT":
            status = "TRADE_COMPLETED"
            status_message = f"✅ TP2 Hit! Full profit taken at ${tp2:.5f}"
            is_active = False  # Trade finished
        elif trade_result == "TP1_HIT":
            status = "TRADE_ACTIVE_TP1_HIT"
            status_message = f"✅ TP1 Hit! Waiting for TP2 at ${tp2:.5f}"
            is_active = True  # Still watching for TP2
        else:
            # Trade still active, no TP/SL hit yet
            status = "TRADE_ACTIVE"

            # Show progress based on price location
            distance_to_tp1 = tp1 - current_price
            distance_to_sl = current_price - sl

            if pnl_pips >= 0:
                if distance_to_tp1 <= 5:
                    status_message = f"🎯 Approaching TP1 (${tp1:.5f}) - {distance_to_tp1:.1f} pips away"
                else:
                    status_message = f"📈 In Profit +{pnl_pips:.1f} pips | Target: TP1 ${tp1:.5f}"
            else:
                if abs(pnl_pips) <= 3:
                    status_message = f"⏸️ At Entry ${entry:.5f} ({pnl_pips:+.1f} pips)"
                else:
                    status_message = f"📉 Pullback {pnl_pips:.1f} pips | SL: ${sl:.5f} ({distance_to_sl:.1f} pips away)"

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
                        f"Entry: ${entry:.5f}",
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
                    "condition": f"Stop Loss at ${sl:.5f}",
                    "reason": f"Trade invalidated if price hits ${sl:.5f}",
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
        """Build steps display for active trade monitoring"""
        steps = []

        # Step 1: Entry confirmation
        steps.append({
            "step": 1,
            "status": "complete",
            "title": f"✅ Entered at ${entry:.5f}",
            "details": f"Trade active with {abs(entry - sl) * 10000:.1f} pip stop loss",
            "explanation": "Position opened based on 5+ confluence points"
        })

        # Step 2: TP1 status
        # For EUR/USD: 1 pip = 0.0001, so multiply dollar difference by 10000
        distance_to_tp1 = (tp1 - current_price) * 10000
        if trade_result == "TP2_HIT" or trade_result == "TP1_HIT":
            tp1_status = "complete"
            tp1_title = f"✅ TP1 Hit at ${tp1:.5f}"
        elif distance_to_tp1 <= 50:  # Within 50 pips = approaching
            tp1_status = "in_progress"
            tp1_title = f"🎯 Approaching TP1 - {distance_to_tp1:.1f} pips away"
        else:
            tp1_status = "pending"
            tp1_title = f"⏳ Target TP1 at ${tp1:.5f} ({distance_to_tp1:.1f} pips)"

        steps.append({
            "step": 2,
            "status": tp1_status,
            "title": tp1_title,
            "details": f"Close 50% position, move SL to breakeven",
            "explanation": "Lock in profit and make trade risk-free"
        })

        # Step 3: TP2 status
        # For EUR/USD: 1 pip = 0.0001, so multiply dollar difference by 10000
        distance_to_tp2 = (tp2 - current_price) * 10000
        if trade_result == "TP2_HIT":
            tp2_status = "complete"
            tp2_title = f"✅ TP2 Hit at ${tp2:.5f} - Trade Complete!"
        elif trade_result == "TP1_HIT":
            tp2_status = "in_progress"
            tp2_title = f"🎯 Targeting TP2 at ${tp2:.5f} ({distance_to_tp2:.1f} pips)"
        else:
            tp2_status = "pending"
            tp2_title = f"⏳ Final Target TP2 at ${tp2:.5f} ({distance_to_tp2:.1f} pips)"

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
            "details": f"Price: ${current_price:.5f} | Entry: ${entry:.5f}",
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
                "condition": f"Price closes back below ${key_level:.5f}",
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
                "condition": f"Current candle closes below ${key_level - 0.0005:.5f}",
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

        elif state == "RETEST_WAITING":
            # Calculate dynamic distance threshold (20 pips for EUR/USD)
            runaway_threshold = key_level + 0.0020

            conditions.append({
                "condition": f"Price moves above ${runaway_threshold:.5f} without pullback",
                "reason": "Setup ran away - strong momentum without retest (20+ pips above breakout)",
                "action": "Cancel setup. Look for new entry or pattern.",
                "severity": "HIGH"
            })
            conditions.append({
                "condition": f"Price closes back below ${key_level:.5f}",
                "reason": "Breakout failed - false breakout",
                "action": "Cancel setup immediately. Breakout reversed.",
                "severity": "CRITICAL"
            })
            conditions.append({
                "condition": "4+ hours passed without retest",
                "reason": "Setup expired - retest took too long",
                "action": "Move on to new patterns. Opportunity missed.",
                "severity": "MEDIUM"
            })

        elif state == "REJECTION_WAITING":
            conditions.append({
                "condition": f"Candle closes below ${key_level:.5f}",
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
                "condition": f"Next candle closes below ${key_level:.5f}",
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
                "condition": f"Entry candle closes below ${key_level:.5f}",
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
            "condition": f"Price falls ${(key_level * 0.01):.5f}+ below support",
            "reason": "Major support break (1% below key level)",
            "action": "Setup completely dead. Look for new pattern.",
            "severity": "CRITICAL"
        })

        return conditions

    def _fvg_invalidation(self, setup: Dict, state: str, key_level: float) -> List[Dict[str, Any]]:
        """Dynamic invalidation for Fair Value Gap (FVG) pattern"""
        fvg_zone = setup.get("fvg_zone", {})
        gap_bottom = fvg_zone.get("bottom", key_level - 0.0005)

        conditions = []

        state = setup.get("state", "DETECTED")

        # State-specific invalidations
        if state == "DETECTED":
            conditions.append({
                "condition": f"Price continues UP without filling FVG (${gap_bottom:.5f})",
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
                "condition": f"Price reverses UP before reaching FVG bottom (${gap_bottom:.5f})",
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
                "condition": f"Price closes BELOW gap bottom (${gap_bottom - 0.0005:.5f})",
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
                "condition": f"Price continues UP without reaching OB (${ob_bottom:.5f})",
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
                "condition": f"Price reverses UP before reaching OB bottom (${ob_bottom:.5f})",
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
                "condition": f"Price closes BELOW OB bottom (${ob_bottom - 3:.5f})",
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
                "condition": f"Price closes below ${key_level - 10:.5f}",
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
    """Get BULLISH professional trader analysis for EUR/USD (BUY setups only)"""
    system = BullishProTraderEURUSD()
    return await system.get_detailed_setup()
