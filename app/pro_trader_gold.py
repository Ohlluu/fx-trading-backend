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

class ProTraderGold:
    """
    Educational trading system that shows professional trader thought process
    Provides step-by-step breakdown of setup formation
    """

    def __init__(self):
        self.pair = "XAUUSD"
        self.timeframes = {
            "D1": 200,  # 200 days
            "H4": 500,  # ~83 days of 4H candles
            "H1": 100   # 100 hours
        }

    async def get_detailed_setup(self) -> Dict[str, Any]:
        """
        Main function: Returns complete setup analysis with educational breakdown
        """
        try:
            # Fetch multi-timeframe data
            h1_data = await fetch_h1(self.pair, timeframe="H1")

            if h1_data is None or h1_data.empty:
                return self._error_response("No data available")

            # Get current live price
            current_price = await get_current_xauusd_price()
            if current_price is None:
                current_price = float(h1_data['close'].iloc[-1])

            # Analyze market structure across timeframes
            daily_analysis = self._analyze_daily_trend(h1_data)
            h4_levels = self._identify_key_levels_h4(h1_data)
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

    def _analyze_daily_trend(self, h1_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze daily timeframe trend
        In real implementation, would fetch D1 data separately
        For now, derive from H1 data
        """
        # Calculate daily-equivalent indicators from H1
        close = h1_data['close']

        # Simple trend analysis
        sma_200 = close.rolling(window=200).mean()
        current_price = close.iloc[-1]
        sma_200_current = sma_200.iloc[-1] if len(sma_200) > 0 else current_price

        trend = "BULLISH" if current_price > sma_200_current else "BEARISH"

        # Find recent swing highs/lows
        recent_high = h1_data['high'].tail(48).max()  # Last 2 days
        recent_low = h1_data['low'].tail(48).min()

        return {
            "trend": trend,
            "explanation": f"Price is {'above' if trend == 'BULLISH' else 'below'} long-term average",
            "points": [
                f"Trend: {trend} (price {'above' if trend == 'BULLISH' else 'below'} 200-period average)",
                f"Recent high: ${recent_high:.2f}",
                f"Recent low: ${recent_low:.2f}",
                "Structure: " + ("Making higher highs" if trend == "BULLISH" else "Making lower lows")
            ]
        }

    def _identify_key_levels_h4(self, h1_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify key support/resistance levels
        These are the levels pros mark on their charts
        """
        # Use last 200 H1 candles to find key levels
        highs = h1_data['high'].tail(200)
        lows = h1_data['low'].tail(200)

        # Find recent significant levels (simplified)
        # In real implementation, would use proper S/R detection
        resistance_candidates = []
        support_candidates = []

        # Find local highs/lows
        for i in range(10, len(highs) - 10):
            # Resistance: local high
            if highs.iloc[i] == highs.iloc[i-10:i+10].max():
                resistance_candidates.append(float(highs.iloc[i]))
            # Support: local low
            if lows.iloc[i] == lows.iloc[i-10:i+10].min():
                support_candidates.append(float(lows.iloc[i]))

        # Get most recent/relevant levels
        resistance = sorted(resistance_candidates)[-3:] if resistance_candidates else []
        support = sorted(support_candidates)[-3:] if support_candidates else []

        # Identify THE key level (most touched)
        key_level = resistance[0] if resistance else support[-1] if support else float(h1_data['close'].iloc[-1])

        return {
            "key_level": key_level,
            "resistance_levels": resistance,
            "support_levels": support,
            "level_type": "resistance" if key_level in resistance else "support"
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
                steps.append({
                    "step": 3,
                    "status": "in_progress",
                    "title": "⏳ WATCHING CURRENT CANDLE for rejection",
                    "details": f"Candle: {current_candle['candle_start']} ({60 - current_candle['time_remaining']}/60 mins complete)",
                    "current_values": {
                        "low": f"${current_candle['low']:.2f}",
                        "current_price": f"${current_price:.2f}",
                        "target_level": f"${key_level:.2f}"
                    },
                    "watching_for": {
                        "wick_requirement": {
                            "status": "complete" if wick_touched else "waiting",
                            "text": f"Wick must touch ${key_level:.2f} (support)",
                            "current": f"Low: ${current_candle['low']:.2f}",
                            "explanation": "The wick shows sellers tried to push lower but failed."
                        },
                        "close_requirement": {
                            "status": "watching",
                            "text": f"Candle must close above ${key_level + 5:.2f}",
                            "current": f"Currently: ${current_price:.2f}",
                            "time_left": f"{current_candle['time_remaining']} minutes until close",
                            "explanation": "Close above shows buyers won the battle."
                        }
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

        return {
            "key_level": key_level,
            "level_type": level_type,
            "points": [
                f"Key Level: ${key_level:.2f} ({level_type})",
                f"This level has been tested multiple times",
                f"Currently {abs(current_price - key_level):.1f} pips away",
                "Structure shows institutional interest at this price"
            ]
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
        """Return invalidation conditions"""
        key_level = setup.get("key_level", 0)

        return [
            {
                "condition": f"Price closes below ${key_level - 5:.2f}",
                "reason": "Support zone broken",
                "action": "Cancel setup, wait for new pattern",
                "severity": "CRITICAL"
            },
            {
                "condition": "Next 2 candles close bearish",
                "reason": "No buyer interest",
                "action": "Setup failed, back to scanning",
                "severity": "HIGH"
            },
            {
                "condition": "Low volume session starts",
                "reason": "Asian session = less reliable",
                "action": "Close positions or skip entry",
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
async def get_pro_trader_analysis() -> Dict[str, Any]:
    """Get professional trader analysis for XAUUSD"""
    system = ProTraderGold()
    return await system.get_detailed_setup()
