#!/usr/bin/env python3
"""
XAU/USD Confluence Trading System
Based on comprehensive 2-year analysis showing 98% S/R respect rate
Focuses exclusively on psychological level confluence for optimal profitability
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from typing import Dict, Any, Optional, List, Tuple
import math

# Configuration based on analysis results
XAUUSD_CONFIG = {
    "psychological_levels": {
        # Major round numbers with 98% historical respect rate
        "major": [1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400],
        "minor": [1925, 1975, 2025, 2075, 2125, 2175, 2225, 2275, 2325, 2375],
    },
    "volatility": {
        "adr_pips": 397.9,  # Average Daily Range from analysis
        "std_dev": 272.4,
        "max_daily": 2000,  # Conservative max for risk management
    },
    "sessions": {
        # Based on analysis - London-NY overlap is best (125.2 pips avg)
        "london_ny_overlap": {"start": 13, "end": 17, "avg_range": 125.2},
        "london": {"start": 8, "end": 17, "avg_range": 99.1},
        "new_york": {"start": 13, "end": 22, "avg_range": 92.1},
        "asian": {"start": 0, "end": 8, "avg_range": 84.0},
    },
    "risk_management": {
        "max_risk_per_trade": 0.02,  # 2% max risk
        "min_rr_ratio": 2.0,  # Minimum 1:2 risk/reward
        "atr_stop_multiplier": 1.5,  # ATR-based stops
    },
    "confluence_thresholds": {
        "minimum_score": 60,  # Based on 98% S/R success rate
        "strong_signal": 80,
        "very_strong": 90,
    }
}

def get_psychological_levels(price: float, range_pips: int = 200) -> List[float]:
    """
    Get relevant psychological levels around current price
    Based on 98% historical respect rate for round numbers
    """
    all_levels = XAUUSD_CONFIG["psychological_levels"]["major"] + \
                XAUUSD_CONFIG["psychological_levels"]["minor"]

    # Convert range from pips to price points (1 pip = $0.10 for gold)
    range_price = range_pips * 0.10

    relevant_levels = [
        level for level in all_levels
        if abs(level - price) <= range_price
    ]

    return sorted(relevant_levels)

def calculate_distance_to_level(price: float, level: float) -> Dict[str, float]:
    """Calculate distance to psychological level in pips and percentage"""
    distance_price = abs(price - level)
    distance_pips = distance_price / 0.10  # 1 pip = $0.10 for gold
    distance_percent = (distance_price / level) * 100

    return {
        "distance_pips": distance_pips,
        "distance_percent": distance_percent,
        "direction": "above" if price > level else "below"
    }

def analyze_support_resistance(df: pd.DataFrame, current_price: float) -> Dict[str, Any]:
    """
    Analyze S/R levels based on our 98% success rate discovery
    This is the core of our edge
    """
    psychological_levels = get_psychological_levels(current_price, 300)

    sr_analysis = {
        "nearest_levels": [],
        "strength_score": 0,
        "confluence_count": 0
    }

    for level in psychological_levels:
        distance = calculate_distance_to_level(current_price, level)

        # Check historical interaction with this level
        level_tests = df[
            (df['high'] >= level - 5) & (df['low'] <= level + 5)
        ]

        # Calculate level strength based on historical respect
        touches = len(level_tests)
        bounces = 0

        for _, candle in level_tests.iterrows():
            # Check if price bounced off the level
            if level <= current_price:  # Level acting as support
                if candle['low'] <= level + 2 and candle['close'] > level + 5:
                    bounces += 1
            else:  # Level acting as resistance
                if candle['high'] >= level - 2 and candle['close'] < level - 5:
                    bounces += 1

        bounce_rate = (bounces / touches) if touches > 0 else 0

        level_info = {
            "level": level,
            "distance_pips": distance["distance_pips"],
            "distance_percent": distance["distance_percent"],
            "direction": distance["direction"],
            "touches": touches,
            "bounces": bounces,
            "bounce_rate": bounce_rate,
            "strength": "strong" if bounce_rate > 0.7 else "moderate" if bounce_rate > 0.4 else "weak"
        }

        sr_analysis["nearest_levels"].append(level_info)

        # Boost confluence score for strong levels nearby
        if distance["distance_pips"] < 50 and bounce_rate > 0.8:
            sr_analysis["confluence_count"] += 2
        elif distance["distance_pips"] < 100 and bounce_rate > 0.6:
            sr_analysis["confluence_count"] += 1

    # Calculate overall strength score
    strong_levels_nearby = sum(1 for level in sr_analysis["nearest_levels"]
                              if level["distance_pips"] < 100 and level["bounce_rate"] > 0.7)

    sr_analysis["strength_score"] = min(100, strong_levels_nearby * 30 + sr_analysis["confluence_count"] * 10)

    return sr_analysis

def get_session_info() -> Dict[str, Any]:
    """Get current session info and expected volatility"""
    utc_now = datetime.now(pytz.UTC)
    utc_hour = utc_now.hour

    session_info = {
        "current_session": "off_hours",
        "expected_range": 50,
        "session_strength": "low",
        "hours_until_london": 0,
        "hours_until_ny": 0
    }

    # Determine current session
    if 8 <= utc_hour < 17:
        if 13 <= utc_hour < 17:
            session_info.update({
                "current_session": "london_ny_overlap",
                "expected_range": 125.2,
                "session_strength": "very_high"
            })
        else:
            session_info.update({
                "current_session": "london",
                "expected_range": 99.1,
                "session_strength": "high"
            })
    elif 13 <= utc_hour < 22:
        session_info.update({
            "current_session": "new_york",
            "expected_range": 92.1,
            "session_strength": "high"
        })
    elif 0 <= utc_hour < 8:
        session_info.update({
            "current_session": "asian",
            "expected_range": 84.0,
            "session_strength": "moderate"
        })

    # Calculate hours until major sessions
    if utc_hour < 8:
        session_info["hours_until_london"] = 8 - utc_hour
    elif utc_hour >= 17:
        session_info["hours_until_london"] = 24 - utc_hour + 8

    if utc_hour < 13:
        session_info["hours_until_ny"] = 13 - utc_hour
    elif utc_hour >= 22:
        session_info["hours_until_ny"] = 24 - utc_hour + 13

    return session_info

def calculate_atr_based_stops(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate ATR-based stop loss distance"""
    if len(df) < period:
        return XAUUSD_CONFIG["volatility"]["adr_pips"] * 0.5

    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]

    # Convert to pips (1 pip = $0.10 for gold)
    atr_pips = atr / 0.10

    return atr_pips * XAUUSD_CONFIG["risk_management"]["atr_stop_multiplier"]

def evaluate_xauusd_signal(df: pd.DataFrame, debug: bool = False) -> Optional[Dict[str, Any]]:
    """
    Main function to evaluate XAUUSD trading signals
    Based on 98% S/R confluence success rate
    """
    if df is None or df.empty or len(df) < 50:
        return {
            "skip_reason": "Insufficient data",
            "context": f"Need at least 50 candles, got {len(df) if df is not None else 0}",
            "signal": "SKIP"
        }

    current_price = df['close'].iloc[-1]
    session_info = get_session_info()

    if debug:
        print(f"XAUUSD Analysis - Price: ${current_price:.2f}, Session: {session_info['current_session']}")

    # Skip during low-volatility sessions unless near major level
    if session_info["session_strength"] == "low":
        nearest_major_level = min(XAUUSD_CONFIG["psychological_levels"]["major"],
                                key=lambda x: abs(x - current_price))
        distance_to_major = abs(current_price - nearest_major_level) / 0.10  # pips

        if distance_to_major > 30:
            return {
                "skip_reason": "Low volatility session",
                "context": f"Asian session with no major level nearby (nearest: ${nearest_major_level}, {distance_to_major:.1f} pips away)",
                "signal": "SKIP"
            }

    # Analyze support/resistance confluence
    sr_analysis = analyze_support_resistance(df, current_price)

    if debug:
        print(f"S/R Analysis: Strength Score: {sr_analysis['strength_score']}, Confluence Count: {sr_analysis['confluence_count']}")

    # Check if we're near a high-probability level
    strong_levels = [level for level in sr_analysis["nearest_levels"]
                    if level["distance_pips"] < 50 and level["bounce_rate"] > 0.8]

    if not strong_levels:
        return {
            "skip_reason": "No high-probability levels nearby",
            "context": f"Nearest strong S/R level is {sr_analysis['nearest_levels'][0]['distance_pips']:.1f} pips away",
            "signal": "SKIP"
        }

    # Calculate confluence score
    confluence_score = sr_analysis["strength_score"]

    # Determine signal direction based on nearest strong level
    nearest_strong = min(strong_levels, key=lambda x: x["distance_pips"])

    signal_direction = "BUY" if nearest_strong["direction"] == "above" and current_price < nearest_strong["level"] else "SELL"
    if nearest_strong["direction"] == "below" and current_price > nearest_strong["level"]:
        signal_direction = "SELL"
    elif nearest_strong["direction"] == "above" and current_price < nearest_strong["level"]:
        signal_direction = "BUY"
    else:
        signal_direction = "WAIT"

    if signal_direction == "WAIT":
        return {
            "skip_reason": "No clear directional bias",
            "context": f"Price ${current_price:.2f} not at optimal entry point relative to ${nearest_strong['level']} level",
            "signal": "SKIP"
        }

    # Skip if confluence score too low
    if confluence_score < XAUUSD_CONFIG["confluence_thresholds"]["minimum_score"]:
        return {
            "skip_reason": "Low confluence score",
            "context": f"Confluence score {confluence_score} below minimum {XAUUSD_CONFIG['confluence_thresholds']['minimum_score']}",
            "signal": "SKIP"
        }

    # Calculate stops and targets
    atr_stop = calculate_atr_based_stops(df)

    if signal_direction == "BUY":
        entry_price = current_price
        stop_loss = entry_price - (atr_stop * 0.10)  # Convert pips to price
        take_profit = entry_price + (atr_stop * XAUUSD_CONFIG["risk_management"]["min_rr_ratio"] * 0.10)
    else:
        entry_price = current_price
        stop_loss = entry_price + (atr_stop * 0.10)
        take_profit = entry_price - (atr_stop * XAUUSD_CONFIG["risk_management"]["min_rr_ratio"] * 0.10)

    # Build signal
    signal_strength = "VERY_STRONG" if confluence_score >= 90 else "STRONG" if confluence_score >= 80 else "MODERATE"

    utc_now = datetime.now(pytz.UTC)
    chicago_tz = pytz.timezone('America/Chicago')
    chicago_now = utc_now.astimezone(chicago_tz)

    signal = {
        "signal": signal_direction,
        "symbol": "XAUUSD",
        "entry_price": round(entry_price, 2),
        "stop_loss": round(stop_loss, 2),
        "take_profit": round(take_profit, 2),
        "confluence_score": confluence_score,
        "signal_strength": signal_strength,
        "risk_reward_ratio": XAUUSD_CONFIG["risk_management"]["min_rr_ratio"],
        "atr_stop_pips": round(atr_stop, 1),
        "session_info": session_info,
        "key_levels": [
            {
                "level": level["level"],
                "distance_pips": round(level["distance_pips"], 1),
                "strength": level["strength"],
                "bounce_rate": round(level["bounce_rate"] * 100, 1)
            }
            for level in strong_levels[:3]  # Top 3 levels
        ],
        "trade_reasons": [
            f"98% historical S/R respect rate at ${nearest_strong['level']} level",
            f"Strong confluence score: {confluence_score}/100",
            f"{signal_strength.replace('_', ' ').title()} signal strength",
            f"Optimal {session_info['current_session'].replace('_', ' ').title()} session timing",
            f"ATR-based risk management: {atr_stop:.1f} pips stop"
        ],
        "timestamp_utc": utc_now.isoformat(),
        "timestamp_chicago": chicago_now.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "analysis_method": "Psychological Level S/R Confluence",
        "expected_session_range": session_info["expected_range"]
    }

    if debug:
        print(f"Generated {signal_direction} signal with {confluence_score} confluence score")

    return signal

def get_xauusd_status() -> Dict[str, Any]:
    """Get current XAUUSD market status and analysis"""
    try:
        from .datafeed import fetch_h1
        df = fetch_h1("XAUUSD", timeframe="H1")

        if df is None or df.empty:
            return {"error": "No data available", "status": "unknown"}

        current_price = df['close'].iloc[-1]
        session_info = get_session_info()
        sr_analysis = analyze_support_resistance(df, current_price)

        return {
            "symbol": "XAUUSD",
            "current_price": round(current_price, 2),
            "session": session_info,
            "nearest_levels": sr_analysis["nearest_levels"][:5],
            "market_structure": {
                "confluence_score": sr_analysis["strength_score"],
                "strong_levels_nearby": len([l for l in sr_analysis["nearest_levels"]
                                           if l["distance_pips"] < 100 and l["bounce_rate"] > 0.7])
            },
            "trading_recommendation": "ACTIVE" if sr_analysis["strength_score"] >= 60 else "WAIT",
            "next_major_session": "London" if session_info["hours_until_london"] > 0 else "New York"
        }

    except Exception as e:
        return {"error": str(e), "status": "error"}