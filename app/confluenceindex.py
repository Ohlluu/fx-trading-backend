# backend/app/confluenceindex.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .confluence import (
    _ensure_utc_index, _is_dtindex, _calc_indicators_h1, _resample_ohlc,
    _calc_indicators_htf, _align_htf_to_h1, _prev_day_levels, _weekly_pivots,
    _pivot_highs_lows, _last_swing_price, _is_bull_engulf, _is_bear_engulf, _is_pin_bar
)

# Index-specific configuration based on research
INDEX_CONFIG = {
    "NAS100": {
        "point_value": 1.0,
        "best_session": "NY",
        "session_hours": (7, 16),  # 7am-4pm Chicago time (NY session)
        "skip_days": [],  # Let user decide - no skip days
        "best_days": [3],  # Thursday is best (3=Thursday)
        "min_score": 6,  # Show weak signals
        "recommended_score": 16,  # Recommended minimum for quality trades
        "psychological_levels": [20000, 21000, 22000, 23000, 24000, 25000],
        "gap_min": 50,  # Minimum gap size to consider (points)
        "volatility_multiplier": 1.0,
        "description": "Tech-heavy US index, best volatility Thursdays"
    },
    "DE30": {
        "point_value": 1.0,
        "best_session": "London",
        "session_hours": (2, 10),  # 2am-10am Chicago time (London session)
        "skip_days": [],  # No skip days
        "best_days": [1, 2, 3],  # Tue-Thu best
        "min_score": 6,  # Show weak signals
        "recommended_score": 15,  # Recommended minimum for quality trades
        "psychological_levels": [18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000],
        "gap_min": 30,
        "volatility_multiplier": 0.8,
        "description": "German industrial index, ECB sensitive, excellent gaps"
    },
    "UK100": {
        "point_value": 1.0,
        "best_session": "London",
        "session_hours": (2, 10),  # 2am-10am Chicago time (London session)
        "skip_days": [],  # No skip days
        "best_days": [1, 2, 3, 4],  # Mon-Thu good
        "min_score": 6,  # Show weak signals
        "recommended_score": 15,  # Recommended minimum for quality trades
        "psychological_levels": [8000, 8500, 9000, 9500, 10000],
        "gap_min": 20,
        "volatility_multiplier": 0.7,
        "description": "UK index, GBP correlation, benefits from weak pound"
    },
    "SPX500": {
        "point_value": 1.0,
        "best_session": "NY",
        "session_hours": (7, 16),  # 7am-4pm Chicago time (NY session)
        "skip_days": [],  # No skip days
        "best_days": [1, 2, 3, 4],  # Mon-Thu good
        "min_score": 6,  # Show weak signals
        "recommended_score": 16,  # Recommended minimum for quality trades
        "psychological_levels": [5000, 5100, 5200, 5300, 5400, 5500, 5600, 5700, 5800, 5900, 6000],
        "gap_min": 10,
        "volatility_multiplier": 1.2,
        "description": "Broad US market index, highest volatility, best performer"
    }
}

def _get_index_config(index: str) -> Dict[str, Any]:
    """Get configuration for specific index"""
    return INDEX_CONFIG.get(index.upper(), INDEX_CONFIG["SPX500"])

def _is_in_session(ts: pd.Timestamp, index: str) -> bool:
    """Check if timestamp is in best trading session for index (Chicago time)"""
    import pytz
    cfg = _get_index_config(index)

    # Convert UTC to Chicago time
    chicago_tz = pytz.timezone('America/Chicago')
    chicago_time = ts.tz_convert(chicago_tz)
    hour = chicago_time.hour

    start_hour, end_hour = cfg["session_hours"]
    return start_hour <= hour < end_hour

def _should_skip_day(ts: pd.Timestamp, index: str) -> bool:
    """Check if we should skip trading this day for the index (Chicago time)"""
    import pytz
    cfg = _get_index_config(index)

    # Convert UTC to Chicago time
    chicago_tz = pytz.timezone('America/Chicago')
    chicago_time = ts.tz_convert(chicago_tz)
    weekday = chicago_time.weekday()  # 0=Monday, 6=Sunday
    return weekday in cfg["skip_days"]

def _is_best_day(ts: pd.Timestamp, index: str) -> bool:
    """Check if this is a best trading day for the index (Chicago time)"""
    import pytz
    cfg = _get_index_config(index)

    # Convert UTC to Chicago time
    chicago_tz = pytz.timezone('America/Chicago')
    chicago_time = ts.tz_convert(chicago_tz)
    weekday = chicago_time.weekday()
    return weekday in cfg["best_days"]

def _detect_gap(df: pd.DataFrame, current_idx: int) -> Optional[Dict[str, Any]]:
    """Detect opening gap from previous close"""
    if current_idx < 1:
        return None

    current_bar = df.iloc[current_idx]
    prev_bar = df.iloc[current_idx - 1]

    # Check if there's a time gap (more than 2 hours)
    time_diff = current_bar.name - prev_bar.name
    if time_diff.total_seconds() < 7200:  # Less than 2 hours
        return None

    gap_size = float(current_bar["open"]) - float(prev_bar["close"])
    gap_pct = (gap_size / float(prev_bar["close"])) * 100

    if abs(gap_size) < 5:  # Minimum gap size
        return None

    return {
        "size": gap_size,
        "percentage": gap_pct,
        "direction": "UP" if gap_size > 0 else "DOWN",
        "fill_target": float(prev_bar["close"])
    }

def _get_psychological_levels(price: float, index: str) -> List[float]:
    """Get nearby psychological levels for the index"""
    cfg = _get_index_config(index)
    levels = cfg["psychological_levels"]

    # Find levels within reasonable range of current price
    nearby_levels = []
    for level in levels:
        if abs(level - price) / price < 0.1:  # Within 10% of current price
            nearby_levels.append(level)

    return nearby_levels

def _calculate_index_score(ctx: pd.DataFrame, i: int, index: str) -> Dict[str, Any]:
    """Calculate confluence score specifically for indices"""
    row = ctx.iloc[i]
    prev_row = ctx.iloc[i-1] if i > 0 else row

    cfg = _get_index_config(index)
    score = 0
    reasons = []

    # HTF Trend alignment (stronger weight for indices)
    h4_up = row.get("H4_close", 0) > row.get("H4_ema200", 0)
    h4_down = row.get("H4_close", 0) < row.get("H4_ema200", 0)
    d1_up = row.get("D1_close", 0) > row.get("D1_ema200", 0)
    d1_down = row.get("D1_close", 0) < row.get("D1_ema200", 0)

    long_score = 0
    short_score = 0

    # H4 trend (3 points for indices vs 2 for forex)
    if h4_up:
        long_score += 3
        reasons.append("H4 uptrend confirmed")
    elif h4_down:
        short_score += 3
        reasons.append("H4 downtrend confirmed")

    # D1 trend (2 points)
    if d1_up:
        long_score += 2
        reasons.append("D1 bullish bias")
    elif d1_down:
        short_score += 2
        reasons.append("D1 bearish bias")

    # H1 baseline trend
    trend_up = row["close"] > row["ema200"]
    trend_down = row["close"] < row["ema200"]

    if trend_up:
        long_score += 2
        reasons.append("H1 above 200 EMA")
    elif trend_down:
        short_score += 2
        reasons.append("H1 below 200 EMA")

    # Fast EMA alignment
    fast_up = row["ema20"] > row["ema50"]
    fast_down = row["ema20"] < row["ema50"]

    if fast_up:
        long_score += 1
        reasons.append("Fast EMAs bullish")
    elif fast_down:
        short_score += 1
        reasons.append("Fast EMAs bearish")

    # RSI momentum (indices respect momentum more)
    rsi = float(row.get("rsi", 50))
    prev_rsi = float(prev_row.get("rsi", 50))

    if 50 <= rsi <= 70 and rsi > prev_rsi:
        long_score += 2
        reasons.append("RSI bullish momentum zone")
    elif 30 <= rsi <= 50 and rsi < prev_rsi:
        short_score += 2
        reasons.append("RSI bearish momentum zone")

    # ADX strength (indices trend well)
    adx = float(row.get("adx", 0))
    if adx >= 25:
        long_score += 1
        short_score += 1
        reasons.append(f"Strong ADX: {adx:.1f}")

    # Psychological level proximity (unique to indices)
    price = float(row["close"])
    psych_levels = _get_psychological_levels(price, index)

    for level in psych_levels:
        distance = abs(price - level)
        if distance < price * 0.005:  # Within 0.5%
            if price < level:
                long_score += 2
                reasons.append(f"Near psychological support {level}")
            else:
                short_score += 2
                reasons.append(f"Near psychological resistance {level}")

    # Choose side
    if long_score > short_score:
        side = "BUY"
        score = long_score
    elif short_score > long_score:
        side = "SELL"
        score = short_score
    else:
        side = None
        score = max(long_score, short_score)

    return {
        "side": side,
        "score": score,
        "long_score": long_score,
        "short_score": short_score,
        "reasons": reasons
    }

def _build_index_context(h1_df: pd.DataFrame) -> pd.DataFrame:
    """Build context dataframe for index analysis"""
    df = h1_df.sort_index().copy()
    df = _ensure_utc_index(df)
    _is_dtindex(df)
    df = _calc_indicators_h1(df)

    # HTF analysis
    h4 = _resample_ohlc(df[["open", "high", "low", "close"]], "4h")
    h4 = _calc_indicators_htf(h4)
    d1 = _resample_ohlc(df[["open", "high", "low", "close"]], "1D")
    d1 = _calc_indicators_htf(d1)
    d1_levels = _prev_day_levels(d1)
    w1 = _resample_ohlc(df[["open", "high", "low", "close"]], "1W")
    weekly = _weekly_pivots(w1)

    # Align everything to H1
    ctx = df.copy()
    ctx = _align_htf_to_h1(ctx, h4[["ema20", "ema50", "ema200", "close"]], prefix="H4")
    ctx = _align_htf_to_h1(ctx, d1[["ema20", "ema50", "ema200", "close"]], prefix="D1")
    ctx = pd.concat([
        ctx,
        d1_levels.reindex(ctx.index, method="ffill"),
        weekly.reindex(ctx.index, method="ffill")
    ], axis=1)

    return ctx

def evaluate_index_signal(h1_df: pd.DataFrame, index: str, debug: bool = False) -> Optional[Dict[str, Any]]:
    """
    Evaluate index trading signal with all intelligence built in
    """
    index = index.upper()
    cfg = _get_index_config(index)

    def _skip(reason: str, context: str = "") -> Optional[Dict[str, Any]]:
        result = {
            "index": index,
            "skip_reason": reason,
            "context": context,
            "time_checked_utc": pd.Timestamp.now(tz="UTC").floor("h")
        }
        return result if debug else None

    # Build context
    try:
        ctx = _build_index_context(h1_df)
    except Exception as e:
        return _skip("Failed to build context", str(e))

    # Lower requirement for NAS100 due to data limitations
    min_bars = 200 if index.upper() == "NAS100" else 300
    if len(ctx) < min_bars:
        return _skip("Insufficient data", f"Need at least {min_bars} H1 bars (have {len(ctx)})")

    # Get last closed bar
    now_utc = pd.Timestamp.now(tz="UTC").floor("h")
    closed_bars = ctx.index[ctx.index < now_utc]

    if len(closed_bars) < 2:
        return _skip("No closed bars available")

    current_ts = closed_bars[-1]
    current_idx = len(closed_bars) - 1

    # Day-of-week intelligence (use current time)
    if _should_skip_day(now_utc, index):
        day_name = now_utc.strftime("%A")
        return _skip(f"Skip {day_name} for {index}", f"{index} has poor performance on {day_name}s")

    # Session timing check with time remaining (use CURRENT time, not candle timestamp)
    import pytz
    if not _is_in_session(now_utc, index):  # Use current time for session check
        session_start, session_end = cfg["session_hours"]
        chicago_tz = pytz.timezone('America/Chicago')
        current_chicago = now_utc.tz_convert(chicago_tz)  # Current time in Chicago
        return _skip("Outside optimal session",
                    f"{index} best session: {session_start:02d}:00-{session_end:02d}:00 Chicago (current: {current_chicago.hour:02d}:00 Chicago)")

    # Calculate time remaining in session (use current time)
    chicago_tz = pytz.timezone('America/Chicago')
    chicago_time = now_utc.tz_convert(chicago_tz)  # Current time in Chicago
    session_start, session_end = cfg["session_hours"]
    hours_remaining = session_end - chicago_time.hour

    # Gap analysis
    gap_info = _detect_gap(ctx, current_idx)

    # Calculate confluence score
    score_info = _calculate_index_score(ctx, current_idx, index)

    if score_info["score"] < cfg["min_score"]:
        return _skip(f"Score {score_info['score']} < minimum {cfg['min_score']}",
                    f"Confluence not strong enough for {index}")

    if not score_info["side"]:
        return _skip("No clear directional bias", "Long and short scores are tied")

    # Determine signal strength
    recommended_score = cfg["recommended_score"]
    is_weak_signal = score_info["score"] < recommended_score

    # Calculate entry, stop, target
    row = ctx.iloc[current_idx]
    side = score_info["side"]

    # Entry price calculation:
    # NOTE: This uses the last candle close as reference
    # In live trading, you would enter at current market price
    last_close = float(row["close"])
    entry = last_close

    # Add entry guidance to trade reasons
    entry_guidance = f"Reference entry: {entry:.2f} (adjust to current market price when trading)"

    # Stop loss using pivot analysis
    pivots = _pivot_highs_lows(ctx)
    swing_stop = _last_swing_price(ctx, current_idx, side, pivots)
    atr_stop = float(row["atr"]) * 1.5

    if side == "BUY":
        if swing_stop:
            stop = min(swing_stop, entry - atr_stop)  # Use the LOWER (safer) stop for BUY
        else:
            stop = entry - atr_stop

        # Target using psychological levels or ATR
        psych_levels = _get_psychological_levels(entry, index)
        resistance_levels = [l for l in psych_levels if l > entry]

        if resistance_levels:
            target = min(resistance_levels)
        else:
            target = entry + (atr_stop * 2)  # 2:1 RR minimum
    else:  # SELL
        if swing_stop:
            stop = max(swing_stop, entry + atr_stop)  # Use the HIGHER (safer) stop for SELL
        else:
            stop = entry + atr_stop

        support_levels = [l for l in _get_psychological_levels(entry, index) if l < entry]

        if support_levels:
            target = max(support_levels)
        else:
            target = entry - (atr_stop * 2)

    # Risk/Reward calculation
    risk = abs(entry - stop)
    reward = abs(target - entry)
    rr = reward / max(risk, 0.01)

    if rr < 1.5:  # Minimum RR for indices
        return _skip(f"RR {rr:.1f} too low", "Need minimum 1.5 RR for index trades")

    # Build trade reasons
    trade_reasons = []

    # Day timing reason
    if _is_best_day(current_ts, index):
        day_name = current_ts.strftime("%A")
        trade_reasons.append(f"{day_name} is a high-performance day for {index}")

    # Session reason
    session_name = cfg["best_session"]
    trade_reasons.append(f"Trading during optimal {session_name} session")

    # Gap reason
    if gap_info:
        gap_dir = "bullish" if gap_info["direction"] == "UP" else "bearish"
        trade_reasons.append(f"{gap_info['direction'].lower()} gap of {abs(gap_info['size']):.0f} points adds {gap_dir} bias")

    # Technical reason with quality warnings
    if is_weak_signal:
        trade_reasons.append(f"⚠️ WEAK SIGNAL: Score {score_info['score']}/{recommended_score} - Consider skipping or reduced size")
        trade_reasons.append(f"ADVISORY: Recommended minimum score is {recommended_score} for quality trades")
    elif score_info["score"] >= recommended_score + 3:
        trade_reasons.append(f"✅ STRONG: Score {score_info['score']}/{recommended_score} - High-quality setup")
    else:
        trade_reasons.append(f"✅ GOOD: Score {score_info['score']}/{recommended_score} - Tradeable setup")

    # Psychological level reason
    psych_levels = _get_psychological_levels(entry, index)
    nearby_levels = [l for l in psych_levels if abs(l - entry) / entry < 0.01]
    if nearby_levels:
        level = nearby_levels[0]
        level_type = "support" if entry > level else "resistance"
        trade_reasons.append(f"Price near key psychological {level_type} at {level}")

    # Session timing reason
    if hours_remaining >= 4:
        trade_reasons.append(f"Excellent timing: {hours_remaining} hours left in {cfg['best_session']} session")
    elif hours_remaining >= 2:
        trade_reasons.append(f"Good timing: {hours_remaining} hours remaining in session")
    else:
        trade_reasons.append(f"⚠️ Caution: Only {hours_remaining} hour(s) left in session - consider reduced size")

    # Add entry guidance
    trade_reasons.append(entry_guidance)

    # Determine signal type based on strength
    if is_weak_signal:
        signal_type = "WEAK"
        signal_advice = "Consider skipping or reduced position size"
    else:
        signal_type = "TRADE"
        signal_advice = "Quality setup - tradeable"

    return {
        "index": index,
        "signal": signal_type,
        "signal_advice": signal_advice,
        "time": current_ts,
        "side": side,
        "price": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "rr": round(rr, 2),
        "score": score_info["score"],
        "recommended_score": recommended_score,
        "is_weak_signal": is_weak_signal,
        "atr": round(float(row["atr"]), 2),
        "session": cfg["best_session"],
        "gap_info": gap_info,
        "trade_reasons": trade_reasons,
        "confluence_breakdown": {
            "long_score": score_info["long_score"],
            "short_score": score_info["short_score"],
            "reasons": score_info["reasons"]
        },
        "timing_intelligence": {
            "is_best_day": _is_best_day(current_ts, index),
            "in_optimal_session": True,
            "session_hours": f"{cfg['session_hours'][0]:02d}:00-{cfg['session_hours'][1]:02d}:00 Chicago",
            "hours_remaining": hours_remaining,
            "session_warning": "⚠️ Less than 2 hours remaining" if hours_remaining < 2 else "✅ Good timing",
            "day_name": current_ts.strftime("%A")
        },
        "skip_reason": None
    }

def get_index_status(index: str) -> Dict[str, Any]:
    """Get current status and timing for an index"""
    import pytz
    cfg = _get_index_config(index)
    now_utc = pd.Timestamp.now(tz="UTC")

    # Convert to Chicago time for display
    chicago_tz = pytz.timezone('America/Chicago')
    now_chicago = now_utc.tz_convert(chicago_tz)

    in_session = _is_in_session(now_utc, index)
    should_skip = _should_skip_day(now_utc, index)
    is_best = _is_best_day(now_utc, index)

    # Calculate next session time in Chicago
    session_start, session_end = cfg["session_hours"]
    current_hour_chicago = now_chicago.hour

    if current_hour_chicago < session_start:
        hours_to_session = session_start - current_hour_chicago
    elif current_hour_chicago >= session_end:
        hours_to_session = (24 - current_hour_chicago) + session_start
    else:
        hours_to_session = 0

    status = "ACTIVE" if in_session and not should_skip else "INACTIVE"
    if should_skip:
        status = "SKIP_DAY"

    return {
        "index": index.upper(),
        "status": status,
        "in_session": in_session,
        "should_skip_day": should_skip,
        "is_best_day": is_best,
        "current_hour_chicago": current_hour_chicago,
        "session_hours": f"{session_start:02d}:00-{session_end:02d}:00 Chicago",
        "hours_to_next_session": hours_to_session,
        "description": cfg["description"],
        "day_name": now_chicago.strftime("%A"),
        "current_time_chicago": now_chicago.strftime("%I:%M %p")
    }