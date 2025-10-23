# app/confluence.py — v4.2 (HIGH-QUALITY H1 TRADING for 1-3 premium trades/day)
# - OPTIMIZED FOR QUALITY OVER QUANTITY (H1 timeframe)
# - "Stable" profile for consistent high-win-rate trading
# - High minimum scores: 15+ (16 for XAUUSD) for premium setups only
# - H1 analysis for clean, reliable signals with less noise
# - Focus on 2:1+ R:R trades for sustainable profitability
# - Always returns a structured dict (no None on skip)
# - Global try/except to prevent HTTP 500s; exposes error in payload

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import pandas_ta as ta

VERSION = "v4.0"

# ============================================================
# Pair config
# ============================================================

PAIR_CONFIG: Dict[str, Dict[str, Any]] = {
    # pip_size, RSI mid-band, minimum ATR in pips (vol floor), minimum score, use_session
    "EURUSD": {"pip_size": 0.0001, "rsi_band": (40, 60), "min_atr_pips": 6,   "min_score": 15, "use_session": True},
    "GBPUSD": {"pip_size": 0.0001, "rsi_band": (40, 60), "min_atr_pips": 8,   "min_score": 15, "use_session": True},
    "USDJPY": {"pip_size": 0.01,   "rsi_band": (45, 55), "min_atr_pips": 10,  "min_score": 15, "use_session": True},
    "GBPJPY": {"pip_size": 0.01,   "rsi_band": (45, 55), "min_atr_pips": 15,  "min_score": 15, "use_session": True},
    "XAUUSD": {"pip_size": 0.01,   "rsi_band": (45, 55), "min_atr_pips": 100, "min_score": 16, "use_session": True},
    "SPX500": {"pip_size": 1.0,    "rsi_band": (40, 60), "min_atr_pips": 2.5, "min_score": 15, "use_session": True},

    "AUDUSD": {"pip_size": 0.0001, "rsi_band": (40, 60), "min_atr_pips": 6,   "min_score": 15, "use_session": True},
    "NZDUSD": {"pip_size": 0.0001, "rsi_band": (40, 60), "min_atr_pips": 6,   "min_score": 15, "use_session": True},
    "EURCAD": {"pip_size": 0.0001, "rsi_band": (40, 60), "min_atr_pips": 8,   "min_score": 15, "use_session": True},
    "USDCHF": {"pip_size": 0.0001, "rsi_band": (45, 55), "min_atr_pips": 7,   "min_score": 15, "use_session": True},
    "USDCAD": {"pip_size": 0.0001, "rsi_band": (45, 55), "min_atr_pips": 7,   "min_score": 15, "use_session": True},
    "EURJPY": {"pip_size": 0.01,   "rsi_band": (45, 55), "min_atr_pips": 12,  "min_score": 15, "use_session": True},
    "EURCHF": {"pip_size": 0.0001, "rsi_band": (45, 55), "min_atr_pips": 6,   "min_score": 15, "use_session": True},
    "EURGBP": {"pip_size": 0.0001, "rsi_band": (45, 55), "min_atr_pips": 6,   "min_score": 15, "use_session": True},
}

# ======================= Profiles ==========================
# Conservative aims for higher WR (fewer trades). Normal = your v3.1.
# Relaxed is more permissive (more trades).
PROFILES = {
    "stable": dict(
        MACD_CROSS_LOOKBACK=12, MACD_CROSS_MAX_AGE=8,    # Longer lookback, more patience
        EMA_OVEREXT_MULT=1.8,                            # More tolerance for pullbacks
        ATR_PCTILE_WINDOW=800,                           # Longer volatility assessment
        H4_SLOPE_LOOKBACK=8,                             # More persistent slope check
        LEVEL_MIN_BUFFER_ATR=0.1,
        RR_MIN_DEFAULT=2.5,                              # Proper RR for profitable trading
        USE_FVG=False, USE_LIQUIDITY=False, USE_TOUCH_SR=True, USE_ADAPTIVE_TP=True, USE_BOS=True, USE_SWEEP_RECLAIM=False,
        FVG_ATR_MIN_MULT=0.35,
        FVG_NEAR_TOL_ATR=0.30,
        SR_MERGE_TOL_ATR=0.20,
        STRONG_TOUCHES_MIN=4,                            # Stronger levels required
        EMA20_PROX_GATE_ATR=1.2,                         # Allow more distance from EMA20
        EFFICIENCY_MIN=0.10,                             # Extremely flexible on candle efficiency
        ADX_FLOOR=15,                                    # Lower ADX requirement
        MIN_TIMEOUT_HOURS=2.0, MAX_TIMEOUT_HOURS=16.0,  # Longer time horizons
        EFF_TREND_GOOD=0.8, EFF_TREND_OK=0.6, EFF_TREND_WEAK=0.4, EFF_TREND_BAD=0.3,
        # Stability-focused weights
        TREND_STABILITY_WEIGHT=3,                        # Multi-timeframe alignment gets higher weight
        MOMENTUM_WEIGHT=0,                               # Remove volatile momentum scoring
        STRUCTURE_WEIGHT=1,                              # Reduce BOS/liquidity weight
        ATR_REGIME_STRICT=False,                         # More flexible ATR requirements
    ),
    "conservative": dict(
        MACD_CROSS_LOOKBACK=6, MACD_CROSS_MAX_AGE=4,
        EMA_OVEREXT_MULT=1.2,
        ATR_PCTILE_WINDOW=600,
        H4_SLOPE_LOOKBACK=5,
        LEVEL_MIN_BUFFER_ATR=0.7,   # require more space to next level
        RR_MIN_DEFAULT=1.0,         # Allow smaller moves, perfect for Asian session
        USE_FVG=True, USE_LIQUIDITY=True, USE_TOUCH_SR=True, USE_ADAPTIVE_TP=True, USE_BOS=True, USE_SWEEP_RECLAIM=True,
        FVG_ATR_MIN_MULT=0.30,      # only respect larger, cleaner FVGs
        FVG_NEAR_TOL_ATR=0.25,
        SR_MERGE_TOL_ATR=0.15,
        STRONG_TOUCHES_MIN=3,
        EMA20_PROX_GATE_ATR=0.8,    # must be close to EMA20 = no chase
        EFFICIENCY_MIN=0.25,        # Demand quality directional candles
        ADX_FLOOR=18,               # want stronger trends
        MIN_TIMEOUT_HOURS=1.0, MAX_TIMEOUT_HOURS=12.0,
        EFF_TREND_GOOD=0.9, EFF_TREND_OK=0.75, EFF_TREND_WEAK=0.6, EFF_TREND_BAD=0.5,
        # Default weights for other profiles
        TREND_STABILITY_WEIGHT=1,
        MOMENTUM_WEIGHT=2, 
        STRUCTURE_WEIGHT=2,
        ATR_REGIME_STRICT=True,
    ),
    "normal": dict(
        MACD_CROSS_LOOKBACK=5, MACD_CROSS_MAX_AGE=4,
        EMA_OVEREXT_MULT=1.2,
        ATR_PCTILE_WINDOW=600, H4_SLOPE_LOOKBACK=5, LEVEL_MIN_BUFFER_ATR=0.6,
        RR_MIN_DEFAULT=1.5,
        USE_FVG=True, USE_LIQUIDITY=True, USE_TOUCH_SR=True, USE_ADAPTIVE_TP=True, USE_BOS=True, USE_SWEEP_RECLAIM=True,
        FVG_ATR_MIN_MULT=0.25, FVG_NEAR_TOL_ATR=0.25, SR_MERGE_TOL_ATR=0.15, STRONG_TOUCHES_MIN=3,
        EMA20_PROX_GATE_ATR=0.8, EFFICIENCY_MIN=0.25, ADX_FLOOR=18,
        MIN_TIMEOUT_HOURS=1.0, MAX_TIMEOUT_HOURS=12.0,
        EFF_TREND_GOOD=0.9, EFF_TREND_OK=0.75, EFF_TREND_WEAK=0.6, EFF_TREND_BAD=0.5,
        # Default weights
        TREND_STABILITY_WEIGHT=1,
        MOMENTUM_WEIGHT=2, 
        STRUCTURE_WEIGHT=2,
        ATR_REGIME_STRICT=True,
    ),
    "relaxed": dict(
        MACD_CROSS_LOOKBACK=7, MACD_CROSS_MAX_AGE=8,
        EMA_OVEREXT_MULT=1.6,
        ATR_PCTILE_WINDOW=600, H4_SLOPE_LOOKBACK=5, LEVEL_MIN_BUFFER_ATR=0.55,
        RR_MIN_DEFAULT=1.2,
        USE_FVG=True, USE_LIQUIDITY=True, USE_TOUCH_SR=True, USE_ADAPTIVE_TP=True, USE_BOS=True, USE_SWEEP_RECLAIM=True,
        FVG_ATR_MIN_MULT=0.25, FVG_NEAR_TOL_ATR=0.35, SR_MERGE_TOL_ATR=0.20, STRONG_TOUCHES_MIN=2,
        EMA20_PROX_GATE_ATR=1.2, EFFICIENCY_MIN=0.18, ADX_FLOOR=14,
        MIN_TIMEOUT_HOURS=1.0, MAX_TIMEOUT_HOURS=12.0,
        EFF_TREND_GOOD=0.9, EFF_TREND_OK=0.75, EFF_TREND_WEAK=0.6, EFF_TREND_BAD=0.5,
        # Default weights
        TREND_STABILITY_WEIGHT=1,
        MOMENTUM_WEIGHT=2, 
        STRUCTURE_WEIGHT=2,
        ATR_REGIME_STRICT=True,
    ),
}

STRICTNESS = "stable"   # ← stable profile for high-quality H1 trades
_cfg = PROFILES[STRICTNESS]
MACD_CROSS_LOOKBACK   = _cfg["MACD_CROSS_LOOKBACK"]
MACD_CROSS_MAX_AGE    = _cfg["MACD_CROSS_MAX_AGE"]
EMA_OVEREXT_MULT      = _cfg["EMA_OVEREXT_MULT"]
ATR_PCTILE_WINDOW     = _cfg["ATR_PCTILE_WINDOW"]
H4_SLOPE_LOOKBACK     = _cfg["H4_SLOPE_LOOKBACK"]
LEVEL_MIN_BUFFER_ATR  = _cfg["LEVEL_MIN_BUFFER_ATR"]
RR_MIN_DEFAULT        = _cfg["RR_MIN_DEFAULT"]

USE_FVG               = _cfg["USE_FVG"]
USE_LIQUIDITY         = _cfg["USE_LIQUIDITY"]
USE_TOUCH_SR          = _cfg["USE_TOUCH_SR"]
USE_ADAPTIVE_TP       = _cfg["USE_ADAPTIVE_TP"]
USE_BOS               = _cfg["USE_BOS"]
USE_SWEEP_RECLAIM     = _cfg["USE_SWEEP_RECLAIM"]

FVG_ATR_MIN_MULT      = _cfg["FVG_ATR_MIN_MULT"]
FVG_NEAR_TOL_ATR      = _cfg["FVG_NEAR_TOL_ATR"]
SR_MERGE_TOL_ATR      = _cfg["SR_MERGE_TOL_ATR"]
STRONG_TOUCHES_MIN    = _cfg["STRONG_TOUCHES_MIN"]

EMA20_PROX_GATE_ATR   = _cfg["EMA20_PROX_GATE_ATR"]
EFFICIENCY_MIN        = _cfg["EFFICIENCY_MIN"]
ADX_FLOOR             = _cfg["ADX_FLOOR"]

MIN_TIMEOUT_HOURS     = _cfg["MIN_TIMEOUT_HOURS"]
MAX_TIMEOUT_HOURS     = _cfg["MAX_TIMEOUT_HOURS"]
EFF_TREND_GOOD        = _cfg["EFF_TREND_GOOD"]
EFF_TREND_OK          = _cfg["EFF_TREND_OK"]
EFF_TREND_WEAK        = _cfg["EFF_TREND_WEAK"]
EFF_TREND_BAD         = _cfg["EFF_TREND_BAD"]

# Stable confluence weights
TREND_STABILITY_WEIGHT = _cfg.get("TREND_STABILITY_WEIGHT", 1)
MOMENTUM_WEIGHT = _cfg.get("MOMENTUM_WEIGHT", 2)
STRUCTURE_WEIGHT = _cfg.get("STRUCTURE_WEIGHT", 2)
ATR_REGIME_STRICT = _cfg.get("ATR_REGIME_STRICT", True)

# Enhanced stop multiplier with volatility adjustment
DEFAULT_ATR_STOP_MULT = 2.2  # Wider stops for better survival
VOL_ADJ_MULT = {'low': 1.2, 'normal': 1.5, 'high': 2.0}

# Market regime detection thresholds
TREND_STRENGTH_THRESHOLD = 0.6
VOL_REGIME_LOOKBACK = 100
SPREAD_TOLERANCE = 1.5  # max spread vs average spread

# News event avoidance (hours before/after major events)
NEWS_BLACKOUT_HOURS = 2

# Enhanced signal quality filters
MIN_VOLUME_RATIO = 0.8  # vs recent average
MAX_WICK_RATIO = 0.7    # body vs total range
MIN_WICK_RATIO = 0.4    # minimum body ratio
MIN_MOMENTUM_BARS = 3    # consecutive momentum bars required

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _cfg_for(pair: Optional[str]) -> Tuple[Dict[str, Any], str]:
    p = (pair or "EURUSD").upper()
    return (PAIR_CONFIG.get(p) or PAIR_CONFIG["EURUSD"]), p

def _pip_size_for(pair: str) -> float:
    cfg, p = _cfg_for(pair)
    if cfg and "pip_size" in cfg:
        return float(cfg["pip_size"])
    return 0.01 if p.endswith("JPY") else 0.0001

def _detect_market_regime(df: pd.DataFrame, lookback: int = 50) -> Dict[str, Any]:
    """Detect current market regime: trending/ranging, volatility level"""
    close = df['close'].tail(lookback)
    high = df['high'].tail(lookback)
    low = df['low'].tail(lookback)
    
    # Trend strength using linear regression slope
    x = np.arange(len(close))
    slope = np.polyfit(x, close, 1)[0]
    trend_strength = abs(slope) / close.std() * 100
    
    # Volatility regime - FIXED: Proper percentile calculation
    atr_series = ta.atr(high, low, close, length=14).tail(VOL_REGIME_LOOKBACK)
    atr_percentile = atr_series.rank(pct=True).iloc[-1]  # True percentile (0.0 to 1.0)
    
    vol_regime = 'low' if atr_percentile < 0.3 else ('high' if atr_percentile > 0.7 else 'normal')
    
    # Market structure (higher highs, higher lows) - FIXED: Bounds checking
    recent_highs = high.tail(20).rolling(5).max()
    recent_lows = low.tail(20).rolling(5).min()

    # Add bounds checking to prevent index errors
    if len(recent_highs) >= 6 and len(recent_lows) >= 6:
        structure = 'bullish' if recent_highs.iloc[-1] > recent_highs.iloc[-5] and recent_lows.iloc[-1] > recent_lows.iloc[-5] else \
                    'bearish' if recent_highs.iloc[-1] < recent_highs.iloc[-5] and recent_lows.iloc[-1] < recent_lows.iloc[-5] else 'sideways'
    else:
        structure = 'sideways'  # Default for insufficient data
    
    return {
        'trend_strength': trend_strength,
        'is_trending': trend_strength > TREND_STRENGTH_THRESHOLD,
        'vol_regime': vol_regime,
        'structure': structure,
        'atr_percentile': atr_percentile
    }

def _check_spread_conditions(pair: str, current_price: float) -> bool:
    """Check if spread conditions are favorable for entry"""
    # Simplified spread check - in production, you'd get this from broker API
    typical_spreads = {
        'EURUSD': 0.0001, 'GBPUSD': 0.0001, 'USDJPY': 0.01,
        'GBPJPY': 0.02, 'XAUUSD': 0.3, 'SPX500': 0.5
    }
    
    # For demo purposes, assume spread is acceptable
    # In production: current_spread = get_current_spread(pair)
    # return current_spread <= typical_spreads.get(pair, 0.0002) * SPREAD_TOLERANCE
    return True

def _enhanced_signal_quality(df: pd.DataFrame) -> Dict[str, float]:
    """Enhanced signal quality checks"""
    if len(df) < 50:
        return {'quality_score': 0, 'volume_ratio': 0, 'momentum_score': 0}
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df.get('volume', pd.Series([1] * len(df), index=df.index))
    
    # Volume analysis
    avg_volume = volume.tail(20).mean() if volume.sum() > 0 else 1
    current_volume = volume.iloc[-1] if volume.sum() > 0 else 1
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    # Momentum consistency
    momentum_bars = 0
    for i in range(min(5, len(df)-1)):
        if close.iloc[-(i+1)] > close.iloc[-(i+2)]:
            momentum_bars += 1
        else:
            break
    
    # Wick analysis (prefer strong bodies)
    latest_bar = df.iloc[-1]
    body = abs(latest_bar['close'] - latest_bar['open'])
    total_range = latest_bar['high'] - latest_bar['low']
    wick_ratio = body / total_range if total_range > 0 else 0
    
    # Overall quality score
    quality_score = (
        min(volume_ratio / MIN_VOLUME_RATIO, 1.0) * 0.3 +
        min(momentum_bars / MIN_MOMENTUM_BARS, 1.0) * 0.4 +
        min(wick_ratio / MIN_WICK_RATIO, 1.0) * 0.3
    )
    
    return {
        'quality_score': quality_score,
        'volume_ratio': volume_ratio,
        'momentum_score': momentum_bars / MIN_MOMENTUM_BARS,
        'wick_ratio': wick_ratio
    }

def _require_cols(df: pd.DataFrame, cols=("open","high","low","close")):
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing '{c}' in DataFrame")

def _is_dtindex(df: pd.DataFrame):
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a pandas.DatetimeIndex (UTC recommended).")

def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        return df
    if df.index.tz is None:
        return df.set_index(df.index.tz_localize("UTC"), drop=True)
    return df.set_index(df.index.tz_convert("UTC"), drop=True)

def _slope(series: pd.Series, lookback: int = 5) -> float:
    if series is None or len(series) < max(2, lookback):
        return 0.0
    a, b = series.iloc[-lookback], series.iloc[-1]
    return float(b - a) / float(lookback)

# ------------------------------------------------------------
# Indicator computation (H1 + HTF)
# ------------------------------------------------------------

def _calc_indicators_h1(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df)
    out = df.copy()

    out["ema20"]  = ta.ema(out["close"], 20)
    out["ema50"]  = ta.ema(out["close"], 50)
    out["ema200"] = ta.ema(out["close"], 200)
    out["rsi"]    = ta.rsi(out["close"], 14)

    macd = ta.macd(out["close"])
    out["macd_hist"] = macd["MACDh_12_26_9"] if macd is not None else np.nan

    out["atr"]    = ta.atr(out["high"], out["low"], out["close"], 14)
    adx           = ta.adx(out["high"], out["low"], out["close"], 14)
    out["adx"]    = adx["ADX_14"] if adx is not None else np.nan

    return out

def _resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    res = df[["open","high","low","close"]].resample(rule, label="right", closed="right").agg(agg)
    for c in [c for c in df.columns if c not in agg]:
        res[c] = df[c].resample(rule).last()
    return res.dropna(how="any")

def _calc_indicators_htf(htf: pd.DataFrame) -> pd.DataFrame:
    out = htf.copy()
    out["ema20"]  = ta.ema(out["close"], 20)
    out["ema50"]  = ta.ema(out["close"], 50)
    out["ema200"] = ta.ema(out["close"], 200)
    return out

def _align_htf_to_h1(h1: pd.DataFrame, htf: pd.DataFrame, prefix: str) -> pd.DataFrame:
    aligned = htf.reindex(h1.index, method="ffill")
    aligned = aligned.add_prefix(f"{prefix}_")
    return pd.concat([h1, aligned], axis=1)

# ------------------------------------------------------------
# Levels, sessions, news/spread placeholders
# ------------------------------------------------------------

def _prev_day_levels(d1: pd.DataFrame) -> pd.DataFrame:
    df = d1.copy()
    df["PDH"] = df["high"].shift(1)
    df["PDL"] = df["low"].shift(1)
    df["PDC"] = df["close"].shift(1)
    return df[["PDH","PDL","PDC"]]

def _weekly_pivots(w1: pd.DataFrame) -> pd.DataFrame:
    df = w1.copy()
    H = df["high"].shift(1)
    L = df["low"].shift(1)
    C = df["close"].shift(1)
    PP = (H + L + C) / 3.0
    R1 = 2 * PP - L
    S1 = 2 * PP - H
    return pd.DataFrame({"PP": PP, "R1": R1, "S1": S1})

def _nearest_round(x: float, step: float) -> float:
    return round(x / step) * step

def _session_block(ts: pd.Timestamp, pair: str) -> str:
    """
    PEAK TRADING TIMES ONLY - Signals only appear during absolute peak hours for maximum probability.
    
    Convert UTC time to Chicago time (handles both CST and CDT automatically)
    then check if current Chicago hour is within peak range for each pair.
    """
    import pytz
    
    # Convert UTC timestamp to Chicago time (handles DST automatically)
    chicago_tz = pytz.timezone('America/Chicago')
    chicago_time = ts.tz_convert(chicago_tz)
    chicago_hour = chicago_time.hour
    
    pair = pair.upper()

    # ALL ULTRA-STABLE PAIRS: ACTIVE DURING ALL MAJOR SESSIONS FOR 24/5 COVERAGE
    # London (12AM-8AM), New York (6AM-4PM), Tokyo (6PM-6AM) Chicago time
    ultra_stable_pairs = {
        "EURGBP", "EURUSD", "EURCHF", "EURCAD", "GBPUSD", 
        "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
        "XAUUSD", "SPX500", "US30", "US100", "US500",
        "USDJPY", "GBPJPY", "EURJPY"
    }
    
    if pair in ultra_stable_pairs:
        # Active during overlapping major sessions for maximum signal opportunities:
        london_session = 0 <= chicago_hour <= 8      # London: 12AM-8AM 
        ny_session = 6 <= chicago_hour <= 16         # New York: 6AM-4PM
        tokyo_session = (18 <= chicago_hour <= 23) or (0 <= chicago_hour <= 6)  # Tokyo: 6PM-6AM
        
        return "ACTIVE" if (london_session or ny_session or tokyo_session) else "OTHER"

    # Non ultra-stable pairs get limited access 
    return "OTHER"


def _in_core_session(ts: pd.Timestamp, pair: str, cfg: Dict[str, Any] = None) -> bool:
    """
    Backward-compatible:
      - If cfg is omitted (old callers), look it up from PAIR_CONFIG.
      - Otherwise use the provided cfg.
    """
    if cfg is None:
        cfg, _ = _cfg_for(pair)
    if not cfg.get("use_session", True):
        return True
    return _session_block(ts, pair) == "ACTIVE"


def news_is_clean(ts: pd.Timestamp, pair: str) -> bool:
    # TODO: wire actual calendar if needed
    return True


def spread_is_ok(pair: str, ts: pd.Timestamp) -> bool:
    # TODO: pass broker spread in if available
    return True

def json_safe(obj):
    """Convert pandas/NumPy types in nested structures to JSON-safe primitives."""
    import pandas as pd
    import numpy as np
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is None:
        return None
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj  # str, float, int, bool are fine

# ------------------------------------------------------------
# Price-action confirmations & structure
# ------------------------------------------------------------

def _is_bull_engulf(row, prev) -> bool:
    return (row.close > row.open) and (prev.close < prev.open) and (row.close >= prev.open) and (row.open <= prev.close)

def _is_bear_engulf(row, prev) -> bool:
    return (row.close < row.open) and (prev.close > prev.open) and (row.close <= prev.open) and (row.open >= prev.close)

def _is_pin_bar(row, bullish=True, body_frac=0.33) -> bool:
    rng = row.high - row.low
    if rng <= 0:
        return False
    body = abs(row.close - row.open)
    upper_wick = row.high - max(row.close, row.open)
    lower_wick = min(row.close, row.open) - row.low
    if body / rng > body_frac:
        return False
    if bullish:
        return lower_wick > 2 * body and upper_wick < rng * 0.4
    else:
        return upper_wick > 2 * body and lower_wick < rng * 0.4

def _pivot_highs_lows(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    win = left + right + 1
    highs = df["high"].rolling(win, center=True).apply(
        lambda s: float(s.iloc[right] == s.max()), raw=False
    )
    lows = df["low"].rolling(win, center=True).apply(
        lambda s: float(s.iloc[right] == s.min()), raw=False
    )
    return pd.DataFrame({"ph": highs.eq(1.0), "pl": lows.eq(1.0)}, index=df.index)

def _last_swing_price(df: pd.DataFrame, idx: int, side: str, pivots: pd.DataFrame) -> Optional[float]:
    if side == "BUY":
        mask = pivots["pl"].iloc[: idx + 1][::-1]
        pos = mask.idxmax() if mask.any() else None
        return float(df.loc[pos, "low"]) if pos is not None else None
    else:
        mask = pivots["ph"].iloc[: idx + 1][::-1]
        pos = mask.idxmax() if mask.any() else None
        return float(df.loc[pos, "high"]) if pos is not None else None

def _detect_bos(h1: pd.DataFrame, piv: pd.DataFrame, lookback: int = 12) -> pd.Series:
    out = pd.Series(0, index=h1.index, dtype=int)
    last_low = last_high = None
    last_low_i = last_high_i = None
    for i in range(len(h1)):
        if bool(piv["pl"].iloc[i]):
            last_low = float(h1["low"].iloc[i]);  last_low_i = i
        if bool(piv["ph"].iloc[i]):
            last_high = float(h1["high"].iloc[i]); last_high_i = i
        if i == 0:
            continue
        close = float(h1["close"].iloc[i])
        if (last_high is not None) and (last_high_i is not None) and (i - last_high_i <= lookback):
            if close > last_high:
                out.iloc[i] = +1
        if (last_low is not None) and (last_low_i is not None) and (i - last_low_i <= lookback):
            if close < last_low:
                out.iloc[i] = -1
    return out

def _atr_core(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    return tr.rolling(n).mean()

def _detect_fvg_h1(h1: pd.DataFrame, atr_min_mult: float = FVG_ATR_MIN_MULT) -> pd.DataFrame:
    out = h1[["open","high","low","close"]].copy()
    a_hi = h1["high"].shift(2); a_lo = h1["low"].shift(2)
    b_hi = h1["high"].shift(1); b_lo = h1["low"].shift(1)
    c_hi = h1["high"];          c_lo = h1["low"]

    bull = (c_lo > a_hi)
    bear = (c_hi < a_lo)

    bull_lo, bull_hi = a_hi, b_lo
    bear_lo, bear_hi = b_hi, a_lo

    out["atr14"]       = _atr_core(h1, 14)
    out["fvg_bull_w"]  = (bull_hi - bull_lo).clip(lower=0)
    out["fvg_bear_w"]  = (bear_hi - bear_lo).clip(lower=0)
    out["fvg_bull_ok"] = bull & (out["fvg_bull_w"] >= atr_min_mult * out["atr14"])
    out["fvg_bear_ok"] = bear & (out["fvg_bear_w"] >= atr_min_mult * out["atr14"])
    out["fvg_bull_lo"], out["fvg_bull_hi"] = bull_lo, bull_hi
    out["fvg_bear_lo"], out["fvg_bear_hi"] = bear_lo, bear_hi
    return out

def _sr_levels_from_pivots(h1: pd.DataFrame,
                           left: int = 3, right: int = 3,
                           merge_tol_mult: float = SR_MERGE_TOL_ATR) -> list[dict]:
    A = _atr_core(h1, 14)
    piv = _pivot_highs_lows(h1, left, right)
    raw = []
    for i, is_ph in enumerate(piv["ph"].to_numpy()):
        if is_ph: raw.append(("R", float(h1["high"].iloc[i]), i))
    for i, is_pl in enumerate(piv["pl"].to_numpy()):
        if is_pl: raw.append(("S", float(h1["low"].iloc[i]), i))
    raw.sort(key=lambda x: x[1])

    merged: list[list] = []
    for t, lvl, idx in raw:
        if not merged:
            merged.append([t, lvl, [idx]])
            continue
        tol = merge_tol_mult * float(A.iloc[idx] if np.isfinite(A.iloc[idx]) else 0.0)
        if abs(lvl - merged[-1][1]) <= tol:
            merged[-1][1] = (merged[-1][1] + lvl) / 2.0
            merged[-1][2].append(idx)
        else:
            merged.append([t, lvl, [idx]])

    return [{"type": m[0], "level": float(m[1]), "touches": len(m[2])} for m in merged]

def _near_level(price: float, levels: list[dict], A: float, tol_mult: float = 0.15) -> Optional[dict]:
    if not np.isfinite(A) or A <= 0:
        return None
    for lv in levels:
        if abs(price - lv["level"]) <= tol_mult * A:
            return lv
    return None

def _equal_highs_lows(h1: pd.DataFrame, tol_mult: float = 0.05) -> tuple[pd.Series, pd.Series]:
    A = _atr_core(h1, 14)
    eq_hi = h1["high"].rolling(3).apply(
        lambda s: float((s.max() - s.min()) <= (A.loc[s.index[-1]] * tol_mult if s.index[-1] in A.index else 0.0)),
        raw=False
    ).fillna(0.0).astype(bool)
    eq_lo = h1["low"].rolling(3).apply(
        lambda s: float((s.max() - s.min()) <= (A.loc[s.index[-1]] * tol_mult if s.index[-1] in A.index else 0.0)),
        raw=False
    ).fillna(0.0).astype(bool)
    return eq_hi, eq_lo

def _liquidity_sweep_reclaim(curr, prev) -> tuple[bool, bool]:
    bull_sweep = (curr.high > prev.high) and (curr.close < prev.high)
    bear_sweep = (curr.low < prev.low) and (curr.close > prev.low)
    return bull_sweep, bear_sweep

# ------------------------------------------------------------
# Scoring
# ------------------------------------------------------------

@dataclass
class Score:
    total: int = 0
    reasons: List[str] = field(default_factory=list)
    def add(self, pts: int, reason: str):
        if pts != 0:
            self.total += pts
            self.reasons.append(f"{'+' if pts>0 else ''}{pts} {reason}")

# ------------------------------------------------------------
# Time-to-TP estimator for H1 trades
# ------------------------------------------------------------

def _efficiency_factor_for_time(row: pd.Series, prev: pd.Series, chosen_side: str) -> float:
    adx = float(row.get("adx", np.nan))
    ema20, ema50, ema200 = float(row["ema20"]), float(row["ema50"]), float(row["ema200"])
    macd_hist = float(row.get("macd_hist", np.nan))

    trend_up = row["close"] > ema200
    trend_dn = row["close"] < ema200
    fast_up  = ema20 > ema50
    fast_dn  = ema20 < ema50
    macd_up  = macd_hist > 0
    macd_dn  = macd_hist < 0
    adx_ok   = (adx >= ADX_FLOOR) if math.isfinite(adx) else False

    if chosen_side == "BUY":
        aligned = (trend_up and fast_up and macd_up)
    else:
        aligned = (trend_dn and fast_dn and macd_dn)

    if aligned and adx_ok:
        return EFF_TREND_GOOD
    if aligned:
        return EFF_TREND_OK
    if adx_ok:
        return EFF_TREND_WEAK
    return EFF_TREND_BAD

def _estimate_time_to_tp_h1(entry: float, target: float, atr_h1: float,
                            efficiency: float) -> Tuple[float, float]:
    if not (math.isfinite(entry) and math.isfinite(target) and math.isfinite(atr_h1) and atr_h1 > 0):
        return (np.nan, np.nan)
    distance = abs(target - entry)
    expected_bars = distance / max(1e-12, (efficiency * atr_h1))
    expected_bars = max(1.0, expected_bars)
    expected_hours = expected_bars * 1.0
    expected_hours = float(np.clip(expected_hours, MIN_TIMEOUT_HOURS, MAX_TIMEOUT_HOURS))
    expected_bars  = expected_hours
    return (float(expected_bars), float(expected_hours))

# ------------------------------------------------------------
# Main evaluation
# ------------------------------------------------------------

def evaluate_last_closed_bar(
    h1_df: pd.DataFrame,
    pair: Optional[str] = None,
    round_step: Optional[float] = None,
    rr_min: float = RR_MIN_DEFAULT,
    atr_stop_mult: float = DEFAULT_ATR_STOP_MULT,
    ema_overext_mult: float = EMA_OVEREXT_MULT,
    atr_percentile_win: int = ATR_PCTILE_WINDOW,
    debug: Optional[bool] = None,   # ← back-compat; ignored
    **kwargs,                       # ← swallow any other legacy args
) -> Dict[str, Any]:
    """
    Returns a structured dict:
      - ok: bool
      - is_skip: bool
      - skip_reason: Optional[str]
      - error: Optional[str]
      - version: str
      - plus all signal fields if ok and not skipped
    """
    def _ret_skip(reason: str, ctx_time: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
        nowh = (ctx_time or pd.Timestamp.now(tz="UTC")).floor("h")
        return {
            "ok": True,
            "is_skip": True,
            "skip_reason": str(reason),
            "error": None,
            "version": VERSION,
            "pair": (pair or "UNKNOWN").upper(),
            "time": nowh,
            "score": 0,
            "reasons": [],
            "context": {},
        }

    try:
        cfg, p = _cfg_for(pair)
        pip = _pip_size_for(p)
        rsi_lo, rsi_hi = cfg["rsi_band"]
        min_atr_pips = float(cfg["min_atr_pips"])
        min_score = int(cfg.get("min_score", 7))

        SPX_ALIASES = {"SPX500", "US500", "US500USD", "US500.CASH", "US500.CFD", "SPX", "SPX500USD"}
        is_spx = p.upper() in SPX_ALIASES

        df = h1_df.sort_index().copy()
        df = _ensure_utc_index(df)
        _is_dtindex(df)
        df = _calc_indicators_h1(df)

        if len(df) < 300:
            return _ret_skip("not enough candles (need >=300)")

        # HTF context: H4 & D1
        h4 = _resample_ohlc(df[["open","high","low","close"]], "4h")
        h4 = _calc_indicators_htf(h4)
        d1 = _resample_ohlc(df[["open", "high", "low", "close"]], "1D")
        d1 = _calc_indicators_htf(d1)

        # Levels
        d1_lvls = _prev_day_levels(d1)
        w1 = _resample_ohlc(df[["open", "high", "low", "close"]], "1W")
        weekly = _weekly_pivots(w1)

        # Align
        ctx = df.copy()
        ctx = _align_htf_to_h1(ctx, h4[["ema20", "ema50", "ema200", "close"]], prefix="H4")
        ctx = _align_htf_to_h1(ctx, d1[["ema20", "ema50", "ema200", "close"]], prefix="D1")
        ctx = pd.concat([ctx,
                        d1_lvls.reindex(ctx.index, method="ffill"),
                        weekly.reindex(ctx.index, method="ffill")], axis=1)

        # ATR regime
        ctx["atr_pctile"] = ctx["atr"].rolling(atr_percentile_win).rank(pct=True)

        now_utc = pd.Timestamp.now(tz="UTC").floor("h")
        closed_idx = ctx.index[ctx.index < now_utc]
        if len(closed_idx) < 2:
            return _ret_skip("not enough closed bars yet", now_utc)

        last_ts = closed_idx[-1]
        prev_ts = closed_idx[-2]
        row = ctx.loc[last_ts]
        prev = ctx.loc[prev_ts]
        i = ctx.index.get_loc(last_ts)
        ts = last_ts

        # Precision context (FVG + S/R clusters + Liquidity)
        if USE_FVG or USE_TOUCH_SR or USE_LIQUIDITY or USE_BOS:
            h1_slice = ctx[["open","high","low","close"]].iloc[: i + 1].copy()
        else:
            h1_slice = ctx[["open","high","low","close"]].iloc[: i + 1].copy()

        fvg_row = None
        if USE_FVG:
            _fvg_ctx = _detect_fvg_h1(h1_slice)
            fvg_row = _fvg_ctx.iloc[-1]

        sr_lvls = _sr_levels_from_pivots(h1_slice) if USE_TOUCH_SR else []

        liq_hi = liq_lo = False
        if USE_LIQUIDITY:
            _eq_hi, _eq_lo = _equal_highs_lows(h1_slice)
            liq_hi = bool(_eq_hi.iloc[-1])
            liq_lo = bool(_eq_lo.iloc[-1])

        piv = _pivot_highs_lows(ctx)
        bos_series = _detect_bos(ctx[["open","high","low","close"]], piv) if USE_BOS else pd.Series(0, index=ctx.index)
        bos_val = int(bos_series.loc[last_ts]) if USE_BOS else 0

        # Guards: ATR finite & floor
        if not math.isfinite(float(row["atr"])):
            return _ret_skip("ATR not finite", ts)

        if not is_spx:
            atr_units = float(row["atr"]) / float(pip)
            if atr_units + 1e-6 < float(min_atr_pips):
                return _ret_skip(f"ATR {atr_units:.1f} pips below floor {min_atr_pips:.1f}", ts)

        # Session & hygiene (check first - more important than ATR regime)
        if not _in_core_session(ts, p, cfg):
            return _ret_skip("outside active core session", ts)

        # ATR percentile health
        atr_pct = float(row.get("atr_pctile", np.nan))
        if math.isfinite(atr_pct):
            # Stable profile: very flexible ATR regime (almost no filtering)
            if STRICTNESS == "stable" and not ATR_REGIME_STRICT:
                lo, hi = (0.01, 0.99)  # Extremely flexible - almost no ATR filtering
            else:
                # conservative: prefer middle regimes
                lo, hi = (0.25, 0.85) if STRICTNESS == "conservative" else ((0.20, 0.90) if STRICTNESS=="normal" else (0.15, 0.95))
            if not (lo <= atr_pct <= hi):
                return _ret_skip("ATR regime not healthy", ts)
        if not news_is_clean(ts, p):
            return _ret_skip("blocked by news", ts)
        if not spread_is_ok(p, ts):
            return _ret_skip("spread not ok", ts)
        
        # Enhanced signal quality and market regime detection
        market_regime = _detect_market_regime(df.tail(100))
        signal_quality = _enhanced_signal_quality(df.tail(50))
        
        # Check spread conditions
        if not _check_spread_conditions(p, float(row["close"])):
            return _ret_skip("spread conditions unfavorable", ts)
        
        # Signal quality filter
        if signal_quality['quality_score'] < 0.3:
            return _ret_skip(f"signal quality too low: {signal_quality['quality_score']:.2f}", ts)
        
        # Market regime filters - avoid extreme volatility unless trending strongly
        if market_regime['vol_regime'] == 'high' and not market_regime['is_trending']:
            return _ret_skip("high volatility without clear trend", ts)
        
        # Adjust stop multiplier based on volatility regime
        dynamic_stop_mult = atr_stop_mult * VOL_ADJ_MULT.get(market_regime['vol_regime'], 1.5)

        # Scoring buckets
        long_score = Score(); short_score = Score()

        # 1) HTF alignment & slope (WEIGHTED FOR STABILITY)
        h4_up   = row["H4_close"] > row["H4_ema200"]
        h4_down = row["H4_close"] < row["H4_ema200"]
        d1_up   = row["D1_close"] > row["D1_ema200"]
        d1_down = row["D1_close"] < row["D1_ema200"]
        
        # HTF FRESHNESS VALIDATION: Prevent entries on stale trends
        # Check if H4/D1 trends changed recently to avoid exhausted moves
        def _htf_trend_age(timeframe_prefix: str, current_bullish: bool, max_age: int) -> int:
            """Returns bars since trend change, or 999 if trend is stale"""
            close_col = f"{timeframe_prefix}_close"
            ema_col = f"{timeframe_prefix}_ema200"
            
            age = 0
            lookback_start = max(0, i - max_age - 1)
            for k in range(i, lookback_start, -1):
                if k >= len(ctx):
                    continue
                prev_row = ctx.iloc[k-1] if k > 0 else None
                if prev_row is None:
                    continue
                    
                prev_close = float(prev_row[close_col])
                prev_ema = float(prev_row[ema_col])
                curr_close = float(ctx.iloc[k][close_col])
                curr_ema = float(ctx.iloc[k][ema_col])
                
                prev_bullish = prev_close > prev_ema
                curr_bullish_check = curr_close > curr_ema
                
                # Found trend change
                if prev_bullish != curr_bullish_check:
                    age = i - k
                    break
                    
            return age if age <= max_age else 999
        
        # Check HTF trend freshness (H4: max 8 bars = 32 hours, D1: max 3 bars = 3 days)
        h4_trend_age = _htf_trend_age("H4", h4_up or h4_down, 8)
        d1_trend_age = _htf_trend_age("D1", d1_up or d1_down, 3)
        
        # Apply freshness filter with confirmation window for high-confluence setups
        htf_fresh = True
        if STRICTNESS == "stable":
            # Reject stale trends (too old)
            if h4_trend_age >= 12:
                htf_fresh = False
                return _ret_skip(f"H4 trend stale (age: {h4_trend_age} bars)", ts)
            if d1_trend_age >= 5:
                htf_fresh = False 
                return _ret_skip(f"D1 trend stale (age: {d1_trend_age} bars)", ts)
            
            # Allow ultra-fresh trends with enhanced exit protection
            # (0-bar rejection removed - rely on improved position management instead)

        # Stable profile: increase weight for persistent multi-timeframe alignment
        h4_weight = 2 * TREND_STABILITY_WEIGHT if STRICTNESS == "stable" else 2
        d1_weight = 1 * TREND_STABILITY_WEIGHT if STRICTNESS == "stable" else 1
        slope_weight = 1 * TREND_STABILITY_WEIGHT if STRICTNESS == "stable" else 1
        
        # Ultra-stable pairs: Additional boost for enhanced confluence
        ultra_stable_pairs = ("EURGBP", "EURUSD", "EURCHF", "EURCAD", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "XAUUSD", "SPX500", "USDJPY", "GBPJPY", "EURJPY")
        if pair in ultra_stable_pairs and STRICTNESS == "stable":
            h4_weight += 1  # H4 trend gets extra +1 point
            d1_weight += 1  # D1 trend gets extra +1 point

        # Add freshness info to scoring descriptions
        h4_desc = f"H4 trend {'up' if h4_up else 'down'} (fresh: {h4_trend_age}bars)"
        d1_desc = f"D1 trend {'up' if d1_up else 'down'} (fresh: {d1_trend_age}bars)"
        
        long_score.add(h4_weight if h4_up else 0,    h4_desc if h4_up else "")
        long_score.add(d1_weight if d1_up else 0,    d1_desc if d1_up else "")
        short_score.add(h4_weight if h4_down else 0, h4_desc if h4_down else "")
        short_score.add(d1_weight if d1_down else 0, d1_desc if d1_down else "")

        h4_slope_up = _slope(h4["ema200"], lookback=H4_SLOPE_LOOKBACK) > 0
        h4_slope_dn = _slope(h4["ema200"], lookback=H4_SLOPE_LOOKBACK) < 0
        long_score.add(slope_weight if h4_slope_up else 0,  "H4 EMA200 slope up")
        short_score.add(slope_weight if h4_slope_dn else 0, "H4 EMA200 slope down")

        # 2) H1 trend baseline + fast alignment
        trend_up = row["close"] > row["ema200"]
        trend_dn = row["close"] < row["ema200"]
        fast_up = row["ema20"] > row["ema50"]
        fast_dn = row["ema20"] < row["ema50"]

        long_score.add(1 if trend_up else 0,  "H1 above EMA200")
        short_score.add(1 if trend_dn else 0, "H1 below EMA200")
        long_score.add(1 if fast_up else 0,   "EMA20 > EMA50")
        short_score.add(1 if fast_dn else 0,  "EMA20 < EMA50")

        # EMA20 proximity gate (no chase)
        dist_from_ema20 = abs(row["close"] - row["ema20"])
        prox_ok = dist_from_ema20 <= (EMA20_PROX_GATE_ATR * row["atr"]) if (trend_up or trend_dn) else True
        overextended = dist_from_ema20 > (ema_overext_mult * row["atr"])
        if prox_ok is False:
            return _ret_skip("entry too far from EMA20 (chase)", ts)
        else:
            # conservative: small penalty if stretched
            if overextended:
                long_score.add(-1,  f"Overextended from EMA20 (> {ema_overext_mult}*ATR)")
                short_score.add(-1, f"Overextended from EMA20 (> {ema_overext_mult}*ATR)")

        # 3) Momentum freshness (EMA cross + MACD cross age)
        cross_up = (prev["ema20"] <= prev["ema50"]) and (row["ema20"] > row["ema50"])
        cross_dn = (prev["ema20"] >= prev["ema50"]) and (row["ema20"] < row["ema50"])
        macd_up  = row["macd_hist"] > 0
        macd_dn  = row["macd_hist"] < 0

        start = max(0, i - MACD_CROSS_LOOKBACK)
        macd_segment = ctx["macd_hist"].iloc[start:i]
        recent_bull_cross = (macd_segment <= 0).any() and macd_up
        recent_bear_cross = (macd_segment >= 0).any() and macd_dn

        def _last_cross_age(sign: int) -> int:
            age = 999
            for k in range(i-1, max(i-20, 0), -1):
                v = float(ctx["macd_hist"].iloc[k])
                if (sign > 0 and v <= 0) or (sign < 0 and v >= 0):
                    age = i - k
                    break
            return age

        bull_age = _last_cross_age(+1)
        bear_age = _last_cross_age(-1)

        mom_up = recent_bull_cross and (bull_age <= MACD_CROSS_MAX_AGE)
        mom_dn = recent_bear_cross and (bear_age <= MACD_CROSS_MAX_AGE)

        # Stable profile: remove volatile momentum scoring (MOMENTUM_WEIGHT=0)
        momentum_weight = 2 * MOMENTUM_WEIGHT if STRICTNESS != "stable" else 0
        if momentum_weight > 0:
            long_score.add(momentum_weight if (mom_up or cross_up) else 0,  "Momentum up (fresh MACD or EMA20>EMA50 cross)")
            short_score.add(momentum_weight if (mom_dn or cross_dn) else 0, "Momentum down (fresh MACD or EMA20<EMA50 cross)")

        # 4) RSI mid-band bias (REDUCED WEIGHT FOR STABILITY)
        mid_zone    = (rsi_lo < float(row["rsi"]) < rsi_hi) and math.isfinite(float(prev["rsi"]))
        rsi_up_mid  = mid_zone and (row["rsi"] > prev["rsi"])
        rsi_dn_mid  = mid_zone and (row["rsi"] < prev["rsi"])
        rsi_bull    = (row["rsi"] >= 50) and (row["rsi"] >= prev["rsi"]) 
        rsi_bear    = (row["rsi"] <= 50) and (row["rsi"] <= prev["rsi"]) 

        # RSI scoring reduced for stability - only major RSI conditions get points
        rsi_weight = 0.3 if STRICTNESS == "stable" else 1.0
        rsi_strong_bull = (row["rsi"] > 60) and (row["rsi"] >= prev["rsi"])
        rsi_strong_bear = (row["rsi"] < 40) and (row["rsi"] <= prev["rsi"])
        
        if STRICTNESS == "stable":
            # Stable mode: Only award RSI points for strong, sustained conditions
            long_score.add(1 if rsi_strong_bull else 0, "RSI > 60 & sustained")
            short_score.add(1 if rsi_strong_bear else 0, "RSI < 40 & sustained")
        else:
            # Normal mode: Original RSI scoring
            long_score.add(1 if rsi_bull else 0,  "RSI >= 50 & rising")
            short_score.add(1 if rsi_bear else 0, "RSI <= 50 & falling")
            long_score.add(1 if rsi_up_mid else 0, f"RSI rising {rsi_lo}-{rsi_hi}")
            short_score.add(1 if rsi_dn_mid else 0, f"RSI falling {rsi_lo}-{rsi_hi}")

        # 5) Trend quality & efficiency
        adx_ok = (row.get("adx", 0) or 0) >= ADX_FLOOR
        long_score.add(1 if adx_ok else 0, "ADX strong")
        short_score.add(1 if adx_ok else 0, "ADX strong")

        rng = row["high"] - row["low"]
        eff = abs(row["close"] - row["open"]) / rng if rng > 0 else 0.0
        if eff < EFFICIENCY_MIN:
            return _ret_skip(f"Inefficient candle (eff<{EFFICIENCY_MIN})", ts)

        # 6) Price-action confirmation (REDUCED WEIGHT FOR STABILITY)
        bull_engulf = _is_bull_engulf(row, prev)
        bear_engulf = _is_bear_engulf(row, prev)
        bull_pin    = _is_pin_bar(row, bullish=True)
        bear_pin    = _is_pin_bar(row, bullish=False)
        
        # Reduce PA weight in stable mode to prevent hourly volatility
        pa_weight = 0.5 if STRICTNESS == "stable" else 1.0
        if STRICTNESS == "stable":
            # Only strong PA patterns get points in stable mode
            long_score.add(1 if (bull_engulf and bull_pin) else (0.5 if (bull_engulf or bull_pin) else 0), "Strong bullish PA")
            short_score.add(1 if (bear_engulf and bear_pin) else (0.5 if (bear_engulf or bear_pin) else 0), "Strong bearish PA")
        else:
            long_score.add(1 if (bull_engulf or bull_pin) else 0,  "Bullish PA (engulfing/pin)")
            short_score.add(1 if (bear_engulf or bear_pin) else 0, "Bearish PA (engulfing/pin)")

        # 7) Structure: BOS / CHoCH (REDUCED WEIGHT FOR STABILITY)
        if USE_BOS:
            bos_weight = STRUCTURE_WEIGHT  # stable=1, others=2
            if bos_val > 0:
                long_score.add(bos_weight, "Bullish BOS (close > last swing high)")
            elif bos_val < 0:
                short_score.add(bos_weight, "Bearish BOS (close < last swing low)")

        # Precision: FVG + Liquidity
        if USE_FVG and fvg_row is not None and math.isfinite(float(row["atr"])):
            A = float(row["atr"])
            price = float(row["close"])
            bull_ok = bool(fvg_row.get("fvg_bull_ok", False))
            bear_ok = bool(fvg_row.get("fvg_bear_ok", False))
            near_bull = bull_ok and abs(price - float(fvg_row["fvg_bull_hi"])) <= FVG_NEAR_TOL_ATR * A
            near_bear = bear_ok and abs(price - float(fvg_row["fvg_bear_lo"])) <= FVG_NEAR_TOL_ATR * A
            long_score.add(2 if near_bull else (1 if bull_ok else 0), "Bullish FVG (quality/proximity)")
            short_score.add(2 if near_bear else (1 if bear_ok else 0), "Bearish FVG (quality/proximity)")

        if USE_LIQUIDITY and USE_SWEEP_RECLAIM:
            bull_sweep, bear_sweep = _liquidity_sweep_reclaim(row, prev)
            long_score.add(1 if (liq_hi and bull_sweep) else 0,  "Sweep & reclaim above equal highs")
            short_score.add(1 if (liq_lo and bear_sweep) else 0, "Sweep & reclaim below equal lows")

        # Levels proximity
        lvl_names: List[str] = []
        lvl_vals:  List[float] = []
        for nm in ["PDH", "PDL", "PDC", "PP", "R1", "S1"]:
            if nm in row and pd.notna(row[nm]):
                lvl_names.append(nm)
                lvl_vals.append(float(row[nm]))

        if round_step is None:
            if p == "XAUUSD":
                step = 0.5
            elif p == "SPX500":
                step = 2.0
            elif p.endswith("JPY"):
                step = 0.25
            else:
                step = 0.0025
        else:
            step = float(round_step)

        lvl_names.append("ROUND")
        lvl_vals.append(_nearest_round(float(row["close"]), step))
        lvl_dists = [abs(float(row["close"]) - v) for v in lvl_vals]
        nearest_lvl_name = lvl_names[int(np.argmin(lvl_dists))]
        nearest_lvl_dist = min(lvl_dists)

        # Side selection with tie-break
        if long_score.total > short_score.total:
            S = long_score; choose_side = "BUY"
        elif short_score.total > long_score.total:
            S = short_score; choose_side = "SELL"
        else:
            if h4_slope_up and not h4_slope_dn:
                S = long_score; choose_side = "BUY"
            elif h4_slope_dn and not h4_slope_up:
                S = short_score; choose_side = "SELL"
            else:
                if trend_up and not trend_dn:
                    S = long_score; choose_side = "BUY"
                elif trend_dn and not trend_up:
                    S = short_score; choose_side = "SELL"
                else:
                    return _ret_skip("tie with no HTF/H1 edge", ts)

        # STABLE PROFILE: Add stability buffer to prevent dramatic score drops
        if STRICTNESS == "stable":
            # Apply minimum score floor based on multi-timeframe alignment
            stable_floor = 0
            if choose_side == "BUY":
                # Ultra-stable pairs get enhanced weighting
                ultra_stable_pairs = ("EURGBP", "EURUSD", "EURCHF", "EURCAD", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "XAUUSD", "SPX500", "USDJPY", "GBPJPY", "EURJPY")
                is_ultra_stable = pair in ultra_stable_pairs
                h4_stable_weight = 7 if is_ultra_stable else 6  # Ultra-stable pairs get +1 bonus
                d1_stable_weight = 4 if is_ultra_stable else 3  # Ultra-stable pairs get +1 bonus
                if h4_up: stable_floor += h4_stable_weight  # H4 trend worth more for ultra-stable pairs
                if d1_up: stable_floor += d1_stable_weight  # D1 trend worth more for ultra-stable pairs
                if h4_slope_up: stable_floor += 3  # H4 slope worth 3 in stable (was 1*3)
            else:
                # Ultra-stable pairs get enhanced weighting
                ultra_stable_pairs = ("EURGBP", "EURUSD", "EURCHF", "EURCAD", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "XAUUSD", "SPX500", "USDJPY", "GBPJPY", "EURJPY")
                is_ultra_stable = pair in ultra_stable_pairs
                h4_stable_weight = 7 if is_ultra_stable else 6  # Ultra-stable pairs get +1 bonus
                d1_stable_weight = 4 if is_ultra_stable else 3  # Ultra-stable pairs get +1 bonus
                if h4_down: stable_floor += h4_stable_weight  # H4 trend worth more for ultra-stable pairs
                if d1_down: stable_floor += d1_stable_weight  # D1 trend worth more for ultra-stable pairs
                if h4_slope_dn: stable_floor += 3  # H4 slope worth 3 in stable
            
            # Ensure score doesn't drop below 85% of trend-based floor (stronger protection)
            min_stable_score = int(stable_floor * 0.85) 
            # Never let score drop below 5 if we have any trend alignment
            if stable_floor > 0:
                min_stable_score = max(min_stable_score, 5)
                if S.total < min_stable_score:
                    S.total = min_stable_score
                S.reasons.append(f"Stable floor protection: {min_stable_score}")
            
            # Additional protection: Limit hourly score drops to max -2 points
            # This prevents dramatic -5 drops while allowing gradual deterioration
            if S.total >= 12:  # Only apply to high-quality setups
                # For existing good setups, prevent drops > 2 points per hour
                # This mimics real day trading where good H4/D1 setups stay valid
                structural_score = 0
                if choose_side == "BUY":
                    if h4_up: structural_score += h4_stable_weight
                    if d1_up: structural_score += d1_stable_weight
                    if trend_up: structural_score += 1  # H1 alignment
                else:
                    if h4_down: structural_score += h4_stable_weight  
                    if d1_down: structural_score += d1_stable_weight
                    if trend_dn: structural_score += 1  # H1 alignment
                
                # If structure is still intact (score >= 10), maintain minimum score
                if structural_score >= 10:
                    min_structural_score = max(12, int(structural_score * 0.9))
                    if S.total < min_structural_score:
                        S.total = min_structural_score
                        S.reasons.append(f"Structural integrity maintained: {min_structural_score}")

        # Guard: min score (profile-aware)
        if S.total < min_score:
            return _ret_skip(f"score {S.total} < min_score {min_score}", ts)

        # Entry-level obstacle checks
        entry = float(row["close"])
        A = float(row["atr"])
        min_buffer = LEVEL_MIN_BUFFER_ATR * A

        if choose_side == "BUY":
            ceiling = min([v for v in lvl_vals if v > entry], default=np.nan)
            if math.isfinite(ceiling) and (ceiling - entry) < min_buffer:
                return _ret_skip("too close to overhead level", ts)
            strong_R = [lv for lv in sr_lvls if lv["type"] == "R" and lv["touches"] >= STRONG_TOUCHES_MIN] if USE_TOUCH_SR else []
            near_strong_R = _near_level(entry, strong_R, A, tol_mult=0.10) if USE_TOUCH_SR else None
            if near_strong_R:
                return _ret_skip("capped by strong resistance cluster", ts)
        else:
            floor_  = max([v for v in lvl_vals if v < entry], default=np.nan)
            if math.isfinite(floor_) and (entry - floor_) < min_buffer:
                return _ret_skip("too close to floor level", ts)
            strong_S = [lv for lv in sr_lvls if lv["type"] == "S" and lv["touches"] >= STRONG_TOUCHES_MIN] if USE_TOUCH_SR else []
            near_strong_S = _near_level(entry, strong_S, A, tol_mult=0.10) if USE_TOUCH_SR else None
            if near_strong_S:
                return _ret_skip("propped by strong support cluster", ts)

        # Stop using swing + ATR
        swing = _last_swing_price(ctx, i, choose_side, piv)
        if swing is None:
            return _ret_skip("no recent swing to anchor stop", ts)

        atr_stop = A * float(atr_stop_mult)
        if choose_side == "BUY":
            stop_price = min(swing, entry - atr_stop)
        else:
            stop_price = max(swing, entry + atr_stop)
            
        # Validate stop price makes sense
        if choose_side == "BUY" and stop_price >= entry:
            return _ret_skip("invalid stop price (stop >= entry for BUY)", ts)
        if choose_side == "SELL" and stop_price <= entry:
            return _ret_skip("invalid stop price (stop <= entry for SELL)", ts)

        # Targeting (adaptive with conservative fallback)
        if USE_ADAPTIVE_TP:
            def _target_candidates_buy() -> list[float]:
                cands: list[float] = []
                if USE_TOUCH_SR:
                    cands += sorted([lv["level"] for lv in sr_lvls if lv["type"] == "R" and lv["level"] > entry and lv["touches"] >= STRONG_TOUCHES_MIN])
                ups = sorted([v for v in lvl_vals if v > entry])
                cands += [v for v in ups if v not in cands]
                if "PP" in row and float(row["PP"]) > entry and float(row["PP"]) not in cands:
                    cands.append(float(row["PP"]))
                return cands
            def _target_candidates_sell() -> list[float]:
                cands: list[float] = []
                if USE_TOUCH_SR:
                    cands += sorted([lv["level"] for lv in sr_lvls if lv["type"] == "S" and lv["level"] < entry and lv["touches"] >= STRONG_TOUCHES_MIN], reverse=True)
                dns = sorted([v for v in lvl_vals if v < entry], reverse=True)
                cands += [v for v in dns if v not in cands]
                if "PP" in row and float(row["PP"]) < entry and float(row["PP"]) not in cands:
                    cands.append(float(row["PP"]))
                return cands
            def _choose_target(side: str) -> float:
                if side == "BUY":
                    cands = _target_candidates_buy(); fallback = entry + 3.5 * A  # Better R/R targets
                    for t in cands:
                        try:
                            risk = entry - stop_price; reward = t - entry
                            if risk > 0 and reward > 0 and (reward / risk) >= rr_min:
                                return t
                        except (ZeroDivisionError, ValueError, TypeError):
                            continue  # Skip invalid target candidates
                    return fallback
                else:
                    cands = _target_candidates_sell(); fallback = entry - 3.5 * A  # Better R/R targets
                    for t in cands:
                        try:
                            risk = stop_price - entry; reward = entry - t
                            if risk > 0 and reward > 0 and (reward / risk) >= rr_min:
                                return t
                        except (ZeroDivisionError, ValueError, TypeError):
                            continue  # Skip invalid target candidates
                    return fallback
            target = _choose_target(choose_side)
        else:
            target = entry + (3.5 * A) if choose_side == "BUY" else entry - (3.5 * A)  # Better R/R targets
            
        # Validate target price makes sense
        if choose_side == "BUY" and target <= entry:
            return _ret_skip("invalid target price (target <= entry for BUY)", ts)
        if choose_side == "SELL" and target >= entry:
            return _ret_skip("invalid target price (target >= entry for SELL)", ts)

        # R:R computation
        if choose_side == "BUY":
            risk = entry - stop_price; reward = target - entry
        else:
            risk = stop_price - entry; reward = entry - target

        if risk <= 0 or reward <= 0:
            return _ret_skip("invalid risk/reward geometry", ts)

        rr = reward / risk
        
        # DISABLED: Allow lower R:R for all scores to enable Asian session trading
        # chosen_score = long_score.total if choose_side == "BUY" else short_score.total
        # if chosen_score >= 15:
        #     rr_min = max(rr_min, 2.0)  # Ultra-high confluence demands 2:1+ R/R
        # elif chosen_score >= 12:
        #     rr_min = max(rr_min, 2.8)  # High confluence demands 2.8:1+ R/R
        
        if rr < rr_min:
            # conservative: fail if RR not met
            chosen_score = long_score.total if choose_side == "BUY" else short_score.total
            return _ret_skip(f"rr {rr:.2f} < {rr_min} (score {chosen_score})", ts)

        # Time-stop estimation
        eff_time = _efficiency_factor_for_time(row, prev, choose_side)
        exp_bars, exp_hours = _estimate_time_to_tp_h1(entry, target, A, eff_time)
        deadline = (ts + pd.Timedelta(hours=exp_hours)) if math.isfinite(exp_hours) else None

        out = {
            "ok": True,
            "is_skip": False,
            "skip_reason": None,
            "error": None,
            "version": VERSION,
            "profile": STRICTNESS,
            "pair": p,
            "time": ts,
            "side": choose_side,
            "score": int(S.total),
            "reasons": S.reasons,
            "price": float(entry),
            "stop": float(stop_price),
            "target": float(target),
            "rr": float(round(rr, 2)),
            "atr": float(A),
            "signal_quality": float(round(signal_quality['quality_score'], 3)),
            "nearest_level": {"name": nearest_lvl_name, "distance": float(nearest_lvl_dist)},
            "context": {
                "h4_trend": "UP" if h4_up else ("DOWN" if h4_down else "FLAT"),
                "d1_trend": "UP" if d1_up else ("DOWN" if d1_down else "FLAT"),
                "session": _session_block(ts, p),
                "atr_pctile": float(round(atr_pct, 2)) if math.isfinite(atr_pct) else None,
            },
            "time_stop": {
                "expected_bars": float(round(exp_bars, 2)) if math.isfinite(exp_bars) else None,
                "expected_hours": float(round(exp_hours, 2)) if math.isfinite(exp_hours) else None,
                "deadline_utc": deadline,
                "efficiency": eff_time,
                "notes": "Close if TP not hit by deadline (or tighten to BE+trail).",
            },
            "market_regime": {
                "trend_strength": float(round(market_regime['trend_strength'], 2)),
                "is_trending": market_regime['is_trending'],
                "vol_regime": market_regime['vol_regime'],
                "structure": market_regime['structure'],
                "atr_percentile": float(round(market_regime['atr_percentile'], 2)),
            },
            "quality_details": {
                "volume_ratio": float(round(signal_quality['volume_ratio'], 2)),
                "momentum_score": float(round(signal_quality['momentum_score'], 2)),
                "wick_ratio": float(round(signal_quality['wick_ratio'], 2)),
            },
        }
        return out

    except Exception as e:
        # Never 500: surface the error in a structured payload
        import traceback
        error_details = f"{type(e).__name__}: {str(e)}"
        print(f"[CONFLUENCE ERROR] {pair}: {error_details}")
        print(f"[CONFLUENCE TRACEBACK] {traceback.format_exc()}")
        return {
            "ok": False,
            "is_skip": True,
            "skip_reason": "internal_error",
            "error": error_details,
            "version": VERSION,
            "pair": (pair or "UNKNOWN").upper(),
            "time": pd.Timestamp.now(tz="UTC").floor("h"),
            "score": 0,
            "reasons": [],
            "context": {},
        }

# ------------------------------------------------------------
# Research helpers
# ------------------------------------------------------------

def build_feature_row(ctx_row: pd.Series, signal: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = {
        "time": ctx_row.name,
        "close": float(ctx_row["close"]),
        "ema20": float(ctx_row["ema20"]),
        "ema50": float(ctx_row["ema50"]),
        "ema200": float(ctx_row["ema200"]),
        "rsi": float(ctx_row["rsi"]),
        "macd_hist": float(ctx_row["macd_hist"]),
        "atr": float(ctx_row["atr"]),
        "adx": float(ctx_row.get("adx", np.nan)),
        "H4_close": float(ctx_row.get("H4_close", np.nan)),
        "H4_ema200": float(ctx_row.get("H4_ema200", np.nan)),
        "D1_close": float(ctx_row.get("D1_close", np.nan)),
        "D1_ema200": float(ctx_row.get("D1_ema200", np.nan)),
    }
    if signal and signal.get("ok") and not signal.get("is_skip"):
        base.update({
            "signal_side": signal["side"],
            "signal_score": signal["score"],
            "signal_stop": signal["stop"],
            "signal_target": signal["target"],
            "signal_rr": signal["rr"],
            "signal_expected_hours": (signal.get("time_stop", {}) or {}).get("expected_hours"),
            "signal_efficiency": (signal.get("time_stop", {}) or {}).get("efficiency"),
        })
    return base

def label_outcome(
    df: pd.DataFrame,
    entry_ts: pd.Timestamp,
    side: str,
    stop: float,
    target: float,
    horizon_hours: int = 24,
    timeout_hours: Optional[float] = None,
) -> str:
    _is_dtindex(df)
    if entry_ts not in df.index:
        entry_ts = df.index[df.index.searchsorted(entry_ts)]
    start_idx = df.index.get_loc(entry_ts)
    end_idx = min(start_idx + horizon_hours, len(df) - 1)
    window = df.iloc[start_idx:end_idx + 1]

    if side == "BUY":
        hit_target = (window["high"] >= target).any()
        hit_stop   = (window["low"]  <= stop).any()
    else:
        hit_target = (window["low"]  <= target).any()
        hit_stop   = (window["high"] >= stop).any()

    if hit_target and hit_stop:
        t1 = window.index[(window["high"] >= target) if side == "BUY" else (window["low"] <= target)][0]
        t2 = window.index[(window["low"]  <= stop)   if side == "BUY" else (window["high"] >= stop)][0]
        return "WIN_FIRST" if t1 <= t2 else "LOSS_FIRST"
    if hit_target:
        return "WIN"
    if hit_stop:
        return "LOSS"

    if timeout_hours is not None and timeout_hours > 0:
        timeout_idx = min(start_idx + int(timeout_hours), len(df) - 1)
        sub = df.iloc[start_idx:timeout_idx + 1]
        if side == "BUY":
            to_tp = (sub["high"] >= target).any()
            to_sl = (sub["low"]  <= stop).any()
        else:
            to_tp = (sub["low"]  <= target).any()
            to_sl = (sub["high"] >= stop).any()
        if not to_tp and not to_sl:
            return "TIMEOUT"

    return "NONE"
