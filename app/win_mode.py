# backend/app/win_mode.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import math
import numpy as np
import pandas as pd

# reuse internals from confluence
from .confluence import (
    _cfg_for, _pip_size_for, _ensure_utc_index, _is_dtindex,
    _calc_indicators_h1, _resample_ohlc, _calc_indicators_htf, _align_htf_to_h1,
    _prev_day_levels, _weekly_pivots, _session_block, _in_core_session,
    _pivot_highs_lows, _last_swing_price, _is_bull_engulf, _is_bear_engulf, _is_pin_bar,
    LEVEL_MIN_BUFFER_ATR, ATR_PCTILE_WINDOW, H4_SLOPE_LOOKBACK, EMA_OVEREXT_MULT,
    PAIR_CONFIG,
)

# -------- Win Mode knobs (can move to PAIR_CONFIG later) ----------
WIN_RR_MIN                = 0.9
WIN_MAX_TIMEOUT_HOURS     = 6.0
WIN_LEVEL_MIN_BUFFER_ATR  = 1.0
WIN_ADX_FLOOR             = 22
WIN_ADX_RISE_BARS         = 3
WIN_PULLBACK_EMA          = "ema20"
WIN_PULLBACK_TOL_ATR      = 0.25
WIN_TRIGGER_PA            = True
WIN_RSI_BAND              = (45, 65)
WIN_ATR_PCT_OK_BAND       = (0.20, 0.85)
WIN_MIN_ACTIVE_FRAC       = 0.5

SPX_ALIASES = {"SPX500","US500","US500USD","US500.CASH","US500.CFD","SPX","SPX500USD"}

def _ols_slope(y: pd.Series) -> float:
    n = len(y)
    if n < 2: return 0.0
    x = np.arange(n)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm)**2).sum()
    if denom == 0: return 0.0
    return float(((x - xm)*(y - ym)).sum() / denom)

def _is_adx_rising(ctx: pd.DataFrame, i: int, bars: int) -> bool:
    if "adx" not in ctx.columns or i < bars+1:
        return False
    seg = ctx["adx"].iloc[i-bars:i]
    return bool(seg.diff().sum() > 0)

def _near_ema_pullback(row: pd.Series, ema_col: str, tol_atr_mult: float) -> bool:
    ema = float(row[ema_col]); price = float(row["close"]); A = float(row["atr"])
    if not (math.isfinite(ema) and math.isfinite(A) and A > 0):
        return False
    return abs(price - ema) <= tol_atr_mult * A

def _trigger_price_action(row, prev, side: str) -> bool:
    return (_is_bull_engulf(row, prev) or _is_pin_bar(row, bullish=True)) if side == "BUY" \
        else (_is_bear_engulf(row, prev) or _is_pin_bar(row, bullish=False))

def _build_ctx(h1_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = h1_df.sort_index().copy()
    df = _ensure_utc_index(df); _is_dtindex(df)
    df = _calc_indicators_h1(df)
    h4 = _resample_ohlc(df[["open","high","low","close"]], "4h")
    h4 = _calc_indicators_htf(h4)
    d1 = _resample_ohlc(df[["open","high","low","close"]], "1D")
    d1 = _calc_indicators_htf(d1)
    d1_lvls = _prev_day_levels(d1)
    w1 = _resample_ohlc(df[["open","high","low","close"]], "1W")
    weekly = _weekly_pivots(w1)

    ctx = df.copy()
    ctx = _align_htf_to_h1(ctx, h4[["ema20","ema50","ema200","close"]], prefix="H4")
    ctx = _align_htf_to_h1(ctx, d1[["ema20","ema50","ema200","close"]], prefix="D1")
    ctx = pd.concat([ctx, d1_lvls.reindex(ctx.index, method="ffill"), weekly.reindex(ctx.index, method="ffill")], axis=1)
    ctx["atr_pctile"] = ctx["atr"].rolling(ATR_PCTILE_WINDOW).rank(pct=True)
    return ctx, df, h4, d1, weekly

def evaluate_last_closed_bar_win(h1_df: pd.DataFrame, pair: Optional[str] = None, debug: bool = False) -> Optional[Dict[str, Any]]:
    """
    Win-mode evaluator: very strict, aims for very high hit-rate (lower avg R).
    """
    cfg, p = _cfg_for(pair)
    pip = _pip_size_for(p)
    rsi_lo, rsi_hi = cfg["rsi_band"]
    min_atr_pips = float(cfg["min_atr_pips"])
    min_score = int(cfg.get("min_score", 5))
    is_spx = p.upper() in SPX_ALIASES

    def _skip(reason: str):
        return {"pair": (pair or "UNKNOWN").upper(), "skip_reason": reason,
                "time_checked_utc": pd.Timestamp.now(tz="UTC").floor("h")} if debug else None

    ctx, df, h4, d1, weekly = _build_ctx(h1_df)
    if len(ctx) < 300:
        return _skip("not enough candles (need >=300)")

    nowh = pd.Timestamp.now(tz="UTC").floor("h")
    closed_idx = ctx.index[ctx.index < nowh]
    if len(closed_idx) < 2:
        return _skip("not enough closed bars yet")

    i = len(closed_idx) - 1
    ts = closed_idx[-1]
    prev_ts = closed_idx[-2]
    row = ctx.loc[ts]; prev = ctx.loc[prev_ts]

    # ATR floor parity (skip SPX)
    if not math.isfinite(float(row["atr"])):
        return _skip("ATR not finite")
    if not is_spx:
        atr_units = float(row["atr"]) / float(pip)
        atr_units_rounded = round(atr_units, 1)
        if atr_units_rounded + 1e-6 < min_atr_pips:
            return _skip(f"ATR {atr_units_rounded:.1f} pips below floor {min_atr_pips:.1f}")

    # Session
    if not _in_core_session(ts, p):
        return _skip("outside active session")

    # Trend alignment (H4 & D1 & H1)
    h4_up = row["H4_close"] > row["H4_ema200"]; h4_down = row["H4_close"] < row["H4_ema200"]
    d1_up = row["D1_close"] > row["D1_ema200"]; d1_down = row["D1_close"] < row["D1_ema200"]
    trend_up = row["close"] > row["ema200"];    trend_dn = row["close"] < row["ema200"]
    fast_up = row["ema20"] > row["ema50"];      fast_dn = row["ema20"] < row["ema50"]

    # choose side deterministically by HTF trend
    side = "BUY" if (h4_up and d1_up and trend_up and fast_up) else ("SELL" if (h4_down and d1_down and trend_dn and fast_dn) else None)
    if side is None:
        return _skip("win-mode: trend alignment not perfect")

    # Momentum strengthening
    adx_ok = (row.get("adx", 0) or 0) >= WIN_ADX_FLOOR
    if not adx_ok:
        return _skip("win-mode: ADX below floor")

    if not _is_adx_rising(ctx, i, WIN_ADX_RISE_BARS):
        return _skip("win-mode: ADX not rising")

    # Pullback to value (EMA20) + trigger PA
    if not _near_ema_pullback(row, WIN_PULLBACK_EMA, WIN_PULLBACK_TOL_ATR):
        return _skip("win-mode: no pullback to EMA20")

    if WIN_TRIGGER_PA and not _trigger_price_action(row, prev, side):
        return _skip("win-mode: no PA trigger at pullback")

    # RSI corridor
    rlo, rhi = WIN_RSI_BAND
    r = float(row["rsi"]); pr = float(prev["rsi"])
    if side == "BUY":
        if not (rlo <= r <= rhi and r >= pr):
            return _skip("win-mode: RSI not in corridor/rising")
    else:
        if not (rlo <= r <= rhi and r <= pr):
            return _skip("win-mode: RSI not in corridor/falling")

    # Regime band
    atr_pct = float(row.get("atr_pctile", np.nan))
    if math.isfinite(atr_pct):
        if not (WIN_ATR_PCT_OK_BAND[0] <= atr_pct <= WIN_ATR_PCT_OK_BAND[1]):
            return _skip("win-mode: ATR regime not healthy")

    # Level proximity (ROUND + daily/weekly)
    lvl_vals: List[float] = []
    for nm in ["PDH","PDL","PDC","PP","R1","S1"]:
        if nm in row and pd.notna(row[nm]):
            lvl_vals.append(float(row[nm]))
    # round step (reuse confluence heuristics)
    if p == "XAUUSD": step = 0.5
    elif p == "SPX500": step = 2.0
    elif p.endswith("JPY"): step = 0.25
    else: step = 0.0025
    lvl_vals.append(round(float(row["close"]) / step) * step)

    entry = float(row["close"])
    min_buffer = WIN_LEVEL_MIN_BUFFER_ATR * float(row["atr"])
    if side == "BUY":
        ceiling = min([v for v in lvl_vals if v > entry], default=np.nan)
        if math.isfinite(ceiling) and (ceiling - entry) < min_buffer:
            return _skip("win-mode: too close to overhead level")
    else:
        floor_ = max([v for v in lvl_vals if v < entry], default=np.nan)
        if math.isfinite(floor_) and (entry - floor_) < min_buffer:
            return _skip("win-mode: too close to floor level")

    # Stop using last swing vs ATR
    piv = _pivot_highs_lows(ctx)
    swing = _last_swing_price(ctx, i, side, piv)
    if swing is None:
        return _skip("win-mode: no swing for stop")
    atr_stop = 1.2 * float(row["atr"])  # same as confluence default
    stop = min(swing, entry - atr_stop) if side == "BUY" else max(swing, entry + atr_stop)

    # Conservative target: nearest qualified level achieving RR >= WIN_RR_MIN, else ATR fallback
    if side == "BUY":
        ups = sorted([v for v in lvl_vals if v > entry])
        target = next((t for t in ups if (t - entry) / max(1e-12, entry - stop) >= WIN_RR_MIN), entry + 1.5 * float(row["atr"]))
        risk, reward = (entry - stop), (target - entry)
    else:
        dns = sorted([v for v in lvl_vals if v < entry], reverse=True)
        target = next((t for t in dns if (entry - t) / max(1e-12, stop - entry) >= WIN_RR_MIN), entry - 1.5 * float(row["atr"]))
        risk, reward = (stop - entry), (entry - target)

    if risk <= 0 or reward <= 0:
        return _skip("win-mode: invalid R geometry")

    rr = reward / risk
    if rr < WIN_RR_MIN:
        return _skip("win-mode: RR below minimum")

    # ETA (simple): distance / (ADX-adjusted ATR). Enforce session alignment.
    eff = 0.8 if (row.get("adx", 0) or 0) >= 25 else 0.7
    distance = abs(target - entry)
    exp_hours = distance / max(1e-12, eff * float(row["atr"]))
    if not math.isfinite(exp_hours) or exp_hours > WIN_MAX_TIMEOUT_HOURS:
        return _skip("win-mode: ETA too long")

    deadline = ts + pd.Timedelta(hours=exp_hours)
    hours_path = pd.date_range(ts, deadline, freq="1H", tz="UTC")
    active_frac = float(np.mean([_in_core_session(h, p) for h in hours_path])) if len(hours_path) else 1.0
    if active_frac < WIN_MIN_ACTIVE_FRAC:
        return _skip("win-mode: ETA mostly outside active session")

    # Output
    return {
        "pair": p,
        "time": ts,
        "side": side,
        "price": float(entry),
        "stop": float(stop),
        "target": float(target),
        "rr": float(round(rr, 2)),
        "atr": float(row["atr"]),
        "context": {
            "h4_trend": "UP" if h4_up else ("DOWN" if h4_down else "FLAT"),
            "d1_trend": "UP" if d1_up else ("DOWN" if d1_down else "FLAT"),
            "session": _session_block(ts, p),
            "atr_pctile": float(round(atr_pct, 2)) if math.isfinite(atr_pct) else None,
        },
        "exit_policy": {
            "win_mode": True,
            "tp1_r_multiple": 0.8,
            "tp1_take_pct": 0.5,
            "move_to_be_after_tp1": True,
            "trail_remainder_atr_mult": 1.0,
        },
        "skip_reason": None,
    }

def scan_win_mode(pair: str, h1_df: pd.DataFrame, hours: int = 48) -> List[Dict[str, Any]]:
    """
    Hourly pass/skip list for Win Mode (for the UI list view).
    """
    ctx, df, h4, d1, weekly = _build_ctx(h1_df)
    if len(ctx) < 300:
        return [{"time": None, "pair": pair.upper(), "skip_reason": "not enough candles (need >=300)"}]
    nowh = pd.Timestamp.now(tz="UTC").floor("h")
    closed_idx = ctx.index[ctx.index < nowh]
    closed_idx = closed_idx[-max(2, hours):]
    out: List[Dict[str, Any]] = []
    for ts in closed_idx[1:]:
        try:
            res = evaluate_last_closed_bar_win(ctx.loc[:ts], pair=pair, debug=True)
            if res and not res.get("skip_reason"):
                out.append({"time": ts.isoformat(), "pair": pair.upper(), "side": res["side"], "rr": res["rr"]})
            else:
                out.append({"time": ts.isoformat(), "pair": pair.upper(), "skip_reason": res["skip_reason"] if res else "no signal"})
        except Exception as e:
            out.append({"time": ts.isoformat(), "pair": pair.upper(), "skip_reason": f"error: {e}"})
    return out
