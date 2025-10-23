# backend/app/debug_scan.py
from __future__ import annotations
from typing import List, Dict, Any
import os
import math
import pandas as pd

# Reuse internals from your evaluator
from .confluence import (
    PAIR_CONFIG,
    _pip_size_for,
    _ensure_utc_index,
    _is_dtindex,
    _calc_indicators_h1,
    _resample_ohlc,
    _calc_indicators_htf,
    _align_htf_to_h1,
    _prev_day_levels,
    _weekly_pivots,
    _session_block,
    _in_core_session,
    LEVEL_MIN_BUFFER_ATR,
    H4_SLOPE_LOOKBACK,
    ATR_PCTILE_WINDOW,
    EMA_OVEREXT_MULT,
    RR_MIN_DEFAULT,
)

SPX_ALIASES = {"SPX500", "US500", "US500USD", "US500.CASH", "US500.CFD", "SPX", "SPX500USD"}


def _csv_path(pair: str, timeframe: str = "H1") -> str:
    return f"data/{pair}_{timeframe}.csv"


def _load_h1(pair: str) -> pd.DataFrame:
    path = _csv_path(pair, "H1")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open", "high", "low", "close"])


def _slope(series: pd.Series, lookback: int = 5) -> float:
    if series is None or len(series) < max(2, lookback):
        return 0.0
    a, b = series.iloc[-lookback], series.iloc[-1]
    return float(b - a) / float(lookback)


def _nearest_round(x: float, step: float) -> float:
    return round(x / step) * step


def _hourly_skip_reasons(pair: str, hours: int = 48) -> List[Dict[str, Any]]:
    """
    Scan the last `hours` CLOSED H1 bars and, per hour:
      - return a trade stub if it WOULD pass (side + pre-RR score)
      - otherwise return a `skip_reason`.

    Notes:
      * This mirrors early/guard + skeleton scoring in confluence.py.
      * It matches confluence's ATR floor behavior (SPX exception + rounded ATR units).
      * This path reports a *pre-RR* score to compare with confluence's `score_before_rr`.
    """
    pair = pair.upper()
    cfg = PAIR_CONFIG.get(pair) or PAIR_CONFIG["EURUSD"]
    pip = _pip_size_for(pair)
    min_atr_pips = float(cfg["min_atr_pips"])
    min_score = int(cfg.get("min_score", 5))
    is_spx = pair in SPX_ALIASES

    df = _load_h1(pair)
    if df.empty:
        return [{"time": None, "pair": pair, "skip_reason": "no data"}]

    df = _ensure_utc_index(df)
    _is_dtindex(df)
    df = _calc_indicators_h1(df)

    if len(df) < 300:
        return [{"time": None, "pair": pair, "skip_reason": "not enough candles (need >=300)"}]

    # HTF context
    h4 = _resample_ohlc(df[["open", "high", "low", "close"]], "4h")
    h4 = _calc_indicators_htf(h4)
    d1 = _resample_ohlc(df[["open", "high", "low", "close"]], "1D")
    d1 = _calc_indicators_htf(d1)
    d1_lvls = _prev_day_levels(d1)
    w1 = _resample_ohlc(df[["open", "high", "low", "close"]], "1W")
    weekly = _weekly_pivots(w1)

    # Align to H1
    ctx = df.copy()
    ctx = _align_htf_to_h1(ctx, h4[["ema20", "ema50", "ema200", "close"]], prefix="H4")
    ctx = _align_htf_to_h1(ctx, d1[["ema20", "ema50", "ema200", "close"]], prefix="D1")
    ctx = pd.concat(
        [
            ctx,
            d1_lvls.reindex(ctx.index, method="ffill"),
            weekly.reindex(ctx.index, method="ffill"),
        ],
        axis=1,
    )
    ctx["atr_pctile"] = ctx["atr"].rolling(ATR_PCTILE_WINDOW).rank(pct=True)

    now_utc = pd.Timestamp.now(tz="UTC").floor("h")
    closed_idx = ctx.index[ctx.index < now_utc]
    if len(closed_idx) < 2:
        return [{"time": None, "pair": pair, "skip_reason": "not enough closed bars yet"}]

    closed_idx = closed_idx[-max(2, hours):]
    out: List[Dict[str, Any]] = []

    for last_ts in closed_idx[1:]:
        prev_ts = closed_idx[closed_idx.get_loc(last_ts) - 1]
        row = ctx.loc[last_ts]
        prev = ctx.loc[prev_ts]

        # 1) Session window (check first - more important than ATR)
        if not _in_core_session(last_ts, pair):
            out.append({"time": last_ts.isoformat(), "pair": pair, "skip_reason": "outside active session"})
            continue

        # 2) ATR finite + floor (match confluence: skip SPX floor, rounded units)
        if not math.isfinite(float(row["atr"])):
            out.append({"time": last_ts.isoformat(), "pair": pair, "skip_reason": "ATR not finite"})
            continue

        if not is_spx:
            atr_units = float(row["atr"]) / float(pip)
            atr_units_rounded = round(atr_units, 1)
            if atr_units_rounded + 1e-6 < min_atr_pips:
                unit = "points" if is_spx else "pips"
                out.append(
                    {
                        "time": last_ts.isoformat(),
                        "pair": pair,
                        "skip_reason": f"ATR {atr_units_rounded:.1f} {unit} below floor {min_atr_pips:.1f}",
                    }
                )
                continue

        # 3) Minimal scoring skeleton (pre-RR) — mirrors confluence categories
        # HTF trend
        h4_up = row["H4_close"] > row["H4_ema200"]
        h4_down = row["H4_close"] < row["H4_ema200"]
        d1_up = row["D1_close"] > row["D1_ema200"]
        d1_down = row["D1_close"] < row["D1_ema200"]

        long_score = 0
        short_score = 0
        long_score += 2 if h4_up else 0
        long_score += 1 if d1_up else 0
        short_score += 2 if h4_down else 0
        short_score += 1 if d1_down else 0

        # H4 slope (tie-breaker hint)
        h4_slope_up = _slope(h4["ema200"], lookback=H4_SLOPE_LOOKBACK) > 0
        h4_slope_dn = _slope(h4["ema200"], lookback=H4_SLOPE_LOOKBACK) < 0

        # H1 baseline + fast alignment
        trend_up = row["close"] > row["ema200"]
        trend_dn = row["close"] < row["ema200"]
        long_score += 1 if trend_up else 0
        short_score += 1 if trend_dn else 0

        fast_up = row["ema20"] > row["ema50"]
        fast_dn = row["ema20"] < row["ema50"]
        long_score += 1 if fast_up else 0
        short_score += 1 if fast_dn else 0

        # RSI bias (mid band is handled in full confluence; we keep the base bias here)
        rsi_lo, rsi_hi = cfg["rsi_band"]
        rsi_bull = (row["rsi"] >= 50) and (row["rsi"] >= prev["rsi"])
        rsi_bear = (row["rsi"] <= 50) and (row["rsi"] <= prev["rsi"])
        long_score += 1 if rsi_bull else 0
        short_score += 1 if rsi_bear else 0

        # ADX mild floor
        adx_ok = (row.get("adx", 0) or 0) >= 18
        long_score += 1 if adx_ok else 0
        short_score += 1 if adx_ok else 0

        # Overextension penalty (same EMA20 vs ATR mult as confluence)
        overextended = abs(row["close"] - row["ema20"]) > (EMA_OVEREXT_MULT * row["atr"])
        if overextended:
            long_score -= 2
            short_score -= 2

        # Choose side (same tie rule)
        if long_score > short_score:
            choose_side = "BUY"
            score_pre_rr = int(long_score)
        elif short_score > long_score:
            choose_side = "SELL"
            score_pre_rr = int(short_score)
        else:
            if h4_slope_up and not h4_slope_dn:
                choose_side = "BUY"
                score_pre_rr = int(long_score)
            elif h4_slope_dn and not h4_slope_up:
                choose_side = "SELL"
                score_pre_rr = int(short_score)
            else:
                out.append({"time": last_ts.isoformat(), "pair": pair, "skip_reason": "tie with no H4 slope edge"})
                continue

        # Min score gate (pre-RR, to match your index behavior)
        if score_pre_rr < min_score:
            out.append(
                {
                    "time": last_ts.isoformat(),
                    "pair": pair,
                    "skip_reason": f"score {score_pre_rr} < min_score {min_score}",
                }
            )
            continue

        # Level proximity guard (ROUND + daily/weekly), same buffers
        lvl_vals: List[float] = []
        for nm in ["PDH", "PDL", "PDC", "PP", "R1", "S1"]:
            if nm in row and pd.notna(row[nm]):
                lvl_vals.append(float(row[nm]))

        # Round step parity with confluence
        if pair == "XAUUSD":
            step = 0.5
        elif pair == "SPX500":
            step = 2.0
        elif pair.endswith("JPY"):
            step = 0.25  # ~25 "pips" (pip=0.01)
        else:
            step = 0.0025  # ~25 pips on majors

        lvl_vals.append(_nearest_round(float(row["close"]), step))

        entry = float(row["close"])
        min_buffer = LEVEL_MIN_BUFFER_ATR * float(row["atr"])

        if choose_side == "BUY":
            ceiling = min([v for v in lvl_vals if v > entry], default=None)
            if ceiling is not None and (ceiling - entry) < min_buffer:
                out.append({"time": last_ts.isoformat(), "pair": pair, "skip_reason": "too close to overhead level"})
                continue
        else:
            floor_ = max([v for v in lvl_vals if v < entry], default=None)
            if floor_ is not None and (entry - floor_) < min_buffer:
                out.append({"time": last_ts.isoformat(), "pair": pair, "skip_reason": "too close to floor level"})
                continue

        # Passed all guards (pre-RR) → report stub
        out.append(
            {
                "time": last_ts.isoformat(),
                "pair": pair,
                "side": choose_side,
                "score_pre_rr": score_pre_rr,   # explicit pre-RR for parity with confluence output
                "rr": None,                     # RR is only computed in confluence
            }
        )

    return out
