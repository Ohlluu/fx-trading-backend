# app/positions.py
# v1.1 — Robust start index clamp, safer time handling, minor idempotency polish.
# Storage: JSON file (data/positions.json). No DB required.
# Exposes:
#   POST   /api/positions            -> start tracking a position (from a signal)
#   GET    /api/positions/open       -> list open positions (for UI)
#   GET    /api/positions/history    -> list closed positions (simple history)
#   PATCH  /api/positions/{id}       -> close_manual / untrack / edit sl/tp
#   POST   /api/positions/tick       -> manager pass: update statuses (can be scheduled)

from __future__ import annotations
from fastapi import APIRouter, HTTPException, Body, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone
import os, json, uuid
import pandas as pd
import math

# Import your existing evaluator (read-only)
from .confluence import evaluate_last_closed_bar

# Enhanced risk management imports
import numpy as np
from typing import Dict, List, Tuple

router = APIRouter()

DATA_PATH = "data"
STORE_PATH = os.path.join(DATA_PATH, "positions.json")

# Enhanced Risk Management Configuration
RISK_CONFIG = {
    'max_portfolio_risk': 0.02,      # Max 2% account risk across all positions
    'max_single_risk': 0.01,         # Max 1% risk per trade
    'max_correlation_risk': 0.015,   # Max 1.5% risk for correlated pairs
    'max_daily_trades': 5,           # Max trades per day
    'kelly_fraction': 0.25,          # Kelly Criterion fraction
    'vol_lookback': 20,              # Volatility calculation period
    'correlation_threshold': 0.7,     # Pairs considered correlated above this
}

# Currency correlation matrix (simplified)
PAIR_CORRELATIONS = {
    ('EURUSD', 'GBPUSD'): 0.65,
    ('EURUSD', 'AUDUSD'): 0.75,
    ('GBPUSD', 'AUDUSD'): 0.70,
    ('USDJPY', 'EURJPY'): 0.80,
    ('USDJPY', 'GBPJPY'): 0.85,
    ('EURJPY', 'GBPJPY'): 0.90,
    ('USDCHF', 'EURUSD'): -0.75,
    ('USDCAD', 'AUDUSD'): -0.60,
}

# ---------------------------
# Enhanced Risk Management Functions
# ---------------------------

def get_pair_correlation(pair1: str, pair2: str) -> float:
    """Get correlation between two currency pairs"""
    key = (pair1, pair2) if (pair1, pair2) in PAIR_CORRELATIONS else (pair2, pair1)
    return PAIR_CORRELATIONS.get(key, 0.0)

def calculate_portfolio_risk(open_positions: List[Dict]) -> float:
    """Calculate current portfolio risk percentage"""
    total_risk = 0.0
    for pos in open_positions:
        # Simplified risk calculation: (entry - sl) / entry
        entry = pos.get('entry', 0)
        sl = pos.get('sl', 0)
        if entry > 0 and sl > 0:
            risk_per_trade = abs(entry - sl) / entry
            total_risk += risk_per_trade
    return min(total_risk, 1.0)  # Cap at 100%

def calculate_correlation_risk(pair: str, open_positions: List[Dict]) -> float:
    """Calculate risk from correlated positions"""
    corr_risk = 0.0
    for pos in open_positions:
        pos_pair = pos.get('pair', '')
        if pos_pair != pair:
            correlation = abs(get_pair_correlation(pair, pos_pair))
            if correlation > RISK_CONFIG['correlation_threshold']:
                entry = pos.get('entry', 0)
                sl = pos.get('sl', 0)
                if entry > 0 and sl > 0:
                    pos_risk = abs(entry - sl) / entry
                    corr_risk += pos_risk * correlation
    return corr_risk

def calculate_optimal_position_size(pair: str, entry: float, sl: float, 
                                  account_balance: float = 10000, 
                                  win_rate: float = 0.5) -> Dict[str, float]:
    """Calculate optimal position size using multiple methods"""
    
    # 1. Fixed risk method (1% risk)
    risk_amount = account_balance * RISK_CONFIG['max_single_risk']
    price_diff = abs(entry - sl)
    fixed_risk_size = risk_amount / price_diff if price_diff > 0 else 0
    
    # 2. Kelly Criterion (simplified)
    # Kelly = (bp - q) / b, where b = avg_win/avg_loss, p = win_rate, q = loss_rate
    avg_win_loss_ratio = 1.8  # Assume 1.8:1 reward:risk
    kelly_fraction = ((avg_win_loss_ratio * win_rate) - (1 - win_rate)) / avg_win_loss_ratio
    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
    kelly_size = (account_balance * kelly_fraction * RISK_CONFIG['kelly_fraction']) / price_diff if price_diff > 0 else 0
    
    # 3. Volatility-adjusted size
    vol_factor = 1.0  # Would be calculated from actual volatility
    vol_adjusted_size = fixed_risk_size / vol_factor
    
    return {
        'fixed_risk': fixed_risk_size,
        'kelly_optimal': kelly_size,
        'volatility_adjusted': vol_adjusted_size,
        'recommended': min(fixed_risk_size, kelly_size, vol_adjusted_size)
    }

def can_open_position(pair: str, entry: float, sl: float, open_positions: List[Dict]) -> Dict[str, Any]:
    """Check if position can be opened based on risk management rules"""
    
    # Check portfolio risk
    current_risk = calculate_portfolio_risk(open_positions)
    if current_risk >= RISK_CONFIG['max_portfolio_risk']:
        return {'allowed': False, 'reason': 'Portfolio risk limit exceeded'}
    
    # Check correlation risk
    corr_risk = calculate_correlation_risk(pair, open_positions)
    if corr_risk >= RISK_CONFIG['max_correlation_risk']:
        return {'allowed': False, 'reason': 'Correlation risk limit exceeded'}
    
    # Check daily trade limit
    today = datetime.now(timezone.utc).date()
    today_positions = [p for p in open_positions if 
                      datetime.fromisoformat(p.get('opened_at', '')).date() == today]
    if len(today_positions) >= RISK_CONFIG['max_daily_trades']:
        return {'allowed': False, 'reason': 'Daily trade limit exceeded'}
    
    return {'allowed': True, 'reason': 'Position allowed'}

def calculate_dynamic_stops(entry: float, sl: float, tp: float, 
                           current_price: float, pair: str) -> Dict[str, float]:
    """Calculate dynamic trailing stops and targets"""
    
    # Load recent price data for volatility calculation
    try:
        df = _load_h1(pair)
        if len(df) > 20:
            recent_atr = df['close'].rolling(14).std().iloc[-1] * 2.5  # Approximated ATR
        else:
            recent_atr = abs(entry - sl)  # Fallback
    except:
        recent_atr = abs(entry - sl)
    
    # Trailing stop calculation
    if entry > sl:  # Long position
        trail_distance = recent_atr * 1.5
        trailing_stop = max(sl, current_price - trail_distance)
        breakeven_stop = entry
        partial_target = entry + (tp - entry) * 0.6
    else:  # Short position
        trail_distance = recent_atr * 1.5
        trailing_stop = min(sl, current_price + trail_distance)
        breakeven_stop = entry
        partial_target = entry - (entry - tp) * 0.6
    
    return {
        'trailing_stop': trailing_stop,
        'breakeven_stop': breakeven_stop,
        'partial_target': partial_target,
        'trail_distance': trail_distance
    }

# ---------------------------
# Models
# ---------------------------

class TimeStop(BaseModel):
    expected_bars: Optional[float] = None
    expected_hours: Optional[float] = None
    deadline_utc: Optional[str] = None  # ISO string
    efficiency: Optional[float] = None
    notes: Optional[str] = None

class Position(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    pair: str
    side: str  # "BUY" | "SELL"
    entry: float
    sl: float
    tp: float
    signal_time: str  # ISO of the H1 signal bar
    opened_at: str    # ISO of when user tapped "I took this"
    score_init: Optional[int] = None
    reasons_init: Optional[List[str]] = None
    time_stop: Optional[TimeStop] = None
    
    # Position sizing (always $5 risk per trade)
    position_size: float = 5.0  # Fixed $5 risk amount

    # Live/derived fields
    status: str = "open"  # open | closed_tp | closed_sl | closed_time | closed_manual
    closed_at: Optional[str] = None
    exit_price: Optional[float] = None

    # health/telemetry
    score_now: Optional[int] = None
    delta_score: Optional[int] = None
    side_flip: Optional[bool] = None
    r_now: Optional[float] = None
    suggested_action: Optional[str] = None  # "hold" | "be" | "partial" | "exit" | "close_tp" | "close_sl" | "close_time"
    
    # Manual control analysis data
    tp_would_hit: Optional[bool] = None      # Would TP have been hit?
    sl_would_hit: Optional[bool] = None      # Would SL have been hit?
    time_expired: Optional[bool] = None      # Has time stop expired?
    hours_open: Optional[float] = None       # How long has position been open?
    current_price: Optional[float] = None    # Current market price
    analysis_timestamp: Optional[str] = None # When was this analysis done?

# ---------------------------
# Storage helpers
# ---------------------------

def _ensure_store():
    os.makedirs(DATA_PATH, exist_ok=True)
    if not os.path.exists(STORE_PATH):
        with open(STORE_PATH, "w", encoding="utf-8") as f:
            json.dump({"open": [], "closed": []}, f)

def _load_store() -> Dict[str, List[Dict[str, Any]]]:
    _ensure_store()
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {"open": [], "closed": []}

def _save_store(store: Dict[str, List[Dict[str, Any]]]):
    tmp = STORE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, STORE_PATH)

# ---------------------------
# Data access (H1 CSV)
# ---------------------------

def _csv_path(pair: str, timeframe: str = "H1") -> str:
    return f"data/{pair}_{timeframe}.csv"

def _load_h1(pair: str) -> pd.DataFrame:
    path = _csv_path(pair, "H1")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"])

# ---------------------------
# Utility
# ---------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _to_iso(x) -> str:
    if hasattr(x, "isoformat"):
        return x.isoformat()
    return str(x)

def _hit_price_window(df: pd.DataFrame, start_ts: pd.Timestamp, side: str, sl: float, tp: float):
    """
    From start_ts onward, did we touch tp or sl? And which came first?
    Robust to start_ts being past the last closed bar.
    """
    # --- FIX: clamp searchsorted result; bail if beyond last index ---
    idx = df.index.searchsorted(start_ts, side="left")
    if idx >= len(df.index):
        return None  # no closed bars at/after start_ts yet

    window = df.iloc[idx:]
    if window.empty:
        return None

    if side == "BUY":
        to_tp = window[window["high"] >= tp]
        to_sl = window[window["low"]  <= sl]
    else:
        to_tp = window[window["low"]  <= tp]
        to_sl = window[window["high"] >= sl]

    tp_time = to_tp.index[0] if not to_tp.empty else None
    sl_time = to_sl.index[0] if not to_sl.empty else None

    if tp_time and sl_time:
        return "tp_first" if tp_time <= sl_time else "sl_first"
    if tp_time:
        return "tp"
    if sl_time:
        return "sl"
    return None

def _r_multiple(entry: float, sl: float, current: float, side: str) -> Optional[float]:
    risk = (entry - sl) if side == "BUY" else (sl - entry)
    if risk == 0:
        return None
    reward = (current - entry) if side == "BUY" else (entry - current)
    return reward / risk

# ---------------------------
# Manager core (single pass)
# ---------------------------

def _assess_position(p: Position) -> Position:
    """One pass assessment: health snapshot, suggestion (NO AUTO-CLOSING FOR MANUAL CONTROL)."""
    # SYNCHRONIZATION FIX: Use same data loading mechanism as signal generation for consistency
    from app.main import load_pair_df
    
    df = load_pair_df(p.pair)
    if df.empty:
        return p

    # Record analysis timestamp immediately to ensure proper synchronization tracking
    analysis_timestamp = _now_iso()
    last_candle_time = df.index[-1] if not df.empty else None
    
    # DEBUG: Log which candle we're analyzing for transparency
    print(f"[POSITION_SYNC] {p.pair} assessment at {analysis_timestamp[:19]} using candle {last_candle_time}")
    
    opened_at = pd.to_datetime(p.opened_at, utc=True, errors="coerce")
    start_ts = opened_at.floor("h")

    # MANUAL CONTROL: No automatic closing for TP/SL/Time stops
    # User retains full control to analyze what went wrong/right
    # Price and time analysis still available for suggested_action calculation
    
    # Check if TP/SL would have been hit (for suggestion logic, not auto-close)
    first = _hit_price_window(df, start_ts, p.side, p.sl, p.tp)
    tp_would_hit = first in ("tp_first", "tp")
    sl_would_hit = first in ("sl_first", "sl")
    
    # Check if time stop deadline passed (for suggestion logic, not auto-close)
    time_expired = False
    if p.time_stop and p.time_stop.deadline_utc:
        try:
            dl = pd.to_datetime(p.time_stop.deadline_utc, utc=True, errors="coerce")
            now_utc = pd.Timestamp.now(tz="UTC")
            time_expired = now_utc >= dl
        except Exception:
            pass

    # 3) Health snapshot (light): re-evaluate current bar context to compare score/side
    # NOTE: Position rescoring uses SAME stability buffer as entry for timer-based consistency
    try:
        idea = evaluate_last_closed_bar(df, pair=p.pair)
        if idea:
            raw_score = int(idea["score"])
            
            # Apply STABLE PROFILE stability buffer to position rescoring (same as entry logic)
            from app.confluence import STRICTNESS
            if STRICTNESS == "stable" and raw_score is not None:
                # Apply same stability buffer logic as new signals
                stable_floor = 0
                
                # Extract trend info from original position reasons (flexible point matching)
                h4_up = any("H4 trend up" in reason for reason in (p.reasons_init or []))
                h4_down = any("H4 trend down" in reason for reason in (p.reasons_init or []))
                d1_up = any("D1 trend up" in reason for reason in (p.reasons_init or []))
                d1_down = any("D1 trend down" in reason for reason in (p.reasons_init or []))
                h4_slope_up = any("H4 EMA200 slope up" in reason for reason in (p.reasons_init or []))
                h4_slope_dn = any("H4 EMA200 slope down" in reason for reason in (p.reasons_init or []))
                
                if p.side == "BUY":
                    # Ultra-stable pairs get enhanced weighting for maximum confluence
                    ultra_stable_pairs = ("EURGBP", "EURUSD", "EURCHF", "EURCAD", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "XAUUSD", "SPX500", "USDJPY", "GBPJPY", "EURJPY")
                    is_ultra_stable = p.pair in ultra_stable_pairs
                    h4_stable_weight = 7 if is_ultra_stable else 6  # Ultra-stable pairs bonus
                    d1_stable_weight = 4 if is_ultra_stable else 3  # Ultra-stable pairs bonus
                    if h4_up: stable_floor += h4_stable_weight  # H4 trend worth more for ultra-stable pairs
                    if d1_up: stable_floor += d1_stable_weight  # D1 trend worth more for ultra-stable pairs  
                    if h4_slope_up: stable_floor += 3  # H4 slope worth 3 in stable (1×3)
                else:
                    # Ultra-stable pairs get enhanced weighting for maximum confluence
                    ultra_stable_pairs = ("EURGBP", "EURUSD", "EURCHF", "EURCAD", "GBPUSD", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD", "XAUUSD", "SPX500", "USDJPY", "GBPJPY", "EURJPY")
                    is_ultra_stable = p.pair in ultra_stable_pairs
                    h4_stable_weight = 7 if is_ultra_stable else 6  # Ultra-stable pairs bonus
                    d1_stable_weight = 4 if is_ultra_stable else 3  # Ultra-stable pairs bonus
                    if h4_down: stable_floor += h4_stable_weight  # H4 trend worth more for ultra-stable pairs
                    if d1_down: stable_floor += d1_stable_weight  # D1 trend worth more for ultra-stable pairs
                    if h4_slope_dn: stable_floor += 3  # H4 slope worth 3 in stable
                
                # Ultra-stable pairs need stronger protection against confluence crashes
                if is_ultra_stable and p.score_init and p.score_init >= 15:
                    # High-confluence ultra-stable positions: 90% retention + minimum 12
                    min_stable_score = max(int(stable_floor * 0.90), 12)
                    # Never drop more than 5 points from ultra-stable high-confluence initial score
                    max_drop_limit = max(p.score_init - 5, min_stable_score)
                    min_stable_score = max(min_stable_score, max_drop_limit)
                else:
                    # Standard protection: 85% of trend-based floor 
                    min_stable_score = int(stable_floor * 0.85)
                    
                # Never let score drop below 5 if we have any trend alignment
                if stable_floor > 0:
                    min_stable_score = max(min_stable_score, 5)
                    if raw_score < min_stable_score:
                        raw_score = min_stable_score
            
            p.score_now = raw_score
            if p.score_init is not None:
                p.delta_score = int(p.score_now - p.score_init)
            p.side_flip = (idea["side"] != p.side)
        else:
            p.score_now = None
            p.delta_score = None
            p.side_flip = None
    except Exception:
        pass

    # 4) Calculate current price first (needed for TP system and r_now)
    current_price = float(df["close"].iloc[-1])
    
    # 5) TP system calculation based on current score
    if p.score_now is not None and p.score_now >= 16:
        from app.main import calculate_tp_probabilities
        p.tp_system = calculate_tp_probabilities(
            score=int(p.score_now),
            current_price=current_price,
            stop_price=p.sl
        )
    else:
        p.tp_system = None

    # 6) r_now & suggested action
    r_now = _r_multiple(p.entry, p.sl, current_price, p.side)
    p.r_now = float(r_now) if r_now is not None and math.isfinite(r_now) else None
    
    # Store analysis data for manual control
    p.tp_would_hit = tp_would_hit
    p.sl_would_hit = sl_would_hit  
    p.time_expired = time_expired
    p.current_price = current_price
    p.analysis_timestamp = analysis_timestamp  # Use the timestamp recorded at the start

    # Enhanced exit logic with trailing stops and smart management
    # Calculate dynamic stops
    dynamic_stops = calculate_dynamic_stops(p.entry, p.sl, p.tp, current_price, p.pair)
    
    # Enhanced exit logic with ultra-stable position protection
    hours_open = 0
    try:
        opened_time = pd.to_datetime(p.opened_at, utc=True)
        hours_open = (pd.Timestamp.now(tz="UTC") - opened_time).total_seconds() / 3600
    except:
        pass
    
    p.hours_open = hours_open
    
    # Ultra-stable position protection: Don't exit high-confluence trades too early
    is_ultra_stable_trade = (p.score_init is not None and p.score_init >= 15)
    
    # Enhanced exit suggestions with manual control context
    if tp_would_hit:
        p.suggested_action = "close_tp"  # TP was hit, suggest manual close at TP
    elif sl_would_hit:
        p.suggested_action = "close_sl"  # SL was hit, suggest manual close at SL
    elif time_expired:
        p.suggested_action = "close_time"  # Time stop expired, suggest manual close
    elif p.side_flip:
        p.suggested_action = "exit"  # Side flip, suggest exit
    elif (p.r_now is not None) and (p.r_now >= 1.5):
        # Position well in profit, suggest trailing stop (manual adjustment)
        p.suggested_action = "trail"
    elif (p.r_now is not None) and (p.r_now >= 0.8):
        # Strong profit, suggest breakeven (manual adjustment)
        if is_ultra_stable_trade and hours_open < 4:
            p.suggested_action = "hold"  # Protect ultra-stable trades from early BE
        else:
            p.suggested_action = "be"
    elif (p.r_now is not None) and (p.r_now >= 0.5):
        # Moderate profit - hold longer for ultra-stable trades
        if is_ultra_stable_trade:
            p.suggested_action = "hold"  # Let ultra-stable trades breathe
        else:
            p.suggested_action = "partial"
    elif (p.r_now is not None) and (p.r_now >= 0.3):
        # Take partial profits only for lower-confluence trades
        if not is_ultra_stable_trade:
            p.suggested_action = "partial"
        else:
            p.suggested_action = "hold"
    elif (p.delta_score is not None) and (p.delta_score <= -6):
        # Only exit on massive score drops (was -4, now -6 for ultra-stable protection)
        if is_ultra_stable_trade and hours_open < 6:
            p.suggested_action = "hold"  # Extra protection for ultra-stable in first 6 hours
        else:
            p.suggested_action = "exit"
    elif (p.r_now is not None) and (p.r_now <= -0.9):
        # Stop is very close, suggest manual tightening
        p.suggested_action = "tighten"
    else:
        p.suggested_action = "hold"

    return p

# ---------------------------
# Public manager entrypoint
# ---------------------------

def manager_tick() -> Dict[str, int]:
    """Run assessment on all open positions (NO AUTO-CLOSING - MANUAL CONTROL ONLY)."""
    store = _load_store()
    open_list = [Position(**p) for p in store.get("open", [])]
    closed = store.get("closed", [])

    updated_open: List[Dict[str, Any]] = []
    moved = 0  # Will always be 0 since we don't auto-close anymore

    for pos in open_list:
        # Only update health metrics and suggestions - NO automatic closing
        pos = _assess_position(pos)
        # Position stays open regardless of TP/SL/time hits - manual control only
        updated_open.append(pos.dict())

    store["open"] = updated_open
    # Closed list only changes via manual actions now
    _save_store(store)
    return {"updated": len(updated_open), "closed_moved": moved, "open": len(updated_open), "closed": len(closed)}

# ---------------------------
# API routes
# ---------------------------

@router.post("", summary="Track a position from a signal")
def create_position(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Body fields expected (minimal):
      pair, side, entry, sl, tp, signal_time (ISO), time_stop (optional), score_init, reasons_init
    """
    required = ["pair", "side", "entry", "sl", "tp", "signal_time"]
    for k in required:
        if body.get(k) is None:
            raise HTTPException(status_code=400, detail=f"Missing field: {k}")

    pos = Position(
        pair=str(body["pair"]).upper(),
        side=str(body["side"]).upper(),
        entry=float(body["entry"]),
        sl=float(body["sl"]),
        tp=float(body["tp"]),
        signal_time=str(body["signal_time"]),
        opened_at=str(body.get("opened_at") or datetime.now(timezone.utc).isoformat()),
        score_init=int(body.get("score_init")) if body.get("score_init") is not None else None,
        reasons_init=[str(x) for x in (body.get("reasons_init") or body.get("reasons") or [])],
        time_stop=TimeStop(**body["time_stop"]) if body.get("time_stop") else None,
    )

    store = _load_store()
    # Idempotency: avoid dup for same pair|signal_time|entry (rounded) still open
    fp = f"{pos.pair}|{pos.signal_time}|{pos.entry:.10f}"
    for p in store.get("open", []):
        other_fp = f'{p.get("pair")}|{p.get("signal_time")}|{float(p.get("entry", 0.0)):.10f}'
        if other_fp == fp:
            return {"ok": True, "position": pos.dict(), "duplicate": True}

    store["open"].append(pos.dict())
    _save_store(store)
    return {"ok": True, "position": pos.dict()}

@router.get("/open", summary="List open tracked positions")
def list_open() -> Dict[str, Any]:
    store = _load_store()
    return {"positions": store.get("open", [])}

@router.get("/history", summary="List closed positions (history)")
def list_history(limit: int = Query(50, ge=1, le=500)) -> Dict[str, Any]:
    store = _load_store()
    hist = store.get("closed", [])[-limit:]
    return {"positions": hist}

@router.patch("/{pos_id}", summary="Modify or close a tracked position")
def patch_position(pos_id: str, body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    """
    Actions:
      { "action": "close_manual", "exit_price": 1.23456 }
      { "action": "untrack" }
      { "action": "edit", "sl": 1.23, "tp": 1.25 }
    """
    store = _load_store()
    open_list = store.get("open", [])
    idx = next((i for i, p in enumerate(open_list) if p.get("id") == pos_id), None)
    if idx is None:
        raise HTTPException(status_code=404, detail="Position not found")

    rec = Position(**open_list[idx])

    action = str(body.get("action") or "").lower()
    if action == "close_manual":
        rec.status = "closed_manual"
        rec.exit_price = float(body.get("exit_price") or rec.entry)
        rec.closed_at = _now_iso()
        # move to closed
        del open_list[idx]
        store.setdefault("closed", []).append(rec.dict())
        _save_store(store)
        return {"ok": True, "position": rec.dict()}

    if action == "untrack":
        # just drop from open without history
        del open_list[idx]
        store["open"] = open_list
        _save_store(store)
        return {"ok": True}

    if action == "edit":
        if body.get("sl") is not None:
            rec.sl = float(body["sl"])
        if body.get("tp") is not None:
            rec.tp = float(body["tp"])
        open_list[idx] = rec.dict()
        store["open"] = open_list
        _save_store(store)
        return {"ok": True, "position": rec.dict()}

    raise HTTPException(status_code=400, detail="Unsupported action")

@router.post("/tick", summary="Run manager tick now")
def tick_now():
    return manager_tick()
