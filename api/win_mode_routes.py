# /Users/user/fx-app/backend/api/win_mode_routes.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime, timezone

# your Win Mode evaluator + scanner
from app.win_mode import evaluate_last_closed_bar_win, scan_win_mode
# reuse CSV loader to mirror how your normal scanner pulls data
from app.debug_scan import _load_h1
from app.confluence import PAIR_CONFIG

router = APIRouter()

# ---------- (optional) programmatic evaluate/scan ----------
class Candle(BaseModel):
    time: str
    open: float
    high: float
    low: float
    close: float

class EvalRequest(BaseModel):
    pair: str
    candles: list[Candle]

@router.post("/win-mode/evaluate")
def win_mode_evaluate(req: EvalRequest):
    df = pd.DataFrame([c.dict() for c in req.candles])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    res = evaluate_last_closed_bar_win(df, pair=req.pair, debug=True)
    if res is None:
        raise HTTPException(400, "no result")
    return res

@router.post("/win-mode/scan")
def win_mode_scan(req: EvalRequest):
    df = pd.DataFrame([c.dict() for c in req.candles])
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()
    return scan_win_mode(req.pair, df, hours=48)

# ---------- endpoints used by your new mobile tab ----------
@router.get("/win-signals/today")
def win_signals_today():
    """
    Build today's Win Mode signals across all configured pairs,
    reading H1 CSVs the same way as your normal scan.
    """
    out = []
    today_utc = datetime.now(timezone.utc).date()
    for pair in PAIR_CONFIG.keys():
        df = _load_h1(pair)
        if df.empty:
            continue
        res = evaluate_last_closed_bar_win(df, pair=pair, debug=False)
        if res and not res.get("skip_reason"):
            # only include if the bar is today (UTC)
            ts = res.get("time")
            try:
                ts_date = pd.to_datetime(ts, utc=True).date()
            except Exception:
                ts_date = None
            if ts_date == today_utc:
                out.append({
                    "pair": res["pair"],
                    "timeframe": "1 hour timeframe",
                    "side": res["side"],
                    "price": res["price"],
                    "entry": res["price"],
                    "stop": res.get("stop"),
                    "target": res.get("target"),
                    "rr": res.get("rr"),
                    "atr": res.get("atr"),
                    "score": res.get("score"),
                    "time": str(res.get("time")),
                    "reasons": res.get("reasons", []),
                    "time_stop": res.get("time_stop"),
                })
    return {"signals": out}

@router.post("/win-fetch-and-scan")
def win_fetch_and_scan():
    """
    Hook for pull-to-refresh. If you have an external market fetcher,
    call it here. For now it’s a no-op so the app proceeds to GET /win-signals/today.
    """
    return {"ok": True}
