# app/skip_debug.py
from fastapi import APIRouter, Query
from typing import Dict, Any
import pandas as pd
from .confluence_eval import evaluate_bar_with_skip_reason  # from earlier helper

router = APIRouter()

@router.get("/skip-reasons")
def skip_reasons(pair: str = Query(...), hours: int = Query(48, ge=2, le=5000)) -> Dict[str, Any]:
    # load your H1 CSV
    path = f"data/{pair}_H1.csv"
    try:
        df = pd.read_csv(path)
    except Exception:
        return {"pair": pair, "hours": hours, "reasons": []}

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()

    # take last N hours of **closed** bars
    end = pd.Timestamp.utcnow().floor("h")
    start = end - pd.Timedelta(hours=hours)
    ctx = df.loc[(df.index >= start) & (df.index < end)]

    out = []
    for ts in ctx.index:
        res = evaluate_bar_with_skip_reason(ctx.loc[:ts])  # returns dict with skip_reason or a signal
        if res is None:
            continue
        out.append(res)

    return {"pair": pair, "hours": hours, "reasons": out}
