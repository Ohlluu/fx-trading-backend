# backend/app/debug_skips.py
from __future__ import annotations

from typing import Dict, Any, Optional
from fastapi import APIRouter, Query
import pandas as pd

# Your helper scan (produces per-hour skip/pass stubs)
from .debug_scan import _hourly_skip_reasons

# ✅ JSON-safe wrapper + evaluator live in app.confluence
from app.confluence import evaluate_last_closed_bar, json_safe

router = APIRouter()


@router.get(
    "/skip-reasons",
    summary="Per-hour skip reasons (or pass stubs) for last N closed H1 bars",
)
def skip_reasons(
    pair: str = Query(..., description="Instrument e.g. SPX500, EURUSD"),
    hours: int = Query(48, ge=2, le=5000),
) -> Dict[str, Any]:
    """
    Returns an object with:
      - pair
      - hours
      - reasons: list of per-hour items with either `skip_reason` or a lightweight pass stub
    Wrapped in json_safe(...) to avoid Pandas/NumPy serialization 500s.
    """
    try:
        payload = {
            "pair": pair.upper(),
            "hours": hours,
            "reasons": _hourly_skip_reasons(pair, hours),
        }
        return json_safe(payload)
    except Exception as e:
        # Defensive: never 500 on debug endpoints
        return json_safe(
            {
                "ok": False,
                "endpoint": "skip-reasons",
                "pair": pair.upper(),
                "hours": hours,
                "error": f"{type(e).__name__}: {e}",
            }
        )


@router.get(
    "/eval-last",
    summary="Evaluate the last closed H1 bar for a pair (confluence engine)",
)
def eval_last(
    pair: str = Query(..., description="Instrument e.g. SPX500, EURUSD"),
    data_path: Optional[str] = Query(
        None, description="Optional override path like data/EURUSD_H1.csv"
    ),
) -> Dict[str, Any]:
    """
    Loads H1 candles for `pair`, runs evaluate_last_closed_bar, and returns the signal/skip
    in a JSON-safe envelope. This mirrors your production call path.
    """
    try:
        # Load H1 data (CSV with `time,open,high,low,close,volume`)
        path = data_path or f"data/{pair.upper()}_H1.csv"
        df = pd.read_csv(path)
        # Ensure UTC DateTimeIndex sorted ascending
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        df = df.dropna(subset=["time"]).set_index("time").sort_index()
        for c in ("open", "high", "low", "close"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close"])

        # Run evaluator (debug arg is accepted/ignored safely by v4)
        payload = evaluate_last_closed_bar(df, pair=pair, debug=True)
        # Return JSON-safe
        return json_safe(payload)

    except FileNotFoundError:
        return json_safe(
            {
                "ok": False,
                "endpoint": "eval-last",
                "pair": pair.upper(),
                "error": f"data file not found for pair at {data_path or f'data/{pair.upper()}_H1.csv'}",
            }
        )
    except Exception as e:
        # Defensive: never 500
        return json_safe(
            {
                "ok": False,
                "endpoint": "eval-last",
                "pair": pair.upper(),
                "error": f"{type(e).__name__}: {e}",
            }
        )
