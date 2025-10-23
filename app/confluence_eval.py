# app/confluence_eval.py
from __future__ import annotations
import pandas as pd
from typing import Optional, Dict, Any

from .confluence import evaluate_last_closed_bar

def evaluate_at_ts(
    h1_df: pd.DataFrame,
    pair: Optional[str] = None,
    asof_ts: Optional[pd.Timestamp] = None,
    **kwargs,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate strategy as if 'asof_ts' was the current hour.
    Truncates the dataframe up to that hour and runs the usual evaluator.
    Returns normal trade dict OR skip_reason if blocked.
    """
    if asof_ts is None:
        return evaluate_last_closed_bar(h1_df, pair=pair, **kwargs)

    ts = pd.to_datetime(asof_ts, utc=True, errors="coerce").floor("h")
    if not isinstance(h1_df.index, pd.DatetimeIndex):
        return None

    df_trunc = h1_df[h1_df.index <= ts]
    if df_trunc.empty:
        return None

    return evaluate_last_closed_bar(df_trunc, pair=pair, **kwargs)
