# /Users/user/fx-app/backend/app/datafeed.py
# v2.1 — Timeframe-aware saving, small robustness fixes, identical public API preserved.

import os
import re
import asyncio
import random
from typing import Tuple, List, Dict, Any, Optional

import httpx
import pandas as pd

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
def _get_twelve_key() -> str:
    key = os.getenv("TWELVE_DATA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY env var is not set. "
            "Set it in your environment before calling fetch_h1()."
        )
    return key

# Friendly names → Twelve Data symbols (non-FX)
INDEX_MAP: Dict[str, str] = {
    # App name      # Twelve Data symbol
    "SPX500": "SPY",    # S&P 500 (SPX symbol returns wrong data, SPY works)
    "US500":  "SPY",    # alias
    "US30":   "DJI",    # Dow Jones
    "NAS100": "QQQ",    # Nasdaq 100 (QQQ ETF has better data than NDX)
    "US100":  "QQQ",    # alias
    "DE30":   "DAX",    # German DAX 30
    "GER30":  "DAX",    # alias
    "UK100":  "UKX",    # FTSE 100
    "FTSE100": "UKX",   # alias
}

# Scaling factors for index proxies (CFD price -> Index price approximation)
# Updated to match exact trading app prices
INDEX_SCALING: Dict[str, float] = {
    "SPX500": 10.04,   # SPY ETF ~$666.78 -> S&P 500 Index ~$6,694 (666.78 * 10.04 = 6,694)
    "US500":  10.04,   # alias
    "NAS100": 41.10,   # QQQ ETF ~602.24 -> Nasdaq 100 Index ~24,750 (602.24 * 41.10 = 24,750)
    "US100":  41.10,   # alias
    "DE30":   532.6,   # DAX CFD ~44.275 -> DAX Index ~23,580 (44.275 * 532.6 = 23,580)
    "GER30":  532.6,   # alias
    "UK100":  87.9,    # UKX CFD ~105.14 -> FTSE 100 ~9,245 (105.14 * 87.9 = 9,245)
    "FTSE100": 87.9,   # alias
}

# -----------------------------------------------------------------------------
# Helpers: symbols, pairs, timeframe & HTTP
# -----------------------------------------------------------------------------
def normalize_pair(pair: str) -> str:
    """
    Normalize inputs like 'eur/usd', 'EUR_USD', ' EurUsd ' -> 'EURUSD'.
    Enforces 6-letter FX/metals format (e.g., EURUSD, GBPUSD, USDJPY, XAUUSD).
    """
    cleaned = re.sub(r"[^A-Za-z]", "", (pair or "")).upper()
    if len(cleaned) != 6:
        raise ValueError("Pair must be exactly 6 letters, e.g. EURUSD, GBPUSD, USDJPY, XAUUSD.")
    return cleaned

def split_pair(pair: str) -> Tuple[str, str]:
    """'EURUSD' -> ('EUR', 'USD')"""
    p = normalize_pair(pair)
    return p[:3], p[3:]

def _is_index(symbol: str) -> bool:
    return (symbol or "").upper() in INDEX_MAP

def _norm_timeframe_for_api(tf: str) -> str:
    """
    Maps common timeframe variants to Twelve Data format.
    Accepts 'H1', '1H', '1h' -> '1h' (hourly)
    Accepts 'H4', '4H', '4h' -> '4h' (4-hour)
    Accepts 'D1', '1D', '1d' -> '1day' (daily)
    Accepts '15m', '15M', 'M15' -> '15min' (for day trading)
    Extend here if you decide to support other frames later.
    """
    if not tf:
        return "1h"
    tf_up = tf.strip().upper()
    if tf_up in ("H1", "1H"):
        return "1h"
    elif tf_up in ("H4", "4H"):
        return "4h"
    elif tf_up in ("D1", "1D"):
        return "1day"
    elif tf_up in ("15M", "M15", "15MIN"):
        return "15min"
    elif tf.lower() == "15m":
        return "15min"
    return tf.lower()

def _norm_timeframe_for_filename(tf: str) -> str:
    """
    Normalize timeframe to filename token (e.g., 'H1', 'H4', 'D1', 'M15').
    """
    if not tf:
        return "H1"
    tf_up = tf.strip().upper()
    if tf_up in ("1H", "H1"):
        return "H1"
    elif tf_up in ("4H", "H4"):
        return "H4"
    elif tf_up in ("1D", "D1"):
        return "D1"
    elif tf_up in ("15M", "M15", "15MIN") or tf.lower() == "15m":
        return "M15"
    return tf_up

# -----------------------------------------------------------------------------
# HTTP: resilient async GET with backoff, jitter, and 429 handling
# -----------------------------------------------------------------------------
async def _http_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    retries: int = 4,
    backoff: float = 0.9,
) -> httpx.Response:
    """
    Async GET with retry/backoff.
    - Retries on 429/5xx/timeouts/transports.
    - Uses `Retry-After` header when present for 429.
    - Adds small jitter to prevent thundering herds.
    """
    attempt = 0
    last_exc: Optional[Exception] = None
    async with httpx.AsyncClient(timeout=timeout) as client:
        while attempt <= retries:
            try:
                resp = await client.get(url, params=params, headers=headers)
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                status = exc.response.status_code
                if attempt == retries:
                    break

                # 429 rate limit: respect Retry-After if present; otherwise exponential backoff
                delay = backoff * (attempt + 1)
                if status == 429:
                    retry_after = exc.response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            delay = max(delay, float(retry_after))
                        except ValueError:
                            pass
                    delay *= 1.5  # a bit extra on 429
                elif 500 <= status < 600:
                    delay *= 1.2  # mild bump for 5xx
                else:
                    # 4xx (non-429) usually shouldn't be retried
                    break

                # jitter
                delay += random.uniform(0.05, 0.25)
                await asyncio.sleep(delay)
                attempt += 1

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt == retries:
                    break
                delay = backoff * (attempt + 1) + random.uniform(0.05, 0.25)
                await asyncio.sleep(delay)
                attempt += 1

    assert last_exc is not None
    raise last_exc

# -----------------------------------------------------------------------------
# Data shaping
# -----------------------------------------------------------------------------
def _rows_to_df(rows: List[List[Any]]) -> pd.DataFrame:
    """
    Build a strict, clean DataFrame in ascending time:
      columns: time, open, high, low, close, volume
    - Ensure UTC tz-aware datetime
    - Drop dupes by time
    - Enforce numeric dtypes
    - Drop rows with missing OHLC
    - Floor timestamps to the hour boundary
    """
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    # Datetime & UTC
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])

    # Floor to hour boundary for H1 candles
    df["time"] = df["time"].dt.floor("h")

    # Deduplicate & sort
    df = df.drop_duplicates(subset=["time"], keep="last").sort_values("time").reset_index(drop=True)

    # Enforce numeric types
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop any incomplete OHLC rows
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Sanity: remove obviously bad rows (negative/zero prices)
    df = df[(df["open"]  > 0) & (df["high"] > 0) & (df["low"]  > 0) & (df["close"] > 0)]

    # Ensure high/low ordering (rare provider oddities)
    bad = df["low"] > df["high"]
    if bad.any():
        # swap where needed
        lo = df.loc[bad, "high"].copy()
        hi = df.loc[bad, "low"].copy()
        df.loc[bad, "low"] = lo
        df.loc[bad, "high"] = hi

    # Set time column as index for confluence system compatibility
    # Keep time column for save_csv compatibility
    df = df.set_index("time", drop=False)

    return df

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
async def fetch_h1(symbol: str, timeframe: str = "H1") -> pd.DataFrame:
    """
    Unified data fetcher using OANDA for FX/metals pairs and TwelveData for indices only.
    Accepts:
      - FX/metals pairs like 'EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD' (uses OANDA exclusively)
      - Indices via friendly names like 'SPX500', 'US30', 'NAS100' (uses TwelveData)

    Returns DataFrame with columns:
      time (UTC), open, high, low, close, volume
    in ascending chronological order (excludes the current/open bar).
    """
    if not symbol:
        raise ValueError("fetch_h1(symbol): symbol is required")

    s = symbol.upper().strip()

    # All FX/metals pairs use OANDA exclusively
    if s == "XAUUSD":
        from .oanda_feed import get_xauusd_candles
        return await get_xauusd_candles(count=1000)

    if s == "GBPUSD":
        from .oanda_feed import get_gbpusd_candles
        return await get_gbpusd_candles(count=1000)

    # Check if it's a 6-letter FX pair (EURUSD, USDJPY, etc.) - route to OANDA
    if len(s) == 6 and s.isalpha() and not _is_index(s):
        from .oanda_feed import get_current_price
        # For other FX pairs, we'd need to add similar functions to oanda_feed.py
        # For now, raise an error indicating we need OANDA implementation
        raise NotImplementedError(f"OANDA implementation needed for {s}. Currently only XAUUSD and GBPUSD are implemented.")

    # Non-FX index path (use TwelveData)
    if _is_index(s):
        key = _get_twelve_key()
        td_interval = _norm_timeframe_for_api(timeframe)
        td_sym = INDEX_MAP[s]  # e.g., 'SPY' for SPX500
        df = await _fetch_h1_twelve_symbol(td_sym, api_key=key, interval=td_interval)

        # Apply scaling for index proxies (e.g., SPY ETF -> S&P 500 Index approximation)
        if s in INDEX_SCALING and not df.empty:
            scale_factor = INDEX_SCALING[s]
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col] * scale_factor

        return df

    raise ValueError(f"Unsupported symbol: {s}. Supported: XAUUSD, GBPUSD, or indices like SPX500, US30, NAS100")

def save_csv(df: pd.DataFrame, symbol: str, timeframe: str = "H1") -> str:
    """
    Save dataframe to data/{SYMBOL}_{TIMEFRAME}.csv
    Works for both FX (EURUSD) and indices (SPX500).
    Ensures the standard column order and dtypes on write.
    """
    if df is None or df.empty:
        raise ValueError("save_csv(): DataFrame is empty")

    # Keep only expected columns & order
    cols = ["time", "open", "high", "low", "close", "volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"save_csv(): missing columns {missing}")

    # Uppercase + alphanumeric only for file name
    safe = re.sub(r"[^A-Za-z0-9]", "", (symbol or "")).upper()
    if not safe:
        raise ValueError("Invalid symbol for save_csv()")

    tf_token = _norm_timeframe_for_filename(timeframe)

    os.makedirs("data", exist_ok=True)
    path = f"data/{safe}_{tf_token}.csv"

    out = df.copy()
    # Ensure tz-aware UTC then write as ISO8601 Z
    if not pd.api.types.is_datetime64_any_dtype(out["time"]):
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
    if out["time"].dt.tz is None:
        out["time"] = out["time"].dt.tz_localize("UTC")
    else:
        out["time"] = out["time"].dt.tz_convert("UTC")
    out["time"] = out["time"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    out.to_csv(path, index=False)
    return path

# -----------------------------------------------------------------------------
# Twelve Data (shared symbol fetcher)
# -----------------------------------------------------------------------------
async def _fetch_h1_twelve_symbol(td_symbol: str, api_key: str, interval: str = "1h") -> pd.DataFrame:
    """
    Fetch candles from Twelve Data for an explicit symbol.
    Examples:
      - 'EUR/USD', 'XAU/USD' (FX/metals)
      - 'SPX', 'DJI', 'NDX' (indices)
    """
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": td_symbol,
        "interval": interval,     # was '1h'; now parameterized (still defaults to 1h)
        "outputsize": "1000",     # reduced for better freshness - still plenty for analysis
        "timezone": "UTC",
        "format": "JSON",
        "apikey": api_key,
    }

    r = await _http_get(url, params=params)
    data = r.json()

    status = (data.get("status") or "").lower()
    if status != "ok":
        msg = data.get("message") or f"Unexpected response keys: {list(data.keys())[:6]}"
        raise RuntimeError(f"TwelveData error: {msg}")

    values = data.get("values") or []
    if not values:
        raise RuntimeError(f"TwelveData returned no values for symbol {td_symbol}")

    # Twelve Data returns latest-first; convert to ascending
    rows: List[List[Any]] = []
    for v in values:
        # Defensive parsing
        dt = pd.to_datetime(v.get("datetime"), utc=True, errors="coerce")
        o  = v.get("open")
        h  = v.get("high")
        l  = v.get("low")
        c  = v.get("close")
        vol = v.get("volume", 0.0)

        if dt is None or pd.isna(dt) or o is None or h is None or l is None or c is None:
            # skip malformed rows
            continue

        rows.append([
            dt,
            float(o),
            float(h),
            float(l),
            float(c),
            float(vol if vol is not None else 0.0),
        ])

    if not rows:
        raise RuntimeError(f"TwelveData returned only malformed rows for {td_symbol}")

    df = _rows_to_df(rows)

    # Drop any rows that are in the future or the current open bar (clock skew safety)
    now_utc = pd.Timestamp.utcnow().floor("h")
    df = df[df["time"] < now_utc]  # exclude the current (open) hour

    return df
