#!/usr/bin/env python3
"""
XAUUSD Trading Backend - v3.0
Simplified system focused exclusively on XAUUSD psychological level confluence
Based on 98% S/R success rate analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import asyncio
from typing import Dict, Any, Optional
import pytz

from .xauusd_confluence import evaluate_xauusd_signal, get_xauusd_status
from .datafeed import fetch_h1

# Scheduler for automated scanning
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

app = FastAPI(title="XAUUSD Trading System - v3.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for XAUUSD signals
LAST_XAUUSD_SIGNAL: Optional[Dict[str, Any]] = None
LAST_XAUUSD_TIMESTAMP: Optional[str] = None
LAST_XAUUSD_SKIP: Optional[Dict[str, Any]] = None
LAST_SCAN_TIME: Optional[str] = None

_SCHEDULER: Optional[BackgroundScheduler] = None

# JSON serialization helper
def to_jsonable(obj: Any) -> Any:
    """Convert objects to JSON-serializable format"""
    if hasattr(obj, 'item'):  # numpy types
        return obj.item()
    elif hasattr(obj, 'isoformat'):  # datetime
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_jsonable(item) for item in obj]
    return obj

async def scan_xauusd() -> Dict[str, Any]:
    """Scan XAUUSD for trading signals"""
    global LAST_XAUUSD_SIGNAL, LAST_XAUUSD_TIMESTAMP, LAST_XAUUSD_SKIP, LAST_SCAN_TIME

    try:
        # Fetch data
        df = await fetch_h1("XAUUSD", timeframe="H1")

        if df is None or df.empty:
            error_result = {
                "error": "No data available for XAUUSD",
                "timestamp": datetime.now(pytz.UTC).isoformat()
            }
            LAST_XAUUSD_SKIP = error_result
            return error_result

        # Evaluate signal
        result = evaluate_xauusd_signal(df, debug=True)

        utc_now = datetime.now(pytz.UTC)
        LAST_SCAN_TIME = utc_now.isoformat()

        if result is None:
            error_result = {
                "error": "Signal evaluation failed",
                "timestamp": utc_now.isoformat()
            }
            LAST_XAUUSD_SKIP = error_result
            return error_result

        if result.get("signal") == "SKIP":
            LAST_XAUUSD_SKIP = result
            LAST_XAUUSD_SKIP["timestamp"] = utc_now.isoformat()
            return {"skip_reason": result.get("skip_reason"), "context": result.get("context")}

        # Valid signal found
        LAST_XAUUSD_SIGNAL = to_jsonable(result)
        LAST_XAUUSD_TIMESTAMP = utc_now.isoformat()
        LAST_XAUUSD_SKIP = None

        return {"signal_generated": True, "signal": LAST_XAUUSD_SIGNAL}

    except Exception as e:
        error_result = {
            "error": f"Scan failed: {str(e)}",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }
        LAST_XAUUSD_SKIP = error_result
        return error_result

# API Endpoints

@app.get("/api/xauusd/signal/latest")
async def get_latest_xauusd_signal():
    """Get the latest XAUUSD signal"""
    if LAST_XAUUSD_SIGNAL:
        return JSONResponse({
            "signal": LAST_XAUUSD_SIGNAL,
            "timestamp": LAST_XAUUSD_TIMESTAMP,
            "scan_time": LAST_SCAN_TIME
        })
    elif LAST_XAUUSD_SKIP:
        return JSONResponse({
            "skip_reason": LAST_XAUUSD_SKIP.get("skip_reason", "No signal"),
            "context": LAST_XAUUSD_SKIP.get("context", ""),
            "timestamp": LAST_XAUUSD_SKIP.get("timestamp"),
            "scan_time": LAST_SCAN_TIME
        })
    else:
        return JSONResponse({
            "message": "No scan performed yet",
            "timestamp": None
        })

@app.get("/api/xauusd/status")
async def get_xauusd_status_endpoint():
    """Get current XAUUSD market status"""
    try:
        status = get_xauusd_status()
        return JSONResponse(to_jsonable(status))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/xauusd/scan")
async def manual_xauusd_scan():
    """Manually trigger XAUUSD scan"""
    result = await scan_xauusd()
    return JSONResponse(to_jsonable(result))

@app.get("/api/xauusd/levels")
async def get_psychological_levels():
    """Get current psychological levels for XAUUSD"""
    try:
        from .xauusd_confluence import get_psychological_levels, calculate_distance_to_level

        # Get current price
        df = await fetch_h1("XAUUSD", timeframe="H1")
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No price data available")

        current_price = df['close'].iloc[-1]
        levels = get_psychological_levels(current_price, 300)

        level_data = []
        for level in levels:
            distance = calculate_distance_to_level(current_price, level)
            level_data.append({
                "level": level,
                "distance_pips": round(distance["distance_pips"], 1),
                "distance_percent": round(distance["distance_percent"], 2),
                "direction": distance["direction"],
                "is_major": level in [1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400]
            })

        return JSONResponse({
            "current_price": round(current_price, 2),
            "levels": sorted(level_data, key=lambda x: x["distance_pips"]),
            "timestamp": datetime.now(pytz.UTC).isoformat()
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/xauusd/data")
async def get_xauusd_data():
    """Get recent XAUUSD price data"""
    try:
        df = await fetch_h1("XAUUSD", timeframe="H1")
        if df is None or df.empty:
            raise HTTPException(status_code=400, detail="No data available")

        # Return last 24 hours of data
        recent_data = df.tail(24)

        data = {
            "symbol": "XAUUSD",
            "timeframe": "H1",
            "candles": [
                {
                    "time": candle["time"].isoformat() if hasattr(candle["time"], 'isoformat') else str(candle["time"]),
                    "open": round(candle["open"], 2),
                    "high": round(candle["high"], 2),
                    "low": round(candle["low"], 2),
                    "close": round(candle["close"], 2),
                    "volume": candle["volume"]
                }
                for _, candle in recent_data.iterrows()
            ],
            "current_price": round(df['close'].iloc[-1], 2),
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }

        return JSONResponse(data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "XAUUSD Trading System",
        "version": "3.0",
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "last_scan": LAST_SCAN_TIME
    }

# Scheduler functions
def scheduled_scan():
    """Scheduled scan function for background scheduler"""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(scan_xauusd())
    except RuntimeError:
        # If no event loop, create one
        asyncio.run(scan_xauusd())

def start_scheduler():
    """Start the background scheduler"""
    global _SCHEDULER

    if _SCHEDULER is not None:
        return

    _SCHEDULER = BackgroundScheduler(timezone=pytz.UTC)

    # Scan every hour at 5 minutes past (when new H1 candle is formed)
    _SCHEDULER.add_job(
        scheduled_scan,
        trigger=CronTrigger(minute=5),
        id="xauusd_hourly_scan",
        replace_existing=True
    )

    # Additional scan during high-volatility London-NY overlap (13:00-17:00 UTC)
    _SCHEDULER.add_job(
        scheduled_scan,
        trigger=CronTrigger(hour="13-17", minute="5,35"),  # Every 30 minutes during overlap
        id="xauusd_overlap_scan",
        replace_existing=True
    )

    _SCHEDULER.start()

@app.on_event("startup")
async def startup_event():
    """Initialize scheduler on startup"""
    start_scheduler()
    # Perform initial scan
    await scan_xauusd()

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    global _SCHEDULER
    if _SCHEDULER:
        _SCHEDULER.shutdown()
        _SCHEDULER = None