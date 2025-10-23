#!/usr/bin/env python3
"""
Multi-Pair Trading Backend - v4.0
Dual trading system: XAUUSD + GBPUSD Smart Confluence
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
import pytz
import asyncio

# Import both systems
from .smart_confluence_system import (
    evaluate_smart_confluence_signal as evaluate_xauusd_signal,
    get_confluence_system_status as get_xauusd_status
)
from .gbpusd_confluence_system import (
    evaluate_gbpusd_confluence_signal,
    get_gbpusd_system_status
)
from .datafeed import fetch_h1
from .current_price import get_current_xauusd_price
from .oanda_feed import get_current_price as get_current_gbpusd_price
from .pro_trader_gold import get_pro_trader_analysis

# Scheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

app = FastAPI(title="Multi-Pair Trading System - XAUUSD + GBPUSD")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DUAL CACHE SYSTEM - One for each pair
class MultiPairDataCache:
    def __init__(self):
        self.xauusd_data = {
            'df': None,
            'last_fetch': None,
            'current_price': None,
            'signal': None,
            'skip_reason': None
        }
        self.gbpusd_data = {
            'df': None,
            'last_fetch': None,
            'current_price': None,
            'signal': None,
            'skip_reason': None
        }
        self.cache_duration = timedelta(minutes=30)  # Longer cache for hourly-based signals

    def is_cache_valid(self, pair: str) -> bool:
        data = self.xauusd_data if pair == 'XAUUSD' else self.gbpusd_data
        if data['last_fetch'] is None:
            return False
        return datetime.now(pytz.UTC) - data['last_fetch'] < self.cache_duration

    async def get_xauusd_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Get XAUUSD data with caching"""
        if not force_refresh and self.is_cache_valid('XAUUSD') and self.xauusd_data['df'] is not None:
            return self.xauusd_data['df']

        # Fresh fetch
        self.xauusd_data['df'] = await fetch_h1("XAUUSD", timeframe="H1")
        self.xauusd_data['last_fetch'] = datetime.now(pytz.UTC)

        # Get real-time price
        try:
            self.xauusd_data['current_price'] = await get_current_xauusd_price()
        except Exception as e:
            print(f"Warning: Failed to get XAUUSD real-time price: {e}")
            if self.xauusd_data['df'] is not None and not self.xauusd_data['df'].empty:
                self.xauusd_data['current_price'] = self.xauusd_data['df']['close'].iloc[-1]

        return self.xauusd_data['df']

    async def get_gbpusd_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Get GBPUSD data with caching"""
        if not force_refresh and self.is_cache_valid('GBPUSD') and self.gbpusd_data['df'] is not None:
            return self.gbpusd_data['df']

        # Fresh fetch using OANDA (same as XAUUSD)
        self.gbpusd_data['df'] = await fetch_h1("GBPUSD", timeframe="H1")
        self.gbpusd_data['last_fetch'] = datetime.now(pytz.UTC)

        # Get real-time price
        try:
            self.gbpusd_data['current_price'] = await get_current_gbpusd_price("GBP_USD")
        except Exception as e:
            print(f"Warning: Failed to get GBPUSD real-time price: {e}")
            if self.gbpusd_data['df'] is not None and not self.gbpusd_data['df'].empty:
                self.gbpusd_data['current_price'] = self.gbpusd_data['df']['close'].iloc[-1]

        return self.gbpusd_data['df']

    def get_current_price(self, pair: str) -> Optional[float]:
        if pair == 'XAUUSD':
            return self.xauusd_data['current_price']
        else:
            return self.gbpusd_data['current_price']

# Global cache instance
cache = MultiPairDataCache()
_SCHEDULER: Optional[BackgroundScheduler] = None

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

async def analyze_xauusd(force_refresh: bool = False) -> Dict[str, Any]:
    """Analyze XAUUSD using proven Smart Confluence System"""
    try:
        df = await cache.get_xauusd_data(force_refresh=force_refresh)

        if df is None or df.empty:
            return {
                "status": "error",
                "pair": "XAUUSD",
                "data": {"error": "No XAUUSD data available"}
            }

        current_price = cache.get_current_price('XAUUSD')
        signal_result = evaluate_xauusd_signal(current_price, df)
        utc_now = datetime.now(pytz.UTC)

        if signal_result is None or signal_result.get("signal") in ["SKIP", "WAIT"]:
            cache.xauusd_data['skip_reason'] = {
                "skip_reason": signal_result.get("skip_reason", "No signal") if signal_result else "Analysis failed",
                "context": signal_result.get("context", "") if signal_result else "",
                "pair": "XAUUSD"
            }
            cache.xauusd_data['signal'] = None

            return {
                "status": "no_signal",
                "pair": "XAUUSD",
                "data": {
                    "skip_info": cache.xauusd_data['skip_reason'],
                    "market_data": {
                        "current_price": round(current_price, 2),
                        "session": {"current_session": "xauusd_confluence", "session_strength": "medium"},
                        "timestamp": utc_now.isoformat()
                    },
                    "system_status": get_xauusd_status(),
                    "last_update": cache.xauusd_data['last_fetch'].isoformat() if cache.xauusd_data['last_fetch'] else None
                }
            }

        # Valid signal found
        cache.xauusd_data['signal'] = to_jsonable(signal_result)
        cache.xauusd_data['skip_reason'] = None

        return {
            "status": "signal",
            "pair": "XAUUSD",
            "data": {
                "signal": cache.xauusd_data['signal'],
                "market_data": {
                    "current_price": round(current_price, 2),
                    "session": signal_result.get("session_info", {"current_session": "xauusd_confluence"}),
                    "timestamp": utc_now.isoformat()
                },
                "system_status": get_xauusd_status(),
                "last_update": cache.xauusd_data['last_fetch'].isoformat()
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "pair": "XAUUSD",
            "data": {"error": f"XAUUSD analysis failed: {str(e)}"}
        }

async def analyze_gbpusd(force_refresh: bool = False) -> Dict[str, Any]:
    """Analyze GBPUSD using research-based Smart Confluence System"""
    try:
        df = await cache.get_gbpusd_data(force_refresh=force_refresh)

        if df is None or df.empty:
            return {
                "status": "error",
                "pair": "GBPUSD",
                "data": {"error": "No GBPUSD data available"}
            }

        current_price = cache.get_current_price('GBPUSD')
        signal_result = evaluate_gbpusd_confluence_signal(current_price, df)
        utc_now = datetime.now(pytz.UTC)

        if signal_result.get("status") != "signal":
            cache.gbpusd_data['skip_reason'] = {
                "skip_reason": signal_result.get("skip_reason", "No signal"),
                "context": signal_result.get("context", ""),
                "pair": "GBPUSD"
            }
            cache.gbpusd_data['signal'] = None

            return {
                "status": "no_signal",
                "pair": "GBPUSD",
                "data": {
                    "skip_info": cache.gbpusd_data['skip_reason'],
                    "market_data": {
                        "current_price": round(current_price, 5),  # GBPUSD needs more precision
                        "session": {"current_session": "gbpusd_confluence", "session_strength": "medium"},
                        "timestamp": utc_now.isoformat()
                    },
                    "system_status": get_gbpusd_system_status(),
                    "confluence_analysis": signal_result.get("confluence_analysis"),
                    "last_update": cache.gbpusd_data['last_fetch'].isoformat() if cache.gbpusd_data['last_fetch'] else None
                }
            }

        # Valid signal found
        cache.gbpusd_data['signal'] = to_jsonable(signal_result.get("signal"))
        cache.gbpusd_data['skip_reason'] = None

        return {
            "status": "signal",
            "pair": "GBPUSD",
            "data": {
                "signal": cache.gbpusd_data['signal'],
                "market_data": {
                    "current_price": round(current_price, 5),
                    "session": cache.gbpusd_data['signal'].get("gbpusd_session_info", {"current_session": "gbpusd_confluence"}),
                    "timestamp": utc_now.isoformat()
                },
                "system_status": get_gbpusd_system_status(),
                "last_update": cache.gbpusd_data['last_fetch'].isoformat()
            }
        }

    except Exception as e:
        return {
            "status": "error",
            "pair": "GBPUSD",
            "data": {"error": f"GBPUSD analysis failed: {str(e)}"}
        }

# API ENDPOINTS

@app.get("/api/xauusd/analysis")
async def get_xauusd_analysis():
    """Get XAUUSD Smart Confluence analysis"""
    result = await analyze_xauusd()
    return JSONResponse(result)

@app.get("/api/gbpusd/analysis")
async def get_gbpusd_analysis():
    """Get GBPUSD Smart Confluence analysis"""
    result = await analyze_gbpusd()
    return JSONResponse(result)

@app.get("/api/multi-pair/analysis")
async def get_multi_pair_analysis():
    """Get analysis for both XAUUSD and GBPUSD"""
    xauusd_result = await analyze_xauusd()
    gbpusd_result = await analyze_gbpusd()

    # Combine results
    combined_result = {
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "pairs": {
            "XAUUSD": xauusd_result,
            "GBPUSD": gbpusd_result
        },
        "summary": {
            "total_signals": 0,
            "active_pairs": []
        }
    }

    # Count active signals
    if xauusd_result.get("status") == "signal":
        combined_result["summary"]["total_signals"] += 1
        combined_result["summary"]["active_pairs"].append("XAUUSD")

    if gbpusd_result.get("status") == "signal":
        combined_result["summary"]["total_signals"] += 1
        combined_result["summary"]["active_pairs"].append("GBPUSD")

    return JSONResponse(combined_result)

@app.post("/api/xauusd/scan")
async def scan_xauusd():
    """Force refresh XAUUSD analysis"""
    result = await analyze_xauusd(force_refresh=True)
    return JSONResponse(result)

@app.post("/api/gbpusd/scan")
async def scan_gbpusd():
    """Force refresh GBPUSD analysis"""
    result = await analyze_gbpusd(force_refresh=True)
    return JSONResponse(result)

@app.post("/api/multi-pair/scan")
async def scan_all_pairs():
    """Force refresh analysis for both pairs"""
    xauusd_result = await analyze_xauusd(force_refresh=True)
    gbpusd_result = await analyze_gbpusd(force_refresh=True)

    return JSONResponse({
        "timestamp": datetime.now(pytz.UTC).isoformat(),
        "pairs": {
            "XAUUSD": xauusd_result,
            "GBPUSD": gbpusd_result
        },
        "message": "Both pairs refreshed successfully"
    })

@app.get("/api/system/status")
async def get_system_status():
    """Get overall system status"""
    return JSONResponse({
        "status": "healthy",
        "service": "Multi-Pair Smart Confluence Trading System",
        "version": "4.0-multi-pair",
        "pairs": ["XAUUSD", "GBPUSD"],
        "systems": {
            "XAUUSD": get_xauusd_status(),
            "GBPUSD": get_gbpusd_system_status()
        },
        "cache_status": {
            "XAUUSD": "valid" if cache.is_cache_valid('XAUUSD') else "stale",
            "GBPUSD": "valid" if cache.is_cache_valid('GBPUSD') else "stale"
        },
        "data_sources": ["OANDA API"],
        "timestamp": datetime.now(pytz.UTC).isoformat()
    })

@app.get("/api/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "service": "Multi-Pair Trading System",
        "pairs_active": ["XAUUSD", "GBPUSD"],
        "version": "4.0",
        "timestamp": datetime.now(pytz.UTC).isoformat()
    }

# PRO TRADER GOLD ENDPOINTS
@app.get("/api/pro-trader-gold/analysis")
async def get_pro_trader_gold_analysis():
    """
    Get Professional Trader Gold analysis
    Educational setup tracker with step-by-step breakdown
    """
    try:
        result = get_pro_trader_analysis()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Pro Trader analysis failed: {str(e)}",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }, status_code=500)

@app.post("/api/pro-trader-gold/scan")
async def scan_pro_trader_gold():
    """Force refresh Pro Trader Gold analysis"""
    try:
        result = get_pro_trader_analysis()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Pro Trader scan failed: {str(e)}",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }, status_code=500)

# SCHEDULER
def scheduled_analysis():
    """Scheduled analysis for both pairs"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Analyze both pairs
        loop.run_until_complete(analyze_xauusd(force_refresh=True))
        loop.run_until_complete(analyze_gbpusd(force_refresh=True))

        loop.close()
        print(f"✅ Multi-pair scheduled analysis completed at {datetime.now()}")
    except Exception as e:
        print(f"❌ Multi-pair scheduled analysis error: {e}")

def start_scheduler():
    """Start scheduler for both pairs"""
    global _SCHEDULER

    if _SCHEDULER is not None:
        return

    _SCHEDULER = BackgroundScheduler(timezone=pytz.UTC)

    # Check both pairs every hour at :05
    _SCHEDULER.add_job(
        scheduled_analysis,
        trigger=CronTrigger(minute="5"),
        id="multi_pair_hourly_check",
        replace_existing=True
    )

    # Additional mid-hour check
    _SCHEDULER.add_job(
        scheduled_analysis,
        trigger=CronTrigger(minute="35"),
        id="multi_pair_mid_hour_check",
        replace_existing=True
    )

    _SCHEDULER.start()
    print("✅ Multi-pair scheduler started")

@app.on_event("startup")
async def startup_event():
    """Initialize multi-pair system"""
    start_scheduler()
    # Perform initial analysis for both pairs
    await analyze_xauusd(force_refresh=True)
    await analyze_gbpusd(force_refresh=True)
    print("🚀 Multi-Pair Trading System initialized: XAUUSD + GBPUSD")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean shutdown"""
    global _SCHEDULER
    if _SCHEDULER:
        _SCHEDULER.shutdown()
        _SCHEDULER = None
    print("🛑 Multi-pair system shutdown complete")