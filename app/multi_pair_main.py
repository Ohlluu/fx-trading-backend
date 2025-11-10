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
from .bullish_pro_trader_gold import get_bullish_pro_trader_analysis
from .bearish_pro_trader_gold import get_bearish_pro_trader_analysis
from .bullish_pro_trader_eurusd import get_bullish_pro_trader_analysis as get_bullish_eurusd_analysis
from .bearish_pro_trader_eurusd import get_bearish_pro_trader_analysis as get_bearish_eurusd_analysis
from .trade_manager import trade_manager
from .telegram_notifier import notifier, send_setup_notification
from pydantic import BaseModel

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

async def check_and_notify_setups(bullish_data: Dict, bearish_data: Dict):
    """
    Check confluence scores and send Telegram notifications if threshold is met
    Threshold: 7+ points for high-quality setups
    """
    threshold = 7  # Minimum score to trigger notification

    try:
        # Check bullish setup
        bullish_score = bullish_data.get('total_score', 0)
        if bullish_score >= threshold:
            await send_setup_notification("BULLISH", bullish_data)

        # Check bearish setup
        bearish_score = bearish_data.get('total_score', 0)
        if bearish_score >= threshold:
            await send_setup_notification("BEARISH", bearish_data)

    except Exception as e:
        print(f"⚠️ Notification check error: {e}")

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
    Get BOTH Bullish and Bearish Professional Trader Gold analysis
    Returns both BUY setups and SELL setups in one response
    Also checks for high-quality setups and sends Telegram notifications
    """
    try:
        # Run both analyses in parallel
        bullish_result, bearish_result = await asyncio.gather(
            get_bullish_pro_trader_analysis(),
            get_bearish_pro_trader_analysis()
        )

        # Check if either setup meets notification threshold (7+ points)
        await check_and_notify_setups(bullish_result, bearish_result)

        return JSONResponse({
            "bullish": bullish_result,
            "bearish": bearish_result,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Pro Trader analysis failed: {str(e)}",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }, status_code=500)

@app.get("/api/pro-trader-gold/bullish")
async def get_bullish_only():
    """Get BULLISH (BUY) setups only"""
    try:
        result = await get_bullish_pro_trader_analysis()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Bullish analysis failed: {str(e)}",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }, status_code=500)

@app.get("/api/pro-trader-gold/bearish")
async def get_bearish_only():
    """Get BEARISH (SELL) setups only"""
    try:
        result = await get_bearish_pro_trader_analysis()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Bearish analysis failed: {str(e)}",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }, status_code=500)

@app.post("/api/pro-trader-gold/scan")
async def scan_pro_trader_gold():
    """Force refresh BOTH Bullish and Bearish analysis"""
    try:
        bullish_result, bearish_result = await asyncio.gather(
            get_bullish_pro_trader_analysis(),
            get_bearish_pro_trader_analysis()
        )

        return JSONResponse({
            "bullish": bullish_result,
            "bearish": bearish_result,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Pro Trader scan failed: {str(e)}",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }, status_code=500)

# PRO TRADER EUR/USD ENDPOINTS
@app.get("/api/pro-trader-eurusd/analysis")
async def get_pro_trader_eurusd_analysis():
    """
    Get BOTH Bullish and Bearish Professional Trader EUR/USD analysis
    Returns both BUY setups and SELL setups in one response
    """
    try:
        # Run both analyses in parallel
        bullish_result, bearish_result = await asyncio.gather(
            get_bullish_eurusd_analysis(),
            get_bearish_eurusd_analysis()
        )

        return JSONResponse({
            "bullish": bullish_result,
            "bearish": bearish_result,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Pro Trader EUR/USD analysis failed: {str(e)}",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }, status_code=500)

@app.get("/api/pro-trader-eurusd/bullish")
async def get_eurusd_bullish_only():
    """Get BULLISH (BUY) EUR/USD setups only"""
    try:
        result = await get_bullish_eurusd_analysis()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Bullish EUR/USD analysis failed: {str(e)}",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }, status_code=500)

@app.get("/api/pro-trader-eurusd/bearish")
async def get_eurusd_bearish_only():
    """Get BEARISH (SELL) EUR/USD setups only"""
    try:
        result = await get_bearish_eurusd_analysis()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Bearish EUR/USD analysis failed: {str(e)}",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }, status_code=500)

@app.post("/api/pro-trader-eurusd/scan")
async def scan_pro_trader_eurusd():
    """Force refresh BOTH Bullish and Bearish EUR/USD analysis"""
    try:
        bullish_result, bearish_result = await asyncio.gather(
            get_bullish_eurusd_analysis(),
            get_bearish_eurusd_analysis()
        )

        return JSONResponse({
            "bullish": bullish_result,
            "bearish": bearish_result,
            "timestamp": datetime.now(pytz.UTC).isoformat()
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": f"Pro Trader EUR/USD scan failed: {str(e)}",
            "timestamp": datetime.now(pytz.UTC).isoformat()
        }, status_code=500)

# TRADE MANAGEMENT MODELS
class EnterTradeRequest(BaseModel):
    entry_price: float
    position_size: int  # 50 or 100
    stop_loss: float
    take_profit_1: float
    take_profit_2: Optional[float] = None
    trade_direction: str = "LONG"

class ExitTradeRequest(BaseModel):
    exit_price: float
    position_size: int  # 50 or 100
    reason: str = "Manual Exit"

class TelegramSettingsRequest(BaseModel):
    chat_id: str
    bot_token: Optional[str] = None
    enabled: bool = True

# TRADE MANAGEMENT ENDPOINTS
@app.post("/api/pro-trader-gold/enter-trade")
async def enter_trade(request: EnterTradeRequest):
    """Record that user entered a trade"""
    try:
        result = trade_manager.enter_trade(
            entry_price=request.entry_price,
            position_size=request.position_size,
            stop_loss=request.stop_loss,
            take_profit_1=request.take_profit_1,
            take_profit_2=request.take_profit_2,
            trade_direction=request.trade_direction
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/api/pro-trader-gold/exit-trade")
async def exit_trade(request: ExitTradeRequest):
    """Record that user exited a trade"""
    try:
        result = trade_manager.exit_trade(
            exit_price=request.exit_price,
            position_size=request.position_size,
            reason=request.reason
        )
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/pro-trader-gold/trade-status")
async def get_trade_status():
    """Get current trade status with P&L and alerts"""
    try:
        # Get current price
        current_price = await get_current_xauusd_price()

        # Get trade status
        trade_status = trade_manager.get_trade_status(current_price)

        # If in trade, get alerts
        if trade_status['in_trade']:
            # Get current candle data for alerts
            h1_data = await fetch_h1("XAUUSD", timeframe="H1")
            last_candle = h1_data.iloc[-1] if not h1_data.empty else None

            current_candle = {
                'open': float(last_candle['open']) if last_candle is not None else current_price,
                'high': float(last_candle['high']) if last_candle is not None else current_price,
                'low': float(last_candle['low']) if last_candle is not None else current_price,
                'close': float(last_candle['close']) if last_candle is not None else current_price
            }

            alerts = trade_manager.get_trade_alerts(current_price, current_candle)
            trade_status['alerts'] = alerts

        return JSONResponse(trade_status)
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "error": str(e)
        }, status_code=500)

@app.get("/api/pro-trader-gold/trade-history")
async def get_trade_history():
    """Get trade history"""
    try:
        history = trade_manager.get_trade_history(limit=20)
        stats = trade_manager.get_statistics()

        return JSONResponse({
            "history": history,
            "statistics": stats
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "error": str(e)
        }, status_code=500)

# TELEGRAM SETTINGS ENDPOINTS
@app.post("/api/settings/telegram")
async def update_telegram_settings(request: TelegramSettingsRequest):
    """
    Update Telegram notification settings
    User provides their chat ID and optionally a bot token
    """
    try:
        # Update notifier settings
        if request.bot_token:
            notifier.bot_token = request.bot_token
            notifier.base_url = f"https://api.telegram.org/bot{request.bot_token}"

        notifier.chat_id = request.chat_id

        # Verify it works by sending a test message
        if request.enabled:
            success = await notifier.send_test_message()
            if not success:
                return JSONResponse({
                    "success": False,
                    "error": "Failed to send test message. Check your chat ID and bot token."
                }, status_code=400)

        return JSONResponse({
            "success": True,
            "message": "Telegram notifications configured successfully!",
            "chat_id": request.chat_id,
            "enabled": request.enabled
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/api/settings/telegram")
async def get_telegram_settings():
    """Get current Telegram notification settings"""
    return JSONResponse({
        "configured": notifier.is_configured(),
        "chat_id": notifier.chat_id if notifier.chat_id else None,
        "enabled": notifier.is_configured()
    })

@app.post("/api/settings/telegram/test")
async def test_telegram_notification():
    """Send a test notification to verify setup"""
    try:
        if not notifier.is_configured():
            return JSONResponse({
                "success": False,
                "error": "Telegram not configured. Please set chat ID first."
            }, status_code=400)

        success = await notifier.send_test_message()

        if success:
            return JSONResponse({
                "success": True,
                "message": "Test notification sent successfully!"
            })
        else:
            return JSONResponse({
                "success": False,
                "error": "Failed to send test notification. Check your configuration."
            }, status_code=400)
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
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