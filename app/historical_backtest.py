# backend/app/historical_backtest.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

from .datafeed import fetch_h1, save_csv
from .confluence import (
    evaluate_last_closed_bar, _cfg_for, _pip_size_for,
    _ensure_utc_index, _is_dtindex, _calc_indicators_h1,
    _resample_ohlc, _calc_indicators_htf, _align_htf_to_h1,
    _prev_day_levels, _weekly_pivots, _session_block,
    _in_core_session, _pivot_highs_lows, _last_swing_price,
    PAIR_CONFIG
)
from .win_mode import evaluate_last_closed_bar_win

# Target pairs for historical analysis (using existing data + indices when available)
TARGET_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "XAUUSD", "SPX500"]
TARGET_INDICES = ["NAS100", "DE30", "UK100", "SPX500"]

class HistoricalBacktester:
    def __init__(self, lookback_months: int = 6):
        self.lookback_months = lookback_months
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        self.performance_stats: Dict[str, Dict[str, float]] = {}

    async def fetch_historical_data(self, pair: str) -> pd.DataFrame:
        """Fetch and save historical data for a pair"""
        print(f"Fetching historical data for {pair}...")
        try:
            df = await fetch_h1(pair, timeframe="H1")
            if not df.empty:
                save_csv(df, pair, "H1")
                print(f"✓ {pair}: {len(df)} H1 candles saved")
            return df
        except Exception as e:
            print(f"✗ {pair}: Failed to fetch data - {e}")
            return pd.DataFrame()

    def load_existing_data(self, pair: str) -> pd.DataFrame:
        """Load existing CSV data if available"""
        csv_path = f"data/{pair}_H1.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df["time"] = pd.to_datetime(df["time"], utc=True)
            df = df.set_index("time").sort_index()
            return df
        return pd.DataFrame()

    def simulate_trade(self, signal: Dict[str, Any], df: pd.DataFrame,
                      start_idx: int) -> Optional[Dict[str, Any]]:
        """Simulate a trade from signal to completion"""
        if not signal or signal.get("skip_reason"):
            return None

        entry_price = float(signal["price"])
        stop_loss = float(signal["stop"])
        target = float(signal["target"])
        side = signal["side"]
        entry_time = signal["time"]

        # Look forward up to 48 hours for trade completion
        max_hours = 48
        future_bars = df.iloc[start_idx+1:start_idx+1+max_hours]

        if future_bars.empty:
            return None

        # Track trade progress bar by bar
        for i, (ts, bar) in enumerate(future_bars.iterrows()):
            high, low = float(bar["high"]), float(bar["low"])

            if side == "BUY":
                # Check stop loss first (more conservative)
                if low <= stop_loss:
                    return {
                        "pair": signal["pair"],
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": stop_loss,
                        "result": "LOSS",
                        "duration_hours": i + 1,
                        "pips": (stop_loss - entry_price) / _pip_size_for(signal["pair"]),
                        "rr_actual": -1.0,
                        "signal_score": signal.get("score", 0)
                    }
                # Check target
                elif high >= target:
                    pips_won = (target - entry_price) / _pip_size_for(signal["pair"])
                    return {
                        "pair": signal["pair"],
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": target,
                        "result": "WIN",
                        "duration_hours": i + 1,
                        "pips": pips_won,
                        "rr_actual": float(signal.get("rr", 0)),
                        "signal_score": signal.get("score", 0)
                    }
            else:  # SELL
                # Check stop loss first
                if high >= stop_loss:
                    return {
                        "pair": signal["pair"],
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": stop_loss,
                        "result": "LOSS",
                        "duration_hours": i + 1,
                        "pips": (entry_price - stop_loss) / _pip_size_for(signal["pair"]),
                        "rr_actual": -1.0,
                        "signal_score": signal.get("score", 0)
                    }
                # Check target
                elif low <= target:
                    pips_won = (entry_price - target) / _pip_size_for(signal["pair"])
                    return {
                        "pair": signal["pair"],
                        "entry_time": entry_time,
                        "exit_time": ts,
                        "side": side,
                        "entry_price": entry_price,
                        "exit_price": target,
                        "result": "WIN",
                        "duration_hours": i + 1,
                        "pips": pips_won,
                        "rr_actual": float(signal.get("rr", 0)),
                        "signal_score": signal.get("score", 0)
                    }

        # Trade timed out
        final_price = float(future_bars.iloc[-1]["close"])
        if side == "BUY":
            pips = (final_price - entry_price) / _pip_size_for(signal["pair"])
        else:
            pips = (entry_price - final_price) / _pip_size_for(signal["pair"])

        return {
            "pair": signal["pair"],
            "entry_time": entry_time,
            "exit_time": future_bars.index[-1],
            "side": side,
            "entry_price": entry_price,
            "exit_price": final_price,
            "result": "TIMEOUT",
            "duration_hours": max_hours,
            "pips": pips,
            "rr_actual": pips / abs((stop_loss - entry_price) / _pip_size_for(signal["pair"])) if stop_loss != entry_price else 0,
            "signal_score": signal.get("score", 0)
        }

    def analyze_pair_historical(self, pair: str, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Run historical analysis on a pair's data"""
        print(f"\nAnalyzing {pair} historical patterns...")

        # Ensure we have enough data
        if len(df) < 500:
            print(f"✗ {pair}: Insufficient data ({len(df)} bars)")
            return []

        # Get last 6 months of data
        end_date = df.index[-1]
        start_date = end_date - pd.Timedelta(days=180)
        analysis_df = df[df.index >= start_date].copy()

        print(f"Analyzing {len(analysis_df)} bars from {start_date.date()} to {end_date.date()}")

        trades: List[Dict[str, Any]] = []
        signals_tested = 0

        # Walk through each hour and test for signals
        for i in range(300, len(analysis_df) - 48):  # Need 300 bars history, 48 bars lookahead
            current_ts = analysis_df.index[i]
            historical_data = analysis_df.iloc[:i+1]  # Data up to current bar

            # Test confluence signal
            try:
                signal = evaluate_last_closed_bar(historical_data, pair=pair)
                if signal and not signal.get("skip_reason"):
                    signals_tested += 1
                    trade_result = self.simulate_trade(signal, analysis_df, i)
                    if trade_result:
                        trades.append(trade_result)

            except Exception as e:
                continue

            # Test win mode signal (stricter criteria)
            try:
                win_signal = evaluate_last_closed_bar_win(historical_data, pair=pair)
                if win_signal and not win_signal.get("skip_reason"):
                    trade_result = self.simulate_trade(win_signal, analysis_df, i)
                    if trade_result:
                        trade_result["strategy"] = "WIN_MODE"
                        trades.append(trade_result)
            except Exception as e:
                continue

        print(f"✓ {pair}: {signals_tested} signals tested, {len(trades)} trades completed")
        return trades

    def calculate_performance_stats(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate comprehensive performance statistics"""
        if not trades:
            return {}

        wins = [t for t in trades if t["result"] == "WIN"]
        losses = [t for t in trades if t["result"] == "LOSS"]
        timeouts = [t for t in trades if t["result"] == "TIMEOUT"]

        total_trades = len(trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0

        total_pips = sum(t["pips"] for t in trades)
        avg_win_pips = np.mean([t["pips"] for t in wins]) if wins else 0
        avg_loss_pips = np.mean([t["pips"] for t in losses]) if losses else 0

        # Session analysis
        session_performance = {}
        for trade in trades:
            session = _session_block(trade["entry_time"], trade["pair"])
            if session not in session_performance:
                session_performance[session] = {"wins": 0, "total": 0, "pips": 0}
            session_performance[session]["total"] += 1
            session_performance[session]["pips"] += trade["pips"]
            if trade["result"] == "WIN":
                session_performance[session]["wins"] += 1

        # Best session
        best_session = max(session_performance.keys(),
                          key=lambda s: session_performance[s]["pips"]) if session_performance else "UNKNOWN"

        return {
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "timeouts": len(timeouts),
            "win_rate": round(win_rate * 100, 1),
            "total_pips": round(total_pips, 1),
            "avg_win_pips": round(avg_win_pips, 1),
            "avg_loss_pips": round(avg_loss_pips, 1),
            "profit_factor": round(abs(avg_win_pips * len(wins)) / max(1, abs(avg_loss_pips * len(losses))), 2),
            "avg_duration_hours": round(np.mean([t["duration_hours"] for t in trades]), 1),
            "best_session": best_session,
            "session_stats": session_performance
        }

    async def run_full_analysis(self) -> Dict[str, Any]:
        """Run complete historical analysis on existing data + attempt new indices"""
        print("=== HISTORICAL BACKTESTING ANALYSIS ===")
        print(f"Available pairs: {TARGET_PAIRS}")
        print(f"Target indices: {TARGET_INDICES}")
        print(f"Lookback period: {self.lookback_months} months\n")

        # Try to fetch fresh data for indices, but continue with existing data
        for pair in TARGET_INDICES:
            try:
                await self.fetch_historical_data(pair)
            except Exception as e:
                print(f"⚠️  {pair}: Using existing data (API unavailable)")

        # Analyze each available pair
        all_results = {}

        # Test both existing forex pairs and any available index data
        test_pairs = list(set(TARGET_PAIRS + TARGET_INDICES))

        for pair in test_pairs:
            df = self.load_existing_data(pair)
            if df.empty:
                print(f"✗ {pair}: No data available")
                continue

            trades = self.analyze_pair_historical(pair, df)
            stats = self.calculate_performance_stats(trades)

            all_results[pair] = {
                "trades": trades,
                "stats": stats
            }

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"backtest_results_{timestamp}.json"

        with open(results_file, "w") as f:
            # Convert timestamps to strings for JSON serialization
            serializable_results = {}
            for pair, data in all_results.items():
                serializable_results[pair] = {
                    "stats": data["stats"],
                    "trades": [
                        {**trade,
                         "entry_time": trade["entry_time"].isoformat() if hasattr(trade["entry_time"], "isoformat") else str(trade["entry_time"]),
                         "exit_time": trade["exit_time"].isoformat() if hasattr(trade["exit_time"], "isoformat") else str(trade["exit_time"])
                        } for trade in data["trades"]
                    ]
                }
            json.dump(serializable_results, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")
        return all_results

    def generate_insights_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive insights from backtest results"""
        report = "\n" + "="*60
        report += "\n    HISTORICAL BACKTESTING INSIGHTS REPORT"
        report += "\n" + "="*60

        # Overall performance ranking
        pair_performance = []
        for pair, data in results.items():
            stats = data["stats"]
            if stats:
                score = stats["win_rate"] * stats["total_pips"] / max(1, stats["total_trades"])
                pair_performance.append((pair, score, stats))

        pair_performance.sort(key=lambda x: x[1], reverse=True)

        report += "\n\n📊 PERFORMANCE RANKING:\n"
        for i, (pair, score, stats) in enumerate(pair_performance, 1):
            report += f"{i}. {pair}: {stats['win_rate']}% WR, {stats['total_pips']} pips, {stats['total_trades']} trades\n"

        # Best patterns found
        report += "\n\n🎯 KEY FINDINGS:\n"

        best_pair = pair_performance[0] if pair_performance else None
        if best_pair:
            pair, score, stats = best_pair
            report += f"• BEST PERFORMER: {pair} with {stats['win_rate']}% win rate\n"
            report += f"• Best session for {pair}: {stats.get('best_session', 'Unknown')}\n"
            report += f"• Average win: +{stats['avg_win_pips']} pips\n"
            report += f"• Average loss: {stats['avg_loss_pips']} pips\n"

        # Session analysis across all pairs
        all_sessions = {}
        for pair, data in results.items():
            for session, session_data in data["stats"].get("session_stats", {}).items():
                if session not in all_sessions:
                    all_sessions[session] = {"wins": 0, "total": 0, "pips": 0}
                all_sessions[session]["wins"] += session_data["wins"]
                all_sessions[session]["total"] += session_data["total"]
                all_sessions[session]["pips"] += session_data["pips"]

        if all_sessions:
            best_session_overall = max(all_sessions.keys(),
                                     key=lambda s: all_sessions[s]["pips"] / max(1, all_sessions[s]["total"]))
            report += f"• BEST SESSION OVERALL: {best_session_overall}\n"

        # Strategy recommendations
        report += "\n\n💡 TRADING RECOMMENDATIONS:\n"

        if pair_performance:
            top_3 = pair_performance[:3]
            report += "• Focus on these top-performing indices:\n"
            for pair, score, stats in top_3:
                report += f"  - {pair}: Trade during {stats.get('best_session', 'core')} session\n"

        report += "\n• Based on 6-month historical analysis"
        report += f"\n• Total signals analyzed: {sum(data['stats'].get('total_trades', 0) for data in results.values())}"
        report += "\n\n" + "="*60

        return report

# CLI function for easy execution
async def main():
    backtester = HistoricalBacktester(lookback_months=6)
    results = await backtester.run_full_analysis()

    # Print summary report
    report = backtester.generate_insights_report(results)
    print(report)

    return results

if __name__ == "__main__":
    asyncio.run(main())