# backend/app/quick_insights.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def load_data(pair: str) -> pd.DataFrame:
    """Load existing CSV data"""
    csv_path = f"data/{pair}_H1.csv"
    if not os.path.exists(csv_path):
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    return df

def calculate_basic_stats(df: pd.DataFrame, pair: str) -> Dict[str, Any]:
    """Calculate basic trading statistics"""
    if df.empty:
        return {}

    # Get last 6 months
    end_date = df.index[-1]
    start_date = end_date - pd.Timedelta(days=180)
    recent_df = df[df.index >= start_date].copy()

    if len(recent_df) < 100:
        return {}

    # Basic price movements
    recent_df["price_change"] = recent_df["close"].pct_change()
    recent_df["range"] = recent_df["high"] - recent_df["low"]
    recent_df["body_size"] = abs(recent_df["close"] - recent_df["open"])

    # Volatility patterns
    hourly_vol = recent_df.groupby(recent_df.index.hour)["range"].mean()
    best_hours = hourly_vol.nlargest(3).index.tolist()

    # Trend analysis (simple)
    recent_df["sma20"] = recent_df["close"].rolling(20).mean()
    recent_df["sma50"] = recent_df["close"].rolling(50).mean()

    trend_up_bars = len(recent_df[recent_df["close"] > recent_df["sma20"]])
    trend_strength = (trend_up_bars / len(recent_df)) * 100

    # Session analysis
    sessions = {
        "Asian": recent_df.between_time("23:00", "07:59"),
        "London": recent_df.between_time("08:00", "15:59"),
        "NY": recent_df.between_time("13:00", "21:59"),
        "Overlap": recent_df.between_time("13:00", "15:59")
    }

    session_stats = {}
    for session_name, session_data in sessions.items():
        if not session_data.empty:
            avg_range = session_data["range"].mean()
            avg_volume = session_data["volume"].mean()
            session_stats[session_name] = {
                "avg_range": round(avg_range, 6),
                "avg_volume": round(avg_volume, 0),
                "bars": len(session_data)
            }

    best_session = max(session_stats.keys(),
                      key=lambda s: session_stats[s]["avg_range"]) if session_stats else "Unknown"

    return {
        "pair": pair,
        "total_bars": len(recent_df),
        "date_range": f"{start_date.date()} to {end_date.date()}",
        "avg_daily_range": round(recent_df.groupby(recent_df.index.date)["range"].sum().mean(), 6),
        "trend_strength": round(trend_strength, 1),
        "best_hours_utc": best_hours,
        "best_session": best_session,
        "session_stats": session_stats,
        "current_price": round(float(recent_df["close"].iloc[-1]), 6),
        "volatility_6m": round(recent_df["price_change"].std() * 100, 2)
    }

def analyze_patterns() -> Dict[str, Any]:
    """Analyze available data and provide insights"""
    print("=== QUICK HISTORICAL INSIGHTS ===")

    # Available pairs
    available_pairs = []
    data_dir = "data"
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if file.endswith("_H1.csv"):
                pair = file.replace("_H1.csv", "")
                available_pairs.append(pair)

    print(f"Found data for: {available_pairs}")

    results = {}

    # Analyze each pair
    for pair in available_pairs:
        if pair == "positions":  # Skip positions.json
            continue

        df = load_data(pair)
        stats = calculate_basic_stats(df, pair)
        if stats:
            results[pair] = stats
            print(f"\n✓ {pair}:")
            print(f"  • 6-month trend: {stats['trend_strength']}% bullish")
            print(f"  • Best session: {stats['best_session']}")
            print(f"  • Daily range: {stats['avg_daily_range']}")
            print(f"  • Volatility: {stats['volatility_6m']}%")

    # Trading recommendations
    recommendations = generate_recommendations(results)

    return {
        "analysis_date": datetime.now().isoformat(),
        "pairs_analyzed": len(results),
        "results": results,
        "recommendations": recommendations
    }

def generate_recommendations(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate trading recommendations from analysis"""
    if not results:
        return {"error": "No data available for analysis"}

    # Rank pairs by trading potential (volatility + trend strength)
    pair_scores = []
    for pair, stats in results.items():
        if pair == "SPX500":  # Index scoring
            score = stats["volatility_6m"] * 2  # Indices typically more volatile
        else:  # Forex scoring
            score = stats["volatility_6m"] * stats["trend_strength"] / 50

        pair_scores.append((pair, score, stats))

    pair_scores.sort(key=lambda x: x[1], reverse=True)

    # Session recommendations
    session_preference = {}
    for pair, score, stats in pair_scores:
        best_session = stats["best_session"]
        if best_session not in session_preference:
            session_preference[best_session] = []
        session_preference[best_session].append(pair)

    top_pairs = [p[0] for p in pair_scores[:5]]

    return {
        "top_trading_pairs": top_pairs,
        "session_focus": session_preference,
        "key_insights": [
            f"Most volatile: {pair_scores[0][0]} ({pair_scores[0][1]:.1f} score)" if pair_scores else "No data",
            f"Best session overall: {max(session_preference.keys(), key=lambda k: len(session_preference[k])) if session_preference else 'Unknown'}",
            f"Index data: {'Available for SPX500' if 'SPX500' in results else 'Limited - need API for DE30, UK100, NAS100'}"
        ],
        "next_steps": [
            "Set up TwelveData API key to fetch DE30, UK100, NAS100 data",
            "Focus initial trading on top-ranked pairs from existing data",
            f"Trade during {max(session_preference.keys(), key=lambda k: len(session_preference[k])) if session_preference else 'core'} session for best volatility",
            "Use confluence system with H1 timeframe as planned"
        ]
    }

if __name__ == "__main__":
    insights = analyze_patterns()

    print("\n" + "="*50)
    print("📊 RECOMMENDATIONS:")
    for rec in insights["recommendations"]["key_insights"]:
        print(f"• {rec}")

    print(f"\n🎯 TOP PAIRS: {', '.join(insights['recommendations']['top_trading_pairs'])}")

    print(f"\n⚡ NEXT STEPS:")
    for step in insights["recommendations"]["next_steps"]:
        print(f"  {step}")
    print("="*50)