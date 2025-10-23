#!/usr/bin/env python3
"""
XAUUSD Realistic R:R Ratio Analysis
Test different R:R ratios to find what actually works
Based on proven confluence factors
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np

async def test_realistic_rr_ratios():
    print("🎯 XAUUSD REALISTIC R:R RATIO ANALYSIS")
    print("=" * 45)
    print("📊 Testing different R:R ratios with proven confluences...")

    try:
        from app.oanda_feed import get_xauusd_candles

        # Get comprehensive data
        df = await get_xauusd_candles(count=5000)
        if df is None or df.empty:
            print("❌ No XAUUSD data available")
            return

        print(f"✅ Testing on {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        chicago_tz = pytz.timezone('America/Chicago')

        # Test different R:R ratios
        rr_ratios = [
            {'name': '1:1', 'tp_pct': 1.0, 'sl_pct': 1.0},
            {'name': '1.5:1', 'tp_pct': 1.5, 'sl_pct': 1.0},
            {'name': '2:1', 'tp_pct': 2.0, 'sl_pct': 1.0},
            {'name': '1:1.5', 'tp_pct': 1.0, 'sl_pct': 1.5},
            {'name': '0.75:1', 'tp_pct': 0.75, 'sl_pct': 1.0},
            {'name': '1:0.75', 'tp_pct': 1.0, 'sl_pct': 0.75},
        ]

        results = []

        for ratio_config in rr_ratios:
            print(f"\n🔍 Testing {ratio_config['name']} ratio...")

            wins = 0
            losses = 0
            total_profit = 0

            trades_analyzed = 0

            for i in range(60, len(df) - 100):
                try:
                    current_price = df['close'].iloc[i]
                    current_time = df.index[i]
                    historical_df = df.iloc[:i+1].copy()

                    # Add proven confluence indicators
                    hist_df = historical_df.copy()
                    hist_df['sma_20'] = hist_df['close'].rolling(window=20).mean()
                    hist_df['sma_50'] = hist_df['close'].rolling(window=50).mean()
                    hist_df['ema_20'] = hist_df['close'].ewm(span=20).mean()
                    hist_df['above_sma20'] = hist_df['close'] > hist_df['sma_20']
                    hist_df['above_sma50'] = hist_df['close'] > hist_df['sma_50']
                    hist_df['above_ema20'] = hist_df['close'] > hist_df['ema_20']
                    hist_df['recent_high_12h'] = hist_df['high'].rolling(window=12).max()
                    hist_df['near_recent_high'] = abs(hist_df['close'] - hist_df['recent_high_12h']) < 10
                    hist_df['is_green'] = hist_df['close'] > hist_df['open']
                    hist_df['ema20_slope'] = hist_df['ema_20'].diff(5)
                    hist_df['uptrend_ema'] = hist_df['ema20_slope'] > 0

                    latest_candle = hist_df.iloc[-1]

                    # Apply PROVEN confluence checklist
                    # MANDATORY: Above EMA20 AND Above SMA50
                    if not (latest_candle.get('above_ema20', False) and latest_candle.get('above_sma50', False)):
                        continue

                    # SCORING: Calculate proven score
                    score = 6  # Base score for mandatory factors

                    if latest_candle.get('above_sma20', False):
                        score += 6
                    if latest_candle.get('near_recent_high', False):
                        score += 5
                    if latest_candle.get('uptrend_ema', False):
                        score += 4
                    if latest_candle.get('is_green', False):
                        score += 2

                    # Must score at least 14 points (proven threshold)
                    if score < 14:
                        continue

                    trades_analyzed += 1

                    # Determine signal direction (simplified: BUY if above all MAs)
                    signal_type = 'BUY'  # For this test, focusing on BUY signals

                    entry_price = current_price
                    tp_price = entry_price * (1 + ratio_config['tp_pct'] / 100)
                    sl_price = entry_price * (1 - ratio_config['sl_pct'] / 100)

                    # Check outcome in next 100 candles
                    future_candles = df.iloc[i+1:i+101]
                    tp_hit = False
                    sl_hit = False
                    tp_hit_time = None
                    sl_hit_time = None

                    for j, future_candle in future_candles.iterrows():
                        if future_candle['high'] >= tp_price and not tp_hit:
                            tp_hit = True
                            tp_hit_time = j
                        if future_candle['low'] <= sl_price and not sl_hit:
                            sl_hit = True
                            sl_hit_time = j

                        # Determine winner
                        if tp_hit and sl_hit:
                            if tp_hit_time <= sl_hit_time:
                                wins += 1
                                total_profit += ratio_config['tp_pct']
                                break
                            else:
                                losses += 1
                                total_profit -= ratio_config['sl_pct']
                                break
                        elif tp_hit:
                            wins += 1
                            total_profit += ratio_config['tp_pct']
                            break
                        elif sl_hit:
                            losses += 1
                            total_profit -= ratio_config['sl_pct']
                            break

                    # If neither hit, consider it a small loss (time decay)
                    if not tp_hit and not sl_hit:
                        losses += 1
                        total_profit -= 0.1  # Small time loss

                except Exception as e:
                    continue

            # Calculate metrics
            total_trades = wins + losses
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            expected_return = total_profit / total_trades if total_trades > 0 else 0

            results.append({
                'ratio': ratio_config['name'],
                'trades': total_trades,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'expected_return': expected_return,
                'profit_per_100': expected_return * 100
            })

            print(f"   {total_trades} trades, {win_rate:.1f}% win rate, {expected_return:+.3f} expected return per trade")

        # Sort results by expected return
        results.sort(key=lambda x: x['expected_return'], reverse=True)

        print(f"\n📊 REALISTIC R:R RATIO RESULTS (Proven Confluences Only):")
        print("=" * 70)
        print(f"{'Ratio':<8} {'Trades':<7} {'Win%':<6} {'Profit/Loss':<12} {'Per 100 Trades':<15} {'Rating'}")
        print("-" * 70)

        for result in results:
            rating = "🏆 BEST" if result == results[0] else "✅ GOOD" if result['expected_return'] > 0 else "❌ POOR"
            profit_per_100 = result['profit_per_100']

            print(f"{result['ratio']:<8} {result['trades']:<7} {result['win_rate']:<6.1f} {result['expected_return']:<+12.3f} {profit_per_100:<+15.1f} {rating}")

        # Best recommendation
        best = results[0]
        print(f"\n🎯 BEST REALISTIC R:R RATIO:")
        print("=" * 35)
        print(f"   Ratio: {best['ratio']}")
        print(f"   Win Rate: {best['win_rate']:.1f}%")
        print(f"   Expected Return: {best['expected_return']:+.3f} per trade")
        print(f"   Total Trades: {best['trades']}")
        print(f"   Profit per 100 trades: {best['profit_per_100']:+.1f}%")

        # Break even analysis
        print(f"\n⚖️  BREAK-EVEN ANALYSIS:")
        print("-" * 25)
        profitable = [r for r in results if r['expected_return'] > 0]
        break_even = [r for r in results if abs(r['expected_return']) < 0.01]

        if profitable:
            print(f"   Profitable ratios: {len(profitable)}/{len(results)}")
            for r in profitable:
                print(f"   • {r['ratio']}: {r['expected_return']:+.3f} per trade")
        else:
            print("   ⚠️  NO profitable ratios found with current confluences")

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_realistic_rr_ratios())