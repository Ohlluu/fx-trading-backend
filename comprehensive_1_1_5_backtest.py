#!/usr/bin/env python3
"""
COMPREHENSIVE 5-YEAR 1:1.5 R:R BACKTEST FOR XAUUSD
Exhaustive analysis of every confluence combination with detailed statistics
Testing ALL possible combinations to find what actually works
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np
from itertools import combinations

async def comprehensive_1_1_5_backtest():
    print("🔍 COMPREHENSIVE 5-YEAR 1:1.5 R:R BACKTEST")
    print("=" * 50)
    print("📊 Testing EVERY confluence combination with detailed stats...")

    try:
        from app.oanda_feed import get_xauusd_candles

        # Get maximum available data
        df = await get_xauusd_candles(count=5000)
        if df is None or df.empty:
            print("❌ No XAUUSD data available")
            return

        print(f"✅ Analyzing {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        print(f"   Time span: {(df.index[-1] - df.index[0]).days} days")

        chicago_tz = pytz.timezone('America/Chicago')

        # Add ALL possible confluence indicators
        print("\n🔧 Adding comprehensive technical indicators...")

        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_100'] = df['close'].rolling(window=100).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()

        # Position relative to MAs
        df['above_sma10'] = df['close'] > df['sma_10']
        df['above_sma20'] = df['close'] > df['sma_20']
        df['above_sma50'] = df['close'] > df['sma_50']
        df['above_sma100'] = df['close'] > df['sma_100']
        df['above_ema10'] = df['close'] > df['ema_10']
        df['above_ema20'] = df['close'] > df['ema_20']
        df['above_ema50'] = df['close'] > df['ema_50']

        # Trend detection (multiple periods)
        df['sma20_slope_3'] = df['sma_20'].diff(3) > 0
        df['sma20_slope_5'] = df['sma_20'].diff(5) > 0
        df['sma50_slope_3'] = df['sma_50'].diff(3) > 0
        df['sma50_slope_5'] = df['sma_50'].diff(5) > 0
        df['ema20_slope_3'] = df['ema_20'].diff(3) > 0
        df['ema20_slope_5'] = df['ema_20'].diff(5) > 0

        # Price action
        df['is_green'] = df['close'] > df['open']
        df['is_red'] = df['close'] < df['open']
        df['prev_green'] = df['is_green'].shift(1)
        df['prev_red'] = df['is_red'].shift(1)
        df['two_green'] = df['is_green'] & df['prev_green']
        df['green_after_red'] = df['is_green'] & df['prev_red']

        # High/Low analysis (multiple periods)
        df['high_6h'] = df['high'].rolling(window=6).max()
        df['high_12h'] = df['high'].rolling(window=12).max()
        df['high_24h'] = df['high'].rolling(window=24).max()
        df['low_6h'] = df['low'].rolling(window=6).min()
        df['low_12h'] = df['low'].rolling(window=12).min()
        df['low_24h'] = df['low'].rolling(window=24).min()

        df['near_high_6h'] = abs(df['close'] - df['high_6h']) < 5
        df['near_high_12h'] = abs(df['close'] - df['high_12h']) < 10
        df['near_high_24h'] = abs(df['close'] - df['high_24h']) < 15
        df['near_low_6h'] = abs(df['close'] - df['low_6h']) < 5
        df['near_low_12h'] = abs(df['close'] - df['low_12h']) < 10
        df['near_low_24h'] = abs(df['close'] - df['low_24h']) < 15

        # Previous high/low breaks
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['break_prev_high'] = df['close'] > df['prev_high']
        df['break_prev_low'] = df['close'] < df['prev_low']

        # Range analysis
        df['range_1h'] = df['high'] - df['low']
        df['avg_range_12h'] = df['range_1h'].rolling(window=12).mean()
        df['large_range'] = df['range_1h'] > df['avg_range_12h'] * 1.5
        df['small_range'] = df['range_1h'] < df['avg_range_12h'] * 0.5

        print(f"✅ Added {len([col for col in df.columns if col not in ['open','high','low','close','volume','time']])} technical indicators")

        # Define ALL confluence factors to test
        confluence_factors = [
            'above_sma10', 'above_sma20', 'above_sma50', 'above_sma100',
            'above_ema10', 'above_ema20', 'above_ema50',
            'sma20_slope_3', 'sma20_slope_5', 'sma50_slope_3', 'sma50_slope_5',
            'ema20_slope_3', 'ema20_slope_5',
            'is_green', 'is_red', 'prev_green', 'prev_red',
            'two_green', 'green_after_red',
            'near_high_6h', 'near_high_12h', 'near_high_24h',
            'near_low_6h', 'near_low_12h', 'near_low_24h',
            'break_prev_high', 'break_prev_low',
            'large_range', 'small_range'
        ]

        print(f"\n🧪 Testing {len(confluence_factors)} individual factors + combinations...")

        # Test individual factors first
        individual_results = []

        for factor in confluence_factors:
            print(f"   Testing: {factor}")

            tp_wins = 0
            sl_losses = 0
            no_exits = 0

            for i in range(200, len(df) - 100):  # Need history for MAs and future for outcomes
                try:
                    current_candle = df.iloc[i]

                    # Skip if factor not present
                    if not current_candle.get(factor, False):
                        continue

                    entry_price = current_candle['close']
                    tp_price = entry_price * 1.01   # 1% TP
                    sl_price = entry_price * 0.985  # 1.5% SL

                    # Check outcome in next 100 candles (4+ days)
                    future_candles = df.iloc[i+1:i+101]
                    tp_hit = False
                    sl_hit = False
                    tp_time = None
                    sl_time = None

                    for j, future_candle in future_candles.iterrows():
                        # Check TP hit
                        if future_candle['high'] >= tp_price and not tp_hit:
                            tp_hit = True
                            tp_time = j
                        # Check SL hit
                        if future_candle['low'] <= sl_price and not sl_hit:
                            sl_hit = True
                            sl_time = j

                        # Determine outcome
                        if tp_hit and sl_hit:
                            if tp_time <= sl_time:
                                tp_wins += 1
                                break
                            else:
                                sl_losses += 1
                                break
                        elif tp_hit:
                            tp_wins += 1
                            break
                        elif sl_hit:
                            sl_losses += 1
                            break

                    # If neither hit
                    if not tp_hit and not sl_hit:
                        no_exits += 1

                except Exception as e:
                    continue

            total_trades = tp_wins + sl_losses + no_exits
            if total_trades > 0:
                tp_pct = (tp_wins / total_trades) * 100
                sl_pct = (sl_losses / total_trades) * 100
                no_exit_pct = (no_exits / total_trades) * 100

                individual_results.append({
                    'factor': factor,
                    'total_trades': total_trades,
                    'tp_wins': tp_wins,
                    'sl_losses': sl_losses,
                    'no_exits': no_exits,
                    'tp_percentage': tp_pct,
                    'sl_percentage': sl_pct,
                    'no_exit_percentage': no_exit_pct,
                    'win_rate': tp_pct,
                    'expected_return': (tp_pct * 0.01 - sl_pct * 0.015) / 100
                })

        # Sort by expected return
        individual_results.sort(key=lambda x: x['expected_return'], reverse=True)

        print(f"\n📊 INDIVIDUAL FACTOR RESULTS (1:1.5 R:R):")
        print("=" * 80)
        print(f"{'Factor':<20} {'Trades':<7} {'TP%':<6} {'SL%':<6} {'NoExit%':<8} {'ExpRet':<8}")
        print("-" * 80)

        for result in individual_results[:15]:  # Top 15
            print(f"{result['factor']:<20} {result['total_trades']:<7} "
                  f"{result['tp_percentage']:<6.1f} {result['sl_percentage']:<6.1f} "
                  f"{result['no_exit_percentage']:<8.1f} {result['expected_return']:<8.3f}")

        # Test combinations of top factors
        print(f"\n🔬 TESTING COMBINATIONS OF TOP FACTORS...")
        top_factors = [r['factor'] for r in individual_results[:10]]  # Top 10 factors

        combination_results = []

        # Test 2-factor combinations
        for combo in combinations(top_factors, 2):
            combo_name = f"{combo[0]}+{combo[1]}"

            tp_wins = 0
            sl_losses = 0
            no_exits = 0

            for i in range(200, len(df) - 100):
                try:
                    current_candle = df.iloc[i]

                    # Both factors must be present
                    if not (current_candle.get(combo[0], False) and current_candle.get(combo[1], False)):
                        continue

                    entry_price = current_candle['close']
                    tp_price = entry_price * 1.01   # 1% TP
                    sl_price = entry_price * 0.985  # 1.5% SL

                    # Check outcome
                    future_candles = df.iloc[i+1:i+101]
                    tp_hit = False
                    sl_hit = False
                    tp_time = None
                    sl_time = None

                    for j, future_candle in future_candles.iterrows():
                        if future_candle['high'] >= tp_price and not tp_hit:
                            tp_hit = True
                            tp_time = j
                        if future_candle['low'] <= sl_price and not sl_hit:
                            sl_hit = True
                            sl_time = j

                        if tp_hit and sl_hit:
                            if tp_time <= sl_time:
                                tp_wins += 1
                                break
                            else:
                                sl_losses += 1
                                break
                        elif tp_hit:
                            tp_wins += 1
                            break
                        elif sl_hit:
                            sl_losses += 1
                            break

                    if not tp_hit and not sl_hit:
                        no_exits += 1

                except Exception as e:
                    continue

            total_trades = tp_wins + sl_losses + no_exits
            if total_trades >= 50:  # Only consider combinations with enough trades
                tp_pct = (tp_wins / total_trades) * 100
                sl_pct = (sl_losses / total_trades) * 100
                no_exit_pct = (no_exits / total_trades) * 100

                combination_results.append({
                    'combination': combo_name,
                    'factors': combo,
                    'total_trades': total_trades,
                    'tp_wins': tp_wins,
                    'sl_losses': sl_losses,
                    'no_exits': no_exits,
                    'tp_percentage': tp_pct,
                    'sl_percentage': sl_pct,
                    'no_exit_percentage': no_exit_pct,
                    'expected_return': (tp_pct * 0.01 - sl_pct * 0.015) / 100
                })

        combination_results.sort(key=lambda x: x['expected_return'], reverse=True)

        print(f"\n📊 TOP 2-FACTOR COMBINATIONS (1:1.5 R:R):")
        print("=" * 90)
        print(f"{'Combination':<35} {'Trades':<7} {'TP%':<6} {'SL%':<6} {'NoExit%':<8} {'ExpRet':<8}")
        print("-" * 90)

        for result in combination_results[:10]:
            print(f"{result['combination']:<35} {result['total_trades']:<7} "
                  f"{result['tp_percentage']:<6.1f} {result['sl_percentage']:<6.1f} "
                  f"{result['no_exit_percentage']:<8.1f} {result['expected_return']:<8.3f}")

        # Final summary
        print(f"\n🎯 COMPREHENSIVE BACKTEST SUMMARY:")
        print("=" * 40)

        best_individual = individual_results[0]
        print(f"BEST INDIVIDUAL FACTOR:")
        print(f"   Factor: {best_individual['factor']}")
        print(f"   Trades: {best_individual['total_trades']}")
        print(f"   TP Rate: {best_individual['tp_percentage']:.1f}%")
        print(f"   SL Rate: {best_individual['sl_percentage']:.1f}%")
        print(f"   No Exit Rate: {best_individual['no_exit_percentage']:.1f}%")
        print(f"   Expected Return: {best_individual['expected_return']:.3f} per trade")

        if combination_results:
            best_combo = combination_results[0]
            print(f"\nBEST 2-FACTOR COMBINATION:")
            print(f"   Combination: {best_combo['combination']}")
            print(f"   Trades: {best_combo['total_trades']}")
            print(f"   TP Rate: {best_combo['tp_percentage']:.1f}%")
            print(f"   SL Rate: {best_combo['sl_percentage']:.1f}%")
            print(f"   No Exit Rate: {best_combo['no_exit_percentage']:.1f}%")
            print(f"   Expected Return: {best_combo['expected_return']:.3f} per trade")

        # Reality check
        print(f"\n⚠️  REALITY CHECK:")
        profitable_individual = [r for r in individual_results if r['expected_return'] > 0]
        profitable_combos = [r for r in combination_results if r['expected_return'] > 0]

        print(f"   Profitable individual factors: {len(profitable_individual)}/{len(individual_results)}")
        print(f"   Profitable combinations: {len(profitable_combos)}/{len(combination_results)}")

        if len(profitable_individual) == 0:
            print(f"   🚨 WARNING: NO individual factors are profitable with 1:1.5 R:R")

        if len(profitable_combos) == 0:
            print(f"   🚨 WARNING: NO combinations are profitable with 1:1.5 R:R")

    except Exception as e:
        print(f"❌ Comprehensive backtest error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(comprehensive_1_1_5_backtest())