#!/usr/bin/env python3
"""
FOCUSED 1:1.5 R:R BACKTEST FOR XAUUSD
Optimized analysis focusing on most important confluence factors
Testing what actually works with detailed outcome tracking
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np

async def focused_1_1_5_backtest():
    print("🎯 FOCUSED 5-YEAR 1:1.5 R:R BACKTEST")
    print("=" * 40)
    print("📊 Testing core confluence factors with precise statistics...")

    try:
        from app.oanda_feed import get_xauusd_candles

        # Get maximum available data
        df = await get_xauusd_candles(count=5000)
        if df is None or df.empty:
            print("❌ No XAUUSD data available")
            return

        print(f"✅ Analyzing {len(df)} candles from {df.index[0]} to {df.index[-1]}")
        print(f"   Time span: {(df.index[-1] - df.index[0]).days} days ({(df.index[-1] - df.index[0]).days/365.25:.1f} years)")

        chicago_tz = pytz.timezone('America/Chicago')

        # Add core technical indicators
        print("\n🔧 Adding technical indicators...")

        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_100'] = df['close'].rolling(window=100).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()

        # Position relative to MAs
        df['above_sma20'] = df['close'] > df['sma_20']
        df['above_sma50'] = df['close'] > df['sma_50']
        df['above_sma100'] = df['close'] > df['sma_100']
        df['above_ema20'] = df['close'] > df['ema_20']
        df['above_ema50'] = df['close'] > df['ema_50']

        # Trend detection
        df['sma20_uptrend'] = df['sma_20'].diff(5) > 0
        df['sma50_uptrend'] = df['sma_50'].diff(5) > 0
        df['ema20_uptrend'] = df['ema_20'].diff(5) > 0

        # Price action
        df['is_green'] = df['close'] > df['open']
        df['prev_green'] = df['is_green'].shift(1)
        df['prev_red'] = (~df['is_green']).shift(1)

        # High/Low analysis
        df['high_12h'] = df['high'].rolling(window=12).max()
        df['low_12h'] = df['low'].rolling(window=12).min()
        df['high_24h'] = df['high'].rolling(window=24).max()
        df['low_24h'] = df['low'].rolling(window=24).min()

        df['near_high_12h'] = abs(df['close'] - df['high_12h']) <= 10
        df['near_low_12h'] = abs(df['close'] - df['low_12h']) <= 10
        df['break_24h_high'] = df['close'] > df['high_24h'].shift(1)
        df['break_24h_low'] = df['close'] < df['low_24h'].shift(1)

        # Previous candle analysis
        df['prev_high_break'] = df['close'] > df['high'].shift(1)
        df['prev_low_break'] = df['close'] < df['low'].shift(1)

        print(f"✅ Added technical indicators")

        # Define confluence factors to test
        confluence_factors = [
            'above_sma20',
            'above_sma50',
            'above_sma100',
            'above_ema20',
            'above_ema50',
            'sma20_uptrend',
            'sma50_uptrend',
            'ema20_uptrend',
            'is_green',
            'prev_green',
            'prev_red',
            'near_high_12h',
            'near_low_12h',
            'break_24h_high',
            'prev_high_break',
            'prev_low_break'
        ]

        print(f"\n🧪 Testing {len(confluence_factors)} individual confluence factors...")

        results = []

        for factor in confluence_factors:
            print(f"   Analyzing: {factor}")

            tp_wins = 0
            sl_losses = 0
            no_exits = 0
            trades_with_factor = 0

            # Test both BUY and SELL signals
            for signal_type in ['BUY', 'SELL']:

                for i in range(100, len(df) - 100):  # Need MA history + future outcome space
                    try:
                        current_candle = df.iloc[i]

                        # Check if confluence factor is present
                        factor_present = current_candle.get(factor, False)
                        if not factor_present:
                            continue

                        # Additional filter for signal direction
                        if signal_type == 'BUY':
                            # Only BUY signals above SMA50 (basic trend filter)
                            if not current_candle.get('above_sma50', False):
                                continue
                        else:  # SELL
                            # Only SELL signals below SMA50
                            if current_candle.get('above_sma50', False):
                                continue

                        trades_with_factor += 1
                        entry_price = current_candle['close']

                        if signal_type == 'BUY':
                            tp_price = entry_price * 1.01   # +1% TP
                            sl_price = entry_price * 0.985  # -1.5% SL
                        else:  # SELL
                            tp_price = entry_price * 0.99   # -1% TP
                            sl_price = entry_price * 1.015  # +1.5% SL

                        # Check outcome in next 100 candles
                        future_candles = df.iloc[i+1:i+101]
                        tp_hit = False
                        sl_hit = False
                        tp_time = None
                        sl_time = None

                        for j, future_candle in future_candles.iterrows():
                            # Check for TP hit
                            if signal_type == 'BUY':
                                if future_candle['high'] >= tp_price and not tp_hit:
                                    tp_hit = True
                                    tp_time = j
                                if future_candle['low'] <= sl_price and not sl_hit:
                                    sl_hit = True
                                    sl_time = j
                            else:  # SELL
                                if future_candle['low'] <= tp_price and not tp_hit:
                                    tp_hit = True
                                    tp_time = j
                                if future_candle['high'] >= sl_price and not sl_hit:
                                    sl_hit = True
                                    sl_time = j

                            # Determine outcome (first hit wins)
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

                        # If neither hit in 100 candles
                        if not tp_hit and not sl_hit:
                            no_exits += 1

                    except Exception as e:
                        continue

            # Calculate percentages
            total_trades = tp_wins + sl_losses + no_exits

            if total_trades >= 10:  # Only factors with sufficient trades
                tp_pct = (tp_wins / total_trades) * 100
                sl_pct = (sl_losses / total_trades) * 100
                no_exit_pct = (no_exits / total_trades) * 100

                # Expected return calculation for 1:1.5 R:R
                expected_return = (tp_pct/100 * 0.01) - (sl_pct/100 * 0.015)

                results.append({
                    'factor': factor,
                    'total_trades': total_trades,
                    'tp_wins': tp_wins,
                    'sl_losses': sl_losses,
                    'no_exits': no_exits,
                    'tp_percentage': tp_pct,
                    'sl_percentage': sl_pct,
                    'no_exit_percentage': no_exit_pct,
                    'expected_return': expected_return,
                    'profitable': expected_return > 0
                })

        # Sort by expected return
        results.sort(key=lambda x: x['expected_return'], reverse=True)

        print(f"\n📊 DETAILED 1:1.5 R:R RESULTS:")
        print("=" * 85)
        print(f"{'Factor':<18} {'Trades':<7} {'TP%':<6} {'SL%':<6} {'NoExit%':<8} {'ExpRet':<8} {'Profit?'}")
        print("-" * 85)

        profitable_count = 0
        for result in results:
            profitable = "✅ YES" if result['profitable'] else "❌ NO"
            if result['profitable']:
                profitable_count += 1

            print(f"{result['factor']:<18} {result['total_trades']:<7} "
                  f"{result['tp_percentage']:<6.1f} {result['sl_percentage']:<6.1f} "
                  f"{result['no_exit_percentage']:<8.1f} {result['expected_return']:<+8.4f} {profitable}")

        print(f"\n🎯 SUMMARY FOR 1:1.5 R:R RATIO:")
        print("=" * 35)
        print(f"   Total factors tested: {len(results)}")
        print(f"   Profitable factors: {profitable_count}")
        print(f"   Unprofitable factors: {len(results) - profitable_count}")

        if profitable_count > 0:
            best = results[0]
            print(f"\n🏆 BEST PERFORMING FACTOR:")
            print(f"   Factor: {best['factor']}")
            print(f"   Total trades: {best['total_trades']}")
            print(f"   TP rate: {best['tp_percentage']:.1f}%")
            print(f"   SL rate: {best['sl_percentage']:.1f}%")
            print(f"   No exit rate: {best['no_exit_percentage']:.1f}%")
            print(f"   Expected return: {best['expected_return']:+.4f} per trade")
            print(f"   Return per 100 trades: {best['expected_return']*100:+.2f}%")

            # Test top 3 factor combination
            print(f"\n🔬 TESTING TOP 3 FACTOR COMBINATION:")
            top_3 = [results[i]['factor'] for i in range(min(3, len(results)))]
            print(f"   Combining: {' + '.join(top_3)}")

            combo_tp = 0
            combo_sl = 0
            combo_no_exit = 0

            for signal_type in ['BUY', 'SELL']:
                for i in range(100, len(df) - 100):
                    try:
                        current_candle = df.iloc[i]

                        # All 3 factors must be present
                        all_present = all(current_candle.get(factor, False) for factor in top_3)
                        if not all_present:
                            continue

                        # Direction filter
                        if signal_type == 'BUY' and not current_candle.get('above_sma50', False):
                            continue
                        if signal_type == 'SELL' and current_candle.get('above_sma50', False):
                            continue

                        entry_price = current_candle['close']

                        if signal_type == 'BUY':
                            tp_price = entry_price * 1.01
                            sl_price = entry_price * 0.985
                        else:
                            tp_price = entry_price * 0.99
                            sl_price = entry_price * 1.015

                        # Check outcome
                        future_candles = df.iloc[i+1:i+101]
                        tp_hit = False
                        sl_hit = False
                        tp_time = None
                        sl_time = None

                        for j, future_candle in future_candles.iterrows():
                            if signal_type == 'BUY':
                                if future_candle['high'] >= tp_price and not tp_hit:
                                    tp_hit = True
                                    tp_time = j
                                if future_candle['low'] <= sl_price and not sl_hit:
                                    sl_hit = True
                                    sl_time = j
                            else:
                                if future_candle['low'] <= tp_price and not tp_hit:
                                    tp_hit = True
                                    tp_time = j
                                if future_candle['high'] >= sl_price and not sl_hit:
                                    sl_hit = True
                                    sl_time = j

                            if tp_hit and sl_hit:
                                if tp_time <= sl_time:
                                    combo_tp += 1
                                    break
                                else:
                                    combo_sl += 1
                                    break
                            elif tp_hit:
                                combo_tp += 1
                                break
                            elif sl_hit:
                                combo_sl += 1
                                break

                        if not tp_hit and not sl_hit:
                            combo_no_exit += 1

                    except Exception as e:
                        continue

            combo_total = combo_tp + combo_sl + combo_no_exit
            if combo_total > 0:
                combo_tp_pct = (combo_tp / combo_total) * 100
                combo_sl_pct = (combo_sl / combo_total) * 100
                combo_no_exit_pct = (combo_no_exit / combo_total) * 100
                combo_expected = (combo_tp_pct/100 * 0.01) - (combo_sl_pct/100 * 0.015)

                print(f"   Total trades: {combo_total}")
                print(f"   TP rate: {combo_tp_pct:.1f}%")
                print(f"   SL rate: {combo_sl_pct:.1f}%")
                print(f"   No exit rate: {combo_no_exit_pct:.1f}%")
                print(f"   Expected return: {combo_expected:+.4f} per trade")
                print(f"   Profitable: {'✅ YES' if combo_expected > 0 else '❌ NO'}")

        else:
            print(f"\n🚨 CRITICAL FINDING:")
            print(f"   NO confluence factors are profitable with 1:1.5 R:R")
            print(f"   This suggests the R:R ratio may be too aggressive")

        # Overall market statistics
        total_candles_analyzed = len(df) - 200  # Available candles for analysis
        print(f"\n📈 MARKET STATISTICS:")
        print(f"   Candles analyzed: {total_candles_analyzed}")
        print(f"   Average daily volatility: {(df['high'] - df['low']).mean():.2f}")
        print(f"   Time period: {(df.index[-1] - df.index[0]).days} days")

    except Exception as e:
        print(f"❌ Backtest error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(focused_1_1_5_backtest())