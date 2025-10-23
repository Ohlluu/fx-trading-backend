#!/usr/bin/env python3
"""
EURUSD vs XAUUSD CONFLUENCE COMPARISON
Direct comparison to validate if EURUSD truly respects confluences more
Testing same confluence factors on both pairs with 1:1 R:R
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np

async def compare_confluence_respect():
    print("🔍 EURUSD vs XAUUSD CONFLUENCE COMPARISON")
    print("=" * 50)
    print("📊 Testing same confluences on both pairs to validate claims...")

    try:
        # Import both data sources
        from app.oanda_feed import get_xauusd_candles
        from eurusd_confluence_research import get_eurusd_candles

        # Get data for both pairs
        print("\n📥 Getting data for both pairs...")
        eurusd_df = await get_eurusd_candles(count=5000)
        xauusd_df = await get_xauusd_candles(count=5000)

        if eurusd_df is None or xauusd_df is None:
            print("❌ Could not get data for both pairs")
            return

        print(f"✅ EURUSD: {len(eurusd_df)} candles ({(eurusd_df.index[-1] - eurusd_df.index[0]).days} days)")
        print(f"✅ XAUUSD: {len(xauusd_df)} candles ({(xauusd_df.index[-1] - xauusd_df.index[0]).days} days)")

        # Function to add same technical indicators to both pairs
        def add_common_indicators(df):
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['sma_100'] = df['close'].rolling(window=100).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()

            df['above_sma20'] = df['close'] > df['sma_20']
            df['above_sma50'] = df['close'] > df['sma_50']
            df['above_sma100'] = df['close'] > df['sma_100']
            df['above_ema20'] = df['close'] > df['ema_20']
            df['above_ema50'] = df['close'] > df['ema_50']

            df['sma20_uptrend'] = df['sma_20'].diff(5) > 0
            df['sma50_uptrend'] = df['sma_50'].diff(5) > 0
            df['ema20_uptrend'] = df['ema_20'].diff(5) > 0

            df['is_green'] = df['close'] > df['open']
            df['prev_green'] = df['is_green'].shift(1)

            df['high_20'] = df['high'].rolling(window=20).max()
            df['low_20'] = df['low'].rolling(window=20).min()
            df['break_resistance_20'] = df['close'] > df['high_20'].shift(1)

            return df

        # Add indicators to both
        print("\n🔧 Adding same technical indicators to both pairs...")
        eurusd_df = add_common_indicators(eurusd_df)
        xauusd_df = add_common_indicators(xauusd_df)

        # Common confluence factors to test
        common_factors = [
            'above_sma20', 'above_sma50', 'above_sma100',
            'above_ema20', 'above_ema50',
            'sma20_uptrend', 'sma50_uptrend', 'ema20_uptrend',
            'is_green', 'prev_green', 'break_resistance_20'
        ]

        # Function to test confluence on a pair
        def test_confluence_on_pair(df, pair_name, tp_pct=0.01, sl_pct=0.01):
            results = []

            for factor in common_factors:
                tp_wins = sl_losses = no_exits = 0

                for signal_type in ['BUY', 'SELL']:
                    for i in range(150, len(df) - 100):
                        try:
                            current_candle = df.iloc[i]

                            if not current_candle.get(factor, False):
                                continue

                            # Basic direction filter
                            if signal_type == 'BUY' and not current_candle.get('above_sma50', False):
                                continue
                            if signal_type == 'SELL' and current_candle.get('above_sma50', False):
                                continue

                            entry_price = current_candle['close']

                            if signal_type == 'BUY':
                                tp_price = entry_price * (1 + tp_pct)
                                sl_price = entry_price * (1 - sl_pct)
                            else:
                                tp_price = entry_price * (1 - tp_pct)
                                sl_price = entry_price * (1 + sl_pct)

                            # Check outcome in next 100 candles
                            future_candles = df.iloc[i+1:i+101]
                            tp_hit = sl_hit = False
                            tp_time = sl_time = None

                            for j, future_candle in future_candles.iterrows():
                                if signal_type == 'BUY':
                                    if future_candle['high'] >= tp_price and not tp_hit:
                                        tp_hit, tp_time = True, j
                                    if future_candle['low'] <= sl_price and not sl_hit:
                                        sl_hit, sl_time = True, j
                                else:
                                    if future_candle['low'] <= tp_price and not tp_hit:
                                        tp_hit, tp_time = True, j
                                    if future_candle['high'] >= sl_price and not sl_hit:
                                        sl_hit, sl_time = True, j

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

                        except Exception:
                            continue

                total_trades = tp_wins + sl_losses + no_exits
                if total_trades >= 10:
                    tp_pct_result = (tp_wins / total_trades) * 100
                    sl_pct_result = (sl_losses / total_trades) * 100
                    expected_return = (tp_pct_result/100 * tp_pct) - (sl_pct_result/100 * sl_pct)

                    results.append({
                        'pair': pair_name,
                        'factor': factor,
                        'total_trades': total_trades,
                        'tp_wins': tp_wins,
                        'sl_losses': sl_losses,
                        'tp_percentage': tp_pct_result,
                        'sl_percentage': sl_pct_result,
                        'expected_return': expected_return,
                        'profitable': expected_return > 0
                    })

            return results

        # Test both pairs
        print(f"\n🧪 Testing {len(common_factors)} common factors on both pairs...")

        print("   Testing EURUSD...")
        eurusd_results = test_confluence_on_pair(eurusd_df, "EURUSD")

        print("   Testing XAUUSD...")
        xauusd_results = test_confluence_on_pair(xauusd_df, "XAUUSD")

        # Create comparison
        print(f"\n📊 DIRECT CONFLUENCE COMPARISON (1:1 R:R):")
        print("=" * 100)
        print(f"{'Factor':<20} {'EURUSD TP%':<12} {'EURUSD SL%':<12} {'EURUSD Exp':<12} {'XAUUSD TP%':<12} {'XAUUSD SL%':<12} {'XAUUSD Exp':<12} {'Winner'}")
        print("-" * 100)

        eurusd_wins = 0
        xauusd_wins = 0
        ties = 0

        for factor in common_factors:
            # Find results for this factor
            eur_result = next((r for r in eurusd_results if r['factor'] == factor), None)
            xau_result = next((r for r in xauusd_results if r['factor'] == factor), None)

            if eur_result and xau_result:
                eur_tp = eur_result['tp_percentage']
                eur_sl = eur_result['sl_percentage']
                eur_exp = eur_result['expected_return']
                xau_tp = xau_result['tp_percentage']
                xau_sl = xau_result['sl_percentage']
                xau_exp = xau_result['expected_return']

                # Determine winner
                if abs(eur_exp - xau_exp) < 0.0001:
                    winner = "TIE"
                    ties += 1
                elif eur_exp > xau_exp:
                    winner = "EURUSD"
                    eurusd_wins += 1
                else:
                    winner = "XAUUSD"
                    xauusd_wins += 1

                print(f"{factor:<20} {eur_tp:<12.1f} {eur_sl:<12.1f} {eur_exp:<+12.4f} "
                      f"{xau_tp:<12.1f} {xau_sl:<12.1f} {xau_exp:<+12.4f} {winner}")

        # Summary stats
        print(f"\n🏆 CONFLUENCE RESPECT COMPARISON SUMMARY:")
        print("=" * 50)
        print(f"   EURUSD wins: {eurusd_wins}")
        print(f"   XAUUSD wins: {xauusd_wins}")
        print(f"   Ties: {ties}")
        print(f"   Total factors compared: {len(common_factors)}")

        # Calculate averages
        eurusd_profitable = [r for r in eurusd_results if r['profitable']]
        xauusd_profitable = [r for r in xauusd_results if r['profitable']]

        print(f"\n📈 PROFITABLE FACTORS ANALYSIS:")
        print("-" * 35)
        print(f"   EURUSD profitable factors: {len(eurusd_profitable)}/{len(eurusd_results)} ({len(eurusd_profitable)/len(eurusd_results)*100:.1f}%)")
        print(f"   XAUUSD profitable factors: {len(xauusd_profitable)}/{len(xauusd_results)} ({len(xauusd_profitable)/len(xauusd_results)*100:.1f}%)")

        if eurusd_profitable:
            eur_avg_tp = np.mean([r['tp_percentage'] for r in eurusd_profitable])
            eur_avg_exp = np.mean([r['expected_return'] for r in eurusd_profitable])
            print(f"   EURUSD avg TP rate (profitable): {eur_avg_tp:.1f}%")
            print(f"   EURUSD avg expected return: {eur_avg_exp:+.4f}")

        if xauusd_profitable:
            xau_avg_tp = np.mean([r['tp_percentage'] for r in xauusd_profitable])
            xau_avg_exp = np.mean([r['expected_return'] for r in xauusd_profitable])
            print(f"   XAUUSD avg TP rate (profitable): {xau_avg_tp:.1f}%")
            print(f"   XAUUSD avg expected return: {xau_avg_exp:+.4f}")

        # Fakeout analysis
        print(f"\n🚨 FAKEOUT ANALYSIS (SL% = Fakeout Rate):")
        print("-" * 45)

        if eurusd_profitable and xauusd_profitable:
            eur_avg_sl = np.mean([r['sl_percentage'] for r in eurusd_profitable])
            xau_avg_sl = np.mean([r['sl_percentage'] for r in xauusd_profitable])

            print(f"   EURUSD avg fakeout rate: {eur_avg_sl:.1f}%")
            print(f"   XAUUSD avg fakeout rate: {xau_avg_sl:.1f}%")

            if eur_avg_sl < xau_avg_sl:
                print(f"   ✅ EURUSD has {xau_avg_sl - eur_avg_sl:.1f}% fewer fakeouts")
            else:
                print(f"   ❌ XAUUSD has {eur_avg_sl - xau_avg_sl:.1f}% fewer fakeouts")

        # Final verdict
        print(f"\n🎯 FINAL VERDICT:")
        print("=" * 20)

        if eurusd_wins > xauusd_wins:
            print(f"   🏆 EURUSD WINS ({eurusd_wins} vs {xauusd_wins})")
            print(f"   ✅ EURUSD respects confluences better")
        elif xauusd_wins > eurusd_wins:
            print(f"   🏆 XAUUSD WINS ({xauusd_wins} vs {eurusd_wins})")
            print(f"   ❌ XAUUSD actually respects confluences better")
        else:
            print(f"   🤝 TIE ({eurusd_wins} vs {xauusd_wins})")
            print(f"   ⚖️  Both pairs respect confluences equally")

        print(f"\n⚠️  DATA LIMITATIONS:")
        print(f"   EURUSD data: {(eurusd_df.index[-1] - eurusd_df.index[0]).days} days")
        print(f"   XAUUSD data: {(xauusd_df.index[-1] - xauusd_df.index[0]).days} days")
        print(f"   Both datasets are < 1 year - limited statistical power")

    except Exception as e:
        print(f"❌ Comparison error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(compare_confluence_respect())