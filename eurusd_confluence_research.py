#!/usr/bin/env python3
"""
EURUSD CONFLUENCE RESEARCH & BACKTEST
Testing if EURUSD truly respects confluences more than other pairs
1:1 R:R analysis with comprehensive confluence testing
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np
import httpx

# OANDA API Configuration
OANDA_API_KEY = "1c2ee716aac27b425f2fd7a8ffbe9b9a-7a3b3da61668a804b56e2974ff21aaa0"
OANDA_BASE_URL = "https://api-fxpractice.oanda.com/v3"
ACCOUNT_ID = "101-001-37143591-001"

async def get_eurusd_candles(count: int = 5000) -> pd.DataFrame:
    """Get EURUSD hourly candles from OANDA"""
    url = f"{OANDA_BASE_URL}/instruments/EUR_USD/candles"
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Accept-Datetime-Format": "UNIX"
    }
    params = {
        "granularity": "H1",
        "count": count,
        "price": "M",  # Mid prices
        "includeIncompleteCandles": "false"
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.get(url, headers=headers, params=params)

            if response.status_code == 200:
                data = response.json()

                if "candles" in data and data["candles"]:
                    candles = []
                    for candle in data["candles"]:
                        if candle["complete"]:
                            timestamp = datetime.fromtimestamp(
                                float(candle["time"]),
                                tz=pytz.UTC
                            )

                            mid = candle["mid"]
                            candles.append([
                                timestamp,
                                float(mid["o"]),  # open
                                float(mid["h"]),  # high
                                float(mid["l"]),  # low
                                float(mid["c"]),  # close
                                int(candle.get("volume", 0))
                            ])

                    if candles:
                        df = pd.DataFrame(candles, columns=[
                            "time", "open", "high", "low", "close", "volume"
                        ])
                        df = df.sort_values("time")
                        df = df.set_index("time")

                        print(f"✅ EURUSD: Retrieved {len(df)} candles from {df.index[0]} to {df.index[-1]}")
                        return df

        print(f"❌ OANDA EURUSD error: {response.status_code}")
        return None

    except Exception as e:
        print(f"❌ EURUSD fetch error: {e}")
        return None

async def eurusd_confluence_research():
    print("🇪🇺 EURUSD CONFLUENCE RESEARCH & BACKTEST")
    print("=" * 50)
    print("🔍 Testing if EURUSD respects confluences more than other pairs...")

    try:
        # Get EURUSD data
        df = await get_eurusd_candles(count=5000)
        if df is None or df.empty:
            print("❌ No EURUSD data available")
            return

        print(f"📊 Data span: {(df.index[-1] - df.index[0]).days} days ({(df.index[-1] - df.index[0]).days/365.25:.2f} years)")

        chicago_tz = pytz.timezone('America/Chicago')

        # Add comprehensive technical indicators
        print("\n🔧 Adding technical indicators...")

        # Moving averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_100'] = df['close'].rolling(window=100).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()

        # MA positions
        df['above_sma10'] = df['close'] > df['sma_10']
        df['above_sma20'] = df['close'] > df['sma_20']
        df['above_sma50'] = df['close'] > df['sma_50']
        df['above_sma100'] = df['close'] > df['sma_100']
        df['above_sma200'] = df['close'] > df['sma_200']
        df['above_ema10'] = df['close'] > df['ema_10']
        df['above_ema20'] = df['close'] > df['ema_20']
        df['above_ema50'] = df['close'] > df['ema_50']

        # MA trends
        df['sma20_uptrend'] = df['sma_20'].diff(5) > 0
        df['sma50_uptrend'] = df['sma_50'].diff(5) > 0
        df['sma200_uptrend'] = df['sma_200'].diff(5) > 0
        df['ema20_uptrend'] = df['ema_20'].diff(5) > 0
        df['ema50_uptrend'] = df['ema_50'].diff(5) > 0

        # MA crossovers (key for EURUSD)
        df['ema20_above_sma50'] = df['ema_20'] > df['sma_50']
        df['sma20_above_sma50'] = df['sma_20'] > df['sma_50']
        df['price_above_all_ma'] = (df['above_sma10'] & df['above_sma20'] &
                                   df['above_sma50'] & df['above_ema20'])

        # Price action
        df['is_green'] = df['close'] > df['open']
        df['is_red'] = df['close'] < df['open']
        df['prev_green'] = df['is_green'].shift(1)
        df['prev_red'] = df['is_red'].shift(1)
        df['body_size'] = abs(df['close'] - df['open'])
        df['wick_size'] = (df['high'] - df['low']) - df['body_size']
        df['strong_candle'] = df['body_size'] > df['wick_size']

        # Support/Resistance levels
        df['high_20'] = df['high'].rolling(window=20).max()
        df['low_20'] = df['low'].rolling(window=20).min()
        df['high_50'] = df['high'].rolling(window=50).max()
        df['low_50'] = df['low'].rolling(window=50).min()

        df['near_resistance_20'] = abs(df['close'] - df['high_20']) <= 0.0010  # 10 pips
        df['near_support_20'] = abs(df['close'] - df['low_20']) <= 0.0010
        df['near_resistance_50'] = abs(df['close'] - df['high_50']) <= 0.0015  # 15 pips
        df['near_support_50'] = abs(df['close'] - df['low_50']) <= 0.0015

        # Break levels
        df['break_resistance_20'] = df['close'] > df['high_20'].shift(1)
        df['break_support_20'] = df['close'] < df['low_20'].shift(1)
        df['break_resistance_50'] = df['close'] > df['high_50'].shift(1)
        df['break_support_50'] = df['close'] < df['low_50'].shift(1)

        # Session analysis (EURUSD loves London session)
        df_with_hour = df.copy()
        df_with_hour['hour_london'] = df.index.tz_convert('Europe/London').hour
        df['london_session'] = (df_with_hour['hour_london'] >= 8) & (df_with_hour['hour_london'] <= 16)
        df['london_open'] = (df_with_hour['hour_london'] >= 8) & (df_with_hour['hour_london'] <= 10)
        df['london_close'] = (df_with_hour['hour_london'] >= 15) & (df_with_hour['hour_london'] <= 16)

        print(f"✅ Added comprehensive technical indicators for EURUSD")

        # Define confluence factors to test
        confluence_factors = [
            'above_sma20', 'above_sma50', 'above_sma100', 'above_sma200',
            'above_ema20', 'above_ema50',
            'sma20_uptrend', 'sma50_uptrend', 'sma200_uptrend',
            'ema20_uptrend', 'ema50_uptrend',
            'ema20_above_sma50', 'sma20_above_sma50', 'price_above_all_ma',
            'is_green', 'prev_green', 'strong_candle',
            'near_resistance_20', 'near_support_20', 'near_resistance_50', 'near_support_50',
            'break_resistance_20', 'break_support_20', 'break_resistance_50', 'break_support_50',
            'london_session', 'london_open', 'london_close'
        ]

        print(f"\n🧪 Testing {len(confluence_factors)} confluence factors for EURUSD (1:1 R:R)...")

        results = []

        for factor in confluence_factors:
            print(f"   Analyzing: {factor}")

            tp_wins = 0
            sl_losses = 0
            no_exits = 0

            # Test both BUY and SELL signals
            for signal_type in ['BUY', 'SELL']:
                for i in range(250, len(df) - 100):  # Need MA history + future space
                    try:
                        current_candle = df.iloc[i]

                        # Check if confluence factor is present
                        factor_present = current_candle.get(factor, False)
                        if not factor_present:
                            continue

                        # Basic direction filter
                        if signal_type == 'BUY':
                            if not current_candle.get('above_sma50', False):
                                continue
                        else:  # SELL
                            if current_candle.get('above_sma50', False):
                                continue

                        entry_price = current_candle['close']

                        # 1:1 R:R calculation
                        if signal_type == 'BUY':
                            tp_price = entry_price * (1 + 0.01)   # +1% TP
                            sl_price = entry_price * (1 - 0.01)   # -1% SL
                        else:  # SELL
                            tp_price = entry_price * (1 - 0.01)   # -1% TP
                            sl_price = entry_price * (1 + 0.01)   # +1% SL

                        # Check outcome in next 100 candles
                        future_candles = df.iloc[i+1:i+101]
                        tp_hit = False
                        sl_hit = False
                        tp_time = None
                        sl_time = None

                        for j, future_candle in future_candles.iterrows():
                            # Check for TP/SL hits
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

                            # First hit wins
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

                        # No exit within 100 candles
                        if not tp_hit and not sl_hit:
                            no_exits += 1

                    except Exception as e:
                        continue

            # Calculate results
            total_trades = tp_wins + sl_losses + no_exits

            if total_trades >= 10:  # Minimum trade threshold
                tp_pct = (tp_wins / total_trades) * 100
                sl_pct = (sl_losses / total_trades) * 100
                no_exit_pct = (no_exits / total_trades) * 100

                # Expected return for 1:1 R:R
                expected_return = (tp_pct/100 * 0.01) - (sl_pct/100 * 0.01)

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

        print(f"\n📊 EURUSD CONFLUENCE RESULTS (1:1 R:R):")
        print("=" * 85)
        print(f"{'Factor':<25} {'Trades':<7} {'TP%':<6} {'SL%':<6} {'NoExit%':<8} {'ExpRet':<8} {'Prof?'}")
        print("-" * 85)

        profitable_count = 0
        for result in results:
            profitable = "✅ YES" if result['profitable'] else "❌ NO"
            if result['profitable']:
                profitable_count += 1

            print(f"{result['factor']:<25} {result['total_trades']:<7} "
                  f"{result['tp_percentage']:<6.1f} {result['sl_percentage']:<6.1f} "
                  f"{result['no_exit_percentage']:<8.1f} {result['expected_return']:<+8.4f} {profitable}")

        # Analysis summary
        print(f"\n🎯 EURUSD CONFLUENCE ANALYSIS SUMMARY:")
        print("=" * 40)
        print(f"   Total factors tested: {len(results)}")
        print(f"   Profitable factors: {profitable_count}")
        print(f"   Success rate: {profitable_count/len(results)*100:.1f}%")

        if profitable_count > 0:
            best = results[0]
            print(f"\n🏆 BEST EURUSD CONFLUENCE:")
            print(f"   Factor: {best['factor']}")
            print(f"   Total trades: {best['total_trades']}")
            print(f"   TP rate: {best['tp_percentage']:.1f}%")
            print(f"   SL rate: {best['sl_percentage']:.1f}%")
            print(f"   Expected return: {best['expected_return']:+.4f} per trade")

            # Test best 3 combination
            print(f"\n🔬 TESTING TOP 3 COMBINATION:")
            top_3 = [results[i]['factor'] for i in range(min(3, len(results)))]
            print(f"   Combining: {' + '.join(top_3)}")

            combo_tp = combo_sl = combo_no_exit = 0

            for signal_type in ['BUY', 'SELL']:
                for i in range(250, len(df) - 100):
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
                            sl_price = entry_price * 0.99
                        else:
                            tp_price = entry_price * 0.99
                            sl_price = entry_price * 1.01

                        # Check outcome
                        future_candles = df.iloc[i+1:i+101]
                        tp_hit = sl_hit = False
                        tp_time = sl_time = None

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
                combo_expected = (combo_tp_pct/100 * 0.01) - (combo_sl_pct/100 * 0.01)

                print(f"   Total trades: {combo_total}")
                print(f"   TP rate: {combo_tp_pct:.1f}%")
                print(f"   SL rate: {combo_sl_pct:.1f}%")
                print(f"   Expected return: {combo_expected:+.4f} per trade")

        # Compare confluence respect
        print(f"\n🔍 CONFLUENCE RESPECT ANALYSIS:")
        print("-" * 35)

        if profitable_count > 0:
            avg_win_rate = np.mean([r['tp_percentage'] for r in results if r['profitable']])
            avg_expected_return = np.mean([r['expected_return'] for r in results if r['profitable']])

            print(f"   Average win rate (profitable factors): {avg_win_rate:.1f}%")
            print(f"   Average expected return: {avg_expected_return:+.4f}")
            print(f"   Profitable factor ratio: {profitable_count}/{len(results)} ({profitable_count/len(results)*100:.1f}%)")
        else:
            print(f"   🚨 NO profitable confluences found with 1:1 R:R")

        print(f"\n📈 DATA QUALITY ASSESSMENT:")
        print(f"   Candles analyzed: {len(df)}")
        print(f"   Time span: {(df.index[-1] - df.index[0]).days} days")
        print(f"   Average daily volatility: {(df['high'] - df['low']).mean():.5f}")

    except Exception as e:
        print(f"❌ EURUSD research error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(eurusd_confluence_research())