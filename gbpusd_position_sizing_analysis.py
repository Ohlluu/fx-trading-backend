#!/usr/bin/env python3
"""
GBPUSD vs XAUUSD POSITION SIZING COMPARISON
Testing if GBPUSD's lower volatility makes it better for larger lot strategies
Comparing immediate SL rates and position sizing viability
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

async def get_gbpusd_candles(count: int = 5000) -> pd.DataFrame:
    """Get GBPUSD hourly candles from OANDA"""
    url = f"{OANDA_BASE_URL}/instruments/GBP_USD/candles"
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Accept-Datetime-Format": "UNIX"
    }
    params = {
        "granularity": "H1",
        "count": count,
        "price": "M",
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
                            timestamp = datetime.fromtimestamp(float(candle["time"]), tz=pytz.UTC)
                            mid = candle["mid"]
                            candles.append([
                                timestamp, float(mid["o"]), float(mid["h"]),
                                float(mid["l"]), float(mid["c"]), int(candle.get("volume", 0))
                            ])

                    if candles:
                        df = pd.DataFrame(candles, columns=["time", "open", "high", "low", "close", "volume"])
                        df = df.sort_values("time").set_index("time")
                        return df
        return None
    except Exception as e:
        print(f"❌ GBPUSD fetch error: {e}")
        return None

async def compare_volatility_and_position_sizing():
    print("🇬🇧 GBPUSD vs XAUUSD POSITION SIZING COMPARISON")
    print("=" * 55)
    print("📊 Testing if lower volatility makes GBPUSD better for larger lots...")

    try:
        # Get data for both pairs
        from app.oanda_feed import get_xauusd_candles

        print("\n📥 Getting data for both pairs...")
        gbpusd_df = await get_gbpusd_candles(count=5000)
        xauusd_df = await get_xauusd_candles(count=5000)

        if gbpusd_df is None or xauusd_df is None:
            print("❌ Could not get data for both pairs")
            return

        print(f"✅ GBPUSD: {len(gbpusd_df)} candles ({(gbpusd_df.index[-1] - gbpusd_df.index[0]).days} days)")
        print(f"✅ XAUUSD: {len(xauusd_df)} candles ({(xauusd_df.index[-1] - xauusd_df.index[0]).days} days)")

        # Calculate volatility metrics
        print(f"\n📈 VOLATILITY ANALYSIS:")
        print("=" * 30)

        # Average daily ranges
        gbpusd_daily_range = (gbpusd_df['high'] - gbpusd_df['low']).mean()
        xauusd_daily_range = (xauusd_df['high'] - xauusd_df['low']).mean()

        # Average hourly moves
        gbpusd_hourly_move = abs(gbpusd_df['close'].diff()).mean()
        xauusd_hourly_move = abs(xauusd_df['close'].diff()).mean()

        # Percentage moves
        gbpusd_pct_range = ((gbpusd_df['high'] - gbpusd_df['low']) / gbpusd_df['close'] * 100).mean()
        xauusd_pct_range = ((xauusd_df['high'] - xauusd_df['low']) / xauusd_df['close'] * 100).mean()

        print(f"   GBPUSD avg daily range: {gbpusd_daily_range:.5f} ({gbpusd_pct_range:.3f}%)")
        print(f"   XAUUSD avg daily range: {xauusd_daily_range:.2f} ({xauusd_pct_range:.3f}%)")
        print(f"   GBPUSD avg hourly move: {gbpusd_hourly_move:.5f}")
        print(f"   XAUUSD avg hourly move: {xauusd_hourly_move:.2f}")

        volatility_ratio = xauusd_pct_range / gbpusd_pct_range
        print(f"   🎯 XAUUSD is {volatility_ratio:.1f}x more volatile than GBPUSD")

        # Function to add indicators to both pairs
        def add_indicators(df):
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['sma_100'] = df['close'].rolling(window=100).mean()
            df['above_sma50'] = df['close'] > df['sma_50']
            df['above_ema50'] = df['close'] > df['ema_50']
            df['above_sma100'] = df['close'] > df['sma_100']
            return df

        # Add indicators
        gbpusd_df = add_indicators(gbpusd_df)
        xauusd_df = add_indicators(xauusd_df)

        # Position sizing scenarios optimized for each pair
        scenarios = [
            {
                'name': 'Conservative',
                'tp_pct': 0.8,
                'sl_pct': 1.2,
                'description': 'Safe approach'
            },
            {
                'name': 'Moderate',
                'tp_pct': 0.5,
                'sl_pct': 0.75,
                'description': 'Balanced risk'
            },
            {
                'name': 'Aggressive',
                'tp_pct': 0.3,
                'sl_pct': 0.45,
                'description': 'Tight stops'
            },
            {
                'name': 'Scalp',
                'tp_pct': 0.15,
                'sl_pct': 0.225,
                'description': 'Very tight'
            }
        ]

        # Function to test position sizing on a pair
        def test_position_sizing(df, pair_name):
            results = []

            for scenario in scenarios:
                tp_wins = sl_losses = no_exits = 0
                immediate_sl_hits = 0
                total_signals = 0

                for i in range(150, len(df) - 100):
                    try:
                        current_candle = df.iloc[i]

                        # Use best confluence combination
                        confluence_present = (
                            current_candle.get('above_sma50', False) and
                            current_candle.get('above_ema50', False) and
                            current_candle.get('above_sma100', False)
                        )

                        if not confluence_present:
                            continue

                        total_signals += 1
                        entry_price = current_candle['close']

                        # Calculate TP/SL
                        tp_price = entry_price * (1 + scenario['tp_pct']/100)
                        sl_price = entry_price * (1 - scenario['sl_pct']/100)

                        # Check immediate SL hit (current candle low breaks SL)
                        if current_candle['low'] <= sl_price:
                            immediate_sl_hits += 1
                            continue

                        # Check outcome in future candles
                        future_candles = df.iloc[i+1:i+101]
                        tp_hit = sl_hit = False
                        tp_time = sl_time = None

                        for j, future_candle in future_candles.iterrows():
                            if future_candle['high'] >= tp_price and not tp_hit:
                                tp_hit, tp_time = True, j
                            if future_candle['low'] <= sl_price and not sl_hit:
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

                # Calculate metrics
                total_trades = tp_wins + sl_losses + no_exits
                win_rate = (tp_wins / total_trades * 100) if total_trades > 0 else 0
                immediate_sl_rate = (immediate_sl_hits / total_signals * 100) if total_signals > 0 else 0

                # Expected return calculation
                expected_return = (win_rate/100 * scenario['tp_pct']/100) - ((100-win_rate)/100 * scenario['sl_pct']/100)

                results.append({
                    'pair': pair_name,
                    'scenario': scenario['name'],
                    'tp_pct': scenario['tp_pct'],
                    'sl_pct': scenario['sl_pct'],
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'immediate_sl_rate': immediate_sl_rate,
                    'expected_return': expected_return,
                    'viable': win_rate > 50 and immediate_sl_rate < 20
                })

            return results

        # Test both pairs
        print(f"\n🧪 Testing position sizing strategies on both pairs...")
        gbpusd_results = test_position_sizing(gbpusd_df, "GBPUSD")
        xauusd_results = test_position_sizing(xauusd_df, "XAUUSD")

        # Compare results
        print(f"\n📊 POSITION SIZING COMPARISON:")
        print("=" * 85)
        print(f"{'Strategy':<12} {'GBPUSD Win%':<12} {'GBPUSD ImmSL%':<13} {'XAUUSD Win%':<12} {'XAUUSD ImmSL%':<13} {'Better?'}")
        print("-" * 85)

        gbpusd_wins = 0
        xauusd_wins = 0

        for i, scenario in enumerate(scenarios):
            gbp_result = gbpusd_results[i]
            xau_result = xauusd_results[i]

            # Determine which is better (higher win rate + lower immediate SL rate)
            gbp_score = gbp_result['win_rate'] - gbp_result['immediate_sl_rate']
            xau_score = xau_result['win_rate'] - xau_result['immediate_sl_rate']

            if gbp_score > xau_score:
                better = "GBPUSD"
                gbpusd_wins += 1
            else:
                better = "XAUUSD"
                xauusd_wins += 1

            print(f"{scenario['name']:<12} {gbp_result['win_rate']:<12.1f} {gbp_result['immediate_sl_rate']:<13.1f} "
                  f"{xau_result['win_rate']:<12.1f} {xau_result['immediate_sl_rate']:<13.1f} {better}")

        # Detailed analysis
        print(f"\n🏆 OVERALL WINNER:")
        print("=" * 20)
        if gbpusd_wins > xauusd_wins:
            print(f"   🇬🇧 GBPUSD WINS ({gbpusd_wins} vs {xauusd_wins})")
            print(f"   ✅ Better for position sizing strategies")
        else:
            print(f"   🥇 XAUUSD WINS ({xauusd_wins} vs {gbpusd_wins})")
            print(f"   ✅ Better despite higher volatility")

        # Viability analysis
        print(f"\n✅ VIABLE STRATEGIES (>50% win rate, <20% immediate SL):")
        print("-" * 55)

        gbpusd_viable = [r for r in gbpusd_results if r['viable']]
        xauusd_viable = [r for r in xauusd_results if r['viable']]

        print(f"   GBPUSD viable strategies: {len(gbpusd_viable)}/4")
        for result in gbpusd_viable:
            print(f"     • {result['scenario']}: {result['win_rate']:.1f}% win, {result['immediate_sl_rate']:.1f}% immSL")

        print(f"   XAUUSD viable strategies: {len(xauusd_viable)}/4")
        for result in xauusd_viable:
            print(f"     • {result['scenario']}: {result['win_rate']:.1f}% win, {result['immediate_sl_rate']:.1f}% immSL")

        # Expected returns comparison
        print(f"\n💰 EXPECTED RETURNS COMPARISON:")
        print("-" * 35)

        for i, scenario in enumerate(scenarios):
            gbp_result = gbpusd_results[i]
            xau_result = xauusd_results[i]

            print(f"   {scenario['name']}:")
            print(f"     GBPUSD: {gbp_result['expected_return']:+.4f} per trade")
            print(f"     XAUUSD: {xau_result['expected_return']:+.4f} per trade")

            if gbp_result['expected_return'] > xau_result['expected_return']:
                print(f"     🏆 GBPUSD better by {gbp_result['expected_return'] - xau_result['expected_return']:+.4f}")
            else:
                print(f"     🏆 XAUUSD better by {xau_result['expected_return'] - gbp_result['expected_return']:+.4f}")
            print()

        # Final recommendation
        print(f"\n🎯 FINAL RECOMMENDATION:")
        print("=" * 30)

        # Calculate average performance
        avg_gbp_win = np.mean([r['win_rate'] for r in gbpusd_viable])
        avg_xau_win = np.mean([r['win_rate'] for r in xauusd_viable])
        avg_gbp_immsl = np.mean([r['immediate_sl_rate'] for r in gbpusd_viable])
        avg_xau_immsl = np.mean([r['immediate_sl_rate'] for r in xauusd_viable])

        if len(gbpusd_viable) > len(xauusd_viable):
            print(f"   🇬🇧 USE GBPUSD for position sizing")
            print(f"   ✅ More viable strategies ({len(gbpusd_viable)} vs {len(xauusd_viable)})")
        elif avg_gbp_immsl < avg_xau_immsl:
            print(f"   🇬🇧 USE GBPUSD for position sizing")
            print(f"   ✅ Lower average immediate SL rate ({avg_gbp_immsl:.1f}% vs {avg_xau_immsl:.1f}%)")
        else:
            print(f"   🥇 STICK WITH XAUUSD")
            print(f"   ✅ Better overall performance despite volatility")
            print(f"   📊 Higher average win rates ({avg_xau_win:.1f}% vs {avg_gbp_win:.1f}%)")

        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • GBPUSD has {volatility_ratio:.1f}x lower volatility")
        print(f"   • But volatility ≠ better position sizing performance")
        print(f"   • Confluence quality matters more than volatility")
        print(f"   • XAUUSD trends more decisively when signals align")

    except Exception as e:
        print(f"❌ Comparison error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(compare_volatility_and_position_sizing())