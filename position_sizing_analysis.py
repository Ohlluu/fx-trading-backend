#!/usr/bin/env python3
"""
POSITION SIZING ANALYSIS FOR XAUUSD
Analyzing the impact of larger lot sizes vs tighter stops
Testing if larger positions with smaller SL distances are viable
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np

async def analyze_position_sizing_strategies():
    print("💰 XAUUSD POSITION SIZING STRATEGY ANALYSIS")
    print("=" * 50)
    print("🔍 Testing larger lots vs tighter stops for daily profit targets...")

    try:
        from app.oanda_feed import get_xauusd_candles

        # Get XAUUSD data
        df = await get_xauusd_candles(count=5000)
        if df is None or df.empty:
            print("❌ No XAUUSD data available")
            return

        print(f"✅ Analyzing {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        chicago_tz = pytz.timezone('America/Chicago')

        # Add indicators for our best confluence (SMA50 + EMA50 + SMA100)
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['sma_100'] = df['close'].rolling(window=100).mean()

        df['above_sma50'] = df['close'] > df['sma_50']
        df['above_ema50'] = df['close'] > df['ema_50']
        df['above_sma100'] = df['close'] > df['sma_100']

        # Define position sizing scenarios
        scenarios = [
            {
                'name': 'Conservative (Original)',
                'tp_pct': 1.0,      # 1% TP
                'sl_pct': 1.5,      # 1.5% SL
                'lot_multiplier': 1.0,
                'daily_target': 100  # $100 daily target
            },
            {
                'name': 'Larger Lot, Tighter SL',
                'tp_pct': 0.5,      # 0.5% TP
                'sl_pct': 0.75,     # 0.75% SL
                'lot_multiplier': 2.0,
                'daily_target': 100
            },
            {
                'name': 'Much Larger Lot, Very Tight',
                'tp_pct': 0.3,      # 0.3% TP
                'sl_pct': 0.45,     # 0.45% SL
                'lot_multiplier': 3.33,
                'daily_target': 100
            },
            {
                'name': 'Aggressive Scalp',
                'tp_pct': 0.2,      # 0.2% TP
                'sl_pct': 0.3,      # 0.3% SL
                'lot_multiplier': 5.0,
                'daily_target': 100
            }
        ]

        results = []

        for scenario in scenarios:
            print(f"\n🧪 Testing: {scenario['name']}")
            print(f"   TP: {scenario['tp_pct']}%, SL: {scenario['sl_pct']}%, Lot: {scenario['lot_multiplier']}x")

            tp_wins = 0
            sl_losses = 0
            no_exits = 0
            total_profit = 0
            trades_taken = []

            # Track intraday movements that would hit tight stops
            tight_sl_hits = 0
            total_signals = 0

            for i in range(150, len(df) - 100):
                try:
                    current_candle = df.iloc[i]

                    # Use our best confluence combination
                    all_factors_present = (
                        current_candle.get('above_sma50', False) and
                        current_candle.get('above_ema50', False) and
                        current_candle.get('above_sma100', False)
                    )

                    if not all_factors_present:
                        continue

                    total_signals += 1
                    entry_price = current_candle['close']
                    entry_time = df.index[i]

                    # Calculate TP/SL based on scenario
                    tp_price = entry_price * (1 + scenario['tp_pct']/100)
                    sl_price = entry_price * (1 - scenario['sl_pct']/100)

                    # Check if current candle already breaches tight SL
                    if current_candle['low'] <= sl_price:
                        tight_sl_hits += 1
                        continue  # Skip this trade as it would hit SL immediately

                    # Position sizing calculation
                    sl_distance_dollars = entry_price * (scenario['sl_pct']/100)
                    base_position_size = scenario['daily_target'] / (scenario['tp_pct']/100 * entry_price)
                    actual_position_size = base_position_size * scenario['lot_multiplier']

                    trade_data = {
                        'entry_time': entry_time.tz_convert(chicago_tz),
                        'entry_price': entry_price,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'position_size': actual_position_size,
                        'scenario': scenario['name']
                    }

                    # Check outcome in next 100 candles
                    future_candles = df.iloc[i+1:i+101]
                    tp_hit = False
                    sl_hit = False
                    tp_time = None
                    sl_time = None

                    for j, future_candle in future_candles.iterrows():
                        # Check for TP hit
                        if future_candle['high'] >= tp_price and not tp_hit:
                            tp_hit = True
                            tp_time = j
                        # Check for SL hit
                        if future_candle['low'] <= sl_price and not sl_hit:
                            sl_hit = True
                            sl_time = j

                        # First hit wins
                        if tp_hit and sl_hit:
                            if tp_time <= sl_time:
                                tp_wins += 1
                                profit = scenario['tp_pct']/100 * entry_price * actual_position_size
                                total_profit += profit
                                trade_data['outcome'] = 'TP'
                                trade_data['profit'] = profit
                                trades_taken.append(trade_data)
                                break
                            else:
                                sl_losses += 1
                                loss = scenario['sl_pct']/100 * entry_price * actual_position_size
                                total_profit -= loss
                                trade_data['outcome'] = 'SL'
                                trade_data['profit'] = -loss
                                trades_taken.append(trade_data)
                                break
                        elif tp_hit:
                            tp_wins += 1
                            profit = scenario['tp_pct']/100 * entry_price * actual_position_size
                            total_profit += profit
                            trade_data['outcome'] = 'TP'
                            trade_data['profit'] = profit
                            trades_taken.append(trade_data)
                            break
                        elif sl_hit:
                            sl_losses += 1
                            loss = scenario['sl_pct']/100 * entry_price * actual_position_size
                            total_profit -= loss
                            trade_data['outcome'] = 'SL'
                            trade_data['profit'] = -loss
                            trades_taken.append(trade_data)
                            break

                    # No exit
                    if not tp_hit and not sl_hit:
                        no_exits += 1
                        trade_data['outcome'] = 'NO_EXIT'
                        trade_data['profit'] = 0
                        trades_taken.append(trade_data)

                except Exception as e:
                    continue

            # Calculate results
            total_trades = tp_wins + sl_losses + no_exits
            win_rate = (tp_wins / total_trades * 100) if total_trades > 0 else 0
            avg_profit_per_trade = total_profit / total_trades if total_trades > 0 else 0

            # Calculate daily profit potential
            trades_per_day = total_trades / ((df.index[-1] - df.index[0]).days) if total_trades > 0 else 0
            daily_profit_potential = avg_profit_per_trade * trades_per_day

            # Immediate SL hit rate
            immediate_sl_rate = (tight_sl_hits / total_signals * 100) if total_signals > 0 else 0

            results.append({
                'scenario': scenario['name'],
                'tp_pct': scenario['tp_pct'],
                'sl_pct': scenario['sl_pct'],
                'lot_multiplier': scenario['lot_multiplier'],
                'total_trades': total_trades,
                'win_rate': win_rate,
                'tp_wins': tp_wins,
                'sl_losses': sl_losses,
                'total_profit': total_profit,
                'avg_profit_per_trade': avg_profit_per_trade,
                'trades_per_day': trades_per_day,
                'daily_profit_potential': daily_profit_potential,
                'immediate_sl_rate': immediate_sl_rate,
                'viable': win_rate > 50 and immediate_sl_rate < 30
            })

            print(f"   Trades: {total_trades}, Win Rate: {win_rate:.1f}%, Immediate SL: {immediate_sl_rate:.1f}%")
            print(f"   Daily Profit Potential: ${daily_profit_potential:.2f}")

        # Results summary
        print(f"\n📊 POSITION SIZING STRATEGY COMPARISON:")
        print("=" * 80)
        print(f"{'Strategy':<25} {'Trades':<7} {'Win%':<6} {'ImmSL%':<7} {'$/Trade':<8} {'$/Day':<8} {'Viable?'}")
        print("-" * 80)

        for result in results:
            viable = "✅ YES" if result['viable'] else "❌ NO"
            print(f"{result['scenario']:<25} {result['total_trades']:<7} "
                  f"{result['win_rate']:<6.1f} {result['immediate_sl_rate']:<7.1f} "
                  f"${result['avg_profit_per_trade']:<7.2f} ${result['daily_profit_potential']:<7.2f} {viable}")

        # Detailed analysis
        print(f"\n🔍 DETAILED ANALYSIS:")
        print("=" * 30)

        viable_strategies = [r for r in results if r['viable']]
        if viable_strategies:
            best = max(viable_strategies, key=lambda x: x['daily_profit_potential'])
            print(f"\n🏆 BEST VIABLE STRATEGY: {best['scenario']}")
            print(f"   TP/SL: {best['tp_pct']}%/{best['sl_pct']}%")
            print(f"   Lot Multiplier: {best['lot_multiplier']}x")
            print(f"   Win Rate: {best['win_rate']:.1f}%")
            print(f"   Immediate SL Rate: {best['immediate_sl_rate']:.1f}%")
            print(f"   Daily Profit Potential: ${best['daily_profit_potential']:.2f}")
        else:
            print(f"\n🚨 NO VIABLE STRATEGIES FOUND")
            print(f"   All tested strategies either have:")
            print(f"   • Win rate below 50%, OR")
            print(f"   • Immediate SL rate above 30%")

        # Risk analysis
        print(f"\n⚠️  RISK ANALYSIS:")
        print("-" * 20)

        for result in results:
            if result['scenario'] != 'Conservative (Original)':
                original = results[0]  # First is original
                risk_increase = (result['immediate_sl_rate'] - original['immediate_sl_rate'])
                win_rate_change = (result['win_rate'] - original['win_rate'])

                print(f"\n{result['scenario']}:")
                print(f"   Immediate SL risk: +{risk_increase:.1f}% vs original")
                print(f"   Win rate change: {win_rate_change:+.1f}% vs original")

                if result['immediate_sl_rate'] > 25:
                    print(f"   🚨 HIGH RISK: {result['immediate_sl_rate']:.1f}% immediate SL rate")
                elif result['immediate_sl_rate'] > 15:
                    print(f"   ⚠️  MODERATE RISK: {result['immediate_sl_rate']:.1f}% immediate SL rate")
                else:
                    print(f"   ✅ LOW RISK: {result['immediate_sl_rate']:.1f}% immediate SL rate")

        # Practical recommendations
        print(f"\n💡 PRACTICAL RECOMMENDATIONS:")
        print("=" * 35)

        if viable_strategies:
            print(f"   ✅ ANSWER: YES, larger lot sizes are possible")
            print(f"   📈 Best approach: {viable_strategies[0]['scenario']}")
            print(f"   🎯 Expected performance:")
            print(f"      • {viable_strategies[0]['win_rate']:.1f}% win rate")
            print(f"      • {viable_strategies[0]['immediate_sl_rate']:.1f}% immediate SL risk")
            print(f"      • ${viable_strategies[0]['daily_profit_potential']:.2f}/day potential")
        else:
            print(f"   ❌ ANSWER: NO, larger lot sizes too risky with XAUUSD")
            print(f"   🚨 All tested approaches have:")
            print(f"      • High immediate SL rates (>30%)")
            print(f"      • OR low win rates (<50%)")
            print(f"   💡 Stick with original 1:1.5 R:R strategy")

        print(f"\n📋 KEY TAKEAWAYS:")
        print(f"   • XAUUSD moves in large ranges - tight stops get hit frequently")
        print(f"   • Confluence signals need room to breathe")
        print(f"   • Position sizing up may work, but requires wider stops")
        print(f"   • Consider multiple smaller trades instead of one large trade")

    except Exception as e:
        print(f"❌ Position sizing analysis error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(analyze_position_sizing_strategies())