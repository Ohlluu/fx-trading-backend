#!/usr/bin/env python3
"""
XAUUSD Proven Confluence Analysis
Analyze which specific confluence factors lead to successful trades
Based on actual TP achievement data
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np

async def analyze_proven_confluences():
    print("🔍 ANALYZING PROVEN XAUUSD CONFLUENCE FACTORS")
    print("=" * 55)
    print("⚡ Finding which confluences actually work...")

    try:
        from app.oanda_feed import get_xauusd_candles
        from app.smart_confluence_system import evaluate_smart_confluence_signal

        # Get comprehensive data
        df = await get_xauusd_candles(count=5000)
        if df is None or df.empty:
            print("❌ No XAUUSD data available")
            return

        print(f"✅ Analyzing {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        chicago_tz = pytz.timezone('America/Chicago')

        # Track successful vs failed trades with confluence details
        successful_trades = []
        failed_trades = []

        print("\n🔍 Analyzing all trades and their confluence factors...")

        for i in range(60, len(df) - 100):
            try:
                current_price = df['close'].iloc[i]
                current_time = df.index[i]
                historical_df = df.iloc[:i+1].copy()

                # Get confluence signal
                result = evaluate_smart_confluence_signal(current_price, historical_df)

                if result and result.get('signal') in ['BUY', 'SELL']:
                    entry_price = result['entry_price']
                    tp_price = result['take_profit']
                    sl_price = result['stop_loss']
                    signal_type = result['signal']
                    confluence_score = result['confluence_score']

                    # Extract detailed confluence factors from the historical data
                    # Add technical indicators to the dataframe
                    hist_df = historical_df.copy()
                    hist_df['sma_20'] = hist_df['close'].rolling(window=20).mean()
                    hist_df['sma_50'] = hist_df['close'].rolling(window=50).mean()
                    hist_df['ema_20'] = hist_df['close'].ewm(span=20).mean()
                    hist_df['above_sma20'] = hist_df['close'] > hist_df['sma_20']
                    hist_df['above_sma50'] = hist_df['close'] > hist_df['sma_50']
                    hist_df['above_ema20'] = hist_df['close'] > hist_df['ema_20']
                    hist_df['sma20_slope'] = hist_df['sma_20'].diff(5)
                    hist_df['ema20_slope'] = hist_df['ema_20'].diff(5)
                    hist_df['uptrend_sma'] = hist_df['sma20_slope'] > 0
                    hist_df['uptrend_ema'] = hist_df['ema20_slope'] > 0
                    hist_df['is_green'] = hist_df['close'] > hist_df['open']
                    hist_df['recent_high_12h'] = hist_df['high'].rolling(window=12).max()
                    hist_df['near_recent_high'] = abs(hist_df['close'] - hist_df['recent_high_12h']) < 10

                    latest_candle = hist_df.iloc[-1]

                    # Create trade data with all confluence factors
                    trade_data = {
                        'entry_time': current_time.tz_convert(chicago_tz),
                        'signal': signal_type,
                        'entry_price': entry_price,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'confluence_score': confluence_score,
                        'above_sma20': bool(latest_candle.get('above_sma20', False)),
                        'above_sma50': bool(latest_candle.get('above_sma50', False)),
                        'above_ema20': bool(latest_candle.get('above_ema20', False)),
                        'uptrend_sma': bool(latest_candle.get('uptrend_sma', False)),
                        'uptrend_ema': bool(latest_candle.get('uptrend_ema', False)),
                        'is_green': bool(latest_candle.get('is_green', False)),
                        'near_recent_high': bool(latest_candle.get('near_recent_high', False)),
                    }

                    # Check if TP was hit in next 100 candles
                    future_candles = df.iloc[i+1:i+101]
                    tp_hit = False
                    sl_hit = False
                    tp_hit_time = None
                    sl_hit_time = None

                    for j, future_candle in future_candles.iterrows():
                        if signal_type == 'BUY':
                            if future_candle['high'] >= tp_price and not tp_hit:
                                tp_hit = True
                                tp_hit_time = j.tz_convert(chicago_tz)
                            if future_candle['low'] <= sl_price and not sl_hit:
                                sl_hit = True
                                sl_hit_time = j.tz_convert(chicago_tz)
                        else:  # SELL
                            if future_candle['low'] <= tp_price and not tp_hit:
                                tp_hit = True
                                tp_hit_time = j.tz_convert(chicago_tz)
                            if future_candle['high'] >= sl_price and not sl_hit:
                                sl_hit = True
                                sl_hit_time = j.tz_convert(chicago_tz)

                        # Determine outcome
                        if tp_hit and sl_hit:
                            if tp_hit_time <= sl_hit_time:
                                trade_data['outcome'] = 'TP_WIN'
                                trade_data['tp_hit_time'] = tp_hit_time
                                successful_trades.append(trade_data)
                                break
                            else:
                                trade_data['outcome'] = 'SL_LOSS'
                                failed_trades.append(trade_data)
                                break
                        elif tp_hit:
                            trade_data['outcome'] = 'TP_WIN'
                            trade_data['tp_hit_time'] = tp_hit_time
                            successful_trades.append(trade_data)
                            break
                        elif sl_hit:
                            trade_data['outcome'] = 'SL_LOSS'
                            failed_trades.append(trade_data)
                            break

                    # If neither hit within 100 candles, consider it a loss
                    if 'outcome' not in trade_data:
                        trade_data['outcome'] = 'NO_EXIT'
                        failed_trades.append(trade_data)

                # Progress indicator
                if i % 1000 == 0:
                    progress = (i / len(df)) * 100
                    print(f"   Progress: {progress:.1f}% - {len(successful_trades)} wins, {len(failed_trades)} losses")

            except Exception as e:
                continue

        print(f"\n📊 CONFLUENCE EFFECTIVENESS ANALYSIS:")
        print(f"   Total successful trades: {len(successful_trades)}")
        print(f"   Total failed trades: {len(failed_trades)}")

        if len(successful_trades) > 0 and len(failed_trades) > 0:
            # Analyze each confluence factor
            confluence_factors = [
                'above_sma20', 'above_sma50', 'above_ema20',
                'uptrend_sma', 'uptrend_ema', 'is_green', 'near_recent_high'
            ]

            print(f"\n🎯 PROVEN CONFLUENCE FACTORS (Win Rate Analysis):")
            print("-" * 60)

            factor_results = []

            for factor in confluence_factors:
                # Calculate win rates when factor is present vs absent
                wins_with_factor = sum(1 for trade in successful_trades if trade.get(factor, False))
                wins_without_factor = len(successful_trades) - wins_with_factor

                losses_with_factor = sum(1 for trade in failed_trades if trade.get(factor, False))
                losses_without_factor = len(failed_trades) - losses_with_factor

                total_with_factor = wins_with_factor + losses_with_factor
                total_without_factor = wins_without_factor + losses_without_factor

                win_rate_with = (wins_with_factor / total_with_factor * 100) if total_with_factor > 0 else 0
                win_rate_without = (wins_without_factor / total_without_factor * 100) if total_without_factor > 0 else 0

                improvement = win_rate_with - win_rate_without

                factor_results.append({
                    'factor': factor,
                    'win_rate_with': win_rate_with,
                    'win_rate_without': win_rate_without,
                    'improvement': improvement,
                    'trades_with': total_with_factor,
                    'trades_without': total_without_factor
                })

            # Sort by improvement (most effective first)
            factor_results.sort(key=lambda x: x['improvement'], reverse=True)

            for result in factor_results:
                factor = result['factor'].replace('_', ' ').title()
                print(f"   {factor:20s}: {result['win_rate_with']:5.1f}% with vs {result['win_rate_without']:5.1f}% without (+{result['improvement']:4.1f}%)")

            # Analyze confluence score effectiveness
            print(f"\n🔢 CONFLUENCE SCORE EFFECTIVENESS:")
            print("-" * 40)

            win_scores = [trade['confluence_score'] for trade in successful_trades]
            loss_scores = [trade['confluence_score'] for trade in failed_trades]

            if win_scores and loss_scores:
                print(f"   Winning trades avg score: {np.mean(win_scores):.1f}")
                print(f"   Losing trades avg score: {np.mean(loss_scores):.1f}")
                print(f"   Score difference: +{np.mean(win_scores) - np.mean(loss_scores):.1f}")

            # Best confluence combinations
            print(f"\n🏆 TOP PERFORMING CONFLUENCE COMBINATIONS:")
            print("-" * 50)

            top_factors = [result['factor'] for result in factor_results[:3]]

            # Find trades with all top 3 factors
            elite_winners = [
                trade for trade in successful_trades
                if all(trade.get(factor, False) for factor in top_factors)
            ]
            elite_losers = [
                trade for trade in failed_trades
                if all(trade.get(factor, False) for factor in top_factors)
            ]

            if elite_winners or elite_losers:
                elite_total = len(elite_winners) + len(elite_losers)
                elite_win_rate = (len(elite_winners) / elite_total * 100) if elite_total > 0 else 0
                print(f"   Top 3 factors combined: {elite_win_rate:.1f}% win rate ({len(elite_winners)}/{elite_total} trades)")

        print(f"\n✅ RECOMMENDED PROVEN CONFLUENCE CHECKLIST:")
        print("=" * 50)

        # Build the proven checklist based on analysis
        if factor_results:
            proven_factors = [result for result in factor_results if result['improvement'] > 0]

            print("📋 MANDATORY CHECKS (must have):")
            mandatory = [f for f in proven_factors if f['improvement'] > 5][:2]
            for i, factor in enumerate(mandatory, 1):
                name = factor['factor'].replace('_', ' ').title()
                print(f"   {i}. {name} (+{factor['improvement']:.1f}% improvement)")

            print("\n📈 SCORING FACTORS (add points):")
            scoring = [f for f in proven_factors if f['improvement'] > 0][2:]
            for i, factor in enumerate(scoring, 1):
                name = factor['factor'].replace('_', ' ').title()
                points = max(1, int(factor['improvement']))
                print(f"   {i}. {name}: +{points} points (+{factor['improvement']:.1f}% improvement)")

            total_factors = len(mandatory) + len(scoring)
            min_score = len(mandatory) * 3 + max(6, len(scoring) * 2)
            print(f"\n🎯 MINIMUM SCORE REQUIRED: {min_score} points")
            print(f"   Based on {total_factors} proven factors")

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(analyze_proven_confluences())