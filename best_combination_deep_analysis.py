#!/usr/bin/env python3
"""
COMPREHENSIVE ANALYSIS OF BEST COMBINATION
SMA50 + EMA50 + SMA100 (1:1.5 R:R) - ALL DETAILS
Every metric you need to know about this combination
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np
from collections import defaultdict

async def analyze_best_combination():
    print("🔍 COMPREHENSIVE ANALYSIS: SMA50 + EMA50 + SMA100 COMBINATION")
    print("=" * 65)
    print("📊 Extracting ALL performance metrics...")

    try:
        from app.oanda_feed import get_xauusd_candles

        # Get data
        df = await get_xauusd_candles(count=5000)
        if df is None or df.empty:
            print("❌ No XAUUSD data available")
            return

        print(f"✅ Analyzing {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        chicago_tz = pytz.timezone('America/Chicago')

        # Add technical indicators
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['sma_100'] = df['close'].rolling(window=100).mean()

        df['above_sma50'] = df['close'] > df['sma_50']
        df['above_ema50'] = df['close'] > df['ema_50']
        df['above_sma100'] = df['close'] > df['sma_100']

        # Storage for all trade details
        all_trades = []

        print("\n🔍 Scanning for combination signals and outcomes...")

        for signal_type in ['BUY', 'SELL']:
            for i in range(150, len(df) - 100):  # Need MA history + future outcome space
                try:
                    current_candle = df.iloc[i]
                    current_time = df.index[i]

                    # Check if ALL 3 factors are present
                    all_factors_present = (
                        current_candle.get('above_sma50', False) and
                        current_candle.get('above_ema50', False) and
                        current_candle.get('above_sma100', False)
                    )

                    if not all_factors_present:
                        continue

                    # Direction filter
                    if signal_type == 'BUY':
                        # Only BUY when above all MAs (trend filter)
                        pass  # Already filtered by the combination
                    else:  # SELL
                        # Only SELL when below SMA50 (but this combination is bullish)
                        if current_candle.get('above_sma50', False):
                            continue  # Skip SELL signals when above SMA50

                    entry_price = current_candle['close']
                    entry_time_utc = current_time
                    entry_time_chicago = current_time.tz_convert(chicago_tz)

                    # Calculate TP/SL levels
                    if signal_type == 'BUY':
                        tp_price = entry_price * 1.01   # +1% TP
                        sl_price = entry_price * 0.985  # -1.5% SL
                    else:  # SELL
                        tp_price = entry_price * 0.99   # -1% TP
                        sl_price = entry_price * 1.015  # +1.5% SL

                    # Track this trade
                    trade = {
                        'signal_type': signal_type,
                        'entry_time_utc': entry_time_utc,
                        'entry_time_chicago': entry_time_chicago,
                        'entry_price': entry_price,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'entry_hour_utc': entry_time_utc.hour,
                        'entry_hour_chicago': entry_time_chicago.hour,
                        'entry_day_of_week': entry_time_chicago.strftime('%A'),
                        'outcome': None,
                        'exit_time': None,
                        'duration_hours': None,
                        'profit_loss_pct': None
                    }

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
                                # TP hit first
                                trade['outcome'] = 'TP'
                                trade['exit_time'] = tp_time.tz_convert(chicago_tz)
                                trade['duration_hours'] = (tp_time - entry_time_utc).total_seconds() / 3600
                                trade['profit_loss_pct'] = 1.0 if signal_type == 'BUY' else -1.0
                                all_trades.append(trade)
                                break
                            else:
                                # SL hit first
                                trade['outcome'] = 'SL'
                                trade['exit_time'] = sl_time.tz_convert(chicago_tz)
                                trade['duration_hours'] = (sl_time - entry_time_utc).total_seconds() / 3600
                                trade['profit_loss_pct'] = -1.5 if signal_type == 'BUY' else 1.5
                                all_trades.append(trade)
                                break
                        elif tp_hit:
                            # Only TP hit
                            trade['outcome'] = 'TP'
                            trade['exit_time'] = tp_time.tz_convert(chicago_tz)
                            trade['duration_hours'] = (tp_time - entry_time_utc).total_seconds() / 3600
                            trade['profit_loss_pct'] = 1.0 if signal_type == 'BUY' else -1.0
                            all_trades.append(trade)
                            break
                        elif sl_hit:
                            # Only SL hit
                            trade['outcome'] = 'SL'
                            trade['exit_time'] = sl_time.tz_convert(chicago_tz)
                            trade['duration_hours'] = (sl_time - entry_time_utc).total_seconds() / 3600
                            trade['profit_loss_pct'] = -1.5 if signal_type == 'BUY' else 1.5
                            all_trades.append(trade)
                            break

                    # If neither hit in 100 candles
                    if trade['outcome'] is None:
                        trade['outcome'] = 'NO_EXIT'
                        trade['duration_hours'] = 100.0  # Max tracking time
                        trade['profit_loss_pct'] = 0.0
                        all_trades.append(trade)

                except Exception as e:
                    continue

        print(f"✅ Found {len(all_trades)} total trades with the combination")

        if len(all_trades) == 0:
            print("❌ No trades found with this combination")
            return

        # Convert to DataFrame for easier analysis
        trades_df = pd.DataFrame(all_trades)

        # COMPREHENSIVE ANALYSIS
        print(f"\n📊 COMPREHENSIVE PERFORMANCE ANALYSIS:")
        print("=" * 50)

        # Basic stats
        total_trades = len(trades_df)
        tp_trades = len(trades_df[trades_df['outcome'] == 'TP'])
        sl_trades = len(trades_df[trades_df['outcome'] == 'SL'])
        no_exit_trades = len(trades_df[trades_df['outcome'] == 'NO_EXIT'])

        print(f"   Total Trades: {total_trades}")
        print(f"   TP Wins: {tp_trades} ({tp_trades/total_trades*100:.1f}%)")
        print(f"   SL Losses: {sl_trades} ({sl_trades/total_trades*100:.1f}%)")
        print(f"   No Exit: {no_exit_trades} ({no_exit_trades/total_trades*100:.1f}%)")

        # Expected return
        total_profit = trades_df['profit_loss_pct'].sum()
        expected_return = total_profit / total_trades
        print(f"   Expected Return: {expected_return:+.4f}% per trade")
        print(f"   Total Return: {total_profit:+.2f}% over {total_trades} trades")

        # TIMING ANALYSIS - TP TRADES
        tp_only = trades_df[trades_df['outcome'] == 'TP']
        if len(tp_only) > 0:
            print(f"\n⏰ TP TIMING ANALYSIS ({len(tp_only)} successful trades):")
            print("-" * 40)
            print(f"   Average time to TP: {tp_only['duration_hours'].mean():.1f} hours")
            print(f"   Median time to TP: {tp_only['duration_hours'].median():.1f} hours")
            print(f"   Fastest TP: {tp_only['duration_hours'].min():.1f} hours")
            print(f"   Slowest TP: {tp_only['duration_hours'].max():.1f} hours")

            # Percentiles
            print(f"   25% hit TP within: {tp_only['duration_hours'].quantile(0.25):.1f} hours")
            print(f"   50% hit TP within: {tp_only['duration_hours'].quantile(0.50):.1f} hours")
            print(f"   75% hit TP within: {tp_only['duration_hours'].quantile(0.75):.1f} hours")
            print(f"   90% hit TP within: {tp_only['duration_hours'].quantile(0.90):.1f} hours")

        # TIMING ANALYSIS - SL TRADES
        sl_only = trades_df[trades_df['outcome'] == 'SL']
        if len(sl_only) > 0:
            print(f"\n💥 SL TIMING ANALYSIS ({len(sl_only)} losing trades):")
            print("-" * 40)
            print(f"   Average time to SL: {sl_only['duration_hours'].mean():.1f} hours")
            print(f"   Median time to SL: {sl_only['duration_hours'].median():.1f} hours")
            print(f"   Fastest SL: {sl_only['duration_hours'].min():.1f} hours")
            print(f"   Slowest SL: {sl_only['duration_hours'].max():.1f} hours")

        # HOURLY DISTRIBUTION - SIGNAL GENERATION
        print(f"\n🕐 SIGNAL GENERATION BY HOUR (Chicago Time):")
        print("-" * 45)
        hourly_signals = trades_df['entry_hour_chicago'].value_counts().sort_index()
        for hour in range(24):
            count = hourly_signals.get(hour, 0)
            percentage = (count / total_trades) * 100 if total_trades > 0 else 0
            bar = "█" * int(percentage / 2)  # Visual bar
            print(f"   {hour:2d}:00 - {count:3d} signals ({percentage:4.1f}%) {bar}")

        # HOURLY WIN RATES
        print(f"\n🎯 WIN RATES BY HOUR (Chicago Time):")
        print("-" * 35)
        for hour in range(24):
            hour_trades = trades_df[trades_df['entry_hour_chicago'] == hour]
            if len(hour_trades) >= 5:  # Only show hours with meaningful data
                hour_tp = len(hour_trades[hour_trades['outcome'] == 'TP'])
                win_rate = (hour_tp / len(hour_trades)) * 100
                print(f"   {hour:2d}:00 - {hour_tp:2d}/{len(hour_trades):2d} trades ({win_rate:4.1f}% win rate)")

        # DAY OF WEEK ANALYSIS
        print(f"\n📅 PERFORMANCE BY DAY OF WEEK:")
        print("-" * 35)
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in days_order:
            day_trades = trades_df[trades_df['entry_day_of_week'] == day]
            if len(day_trades) > 0:
                day_tp = len(day_trades[day_trades['outcome'] == 'TP'])
                day_sl = len(day_trades[day_trades['outcome'] == 'SL'])
                win_rate = (day_tp / len(day_trades)) * 100
                print(f"   {day:9s}: {day_tp:2d} TP, {day_sl:2d} SL, {len(day_trades):2d} total ({win_rate:4.1f}% win)")

        # BEST PERFORMING SESSIONS
        print(f"\n🏆 BEST PERFORMING SESSIONS:")
        print("-" * 30)

        # Define trading sessions (Chicago time)
        sessions = {
            'Asian': (18, 3),      # 6 PM - 3 AM Chicago
            'London': (3, 12),     # 3 AM - 12 PM Chicago
            'US': (8, 17),         # 8 AM - 5 PM Chicago
            'US_Close': (15, 18)   # 3 PM - 6 PM Chicago
        }

        for session_name, (start_hour, end_hour) in sessions.items():
            if start_hour < end_hour:
                session_trades = trades_df[
                    (trades_df['entry_hour_chicago'] >= start_hour) &
                    (trades_df['entry_hour_chicago'] < end_hour)
                ]
            else:  # Overnight session (Asian)
                session_trades = trades_df[
                    (trades_df['entry_hour_chicago'] >= start_hour) |
                    (trades_df['entry_hour_chicago'] < end_hour)
                ]

            if len(session_trades) > 0:
                session_tp = len(session_trades[session_trades['outcome'] == 'TP'])
                win_rate = (session_tp / len(session_trades)) * 100
                avg_duration = session_trades[session_trades['outcome'] == 'TP']['duration_hours'].mean()
                print(f"   {session_name:9s}: {session_tp:2d}/{len(session_trades):2d} ({win_rate:4.1f}% win, {avg_duration:.1f}h avg TP time)")

        # RECENT PERFORMANCE
        print(f"\n📈 MOST RECENT TRADES:")
        print("-" * 25)
        recent_trades = trades_df.tail(5)
        for i, trade in recent_trades.iterrows():
            entry_time = trade['entry_time_chicago'].strftime('%m/%d %H:%M')
            outcome = trade['outcome']
            duration = trade['duration_hours']
            profit = trade['profit_loss_pct']
            print(f"   {entry_time} {trade['signal_type']:4s} → {outcome:6s} {duration:5.1f}h {profit:+4.1f}%")

        # STATISTICAL SIGNIFICANCE
        print(f"\n📊 STATISTICAL ASSESSMENT:")
        print("-" * 30)
        if total_trades >= 100:
            print(f"   ✅ Good sample size ({total_trades} trades)")
        elif total_trades >= 50:
            print(f"   ⚠️  Moderate sample size ({total_trades} trades)")
        else:
            print(f"   🚨 Small sample size ({total_trades} trades)")

        confidence_interval = 1.96 * np.sqrt((tp_trades/total_trades) * (1-tp_trades/total_trades) / total_trades)
        win_rate_ci_lower = (tp_trades/total_trades) - confidence_interval
        win_rate_ci_upper = (tp_trades/total_trades) + confidence_interval

        print(f"   Win rate 95% CI: {win_rate_ci_lower*100:.1f}% - {win_rate_ci_upper*100:.1f}%")
        print(f"   Data span: {(df.index[-1] - df.index[0]).days} days")

    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(analyze_best_combination())