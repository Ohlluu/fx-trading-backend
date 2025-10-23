#!/usr/bin/env python3
"""
XAUUSD 2-Year Backtesting - Real Data Validation
Test our confluence strategies against historical data to get REAL win rates
"""

import asyncio
import httpx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Any, Optional
import json

# Create data directory
import os
os.makedirs("data", exist_ok=True)

async def fetch_historical_data():
    """Fetch 2+ years of XAUUSD data for backtesting"""

    print("🔄 FETCHING 2+ YEARS OF XAUUSD DATA FOR BACKTESTING")
    print("This will get maximum available historical data...")

    api_key = "0e24ff3eb6ef415dba0cebcf04593e4f"

    # Get maximum historical data
    params = {
        'symbol': 'XAU/USD',
        'interval': '1h',
        'outputsize': '5000',  # Maximum allowed
        'timezone': 'UTC',
        'format': 'JSON',
        'apikey': api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            print("📡 Fetching XAUUSD hourly data...")
            response = await client.get('https://api.twelvedata.com/time_series', params=params)
            data = response.json()

        if data.get('status') == 'ok' and data.get('values'):
            values = data['values']
            print(f"✅ Got {len(values)} hourly candles")

            # Convert to DataFrame
            df = pd.DataFrame(values)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')

            # Convert price columns to float
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                df[col] = df[col].astype(float)

            # Add NY time (localize first, then convert)
            ny_tz = pytz.timezone('America/New_York')
            utc_tz = pytz.timezone('UTC')
            df['datetime'] = df['datetime'].dt.tz_localize(utc_tz)
            df['ny_time'] = df['datetime'].dt.tz_convert(ny_tz)
            df['ny_hour'] = df['ny_time'].dt.hour
            df['ny_date'] = df['ny_time'].dt.date

            print(f"📊 Data range: {df.iloc[0]['datetime']} to {df.iloc[-1]['datetime']}")
            print(f"📈 Price range: ${df['low'].min():.2f} to ${df['high'].max():.2f}")
            days = (df.iloc[-1]['datetime'] - df.iloc[0]['datetime']).days
            print(f"⏰ Coverage: {days} days ({days/365:.1f} years)")

            # Save for backtesting
            df.to_csv('data/XAUUSD_BACKTEST_DATA.csv', index=False)
            print("💾 Saved to: data/XAUUSD_BACKTEST_DATA.csv")

            return df

        else:
            print("❌ API Error:", data.get('message', 'Unknown error'))
            return None

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def identify_asian_session_range(day_data: pd.DataFrame) -> Dict[str, Any]:
    """Find Asian session range for a given day (NY time 7PM-4AM)"""

    # Asian session hours in NY time: 19, 20, 21, 22, 23, 0, 1, 2, 3
    asian_hours = [19, 20, 21, 22, 23, 0, 1, 2, 3]
    asian_data = day_data[day_data['ny_hour'].isin(asian_hours)]

    if len(asian_data) < 3:  # Need minimum data
        return None

    asian_high = asian_data['high'].max()
    asian_low = asian_data['low'].min()
    asian_range = asian_high - asian_low

    return {
        'asian_high': asian_high,
        'asian_low': asian_low,
        'asian_range': asian_range,
        'candles_count': len(asian_data)
    }

def check_london_fix_breakout(candle: pd.Series, prev_candles: pd.DataFrame) -> Dict[str, Any]:
    """Check if current candle is a London Fix breakout"""

    # London Fix times: 5:30 AM and 10:00 AM NY time
    if candle['ny_hour'] not in [5, 10]:
        return None

    # Calculate volatility
    volatility = candle['high'] - candle['low']

    # Need minimum $15 movement for London Fix
    if volatility < 15.0:
        return None

    # Determine direction
    if candle['close'] > candle['open']:
        direction = "BUY"
        entry_price = candle['high']
    else:
        direction = "SELL"
        entry_price = candle['low']

    return {
        'type': 'london_fix',
        'direction': direction,
        'entry_price': entry_price,
        'volatility': volatility,
        'fix_time': 'AM' if candle['ny_hour'] == 5 else 'PM'
    }

def check_asian_range_breakout(candle: pd.Series, asian_range: Dict[str, Any]) -> Dict[str, Any]:
    """Check if current candle breaks Asian range"""

    if not asian_range:
        return None

    # Need to be in London session (8AM-12PM NY)
    if candle['ny_hour'] not in [8, 9, 10, 11]:
        return None

    asian_high = asian_range['asian_high']
    asian_low = asian_range['asian_low']

    # Check for breakouts with minimum $5 break
    if candle['high'] > asian_high + 5.0:
        return {
            'type': 'asian_breakout',
            'direction': 'BUY',
            'entry_price': asian_high + 5.0,
            'asian_high': asian_high,
            'asian_low': asian_low
        }
    elif candle['low'] < asian_low - 5.0:
        return {
            'type': 'asian_breakout',
            'direction': 'SELL',
            'entry_price': asian_low - 5.0,
            'asian_high': asian_high,
            'asian_low': asian_low
        }

    return None

def simulate_trade(signal: Dict[str, Any], future_data: pd.DataFrame) -> Dict[str, Any]:
    """Simulate a trade based on signal and return results"""

    entry_price = signal['entry_price']
    direction = signal['direction']

    # Risk management: $5 risk, 2:1 R/R minimum
    risk_amount = 5.0

    if direction == "BUY":
        stop_loss = entry_price - risk_amount
        take_profit = entry_price + (risk_amount * 2)  # 2:1 R/R
    else:  # SELL
        stop_loss = entry_price + risk_amount
        take_profit = entry_price - (risk_amount * 2)

    # Check what happens in next 24 hours (max trade duration)
    max_duration = 24
    trade_data = future_data.head(max_duration)

    for i, (_, candle) in enumerate(trade_data.iterrows()):
        hours_elapsed = i + 1

        if direction == "BUY":
            # Check for take profit hit
            if candle['high'] >= take_profit:
                return {
                    'result': 'WIN',
                    'exit_price': take_profit,
                    'profit': risk_amount * 2,  # 2:1 win
                    'duration_hours': hours_elapsed,
                    'exit_reason': 'take_profit'
                }
            # Check for stop loss hit
            elif candle['low'] <= stop_loss:
                return {
                    'result': 'LOSS',
                    'exit_price': stop_loss,
                    'profit': -risk_amount,
                    'duration_hours': hours_elapsed,
                    'exit_reason': 'stop_loss'
                }
        else:  # SELL
            # Check for take profit hit
            if candle['low'] <= take_profit:
                return {
                    'result': 'WIN',
                    'exit_price': take_profit,
                    'profit': risk_amount * 2,
                    'duration_hours': hours_elapsed,
                    'exit_reason': 'take_profit'
                }
            # Check for stop loss hit
            elif candle['high'] >= stop_loss:
                return {
                    'result': 'LOSS',
                    'exit_price': stop_loss,
                    'profit': -risk_amount,
                    'duration_hours': hours_elapsed,
                    'exit_reason': 'stop_loss'
                }

    # If no exit after 24 hours, close at last price
    last_candle = trade_data.iloc[-1]
    final_price = last_candle['close']

    if direction == "BUY":
        profit = final_price - entry_price
    else:
        profit = entry_price - final_price

    return {
        'result': 'WIN' if profit > 0 else 'LOSS',
        'exit_price': final_price,
        'profit': profit,
        'duration_hours': max_duration,
        'exit_reason': 'timeout'
    }

def run_backtest(df: pd.DataFrame) -> Dict[str, Any]:
    """Run complete backtest on historical data"""

    print("\n🎯 STARTING COMPREHENSIVE BACKTEST")
    print("=" * 60)

    trades = []
    daily_trades = {}

    # Group data by date for daily analysis
    for date, day_data in df.groupby('ny_date'):
        daily_trades[date] = 0

        # Get Asian range for this day
        asian_range = identify_asian_session_range(day_data)

        # Check each candle for signals
        for i, (idx, candle) in enumerate(day_data.iterrows()):

            # Skip if we've hit daily limit (3 trades max per day)
            if daily_trades[date] >= 3:
                continue

            # Get remaining data for trade simulation
            remaining_data = df.iloc[idx+1:idx+25]  # Next 24 hours
            if len(remaining_data) < 5:  # Need minimum data
                continue

            signal = None

            # Check for London Fix breakout
            london_signal = check_london_fix_breakout(candle, day_data.iloc[:i])
            if london_signal:
                signal = london_signal

            # Check for Asian range breakout (if no London signal)
            elif asian_range:
                asian_signal = check_asian_range_breakout(candle, asian_range)
                if asian_signal:
                    signal = asian_signal

            # If we have a signal, simulate the trade
            if signal:
                trade_result = simulate_trade(signal, remaining_data)

                trade_record = {
                    'date': date,
                    'entry_time': candle['ny_time'],
                    'strategy': signal['type'],
                    'direction': signal['direction'],
                    'entry_price': signal['entry_price'],
                    **trade_result
                }

                trades.append(trade_record)
                daily_trades[date] += 1

                print(f"📈 Trade {len(trades)}: {date} {signal['direction']} @${signal['entry_price']:.2f} → {trade_result['result']} (${trade_result['profit']:+.2f})")

    return analyze_backtest_results(trades)

def analyze_backtest_results(trades: List[Dict]) -> Dict[str, Any]:
    """Analyze backtest results and calculate performance metrics"""

    if not trades:
        print("❌ No trades found in backtest")
        return {}

    print(f"\n📊 BACKTEST RESULTS ANALYSIS ({len(trades)} trades)")
    print("=" * 60)

    # Basic stats
    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']

    win_count = len(wins)
    loss_count = len(losses)
    total_trades = len(trades)

    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0

    # Profit analysis
    total_profit = sum(t['profit'] for t in trades)
    avg_win = sum(t['profit'] for t in wins) / win_count if wins else 0
    avg_loss = sum(t['profit'] for t in losses) / loss_count if losses else 0

    # Strategy breakdown
    london_fix_trades = [t for t in trades if t['strategy'] == 'london_fix']
    asian_breakout_trades = [t for t in trades if t['strategy'] == 'asian_breakout']

    london_wins = len([t for t in london_fix_trades if t['result'] == 'WIN'])
    asian_wins = len([t for t in asian_breakout_trades if t['result'] == 'WIN'])

    london_win_rate = (london_wins / len(london_fix_trades)) * 100 if london_fix_trades else 0
    asian_win_rate = (asian_wins / len(asian_breakout_trades)) * 100 if asian_breakout_trades else 0

    # Print results
    print(f"🎯 OVERALL PERFORMANCE:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Wins: {win_count} ({win_rate:.1f}%)")
    print(f"   Losses: {loss_count} ({100-win_rate:.1f}%)")
    print(f"   Total P&L: ${total_profit:+.2f}")
    print(f"   Average Win: ${avg_win:.2f}")
    print(f"   Average Loss: ${avg_loss:.2f}")

    print(f"\n📈 STRATEGY BREAKDOWN:")
    print(f"   London Fix: {len(london_fix_trades)} trades, {london_win_rate:.1f}% win rate")
    print(f"   Asian Range: {len(asian_breakout_trades)} trades, {asian_win_rate:.1f}% win rate")

    # Save detailed results
    results_df = pd.DataFrame(trades)
    results_df.to_csv('data/BACKTEST_RESULTS.csv', index=False)
    print(f"\n💾 Detailed results saved to: data/BACKTEST_RESULTS.csv")

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'london_fix_win_rate': london_win_rate,
        'asian_breakout_win_rate': asian_win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }

async def main():
    """Main backtesting function"""

    print("🚀 XAUUSD 2-YEAR BACKTEST STARTING")
    print("=" * 60)

    # Step 1: Get historical data
    df = await fetch_historical_data()
    if df is None:
        print("❌ Failed to fetch data")
        return

    # Step 2: Run backtest
    results = run_backtest(df)

    # Step 3: Final summary
    print("\n🏆 FINAL BACKTEST SUMMARY")
    print("=" * 60)

    if results:
        print(f"✅ REAL WIN RATE: {results['win_rate']:.1f}% (not the made-up 82.9%)")
        print(f"✅ TOTAL PROFIT: ${results['total_profit']:+.2f} over {results['total_trades']} trades")
        print(f"✅ London Fix Strategy: {results['london_fix_win_rate']:.1f}% win rate")
        print(f"✅ Asian Range Strategy: {results['asian_breakout_win_rate']:.1f}% win rate")

        if results['total_profit'] > 0:
            print("🎉 PROFITABLE SYSTEM - This strategy works!")
        else:
            print("⚠️  LOSING SYSTEM - Need to revise strategy")

    print("\n💡 Now we know what ACTUALLY works instead of guessing!")

if __name__ == "__main__":
    asyncio.run(main())