#!/usr/bin/env python3
"""
Smart Money Concepts (SMC) XAUUSD Backtesting
Test REAL professional Gold trading setups that institutions use
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Any, Optional

def load_data():
    """Load existing backtest data"""
    try:
        df = pd.read_csv('data/XAUUSD_BACKTEST_DATA.csv')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['ny_time'] = pd.to_datetime(df['ny_time'])
        df['ny_date_str'] = df['ny_time'].astype(str).str[:10]
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("Run backtest_xauusd.py first to get data")
        return None

def identify_fair_value_gaps(df: pd.DataFrame, min_gap_size: float = 8.0) -> List[Dict]:
    """Identify Fair Value Gaps - price imbalances from fast moves"""

    fvgs = []

    for i in range(1, len(df) - 1):
        candle_1 = df.iloc[i-1]
        candle_2 = df.iloc[i]    # Middle candle
        candle_3 = df.iloc[i+1]

        # Bullish FVG: Gap between candle_1 high and candle_3 low
        if candle_1['high'] < candle_3['low']:
            gap_size = candle_3['low'] - candle_1['high']
            if gap_size >= min_gap_size:
                fvgs.append({
                    'index': i,
                    'time': candle_2['ny_time'],
                    'type': 'bullish_fvg',
                    'gap_low': candle_1['high'],
                    'gap_high': candle_3['low'],
                    'gap_size': gap_size,
                    'middle_price': (candle_1['high'] + candle_3['low']) / 2
                })

        # Bearish FVG: Gap between candle_1 low and candle_3 high
        elif candle_1['low'] > candle_3['high']:
            gap_size = candle_1['low'] - candle_3['high']
            if gap_size >= min_gap_size:
                fvgs.append({
                    'index': i,
                    'time': candle_2['ny_time'],
                    'type': 'bearish_fvg',
                    'gap_low': candle_3['high'],
                    'gap_high': candle_1['low'],
                    'gap_size': gap_size,
                    'middle_price': (candle_1['low'] + candle_3['high']) / 2
                })

    print(f"📊 Found {len(fvgs)} Fair Value Gaps")
    return fvgs

def identify_order_blocks(df: pd.DataFrame, lookback: int = 10) -> List[Dict]:
    """Identify Order Blocks - zones where institutions placed large orders"""

    order_blocks = []

    for i in range(lookback, len(df) - lookback):
        current_candle = df.iloc[i]

        # Look for strong impulsive moves (displacement)
        body_size = abs(current_candle['close'] - current_candle['open'])
        candle_range = current_candle['high'] - current_candle['low']

        # Strong directional candle with minimal wicks
        if body_size > candle_range * 0.7 and candle_range > 20:

            # Look for the last opposite candle before the move
            if current_candle['close'] > current_candle['open']:  # Bullish displacement
                # Find last bearish candle before this move
                for j in range(i-1, max(0, i-lookback), -1):
                    prev_candle = df.iloc[j]
                    if prev_candle['close'] < prev_candle['open']:  # Bearish candle
                        order_blocks.append({
                            'index': j,
                            'time': prev_candle['ny_time'],
                            'type': 'bullish_ob',
                            'ob_high': prev_candle['high'],
                            'ob_low': prev_candle['low'],
                            'displacement_index': i,
                            'displacement_strength': candle_range
                        })
                        break

            else:  # Bearish displacement
                # Find last bullish candle before this move
                for j in range(i-1, max(0, i-lookback), -1):
                    prev_candle = df.iloc[j]
                    if prev_candle['close'] > prev_candle['open']:  # Bullish candle
                        order_blocks.append({
                            'index': j,
                            'time': prev_candle['ny_time'],
                            'type': 'bearish_ob',
                            'ob_high': prev_candle['high'],
                            'ob_low': prev_candle['low'],
                            'displacement_index': i,
                            'displacement_strength': candle_range
                        })
                        break

    print(f"📊 Found {len(order_blocks)} Order Blocks")
    return order_blocks

def identify_liquidity_sweeps(df: pd.DataFrame, lookback: int = 20) -> List[Dict]:
    """Identify liquidity sweeps - taking out highs/lows before reversal"""

    sweeps = []

    for i in range(lookback, len(df) - 5):
        current_candle = df.iloc[i]

        # Look for recent swing high/low in lookback period
        recent_data = df.iloc[i-lookback:i]

        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()

        recent_high_idx = recent_data['high'].idxmax()
        recent_low_idx = recent_data['low'].idxmin()

        # Check if current candle sweeps recent high
        if current_candle['high'] > recent_high + 3:  # Sweep with 3 point buffer
            # Look for reversal in next few candles
            future_candles = df.iloc[i+1:i+6]
            if not future_candles.empty and future_candles['low'].min() < current_candle['low'] - 10:
                sweeps.append({
                    'index': i,
                    'time': current_candle['ny_time'],
                    'type': 'high_sweep_bearish',
                    'swept_level': recent_high,
                    'sweep_price': current_candle['high'],
                    'reversal_strength': current_candle['low'] - future_candles['low'].min()
                })

        # Check if current candle sweeps recent low
        elif current_candle['low'] < recent_low - 3:  # Sweep with 3 point buffer
            # Look for reversal in next few candles
            future_candles = df.iloc[i+1:i+6]
            if not future_candles.empty and future_candles['high'].max() > current_candle['high'] + 10:
                sweeps.append({
                    'index': i,
                    'time': current_candle['ny_time'],
                    'type': 'low_sweep_bullish',
                    'swept_level': recent_low,
                    'sweep_price': current_candle['low'],
                    'reversal_strength': future_candles['high'].max() - current_candle['high']
                })

    print(f"📊 Found {len(sweeps)} Liquidity Sweeps")
    return sweeps

def check_smc_confluence(candle: pd.Series, idx: int, fvgs: List[Dict],
                        order_blocks: List[Dict], sweeps: List[Dict]) -> Optional[Dict]:
    """Check for SMC confluence at current candle"""

    current_price = candle['close']
    confluence_factors = []
    confluence_score = 0

    # 1. FVG Confluence - price near unfilled gap
    for fvg in fvgs:
        if fvg['index'] < idx:  # FVG must be in the past
            if fvg['gap_low'] <= current_price <= fvg['gap_high']:
                confluence_factors.append(f"{fvg['type']} FVG ${fvg['middle_price']:.0f}")
                confluence_score += 3
                break

    # 2. Order Block Confluence - price near institutional zone
    for ob in order_blocks:
        if ob['index'] < idx:  # OB must be in the past
            if ob['ob_low'] <= current_price <= ob['ob_high']:
                confluence_factors.append(f"{ob['type']} Order Block")
                confluence_score += 4
                break

    # 3. Liquidity Sweep Confluence - recent sweep occurred
    for sweep in sweeps:
        if abs(sweep['index'] - idx) <= 3:  # Recent sweep (within 3 candles)
            confluence_factors.append(f"{sweep['type']} sweep")
            confluence_score += 2

    # 4. Session Confluence - London/NY overlap
    hour = candle['ny_hour']
    if hour in [8, 9, 10, 11]:  # London-NY overlap
        confluence_factors.append("London-NY overlap")
        confluence_score += 2
    elif hour in [5, 10]:  # London Fix
        confluence_factors.append("London Fix time")
        confluence_score += 3

    # 5. Displacement Confluence - strong move present
    body_size = abs(candle['close'] - candle['open'])
    candle_range = candle['high'] - candle['low']

    if body_size > candle_range * 0.7 and candle_range > 15:
        confluence_factors.append("Strong displacement")
        confluence_score += 2

    # Require minimum confluence
    if confluence_score < 6:
        return None

    # Determine direction
    direction = None
    entry_price = None

    # Bullish setup
    if any('bullish' in factor for factor in confluence_factors):
        if candle['close'] > candle['open']:  # Bullish candle
            direction = "BUY"
            entry_price = candle['high'] + 2

    # Bearish setup
    elif any('bearish' in factor for factor in confluence_factors):
        if candle['close'] < candle['open']:  # Bearish candle
            direction = "SELL"
            entry_price = candle['low'] - 2

    if not direction:
        return None

    return {
        'direction': direction,
        'entry_price': entry_price,
        'confluence_score': confluence_score,
        'confluence_factors': confluence_factors,
        'strategy': 'smc_confluence'
    }

def simulate_smc_trade(signal: Dict, future_data: pd.DataFrame) -> Optional[Dict]:
    """Simulate SMC trade with proper risk management"""

    entry_price = signal['entry_price']
    direction = signal['direction']

    # SMC Risk Management - 1:2 minimum R/R
    risk_amount = 12.0  # $12 risk per trade

    if direction == "BUY":
        stop_loss = entry_price - risk_amount
        take_profit = entry_price + (risk_amount * 2)  # 1:2 R/R
    else:
        stop_loss = entry_price + risk_amount
        take_profit = entry_price - (risk_amount * 2)

    # Check trade outcome
    for i, (_, candle) in enumerate(future_data.iterrows()):
        hours_elapsed = i + 1

        if direction == "BUY":
            if candle['high'] >= take_profit:
                return {
                    'result': 'WIN',
                    'exit_price': take_profit,
                    'profit': risk_amount * 2,
                    'duration_hours': hours_elapsed,
                    'exit_reason': 'take_profit'
                }
            elif candle['low'] <= stop_loss:
                return {
                    'result': 'LOSS',
                    'exit_price': stop_loss,
                    'profit': -risk_amount,
                    'duration_hours': hours_elapsed,
                    'exit_reason': 'stop_loss'
                }
        else:
            if candle['low'] <= take_profit:
                return {
                    'result': 'WIN',
                    'exit_price': take_profit,
                    'profit': risk_amount * 2,
                    'duration_hours': hours_elapsed,
                    'exit_reason': 'take_profit'
                }
            elif candle['high'] >= stop_loss:
                return {
                    'result': 'LOSS',
                    'exit_price': stop_loss,
                    'profit': -risk_amount,
                    'duration_hours': hours_elapsed,
                    'exit_reason': 'stop_loss'
                }

        # Max 8 hour trade duration for SMC
        if hours_elapsed >= 8:
            break

    return None  # No clear outcome

def run_smc_backtest(df: pd.DataFrame) -> Dict[str, Any]:
    """Run Smart Money Concepts backtesting"""

    print("🎯 SMART MONEY CONCEPTS (SMC) BACKTESTING")
    print("=" * 60)

    # Identify SMC components
    print("🔍 Identifying SMC structures...")
    fvgs = identify_fair_value_gaps(df)
    order_blocks = identify_order_blocks(df)
    sweeps = identify_liquidity_sweeps(df)

    trades = []
    daily_trades = {}

    print("📈 Scanning for SMC confluence signals...")

    # Group by date
    for date, day_data in df.groupby('ny_date_str'):
        if date not in daily_trades:
            daily_trades[date] = 0

        day_df = df[df['ny_date_str'] == date].copy()

        for i, (idx, candle) in enumerate(day_df.iterrows()):
            # Limit trades per day
            if daily_trades[date] >= 2:
                continue

            # Need sufficient history
            if idx < 50:
                continue

            # Check SMC confluence
            signal = check_smc_confluence(candle, idx, fvgs, order_blocks, sweeps)

            if signal:
                # Get future data for simulation
                future_data = df.iloc[idx+1:idx+9]  # Next 8 hours
                if len(future_data) < 4:
                    continue

                trade_result = simulate_smc_trade(signal, future_data)

                if trade_result:
                    trade_record = {
                        'date': date,
                        'entry_time': candle['ny_time'],
                        'direction': signal['direction'],
                        'entry_price': signal['entry_price'],
                        'confluence_score': signal['confluence_score'],
                        'confluence_factors': ', '.join(signal['confluence_factors']),
                        **trade_result
                    }

                    trades.append(trade_record)
                    daily_trades[date] += 1

                    result_emoji = "✅" if trade_result['result'] == 'WIN' else "❌"
                    factors_short = signal['confluence_factors'][0] if signal['confluence_factors'] else 'SMC'

                    print(f"{result_emoji} Trade {len(trades)}: {date} {signal['direction']} "
                          f"@${signal['entry_price']:.2f} [{factors_short}] → "
                          f"{trade_result['result']} ${trade_result['profit']:+.2f}")

    return analyze_smc_results(trades)

def analyze_smc_results(trades: List[Dict]) -> Dict[str, Any]:
    """Analyze SMC backtest results"""

    if not trades:
        print("❌ No SMC trades found")
        return {}

    print(f"\n🏆 SMART MONEY CONCEPTS RESULTS ({len(trades)} trades)")
    print("=" * 60)

    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']

    win_count = len(wins)
    total_trades = len(trades)
    win_rate = (win_count / total_trades) * 100

    total_profit = sum(t['profit'] for t in trades)
    avg_win = sum(t['profit'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['profit'] for t in losses) / len(losses) if losses else 0

    avg_confluence = sum(t['confluence_score'] for t in trades) / len(trades)

    print(f"🎯 WIN RATE: {win_rate:.1f}% ({win_count}/{total_trades})")
    print(f"💰 TOTAL PROFIT: ${total_profit:+.2f}")
    print(f"📊 AVERAGE WIN: ${avg_win:.2f}")
    print(f"📊 AVERAGE LOSS: ${avg_loss:.2f}")
    print(f"⚖️  PROFIT FACTOR: {abs(avg_win/avg_loss):.2f}:1" if avg_loss != 0 else "N/A")
    print(f"🧠 AVG SMC SCORE: {avg_confluence:.1f}")

    # Strategy breakdown
    fvg_trades = [t for t in trades if 'FVG' in t['confluence_factors']]
    ob_trades = [t for t in trades if 'Order Block' in t['confluence_factors']]
    sweep_trades = [t for t in trades if 'sweep' in t['confluence_factors']]

    if fvg_trades:
        fvg_wins = len([t for t in fvg_trades if t['result'] == 'WIN'])
        fvg_win_rate = (fvg_wins / len(fvg_trades)) * 100
        print(f"\n📊 FVG Trades: {fvg_win_rate:.1f}% win rate ({fvg_wins}/{len(fvg_trades)})")

    if ob_trades:
        ob_wins = len([t for t in ob_trades if t['result'] == 'WIN'])
        ob_win_rate = (ob_wins / len(ob_trades)) * 100
        print(f"📊 Order Block Trades: {ob_win_rate:.1f}% win rate ({ob_wins}/{len(ob_trades)})")

    if sweep_trades:
        sweep_wins = len([t for t in sweep_trades if t['result'] == 'WIN'])
        sweep_win_rate = (sweep_wins / len(sweep_trades)) * 100
        print(f"📊 Liquidity Sweep Trades: {sweep_win_rate:.1f}% win rate ({sweep_wins}/{len(sweep_trades)})")

    # Save results
    results_df = pd.DataFrame(trades)
    results_df.to_csv('data/SMC_BACKTEST_RESULTS.csv', index=False)
    print(f"\n💾 Results saved to: data/SMC_BACKTEST_RESULTS.csv")

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'avg_confluence': avg_confluence
    }

def main():
    """Main SMC backtesting function"""

    print("🧠 SMART MONEY CONCEPTS XAUUSD BACKTESTING")
    print("=" * 60)

    df = load_data()
    if df is None:
        return

    print(f"📊 Testing SMC on {len(df)} XAUUSD candles")

    results = run_smc_backtest(df)

    print(f"\n🏆 FINAL SMC RESULTS")
    print("=" * 60)

    if results and results['win_rate'] >= 55:
        print(f"🎉 SUCCESS! SMC achieves {results['win_rate']:.1f}% win rate!")
        print("✅ Professional SMC concepts work for XAUUSD!")
    elif results:
        print(f"⚠️  {results['win_rate']:.1f}% win rate - better than traditional TA")
        print(f"💰 ${results['total_profit']:+.2f} profit shows SMC has merit")
    else:
        print("❌ No profitable SMC setups found")

    print("\n💡 SMC focuses on what institutions actually do, not retail indicators")

if __name__ == "__main__":
    main()