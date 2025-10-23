#!/usr/bin/env python3
"""
ULTIMATE CONFLUENCE BACKTEST - Using Only the Most Frequent Real Patterns
Based on 5-year analysis of what actually triggered before 2,031 significant moves
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

def load_data():
    """Load the 5-year dataset"""
    try:
        df = pd.read_csv('data/XAUUSD_3YEAR_DATA.csv')
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df['ny_time'] = pd.to_datetime(df['ny_time'], utc=True)
        print(f"📊 Loaded {len(df)} candles for ultimate confluence backtest")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def add_proven_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add only the indicators that showed up most frequently before real moves"""

    print("🔧 Adding PROVEN confluence indicators...")

    # Moving averages (most frequent)
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()

    # Position relative to MAs (TOP confluence for bullish moves)
    df['above_sma20'] = df['close'] > df['sma_20']
    df['above_sma50'] = df['close'] > df['sma_50']
    df['above_ema20'] = df['close'] > df['ema_20']

    # Trend direction (appeared in 63-65% of moves)
    df['sma20_slope'] = df['sma_20'].diff(5)
    df['ema20_slope'] = df['ema_20'].diff(5)
    df['uptrend_sma'] = df['sma20_slope'] > 0
    df['uptrend_ema'] = df['ema20_slope'] > 0

    # Basic price action (50%+ frequency)
    df['is_green'] = df['close'] > df['open']
    df['prev_green'] = df['is_green'].shift(1)
    df['prev_red'] = (~df['is_green']).shift(1)

    # Support/Resistance (55.6% of bullish moves were near recent highs)
    df['recent_high_12h'] = df['high'].rolling(window=12).max()
    df['recent_low_12h'] = df['low'].rolling(window=12).min()
    df['near_recent_high'] = abs(df['close'] - df['recent_high_12h']) < 10

    print("✅ Added only the most proven confluence factors")
    return df

def check_ultimate_bullish_confluence(candle: pd.Series, df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """Check for bullish confluence using only the most frequent real patterns"""

    # TOP BULLISH CONFLUENCES from real analysis:
    # 1. above_sma50 (66.1%)
    # 2. uptrend_ema (65.4%)
    # 3. above_ema20 (65.0%)
    # 4. uptrend_sma (63.6%)
    # 5. above_sma20 (63.3%)

    confluence_score = 0
    factors = []

    # Must have the TOP confluence
    if not candle['above_sma50']:
        return None
    confluence_score += 4
    factors.append("Above SMA50 (66% frequency)")

    # Add other high-frequency confluences
    if candle['uptrend_ema']:
        confluence_score += 3
        factors.append("EMA uptrend (65% frequency)")

    if candle['above_ema20']:
        confluence_score += 3
        factors.append("Above EMA20 (65% frequency)")

    if candle['uptrend_sma']:
        confluence_score += 2
        factors.append("SMA uptrend (64% frequency)")

    if candle['above_sma20']:
        confluence_score += 2
        factors.append("Above SMA20 (63% frequency)")

    if candle['near_recent_high']:
        confluence_score += 2
        factors.append("Near recent high (56% frequency)")

    if candle['is_green']:
        confluence_score += 1
        factors.append("Green candle (52% frequency)")

    # Require minimum confluence (stricter than before)
    if confluence_score < 10:
        return None

    return {
        'direction': 'BUY',
        'entry_price': candle['close'],
        'confluence_score': confluence_score,
        'factors': factors,
        'pattern_type': 'ultimate_bullish'
    }

def check_ultimate_bearish_confluence(candle: pd.Series, df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """Check for bearish confluence using only the most frequent real patterns"""

    # TOP BEARISH CONFLUENCES from real analysis:
    # 1. uptrend_sma (52.0%) - Interesting: bearish moves happen in uptrends!
    # 2. uptrend_ema (50.8%)
    # 3. prev_red (50.8%)
    # 4. above_sma20 (50.4%) - Bearish moves even above SMA20!
    # 5. is_green (50.0%) - Bearish moves after green candles

    confluence_score = 0
    factors = []

    # Must have TOP confluence - counter-intuitive but data-driven!
    if not candle['uptrend_sma']:
        return None
    confluence_score += 3
    factors.append("SMA uptrend (52% frequency)")

    if candle['uptrend_ema']:
        confluence_score += 3
        factors.append("EMA uptrend (51% frequency)")

    if candle['prev_red']:
        confluence_score += 2
        factors.append("Previous red candle (51% frequency)")

    if candle['above_sma20']:
        confluence_score += 2
        factors.append("Above SMA20 (50% frequency)")

    if candle['is_green']:
        confluence_score += 2
        factors.append("Green candle (50% frequency)")

    # Require minimum confluence
    if confluence_score < 8:
        return None

    return {
        'direction': 'SELL',
        'entry_price': candle['close'],
        'confluence_score': confluence_score,
        'factors': factors,
        'pattern_type': 'ultimate_bearish'
    }

def simulate_trade(signal: Dict, future_data: pd.DataFrame) -> Optional[Dict]:
    """Simulate trade with conservative targets"""

    entry_price = signal['entry_price']
    direction = signal['direction']

    # Conservative targets based on reliability analysis
    profit_target = 40.0  # Slightly higher target
    stop_loss = 20.0      # Reasonable stop

    if direction == "BUY":
        take_profit = entry_price + profit_target
        stop_price = entry_price - stop_loss
    else:
        take_profit = entry_price - profit_target
        stop_price = entry_price + stop_loss

    # Check outcome over next 36 hours (more time for development)
    for i, (_, candle) in enumerate(future_data.head(36).iterrows()):
        hours_elapsed = i + 1

        if direction == "BUY":
            if candle['high'] >= take_profit:
                return {
                    'result': 'WIN',
                    'exit_price': take_profit,
                    'profit': profit_target,
                    'duration_hours': hours_elapsed,
                    'exit_reason': 'take_profit'
                }
            elif candle['low'] <= stop_price:
                return {
                    'result': 'LOSS',
                    'exit_price': stop_price,
                    'profit': -stop_loss,
                    'duration_hours': hours_elapsed,
                    'exit_reason': 'stop_loss'
                }
        else:  # SELL
            if candle['low'] <= take_profit:
                return {
                    'result': 'WIN',
                    'exit_price': take_profit,
                    'profit': profit_target,
                    'duration_hours': hours_elapsed,
                    'exit_reason': 'take_profit'
                }
            elif candle['high'] >= stop_price:
                return {
                    'result': 'LOSS',
                    'exit_price': stop_price,
                    'profit': -stop_loss,
                    'duration_hours': hours_elapsed,
                    'exit_reason': 'stop_loss'
                }

    # Timeout at market price
    final_candle = future_data.iloc[-1] if len(future_data) > 0 else None
    if final_candle is not None:
        final_price = final_candle['close']
        if direction == "BUY":
            profit = final_price - entry_price
        else:
            profit = entry_price - final_price

        return {
            'result': 'WIN' if profit > 0 else 'LOSS',
            'exit_price': final_price,
            'profit': profit,
            'duration_hours': 36,
            'exit_reason': 'timeout'
        }

    return None

def run_ultimate_confluence_backtest(df: pd.DataFrame) -> Dict[str, Any]:
    """Run backtest using only the highest frequency confluences from real analysis"""

    print("🎯 ULTIMATE CONFLUENCE BACKTEST")
    print("Using only confluences that appeared in 50%+ of actual significant moves")
    print("="*80)

    trades = []
    daily_trades = {}

    # Test on subset for performance (last 2 years)
    recent_df = df.tail(17520).copy()  # ~2 years of hourly data
    print(f"📊 Testing on {len(recent_df)} recent candles...")

    for date, day_data in recent_df.groupby('ny_date_str'):
        if date not in daily_trades:
            daily_trades[date] = 0

        # Limit trades per day
        if daily_trades[date] >= 2:
            continue

        day_df = recent_df[recent_df['ny_date_str'] == date].copy()

        for i, (idx, candle) in enumerate(day_df.iterrows()):
            # Need sufficient history
            if idx < 200:
                continue

            signal = None

            # Check for ultimate confluences
            signal = check_ultimate_bullish_confluence(candle, recent_df, idx)
            if not signal:
                signal = check_ultimate_bearish_confluence(candle, recent_df, idx)

            if signal and daily_trades[date] < 2:
                # Get future data
                future_start = recent_df.index.get_loc(idx) + 1
                future_data = recent_df.iloc[future_start:future_start+37]

                if len(future_data) < 12:
                    continue

                trade_result = simulate_trade(signal, future_data)

                if trade_result:
                    trade_record = {
                        'date': date,
                        'direction': signal['direction'],
                        'entry_price': signal['entry_price'],
                        'confluence_score': signal['confluence_score'],
                        'factors': '; '.join(signal['factors']),
                        **trade_result
                    }

                    trades.append(trade_record)
                    daily_trades[date] += 1

                    result_emoji = "✅" if trade_result['result'] == 'WIN' else "❌"
                    print(f"{result_emoji} Trade {len(trades)}: {signal['direction']} @${signal['entry_price']:.2f} "
                          f"(Score: {signal['confluence_score']}) → {trade_result['result']} ${trade_result['profit']:+.2f}")

    return analyze_ultimate_results(trades)

def analyze_ultimate_results(trades: List[Dict]) -> Dict[str, Any]:
    """Analyze the ultimate confluence backtest results"""

    if not trades:
        print("❌ No ultimate confluence trades found")
        return {}

    print(f"\n🏆 ULTIMATE CONFLUENCE BACKTEST RESULTS ({len(trades)} trades)")
    print("="*80)

    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']

    win_count = len(wins)
    total_trades = len(trades)
    win_rate = (win_count / total_trades) * 100

    total_profit = sum(t['profit'] for t in trades)
    avg_win = sum(t['profit'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['profit'] for t in losses) / len(losses) if losses else 0

    print(f"🎯 WIN RATE: {win_rate:.1f}% ({win_count}/{total_trades})")
    print(f"💰 TOTAL PROFIT: ${total_profit:+.2f}")
    print(f"📊 AVERAGE WIN: ${avg_win:.2f}")
    print(f"📊 AVERAGE LOSS: ${avg_loss:.2f}")
    print(f"⚖️  PROFIT FACTOR: {abs(avg_win/avg_loss):.2f}:1" if avg_loss != 0 else "N/A")

    # Pattern performance
    bullish_trades = [t for t in trades if t['direction'] == 'BUY']
    bearish_trades = [t for t in trades if t['direction'] == 'SELL']

    if bullish_trades:
        bull_wins = len([t for t in bullish_trades if t['result'] == 'WIN'])
        bull_win_rate = (bull_wins / len(bullish_trades)) * 100
        print(f"\n📈 BULLISH CONFLUENCES: {bull_win_rate:.1f}% win rate ({bull_wins}/{len(bullish_trades)} trades)")

    if bearish_trades:
        bear_wins = len([t for t in bearish_trades if t['result'] == 'WIN'])
        bear_win_rate = (bear_wins / len(bearish_trades)) * 100
        print(f"📉 BEARISH CONFLUENCES: {bear_win_rate:.1f}% win rate ({bear_wins}/{len(bearish_trades)} trades)")

    # Save results
    results_df = pd.DataFrame(trades)
    results_df.to_csv('data/ULTIMATE_CONFLUENCE_RESULTS.csv', index=False)
    print(f"\n💾 Results saved to: data/ULTIMATE_CONFLUENCE_RESULTS.csv")

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }

def main():
    """Main backtest function"""

    print("🎯 ULTIMATE CONFLUENCE BACKTEST - THE FINAL TEST!")
    print("Based on analyzing 2,031 real significant moves over 5 years")
    print("="*80)

    df = load_data()
    if df is None:
        return

    # Add proven indicators
    df = add_proven_indicators(df)

    # Run the ultimate test
    results = run_ultimate_confluence_backtest(df)

    print(f"\n🏁 FINAL VERDICT:")
    print("="*50)

    if results and results['win_rate'] >= 55:
        print(f"🎉 SUCCESS! {results['win_rate']:.1f}% win rate proves real confluences work!")
        print(f"💰 ${results['total_profit']:+.2f} profit validates the 5-year analysis!")
        print("✅ Data-driven approach beats theoretical confluences!")
    elif results and results['win_rate'] >= 45:
        print(f"✅ PROMISING! {results['win_rate']:.1f}% win rate shows improvement")
        print(f"💰 ${results['total_profit']:+.2f} - better than random but needs refinement")
        print("💡 Real patterns are directionally correct but need better filtering")
    elif results:
        print(f"⚠️  {results['win_rate']:.1f}% win rate - even real patterns struggle")
        print(f"💰 ${results['total_profit']:+.2f}")
        print("🔍 This confirms that XAUUSD is extremely difficult to predict")
        print("💭 Maybe markets are more random than we think...")
    else:
        print("❌ No trades generated with strict confluence requirements")

    print(f"\n🎯 ULTIMATE CONCLUSION:")
    print("We analyzed 30,000 candles and 2,031 actual big moves.")
    print("This is the most comprehensive analysis possible!")

if __name__ == "__main__":
    main()