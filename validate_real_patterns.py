#!/usr/bin/env python3
"""
VALIDATE REAL XAUUSD PATTERNS - Backtest Based on Actual Pre-Move Analysis
Test the patterns we discovered from analyzing what preceded big moves
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
        return None

def add_pattern_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the specific indicators our pattern analysis revealed"""

    # Basic price action
    df['range_size'] = df['high'] - df['low']
    df['body_size'] = abs(df['close'] - df['open'])

    # 20-period SMA (key from our analysis)
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['above_sma20'] = df['close'] > df['sma_20']

    # Recent high/low analysis (12-hour lookback from our analysis)
    df['recent_high_12h'] = df['high'].rolling(window=12).max()
    df['recent_low_12h'] = df['low'].rolling(window=12).min()
    df['distance_from_high'] = df['close'] - df['recent_high_12h']
    df['distance_from_low'] = df['close'] - df['recent_low_12h']

    # Average volatility (for context)
    df['avg_range_12h'] = df['range_size'].rolling(window=12).mean()

    # London Fix times
    df['london_fix'] = df['ny_hour'].isin([5, 10])

    return df

def check_bullish_pattern(candle: pd.Series, df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """Check for bullish pattern based on our real analysis"""

    # Key criteria from analysis:
    # - Best times: 7 AM or 9 AM NY
    # - Price above SMA20 (64% of successful cases)
    # - Distance from high: around -$15.5 (below recent high)
    # - Distance from low: around +$21.4 (above recent low)
    # - London Fix present in 73.5% of cases

    hour = candle['ny_hour']

    # Must be at optimal times
    if hour not in [7, 9]:
        return None

    # Need sufficient data
    if idx < 20:
        return None

    # Price should be above SMA20 (uptrend bias)
    if not candle['above_sma20']:
        return None

    # Check position in recent range
    dist_from_high = candle['distance_from_high']
    dist_from_low = candle['distance_from_low']

    # Should be below recent high but not too far (-$10 to -$25 range)
    if not (-25 <= dist_from_high <= -5):
        return None

    # Should be above recent low (+$10 to +$35 range)
    if not (10 <= dist_from_low <= 40):
        return None

    # Check if we're in London Fix vicinity (within 2 hours)
    london_fix_nearby = any(df.iloc[max(0, idx-2):idx+3]['london_fix'])

    # Calculate confluence score
    confluence_score = 0
    factors = []

    if candle['above_sma20']:
        confluence_score += 3
        factors.append("Above SMA20")

    if hour in [7, 9]:
        confluence_score += 3
        factors.append(f"Optimal time {hour}:00")

    if london_fix_nearby:
        confluence_score += 2
        factors.append("London Fix nearby")

    if -20 <= dist_from_high <= -10:
        confluence_score += 2
        factors.append("Good distance from high")

    if 15 <= dist_from_low <= 30:
        confluence_score += 2
        factors.append("Good distance from low")

    # Require minimum confluence
    if confluence_score < 8:
        return None

    return {
        'direction': 'BUY',
        'entry_price': candle['close'],
        'confluence_score': confluence_score,
        'factors': factors,
        'pattern_type': 'real_bullish'
    }

def check_bearish_pattern(candle: pd.Series, df: pd.DataFrame, idx: int) -> Optional[Dict]:
    """Check for bearish pattern based on our real analysis"""

    # Key criteria from analysis:
    # - Best time: 7 PM NY (19:00)
    # - Can work with price above or below SMA20 (55% above)
    # - Distance from high: around -$18.9 (closer to recent high)
    # - Distance from low: around +$23.1 (above recent low)
    # - London Fix present in 74.6% of cases

    hour = candle['ny_hour']

    # Must be at optimal time
    if hour != 19:
        return None

    # Need sufficient data
    if idx < 20:
        return None

    # Check position in recent range
    dist_from_high = candle['distance_from_high']
    dist_from_low = candle['distance_from_low']

    # Should be closer to recent high than bullish setups (-$25 to -$10)
    if not (-30 <= dist_from_high <= -5):
        return None

    # Should be well above recent low (+$15 to +$35)
    if not (15 <= dist_from_low <= 40):
        return None

    # Check if we're in London Fix vicinity
    london_fix_nearby = any(df.iloc[max(0, idx-2):idx+3]['london_fix'])

    # Calculate confluence score
    confluence_score = 0
    factors = []

    if hour == 19:
        confluence_score += 4
        factors.append("Optimal Asian open time")

    if london_fix_nearby:
        confluence_score += 2
        factors.append("London Fix nearby")

    if -25 <= dist_from_high <= -10:
        confluence_score += 3
        factors.append("Close to recent high")

    if 20 <= dist_from_low <= 30:
        confluence_score += 2
        factors.append("Good distance from low")

    # SMA20 position (less important for bears)
    if candle['above_sma20']:
        confluence_score += 1
        factors.append("Above SMA20")
    else:
        confluence_score += 1
        factors.append("Below SMA20")

    # Require minimum confluence
    if confluence_score < 8:
        return None

    return {
        'direction': 'SELL',
        'entry_price': candle['close'],
        'confluence_score': confluence_score,
        'factors': factors,
        'pattern_type': 'real_bearish'
    }

def simulate_pattern_trade(signal: Dict, future_data: pd.DataFrame, expected_move: float = 60) -> Optional[Dict]:
    """Simulate trade based on real pattern expectations"""

    entry_price = signal['entry_price']
    direction = signal['direction']

    # Based on our analysis: ~$62 average moves in ~18 hours
    # Use conservative targets: $30 profit, $15 stop
    profit_target = 30.0
    stop_loss = 15.0

    if direction == "BUY":
        take_profit = entry_price + profit_target
        stop_price = entry_price - stop_loss
    else:
        take_profit = entry_price - profit_target
        stop_price = entry_price + stop_loss

    # Check outcome over next 24 hours (our analysis showed ~18h average)
    for i, (_, candle) in enumerate(future_data.head(24).iterrows()):
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

    # If no clear outcome, exit at market
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
            'duration_hours': 24,
            'exit_reason': 'timeout'
        }

    return None

def run_pattern_validation_backtest(df: pd.DataFrame) -> Dict[str, Any]:
    """Run backtest based on real discovered patterns"""

    print("🎯 VALIDATING REAL XAUUSD PATTERNS")
    print("=" * 60)

    trades = []
    daily_trades = {}

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

            signal = None

            # Check for bullish pattern (7 AM, 9 AM NY)
            if candle['ny_hour'] in [7, 9]:
                signal = check_bullish_pattern(candle, df, idx)

            # Check for bearish pattern (7 PM NY)
            elif candle['ny_hour'] == 19:
                signal = check_bearish_pattern(candle, df, idx)

            if signal:
                # Get future data for simulation
                future_data = df.iloc[idx+1:idx+25]  # Next 24 hours
                if len(future_data) < 12:
                    continue

                trade_result = simulate_pattern_trade(signal, future_data)

                if trade_result:
                    trade_record = {
                        'date': date,
                        'entry_time': candle['ny_time'],
                        'pattern_type': signal['pattern_type'],
                        'direction': signal['direction'],
                        'entry_price': signal['entry_price'],
                        'confluence_score': signal['confluence_score'],
                        'factors': ', '.join(signal['factors']),
                        **trade_result
                    }

                    trades.append(trade_record)
                    daily_trades[date] += 1

                    result_emoji = "✅" if trade_result['result'] == 'WIN' else "❌"
                    print(f"{result_emoji} Trade {len(trades)}: {date} {signal['direction']} "
                          f"@${signal['entry_price']:.2f} [{candle['ny_hour']}:00] "
                          f"→ {trade_result['result']} ${trade_result['profit']:+.2f}")

    return analyze_pattern_results(trades)

def analyze_pattern_results(trades: List[Dict]) -> Dict[str, Any]:
    """Analyze the pattern-based backtest results"""

    if not trades:
        print("❌ No pattern trades found")
        return {}

    print(f"\n🏆 REAL PATTERN VALIDATION RESULTS ({len(trades)} trades)")
    print("=" * 60)

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

    # Pattern breakdown
    bullish_trades = [t for t in trades if t['pattern_type'] == 'real_bullish']
    bearish_trades = [t for t in trades if t['pattern_type'] == 'real_bearish']

    if bullish_trades:
        bull_wins = len([t for t in bullish_trades if t['result'] == 'WIN'])
        bull_win_rate = (bull_wins / len(bullish_trades)) * 100
        print(f"\n📈 BULLISH (7,9 AM): {bull_win_rate:.1f}% win rate ({bull_wins}/{len(bullish_trades)} trades)")

    if bearish_trades:
        bear_wins = len([t for t in bearish_trades if t['result'] == 'WIN'])
        bear_win_rate = (bear_wins / len(bearish_trades)) * 100
        print(f"📉 BEARISH (7 PM): {bear_win_rate:.1f}% win rate ({bear_wins}/{len(bearish_trades)} trades)")

    # Time analysis
    time_analysis = {}
    for trade in trades:
        if isinstance(trade['entry_time'], str):
            hour = int(trade['entry_time'].split(':')[0]) % 24
        else:
            hour = trade['entry_time'].hour
        if hour not in time_analysis:
            time_analysis[hour] = {'wins': 0, 'total': 0}
        time_analysis[hour]['total'] += 1
        if trade['result'] == 'WIN':
            time_analysis[hour]['wins'] += 1

    print(f"\n⏰ TIME-BASED PERFORMANCE:")
    for hour in sorted(time_analysis.keys()):
        wins = time_analysis[hour]['wins']
        total = time_analysis[hour]['total']
        rate = (wins / total) * 100 if total > 0 else 0
        print(f"   {hour:02d}:00 - {rate:.1f}% ({wins}/{total})")

    # Save results
    results_df = pd.DataFrame(trades)
    results_df.to_csv('data/PATTERN_VALIDATION_RESULTS.csv', index=False)
    print(f"\n💾 Results saved to: data/PATTERN_VALIDATION_RESULTS.csv")

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }

def main():
    """Main validation function"""

    print("🔍 VALIDATING DISCOVERED XAUUSD PATTERNS")
    print("=" * 60)

    df = load_data()
    if df is None:
        return

    print(f"📊 Testing patterns on {len(df)} candles")
    print("🎯 Bullish: 7 AM & 9 AM NY, above SMA20, specific positioning")
    print("🎯 Bearish: 7 PM NY, near highs, London Fix vicinity")

    # Add pattern indicators
    df = add_pattern_indicators(df)

    # Run validation backtest
    results = run_pattern_validation_backtest(df)

    print(f"\n🏆 FINAL VALIDATION RESULTS")
    print("=" * 60)

    if results and results['win_rate'] >= 60:
        print(f"🎉 EXCELLENT! {results['win_rate']:.1f}% win rate validates the patterns!")
        print(f"💰 ${results['total_profit']:+.2f} profit confirms these patterns work!")
        print("✅ REAL data-driven approach beats theoretical confluences!")
    elif results and results['win_rate'] >= 45:
        print(f"✅ GOOD! {results['win_rate']:.1f}% win rate shows patterns have merit")
        print(f"💰 ${results['total_profit']:+.2f} profit - better than previous attempts")
        print("💡 These time-based patterns are more reliable than indicators")
    elif results:
        print(f"⚠️  {results['win_rate']:.1f}% win rate - patterns need refinement")
        print(f"💰 ${results['total_profit']:+.2f} - let's analyze what's missing")
    else:
        print("❌ Patterns didn't generate enough trades for validation")

    print(f"\n💡 Pattern validation complete!")
    print("🎯 Now we know if time-based confluences actually work!")

if __name__ == "__main__":
    main()