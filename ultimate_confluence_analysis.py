#!/usr/bin/env python3
"""
ULTIMATE CONFLUENCE ANALYSIS - Find What ACTUALLY Works
Analyze 5 years of XAUUSD data to find reliable confluence patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Any, Optional, Tuple

def load_3year_data():
    """Load the comprehensive XAUUSD dataset"""
    try:
        df = pd.read_csv('data/XAUUSD_3YEAR_DATA.csv')
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df['ny_time'] = pd.to_datetime(df['ny_time'], utc=True)
        print(f"📊 Loaded {len(df)} candles from {df.iloc[0]['ny_time']} to {df.iloc[-1]['ny_time']}")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def add_comprehensive_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add ALL possible confluence indicators"""

    print("🔧 Adding comprehensive technical indicators...")

    # Basic price action
    df['body_size'] = abs(df['close'] - df['open'])
    df['range_size'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['is_green'] = df['close'] > df['open']
    df['is_doji'] = df['body_size'] < df['range_size'] * 0.1
    df['is_hammer'] = df['lower_wick'] > df['body_size'] * 2
    df['is_shooting_star'] = df['upper_wick'] > df['body_size'] * 2

    # Moving Averages
    for period in [9, 20, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

    # MA relationships
    df['above_sma9'] = df['close'] > df['sma_9']
    df['above_sma20'] = df['close'] > df['sma_20']
    df['above_sma50'] = df['close'] > df['sma_50']
    df['above_ema9'] = df['close'] > df['ema_9']
    df['above_ema20'] = df['close'] > df['ema_20']

    # Trend analysis
    df['sma20_slope'] = df['sma_20'].diff(5)  # 5-period slope
    df['ema20_slope'] = df['ema_20'].diff(5)
    df['uptrend_sma'] = df['sma20_slope'] > 0
    df['uptrend_ema'] = df['ema20_slope'] > 0

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_oversold'] = df['rsi'] < 30
    df['rsi_overbought'] = df['rsi'] > 70
    df['rsi_bullish'] = (df['rsi'] > 50) & (df['rsi'] < 70)
    df['rsi_bearish'] = (df['rsi'] < 50) & (df['rsi'] > 30)

    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    df['macd_bullish'] = (df['macd'] > df['macd_signal']) & (df['macd'] > 0)
    df['macd_bearish'] = (df['macd'] < df['macd_signal']) & (df['macd'] < 0)

    # Volatility
    df['atr_14'] = df['range_size'].rolling(window=14).mean()
    df['high_volatility'] = df['range_size'] > df['atr_14'] * 1.5
    df['low_volatility'] = df['range_size'] < df['atr_14'] * 0.5

    # Support/Resistance levels
    df['recent_high_12h'] = df['high'].rolling(window=12).max()
    df['recent_low_12h'] = df['low'].rolling(window=12).min()
    df['recent_high_24h'] = df['high'].rolling(window=24).max()
    df['recent_low_24h'] = df['low'].rolling(window=24).min()

    # Distance from key levels
    df['dist_from_high_12h'] = df['close'] - df['recent_high_12h']
    df['dist_from_low_12h'] = df['close'] - df['recent_low_12h']
    df['dist_from_high_24h'] = df['close'] - df['recent_high_24h']
    df['dist_from_low_24h'] = df['close'] - df['recent_low_24h']

    # Near key levels
    df['near_recent_high'] = abs(df['dist_from_high_12h']) < 10
    df['near_recent_low'] = abs(df['dist_from_low_12h']) < 10

    # Session information
    df['london_fix'] = df['ny_hour'].isin([5, 10])  # 10:30 AM & 3:00 PM GMT
    df['asian_session'] = df['ny_hour'].isin([19, 20, 21, 22, 23, 0, 1, 2, 3])
    df['london_session'] = df['ny_hour'].isin([3, 4, 5, 6, 7, 8, 9, 10, 11])
    df['ny_session'] = df['ny_hour'].isin([8, 9, 10, 11, 12, 13, 14, 15, 16])

    # Day of week patterns
    df['weekday'] = df['ny_time'].dt.dayofweek
    df['is_monday'] = df['weekday'] == 0
    df['is_friday'] = df['weekday'] == 4

    # Previous candle analysis
    df['prev_green'] = df['is_green'].shift(1)
    df['prev_red'] = (~df['is_green']).shift(1)
    df['prev_doji'] = df['is_doji'].shift(1)

    # Consecutive patterns
    df['consecutive_green'] = 0
    df['consecutive_red'] = 0

    for i in range(1, len(df)):
        if df.iloc[i]['is_green'] and df.iloc[i-1]['is_green']:
            df.iloc[i, df.columns.get_loc('consecutive_green')] = df.iloc[i-1]['consecutive_green'] + 1
        if not df.iloc[i]['is_green'] and not df.iloc[i-1]['is_green']:
            df.iloc[i, df.columns.get_loc('consecutive_red')] = df.iloc[i-1]['consecutive_red'] + 1

    print(f"✅ Added {len([col for col in df.columns if col not in ['datetime', 'ny_time', 'open', 'high', 'low', 'close', 'volume', 'ny_hour', 'ny_date_str']])} indicators")

    return df

def find_significant_moves(df: pd.DataFrame, min_move_size: float = 50.0, hours_ahead: int = 24) -> List[Dict]:
    """Find significant price moves and what triggered them"""

    print(f"🔍 Finding moves ≥ ${min_move_size} within {hours_ahead} hours...")

    moves = []

    for i in range(len(df) - hours_ahead):
        current = df.iloc[i]
        future_data = df.iloc[i+1:i+hours_ahead+1]

        if len(future_data) < hours_ahead // 2:
            continue

        # Find potential moves
        future_high = future_data['high'].max()
        future_low = future_data['low'].min()

        bullish_move = future_high - current['close']
        bearish_move = current['close'] - future_low

        if bullish_move >= min_move_size:
            # Find when high was reached
            high_idx = future_data['high'].idxmax()
            hours_to_target = future_data.index.get_loc(high_idx) + 1

            moves.append({
                'index': i,
                'direction': 'BULLISH',
                'move_size': bullish_move,
                'hours_to_target': hours_to_target,
                'entry_price': current['close'],
                'target_price': future_high,
                'entry_candle': current
            })

        elif bearish_move >= min_move_size:
            # Find when low was reached
            low_idx = future_data['low'].idxmin()
            hours_to_target = future_data.index.get_loc(low_idx) + 1

            moves.append({
                'index': i,
                'direction': 'BEARISH',
                'move_size': bearish_move,
                'hours_to_target': hours_to_target,
                'entry_price': current['close'],
                'target_price': future_low,
                'entry_candle': current
            })

    print(f"📊 Found {len(moves)} significant moves")
    bullish_count = len([m for m in moves if m['direction'] == 'BULLISH'])
    bearish_count = len([m for m in moves if m['direction'] == 'BEARISH'])
    print(f"   📈 Bullish: {bullish_count}")
    print(f"   📉 Bearish: {bearish_count}")

    return moves

def analyze_confluence_patterns(moves: List[Dict], df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze what confluences actually triggered before big moves"""

    print("🔍 Analyzing confluence patterns before significant moves...")

    # All possible confluence factors to test
    confluence_factors = [
        # Moving Average confluences
        'above_sma9', 'above_sma20', 'above_sma50', 'above_ema9', 'above_ema20',
        'uptrend_sma', 'uptrend_ema',

        # RSI confluences
        'rsi_oversold', 'rsi_overbought', 'rsi_bullish', 'rsi_bearish',

        # MACD confluences
        'macd_bullish', 'macd_bearish',

        # Price action confluences
        'is_green', 'is_doji', 'is_hammer', 'is_shooting_star',
        'prev_green', 'prev_red', 'prev_doji',

        # Volatility confluences
        'high_volatility', 'low_volatility',

        # Level confluences
        'near_recent_high', 'near_recent_low',

        # Session confluences
        'london_fix', 'asian_session', 'london_session', 'ny_session',
        'is_monday', 'is_friday'
    ]

    bullish_moves = [m for m in moves if m['direction'] == 'BULLISH']
    bearish_moves = [m for m in moves if m['direction'] == 'BEARISH']

    results = {
        'bullish_confluences': {},
        'bearish_confluences': {},
        'total_bullish_moves': len(bullish_moves),
        'total_bearish_moves': len(bearish_moves)
    }

    # Analyze bullish confluences
    print("📈 Analyzing BULLISH confluence triggers...")
    for factor in confluence_factors:
        if factor not in df.columns:
            continue

        factor_present = sum(1 for move in bullish_moves if move['entry_candle'].get(factor, False))
        percentage = (factor_present / len(bullish_moves)) * 100 if bullish_moves else 0

        results['bullish_confluences'][factor] = {
            'count': factor_present,
            'percentage': percentage,
            'total_moves': len(bullish_moves)
        }

    # Analyze bearish confluences
    print("📉 Analyzing BEARISH confluence triggers...")
    for factor in confluence_factors:
        if factor not in df.columns:
            continue

        factor_present = sum(1 for move in bearish_moves if move['entry_candle'].get(factor, False))
        percentage = (factor_present / len(bearish_moves)) * 100 if bearish_moves else 0

        results['bearish_confluences'][factor] = {
            'count': factor_present,
            'percentage': percentage,
            'total_moves': len(bearish_moves)
        }

    return results

def calculate_confluence_reliability(df: pd.DataFrame, moves: List[Dict]) -> Dict[str, Dict]:
    """Calculate how often each confluence appears but DOESN'T lead to a move"""

    print("⚖️ Calculating confluence reliability (success vs failure rates)...")

    confluence_factors = [
        'above_sma20', 'above_ema20', 'uptrend_sma', 'rsi_bullish', 'rsi_bearish',
        'macd_bullish', 'macd_bearish', 'is_green', 'is_hammer', 'is_shooting_star',
        'london_fix', 'asian_session', 'near_recent_high', 'near_recent_low'
    ]

    # Get indices where moves occurred
    move_indices = set(m['index'] for m in moves)
    bullish_indices = set(m['index'] for m in moves if m['direction'] == 'BULLISH')
    bearish_indices = set(m['index'] for m in moves if m['direction'] == 'BEARISH')

    reliability_results = {}

    for factor in confluence_factors:
        if factor not in df.columns:
            continue

        # Count total occurrences of this confluence
        total_occurrences = df[factor].sum()

        # Count how many led to bullish moves
        bullish_successes = sum(1 for idx in bullish_indices if idx < len(df) and df.iloc[idx][factor])

        # Count how many led to bearish moves
        bearish_successes = sum(1 for idx in bearish_indices if idx < len(df) and df.iloc[idx][factor])

        # Count total successes
        total_successes = bullish_successes + bearish_successes

        # Calculate reliability
        bullish_reliability = (bullish_successes / total_occurrences * 100) if total_occurrences > 0 else 0
        bearish_reliability = (bearish_successes / total_occurrences * 100) if total_occurrences > 0 else 0
        overall_reliability = (total_successes / total_occurrences * 100) if total_occurrences > 0 else 0

        reliability_results[factor] = {
            'total_occurrences': total_occurrences,
            'bullish_successes': bullish_successes,
            'bearish_successes': bearish_successes,
            'total_successes': total_successes,
            'bullish_reliability': bullish_reliability,
            'bearish_reliability': bearish_reliability,
            'overall_reliability': overall_reliability,
            'failure_rate': 100 - overall_reliability
        }

    return reliability_results

def print_confluence_results(confluence_results: Dict, reliability_results: Dict):
    """Print comprehensive confluence analysis results"""

    print("\n" + "="*80)
    print("🎯 COMPREHENSIVE CONFLUENCE ANALYSIS RESULTS")
    print("="*80)

    # Top bullish confluences
    print("\n📈 TOP BULLISH CONFLUENCES (most frequent before big bullish moves):")
    bullish_sorted = sorted(confluence_results['bullish_confluences'].items(),
                           key=lambda x: x[1]['percentage'], reverse=True)

    for i, (factor, data) in enumerate(bullish_sorted[:10]):
        reliability = reliability_results.get(factor, {})
        print(f"   {i+1:2d}. {factor:<20} | {data['percentage']:5.1f}% ({data['count']:3d}/{data['total_moves']:3d}) | "
              f"Reliability: {reliability.get('bullish_reliability', 0):.2f}%")

    # Top bearish confluences
    print("\n📉 TOP BEARISH CONFLUENCES (most frequent before big bearish moves):")
    bearish_sorted = sorted(confluence_results['bearish_confluences'].items(),
                           key=lambda x: x[1]['percentage'], reverse=True)

    for i, (factor, data) in enumerate(bearish_sorted[:10]):
        reliability = reliability_results.get(factor, {})
        print(f"   {i+1:2d}. {factor:<20} | {data['percentage']:5.1f}% ({data['count']:3d}/{data['total_moves']:3d}) | "
              f"Reliability: {reliability.get('bearish_reliability', 0):.2f}%")

    # Most reliable confluences (low failure rate)
    print("\n⚖️ MOST RELIABLE CONFLUENCES (lowest failure rates):")
    reliable_sorted = sorted(reliability_results.items(),
                           key=lambda x: x[1]['overall_reliability'], reverse=True)

    for i, (factor, data) in enumerate(reliable_sorted[:10]):
        print(f"   {i+1:2d}. {factor:<20} | Success: {data['overall_reliability']:5.2f}% "
              f"| Failure: {data['failure_rate']:5.2f}% | Occurs: {data['total_occurrences']} times")

    # Identify best combinations
    print("\n🏆 KEY INSIGHTS:")

    # Best bullish factors
    best_bullish = [(k, v) for k, v in bullish_sorted[:5]]
    print(f"🎯 Best Bullish Setup: ", end="")
    print(" + ".join([f"{k} ({v['percentage']:.0f}%)" for k, v in best_bullish[:3]]))

    # Best bearish factors
    best_bearish = [(k, v) for k, v in bearish_sorted[:5]]
    print(f"🎯 Best Bearish Setup: ", end="")
    print(" + ".join([f"{k} ({v['percentage']:.0f}%)" for k, v in best_bearish[:3]]))

    # Most reliable overall
    most_reliable = reliable_sorted[0]
    print(f"🔥 Most Reliable Factor: {most_reliable[0]} ({most_reliable[1]['overall_reliability']:.1f}% success rate)")

def main():
    """Main analysis function"""

    print("🎯 ULTIMATE CONFLUENCE ANALYSIS - FIND WHAT ACTUALLY WORKS!")
    print("="*80)

    # Load comprehensive dataset
    df = load_3year_data()
    if df is None:
        return

    print(f"📊 Analyzing {len(df)} candles spanning 5 years of XAUUSD data")

    # Add all technical indicators
    df = add_comprehensive_indicators(df)

    # Find significant moves
    significant_moves = find_significant_moves(df, min_move_size=50.0, hours_ahead=24)

    if not significant_moves:
        print("❌ No significant moves found")
        return

    # Analyze confluence patterns
    confluence_results = analyze_confluence_patterns(significant_moves, df)

    # Calculate reliability
    reliability_results = calculate_confluence_reliability(df, significant_moves)

    # Print comprehensive results
    print_confluence_results(confluence_results, reliability_results)

    # Save results
    results_summary = {
        'analysis_date': datetime.now().isoformat(),
        'total_candles': len(df),
        'total_moves': len(significant_moves),
        'bullish_moves': confluence_results['total_bullish_moves'],
        'bearish_moves': confluence_results['total_bearish_moves'],
        'confluence_analysis': confluence_results,
        'reliability_analysis': reliability_results
    }

    # Save to CSV for further analysis
    moves_df = pd.DataFrame(significant_moves)
    moves_df.to_csv('data/SIGNIFICANT_MOVES_5YEAR.csv', index=False)

    print(f"\n💾 Results saved to data/SIGNIFICANT_MOVES_5YEAR.csv")
    print(f"🎉 ANALYSIS COMPLETE! Now you know what confluences ACTUALLY work!")

if __name__ == "__main__":
    main()