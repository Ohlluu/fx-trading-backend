#!/usr/bin/env python3
"""
Analyze What Actually Happened Before XAUUSD's Biggest Moves
Instead of guessing confluences, let's see what preceded real profitable trades
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

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic technical indicators to analyze"""

    # Price action
    df['body_size'] = abs(df['close'] - df['open'])
    df['range_size'] = df['high'] - df['low']
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()

    # Volatility
    df['atr_14'] = df['range_size'].rolling(window=14).mean()
    df['high_vol'] = df['range_size'] > df['atr_14'] * 1.5

    # Price position
    df['above_sma20'] = df['close'] > df['sma_20']
    df['above_sma50'] = df['close'] > df['sma_50']

    # Session info
    df['london_fix'] = df['ny_hour'].isin([5, 10])
    df['london_session'] = df['ny_hour'].isin([8, 9, 10, 11])
    df['asian_session'] = df['ny_hour'].isin([19, 20, 21, 22, 23, 0, 1, 2, 3])

    return df

def find_significant_moves(df: pd.DataFrame, min_move_size: float = 40.0) -> List[Dict]:
    """Find significant price moves in the data"""

    print(f"🔍 Finding moves ≥ ${min_move_size}")

    significant_moves = []

    for i in range(len(df) - 24):  # Look ahead 24 hours
        current_price = df.iloc[i]['close']
        current_time = df.iloc[i]['ny_time']

        # Look at next 24 hours for significant moves
        future_data = df.iloc[i+1:i+25]

        if len(future_data) < 12:  # Need minimum future data
            continue

        # Find highest high and lowest low in next 24 hours
        future_high = future_data['high'].max()
        future_low = future_data['low'].min()

        # Calculate potential moves
        bullish_move = future_high - current_price
        bearish_move = current_price - future_low

        # Check for significant bullish move
        if bullish_move >= min_move_size:
            # Find when the high was reached
            high_idx = future_data['high'].idxmax()
            high_candle = df.loc[high_idx]
            hours_to_high = future_data.index.get_loc(high_idx) + 1

            significant_moves.append({
                'index': i,
                'time': current_time,
                'direction': 'BULLISH',
                'entry_price': current_price,
                'target_price': future_high,
                'move_size': bullish_move,
                'hours_to_target': hours_to_high,
                'target_time': high_candle['ny_time']
            })

        # Check for significant bearish move
        elif bearish_move >= min_move_size:
            # Find when the low was reached
            low_idx = future_data['low'].idxmin()
            low_candle = df.loc[low_idx]
            hours_to_low = future_data.index.get_loc(low_idx) + 1

            significant_moves.append({
                'index': i,
                'time': current_time,
                'direction': 'BEARISH',
                'entry_price': current_price,
                'target_price': future_low,
                'move_size': bearish_move,
                'hours_to_target': hours_to_low,
                'target_time': low_candle['ny_time']
            })

    print(f"📊 Found {len(significant_moves)} significant moves")
    return significant_moves

def analyze_pre_move_conditions(df: pd.DataFrame, moves: List[Dict], lookback: int = 12) -> Dict:
    """Analyze conditions before significant moves"""

    print(f"🔍 Analyzing conditions {lookback} hours before big moves...")

    bullish_setups = []
    bearish_setups = []

    for move in moves:
        move_idx = move['index']

        # Skip if not enough history
        if move_idx < lookback:
            continue

        # Get pre-move data (lookback hours before)
        pre_move_data = df.iloc[move_idx-lookback:move_idx]
        current_candle = df.iloc[move_idx]

        # Calculate pre-move characteristics
        setup_analysis = {
            'move_size': move['move_size'],
            'hours_to_target': move['hours_to_target'],

            # Recent volatility
            'avg_range_12h': pre_move_data['range_size'].mean(),
            'high_vol_count': pre_move_data['high_vol'].sum(),
            'max_range_12h': pre_move_data['range_size'].max(),

            # Price action patterns
            'consecutive_green': 0,
            'consecutive_red': 0,
            'doji_count': 0,
            'hammer_count': 0,

            # Position relative to MAs
            'above_sma20_pct': (pre_move_data['above_sma20'].sum() / len(pre_move_data)) * 100,
            'above_sma50_pct': (pre_move_data['above_sma50'].sum() / len(pre_move_data)) * 100,
            'sma20_slope': (current_candle['sma_20'] - pre_move_data.iloc[0]['sma_20']),

            # Session timing
            'london_fix_present': pre_move_data['london_fix'].any(),
            'london_session_pct': (pre_move_data['london_session'].sum() / len(pre_move_data)) * 100,
            'asian_session_pct': (pre_move_data['asian_session'].sum() / len(pre_move_data)) * 100,

            # Current candle properties
            'entry_hour': current_candle['ny_hour'],
            'entry_body_size': current_candle['body_size'],
            'entry_range_size': current_candle['range_size'],
            'entry_is_green': current_candle['close'] > current_candle['open'],

            # Recent highs/lows
            'recent_high': pre_move_data['high'].max(),
            'recent_low': pre_move_data['low'].min(),
            'distance_from_high': current_candle['close'] - pre_move_data['high'].max(),
            'distance_from_low': current_candle['close'] - pre_move_data['low'].min(),
        }

        # Calculate consecutive patterns
        for j in range(len(pre_move_data)):
            candle = pre_move_data.iloc[-(j+1)]  # Go backwards
            if candle['close'] > candle['open']:
                if setup_analysis['consecutive_green'] == j:
                    setup_analysis['consecutive_green'] += 1
            else:
                break

        for j in range(len(pre_move_data)):
            candle = pre_move_data.iloc[-(j+1)]  # Go backwards
            if candle['close'] < candle['open']:
                if setup_analysis['consecutive_red'] == j:
                    setup_analysis['consecutive_red'] += 1
            else:
                break

        # Count special candle patterns
        for _, candle in pre_move_data.iterrows():
            # Doji (small body relative to range)
            if candle['body_size'] < candle['range_size'] * 0.1:
                setup_analysis['doji_count'] += 1

            # Hammer-like (long lower wick)
            if candle['lower_wick'] > candle['body_size'] * 2:
                setup_analysis['hammer_count'] += 1

        # Store in appropriate list
        if move['direction'] == 'BULLISH':
            bullish_setups.append(setup_analysis)
        else:
            bearish_setups.append(setup_analysis)

    return {
        'bullish_setups': bullish_setups,
        'bearish_setups': bearish_setups
    }

def find_common_patterns(setups: List[Dict], direction: str) -> Dict:
    """Find common patterns in the setups"""

    if not setups:
        return {}

    print(f"\n📊 ANALYZING {len(setups)} {direction} SETUPS")
    print("=" * 50)

    # Convert to DataFrame for easier analysis
    df_setups = pd.DataFrame(setups)

    patterns = {
        'count': len(setups),
        'avg_move_size': df_setups['move_size'].mean(),
        'avg_hours_to_target': df_setups['hours_to_target'].mean(),

        # Volatility patterns
        'avg_recent_volatility': df_setups['avg_range_12h'].mean(),
        'high_vol_sessions_avg': df_setups['high_vol_count'].mean(),
        'max_range_avg': df_setups['max_range_12h'].mean(),

        # MA patterns
        'above_sma20_avg': df_setups['above_sma20_pct'].mean(),
        'above_sma50_avg': df_setups['above_sma50_pct'].mean(),
        'sma20_slope_avg': df_setups['sma20_slope'].mean(),

        # Session timing
        'london_fix_frequency': (df_setups['london_fix_present'].sum() / len(setups)) * 100,
        'london_session_avg': df_setups['london_session_pct'].mean(),
        'asian_session_avg': df_setups['asian_session_pct'].mean(),

        # Entry timing
        'common_entry_hours': df_setups['entry_hour'].mode().tolist() if not df_setups['entry_hour'].mode().empty else [],
        'avg_entry_body': df_setups['entry_body_size'].mean(),
        'avg_entry_range': df_setups['entry_range_size'].mean(),
        'green_entry_pct': (df_setups['entry_is_green'].sum() / len(setups)) * 100,

        # Position patterns
        'avg_distance_from_high': df_setups['distance_from_high'].mean(),
        'avg_distance_from_low': df_setups['distance_from_low'].mean(),

        # Consecutive patterns
        'avg_consecutive_green': df_setups['consecutive_green'].mean(),
        'avg_consecutive_red': df_setups['consecutive_red'].mean(),
        'avg_doji_count': df_setups['doji_count'].mean(),
        'avg_hammer_count': df_setups['hammer_count'].mean(),
    }

    # Print key findings
    print(f"💰 AVERAGE MOVE SIZE: ${patterns['avg_move_size']:.1f}")
    print(f"⏰ AVERAGE TIME TO TARGET: {patterns['avg_hours_to_target']:.1f} hours")
    print(f"📈 VOLATILITY BEFORE MOVE: ${patterns['avg_recent_volatility']:.1f} avg range")
    print(f"🎯 POSITION VS SMA20: {patterns['above_sma20_avg']:.1f}% of time above")
    print(f"🕐 COMMON ENTRY HOURS: {patterns['common_entry_hours']}")
    print(f"🔥 LONDON FIX PRESENT: {patterns['london_fix_frequency']:.1f}% of setups")
    print(f"📊 GREEN ENTRY CANDLES: {patterns['green_entry_pct']:.1f}%")
    print(f"📏 AVG DISTANCE FROM HIGH: ${patterns['avg_distance_from_high']:+.1f}")
    print(f"📏 AVG DISTANCE FROM LOW: ${patterns['avg_distance_from_low']:+.1f}")

    return patterns

def compare_patterns(bullish_patterns: Dict, bearish_patterns: Dict):
    """Compare bullish vs bearish setup patterns"""

    print(f"\n🔀 BULLISH vs BEARISH PATTERN COMPARISON")
    print("=" * 60)

    if not bullish_patterns or not bearish_patterns:
        print("❌ Need both bullish and bearish data for comparison")
        return

    comparisons = [
        ('Move Size', 'avg_move_size', '$'),
        ('Hours to Target', 'avg_hours_to_target', 'h'),
        ('Recent Volatility', 'avg_recent_volatility', '$'),
        ('Above SMA20 %', 'above_sma20_avg', '%'),
        ('London Fix %', 'london_fix_frequency', '%'),
        ('Green Entry %', 'green_entry_pct', '%'),
        ('Distance from High', 'avg_distance_from_high', '$'),
        ('Distance from Low', 'avg_distance_from_low', '$'),
    ]

    print(f"{'METRIC':<20} {'BULLISH':<12} {'BEARISH':<12} {'DIFFERENCE':<12}")
    print("-" * 60)

    for metric, key, unit in comparisons:
        bull_val = bullish_patterns.get(key, 0)
        bear_val = bearish_patterns.get(key, 0)
        diff = bull_val - bear_val

        print(f"{metric:<20} {bull_val:>8.1f}{unit:<3} {bear_val:>8.1f}{unit:<3} {diff:>+8.1f}{unit}")

    # Key insights
    print(f"\n💡 KEY INSIGHTS:")

    if bullish_patterns['above_sma20_avg'] > bearish_patterns['above_sma20_avg']:
        print("• Bullish moves more likely when price is above SMA20")
    else:
        print("• Bearish moves more likely when price is above SMA20")

    if bullish_patterns['london_fix_frequency'] > bearish_patterns['london_fix_frequency']:
        print("• Bullish moves more often occur around London Fix times")
    else:
        print("• Bearish moves more often occur around London Fix times")

    bull_from_high = bullish_patterns['avg_distance_from_high']
    bear_from_high = bearish_patterns['avg_distance_from_high']

    if bull_from_high < bear_from_high:
        print(f"• Bullish moves start closer to recent highs (${bull_from_high:.1f} vs ${bear_from_high:.1f})")
    else:
        print(f"• Bearish moves start closer to recent highs (${bear_from_high:.1f} vs ${bull_from_high:.1f})")

def main():
    """Main analysis function"""

    print("🔍 ANALYZING WHAT PRECEDED XAUUSD'S BIGGEST MOVES")
    print("=" * 60)

    df = load_data()
    if df is None:
        return

    print(f"📊 Loaded {len(df)} XAUUSD hourly candles")

    # Add technical indicators
    df = add_technical_indicators(df)

    # Find significant moves (≥$40)
    significant_moves = find_significant_moves(df, min_move_size=40.0)

    if not significant_moves:
        print("❌ No significant moves found")
        return

    bullish_moves = [m for m in significant_moves if m['direction'] == 'BULLISH']
    bearish_moves = [m for m in significant_moves if m['direction'] == 'BEARISH']

    print(f"📈 Bullish moves ≥$40: {len(bullish_moves)}")
    print(f"📉 Bearish moves ≥$40: {len(bearish_moves)}")

    # Analyze pre-move conditions
    setup_analysis = analyze_pre_move_conditions(df, significant_moves, lookback=12)

    # Find patterns
    bullish_patterns = find_common_patterns(setup_analysis['bullish_setups'], 'BULLISH')
    bearish_patterns = find_common_patterns(setup_analysis['bearish_setups'], 'BEARISH')

    # Compare patterns
    compare_patterns(bullish_patterns, bearish_patterns)

    # Save detailed results
    if bullish_patterns or bearish_patterns:
        results = {
            'bullish_patterns': bullish_patterns,
            'bearish_patterns': bearish_patterns,
            'significant_moves': significant_moves
        }

        # Save move data
        moves_df = pd.DataFrame(significant_moves)
        moves_df.to_csv('data/SIGNIFICANT_MOVES.csv', index=False)

        print(f"\n💾 Results saved to data/SIGNIFICANT_MOVES.csv")

        # Final summary
        print(f"\n🎯 FINAL INSIGHTS FOR TRADING SYSTEM:")
        print("=" * 50)

        if bullish_patterns and bearish_patterns:
            best_bull_hour = bullish_patterns['common_entry_hours'][0] if bullish_patterns['common_entry_hours'] else 'N/A'
            best_bear_hour = bearish_patterns['common_entry_hours'][0] if bearish_patterns['common_entry_hours'] else 'N/A'

            print(f"• Best bullish entry time: {best_bull_hour}:00 NY")
            print(f"• Best bearish entry time: {best_bear_hour}:00 NY")
            print(f"• Bullish moves average ${bullish_patterns['avg_move_size']:.1f} in {bullish_patterns['avg_hours_to_target']:.1f}h")
            print(f"• Bearish moves average ${bearish_patterns['avg_move_size']:.1f} in {bearish_patterns['avg_hours_to_target']:.1f}h")

            print("\n✅ NOW WE KNOW WHAT ACTUALLY WORKS!")
            print("💡 Build system around these REAL patterns, not theory!")

if __name__ == "__main__":
    main()