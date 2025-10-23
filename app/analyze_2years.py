#!/usr/bin/env python3
"""
Analyze 2+ years of XAUUSD data to find what professional patterns actually work
"""

import pandas as pd
import numpy as np
from datetime import datetime
import pytz

def load_and_analyze_data():
    print("=== ANALYZING 2+ YEARS XAUUSD PATTERNS ===")

    # Load the data
    df = pd.read_csv("data/XAUUSD_FULL_2YEARS.csv")
    df['time'] = pd.to_datetime(df['time'])

    print(f"Loaded {len(df)} candles from {df['time'].min().date()} to {df['time'].max().date()}")

    # Basic stats
    print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    print(f"Current vs start: ${df.iloc[-1]['close']:.2f} vs ${df.iloc[0]['close']:.2f}")

    # Convert to NY time
    chicago_tz = pytz.timezone('America/Chicago')
    df['ny_time'] = df['time'].dt.tz_convert(chicago_tz)
    df['ny_hour'] = df['ny_time'].dt.hour
    df['ny_dow'] = df['ny_time'].dt.dayofweek  # 0=Monday
    df['volatility'] = df['high'] - df['low']

    return df

def analyze_sessions(df):
    print("\n=== NY TIME SESSION ANALYSIS ===")

    # Define sessions (NY time)
    sessions = {
        'Asian_Night': list(range(19, 24)) + list(range(0, 3)),  # 7PM-3AM
        'London_Fix_AM': [5, 6],  # 5-6AM (London 10:30)
        'London_NY_Overlap': list(range(8, 12)),  # 8AM-12PM
        'London_Fix_PM': [10, 11],  # 10-11AM (London 15:00)
        'NY_Afternoon': list(range(14, 17)),  # 2-5PM
    }

    session_results = {}

    for session_name, hours in sessions.items():
        session_data = df[df['ny_hour'].isin(hours)]

        if not session_data.empty:
            avg_vol = session_data['volatility'].mean()
            big_moves = len(session_data[session_data['volatility'] > 20])
            total = len(session_data)
            pct_big = (big_moves / total * 100) if total > 0 else 0

            session_results[session_name] = {
                'avg_volatility': avg_vol,
                'big_moves_pct': pct_big,
                'total_candles': total
            }

            print(f"{session_name:18}: ${avg_vol:5.2f} avg range, {pct_big:4.1f}% big moves (>$20), {total:5} candles")

    return session_results

def analyze_round_numbers(df):
    print("\n=== ROUND NUMBER ANALYSIS ===")

    # Generate round numbers based on actual price range
    min_price = int(df['low'].min() / 50) * 50
    max_price = int(df['high'].max() / 50) * 50 + 50

    # Major round numbers every $50
    round_numbers = list(range(min_price, max_price + 1, 50))

    level_analysis = []

    for level in round_numbers:
        # Find touches within 15 points of level
        tolerance = 15
        touches = df[(df['low'] <= level + tolerance) & (df['high'] >= level - tolerance)]

        if len(touches) >= 5:  # Only analyze levels with 5+ touches

            # Analyze bounces vs breaks
            bounces = 0
            breaks = 0

            for idx, candle in touches.iterrows():
                if candle['low'] <= level <= candle['high']:  # Price touched level
                    # Check next 2 candles for bounce (stayed within 30 points)
                    try:
                        next_idx = df.index[df.index > idx][:2]
                        if len(next_idx) >= 2:
                            next_candles = df.loc[next_idx]

                            # Define bounce as not breaking 30 points away
                            max_distance = max(abs(candle['close'] - level) for candle in next_candles.itertuples())

                            if max_distance <= 30:
                                bounces += 1
                            else:
                                breaks += 1
                    except:
                        continue

            total_reactions = bounces + breaks
            if total_reactions >= 3:  # Need minimum reactions to be meaningful
                bounce_rate = bounces / total_reactions

                level_analysis.append({
                    'level': level,
                    'touches': len(touches),
                    'bounces': bounces,
                    'breaks': breaks,
                    'bounce_rate': bounce_rate,
                    'strength': 'MAJOR' if level % 100 == 0 else 'MINOR'
                })

    # Sort by bounce rate and touches
    level_analysis.sort(key=lambda x: (x['bounce_rate'], x['touches']), reverse=True)

    print("TOP RESPECTED LEVELS (Min 5 touches, 3+ reactions):")
    for i, level in enumerate(level_analysis[:20]):  # Top 20
        strength_mark = "★" if level['strength'] == 'MAJOR' else "•"
        print(f"{strength_mark} ${level['level']:4}: {level['bounce_rate']:5.1%} bounce rate ({level['bounces']}/{level['bounces'] + level['breaks']}) over {level['touches']} touches")

        if i == 9:  # Show top 10, then separator
            print("  " + "-" * 50)

    return level_analysis

def analyze_day_patterns(df):
    print("\n=== DAY OF WEEK ANALYSIS ===")

    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    day_results = {}

    for dow in range(7):
        dow_data = df[df['ny_dow'] == dow]
        if not dow_data.empty:
            avg_range = dow_data['volatility'].mean()
            big_moves = len(dow_data[dow_data['volatility'] > 25])
            total = len(dow_data)
            pct = (big_moves / total * 100) if total > 0 else 0

            day_results[dow_names[dow]] = {
                'avg_volatility': avg_range,
                'big_moves_pct': pct,
                'total_sessions': total
            }

            print(f"{dow_names[dow]:9}: ${avg_range:5.2f} avg range, {pct:4.1f}% big moves (>$25), {total:4} sessions")

    return day_results

def find_best_setups(df, level_analysis, session_results):
    print("\n=== BEST PROFESSIONAL SETUPS IDENTIFIED ===")

    # Get top levels
    top_levels = [l for l in level_analysis[:10] if l['bounce_rate'] >= 0.7]

    # Get best sessions
    best_sessions = sorted(session_results.items(),
                          key=lambda x: x[1]['big_moves_pct'], reverse=True)[:3]

    print("🎯 HIGHEST PROBABILITY CONFLUENCE SETUPS:")
    print(f"1. TOP PSYCHOLOGICAL LEVELS: {len(top_levels)} levels with 70%+ bounce rates")
    for level in top_levels[:5]:
        print(f"   • ${level['level']} ({level['strength']}): {level['bounce_rate']:.1%} success rate")

    print(f"\n2. BEST TRADING SESSIONS:")
    for session_name, stats in best_sessions:
        session_display = session_name.replace('_', ' ').title()
        print(f"   • {session_display}: {stats['big_moves_pct']:.1f}% big moves, ${stats['avg_volatility']:.2f} avg range")

    print(f"\n3. CONFLUENCE COMBINATION:")
    print(f"   • Trade ONLY when price is within 15 points of a 70%+ bounce level")
    print(f"   • During {best_sessions[0][0].replace('_', ' ')} session")
    print(f"   • Expected success rate: 70%+ (based on historical analysis)")
    print(f"   • Average volatility: ${best_sessions[0][1]['avg_volatility']:.2f} (sufficient for profitable R:R)")

if __name__ == "__main__":
    # Run the complete analysis
    df = load_and_analyze_data()
    session_results = analyze_sessions(df)
    level_analysis = analyze_round_numbers(df)
    day_results = analyze_day_patterns(df)
    find_best_setups(df, level_analysis, session_results)

    print(f"\n✅ Analysis complete - {len(df)} candles over {(df['time'].max() - df['time'].min()).days} days")
    print("💡 Ready to build professional trading system based on these findings!")