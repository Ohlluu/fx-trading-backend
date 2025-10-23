#!/usr/bin/env python3
"""
Real XAUUSD Analysis - Finding Actual Psychological Levels
Based on 9.2 months of historical hourly data
"""

import pandas as pd
import numpy as np

def analyze_real_psychological_levels():
    """Analyze real XAUUSD data to find actual psychological levels"""

    # Load historical data
    print('=== ANALYZING REAL XAUUSD HISTORICAL DATA ===')
    df = pd.read_csv('xauusd_historical.csv', index_col=0, parse_dates=True)
    print(f'Analyzing {len(df)} hourly candles')
    print(f'Price range: ${df["low"].min():.2f} - ${df["high"].max():.2f}')
    print()

    # Generate potential psychological levels every $25
    min_price = int(df['low'].min() / 25) * 25
    max_price = int(df['high'].max() / 25 + 1) * 25

    psychological_levels = list(range(min_price, max_price + 25, 25))
    print(f'Testing {len(psychological_levels)} potential levels from ${min_price} to ${max_price}')

    # Test each level for market reactions
    level_data = []
    for level in psychological_levels:
        # Find candles that tested this level (within $5)
        tolerance = 5
        touches = df[(df['low'] <= level + tolerance) & (df['high'] >= level - tolerance)]

        if len(touches) >= 3:  # Need minimum 3 tests
            bounces = 0
            breaks = 0

            for i, (idx, candle) in enumerate(touches.iterrows()):
                # Skip if this is the last candle
                try:
                    next_idx = df.index.get_loc(idx) + 1
                    if next_idx < len(df):
                        next_candle = df.iloc[next_idx]

                        # Determine reaction type
                        if candle['low'] <= level <= candle['high']:
                            if level <= candle['close']:  # Support test
                                if next_candle['close'] > level + 3:
                                    bounces += 1
                                elif next_candle['close'] < level - 3:
                                    breaks += 1
                            else:  # Resistance test
                                if next_candle['close'] < level - 3:
                                    bounces += 1
                                elif next_candle['close'] > level + 3:
                                    breaks += 1
                except:
                    continue

            total_reactions = bounces + breaks
            if total_reactions > 0:
                bounce_rate = bounces / total_reactions

                level_data.append({
                    'level': level,
                    'touches': len(touches),
                    'bounces': bounces,
                    'breaks': breaks,
                    'bounce_rate': bounce_rate,
                    'significance': 'MAJOR' if level % 50 == 0 else 'MINOR'
                })

    # Sort by reliability
    level_data.sort(key=lambda x: (x['bounce_rate'], x['touches']), reverse=True)

    print()
    print('=== TOP 15 REAL PSYCHOLOGICAL LEVELS ===')
    print('Level     | Type  | Touches | Bounces | Breaks | Bounce Rate | Strength')
    print('----------|-------|---------|---------|--------|-------------|----------')

    for level in level_data[:15]:
        if level['bounce_rate'] >= 0.7:
            strength = 'STRONG'
        elif level['bounce_rate'] >= 0.5:
            strength = 'MODERATE'
        else:
            strength = 'WEAK'

        print(f"${level['level']:<8} | {level['significance']:<5} | "
              f"{level['touches']:<7} | {level['bounces']:<7} | {level['breaks']:<6} | "
              f"{level['bounce_rate']:<11.1%} | {strength}")

    print()
    print('=== KEY INSIGHTS ===')

    # Find truly strong levels
    strong_levels = [l for l in level_data if l['bounce_rate'] >= 0.7 and l['touches'] >= 5]
    print(f'Truly strong levels (≥70% bounce rate, ≥5 touches): {len(strong_levels)}')

    if strong_levels:
        print('Most reliable levels for trading:')
        for i, level in enumerate(strong_levels[:5], 1):
            print(f'  {i}. ${level["level"]} ({level["significance"]}): '
                  f'{level["bounce_rate"]:.1%} bounce rate over {level["touches"]} touches')

    print()
    print('=== CURRENT MARKET CONTEXT ===')
    current_price = df['close'].iloc[-1]
    print(f'Current Gold price: ${current_price:.2f}')

    # Find nearest strong levels
    nearby_strong = []
    for level in strong_levels:
        distance = abs(level['level'] - current_price)
        if distance <= 100:  # Within 100 points ($10)
            nearby_strong.append({
                'level': level['level'],
                'distance': distance,
                'bounce_rate': level['bounce_rate'],
                'touches': level['touches'],
                'type': level['significance']
            })

    if nearby_strong:
        nearby_strong.sort(key=lambda x: x['distance'])
        print('Nearby strong levels:')
        for level in nearby_strong:
            direction = 'above' if current_price > level['level'] else 'below'
            print(f'  ${level["level"]} ({level["type"]}): '
                  f'{level["distance"]:.0f} points away ({direction}), '
                  f'{level["bounce_rate"]:.1%} bounce rate')
    else:
        print('No strong levels within 100 points of current price')

    return level_data, strong_levels, nearby_strong

if __name__ == '__main__':
    analyze_real_psychological_levels()