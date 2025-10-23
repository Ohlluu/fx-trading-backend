#!/usr/bin/env python3
"""
REVERSE PSYCHOLOGY TEST - Trade the OPPOSITE!
If confluences fail 76% of the time, can we win 76% by doing the opposite?
"""

import pandas as pd
import numpy as np
from typing import Dict, List

def load_data():
    """Load our dataset"""
    try:
        df = pd.read_csv('data/XAUUSD_3YEAR_DATA.csv')
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        print(f"🎯 Loaded {len(df)} races for REVERSE PSYCHOLOGY test!")
        return df
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the indicators that usually fail"""

    print("🔧 Adding the 'failing' indicators...")

    # The usual suspects that fail most of the time
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()

    # Normal confluences (that usually fail)
    df['above_sma20'] = df['close'] > df['sma_20']
    df['above_sma50'] = df['close'] > df['sma_50']
    df['above_ema20'] = df['close'] > df['ema_20']

    df['sma20_slope'] = df['sma_20'].diff(5)
    df['uptrend_sma'] = df['sma20_slope'] > 0

    df['is_green'] = df['close'] > df['open']

    # REVERSE confluences (do the opposite!)
    df['below_sma20'] = df['close'] < df['sma_20']      # OPPOSITE!
    df['below_sma50'] = df['close'] < df['sma_50']      # OPPOSITE!
    df['below_ema20'] = df['close'] < df['ema_20']      # OPPOSITE!
    df['downtrend_sma'] = df['sma20_slope'] < 0         # OPPOSITE!
    df['is_red'] = df['close'] < df['open']             # OPPOSITE!

    print("✅ Added normal AND reverse indicators!")
    return df

def test_reverse_psychology(df: pd.DataFrame):
    """Test if doing the OPPOSITE works better"""

    print("\n🎭 REVERSE PSYCHOLOGY TEST!")
    print("If normal confluences fail 76% of time, can we win 76% by doing opposite?")
    print("="*80)

    # Test normal vs reverse confluences
    test_cases = [
        # Normal confluences (that usually fail)
        ('above_sma20', 'BUY when above blue line', 'BUY'),
        ('above_sma50', 'BUY when above red line', 'BUY'),
        ('uptrend_sma', 'BUY when going uphill', 'BUY'),
        ('is_green', 'BUY after green candle', 'BUY'),

        # REVERSE confluences (do opposite!)
        ('below_sma20', 'SELL when below blue line', 'SELL'),
        ('below_sma50', 'SELL when below red line', 'SELL'),
        ('downtrend_sma', 'SELL when going downhill', 'SELL'),
        ('is_red', 'SELL after red candle', 'SELL'),
    ]

    results = {}

    for confluence, description, direction in test_cases:
        print(f"\n🔍 Testing: {description}")

        # Find when this confluence is true
        signal_true = df[confluence]
        total_occurrences = signal_true.sum()

        if total_occurrences < 500:
            print(f"   ⏭️  Only {total_occurrences} times - not enough")
            continue

        # Test what happens next
        profits = []
        for idx in range(len(df) - 24):
            if signal_true.iloc[idx]:
                current_price = df.iloc[idx]['close']
                future_data = df.iloc[idx+1:idx+25]

                if len(future_data) < 12:
                    continue

                # Simulate trade based on direction
                if direction == 'BUY':
                    # Look for upward moves
                    max_up = future_data['high'].max() - current_price
                    max_down = current_price - future_data['low'].min()

                    if max_up >= 40:  # Win!
                        profits.append(40)
                    elif max_down >= 20:  # Loss
                        profits.append(-20)

                else:  # SELL
                    # Look for downward moves
                    max_down = current_price - future_data['low'].min()
                    max_up = future_data['high'].max() - current_price

                    if max_down >= 40:  # Win!
                        profits.append(40)
                    elif max_up >= 20:  # Loss
                        profits.append(-20)

        if len(profits) < 100:
            print(f"   ⏭️  Not enough valid trades")
            continue

        # Calculate results
        wins = len([p for p in profits if p > 0])
        losses = len([p for p in profits if p < 0])
        total_trades = wins + losses

        if total_trades == 0:
            continue

        win_rate = (wins / total_trades) * 100
        total_profit = sum(profits)

        results[confluence] = {
            'description': description,
            'direction': direction,
            'total_occurrences': total_occurrences,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit': total_profit
        }

        is_reverse = confluence.startswith(('below', 'downtrend', 'is_red'))
        reverse_emoji = "🔄" if is_reverse else "➡️"

        print(f"   {reverse_emoji} Happened {total_occurrences} times")
        print(f"   🎯 Win Rate: {win_rate:.1f}% ({wins}/{total_trades})")
        print(f"   💰 Profit: ${total_profit:+.2f}")

    return results

def analyze_reverse_results(results: Dict):
    """Compare normal vs reverse psychology results"""

    print(f"\n🏆 REVERSE PSYCHOLOGY ANALYSIS")
    print("="*60)

    # Separate normal and reverse
    normal_results = {k: v for k, v in results.items()
                     if not k.startswith(('below', 'downtrend', 'is_red'))}
    reverse_results = {k: v for k, v in results.items()
                      if k.startswith(('below', 'downtrend', 'is_red'))}

    print("➡️  NORMAL CONFLUENCES (Do what everyone does):")
    for conf, data in normal_results.items():
        print(f"   {data['description']}: {data['win_rate']:.1f}% win rate, ${data['total_profit']:+.2f}")

    print(f"\n🔄 REVERSE CONFLUENCES (Do the OPPOSITE!):")
    for conf, data in reverse_results.items():
        print(f"   {data['description']}: {data['win_rate']:.1f}% win rate, ${data['total_profit']:+.2f}")

    # Calculate averages
    if normal_results:
        avg_normal_win_rate = sum(r['win_rate'] for r in normal_results.values()) / len(normal_results)
        total_normal_profit = sum(r['total_profit'] for r in normal_results.values())
    else:
        avg_normal_win_rate = 0
        total_normal_profit = 0

    if reverse_results:
        avg_reverse_win_rate = sum(r['win_rate'] for r in reverse_results.values()) / len(reverse_results)
        total_reverse_profit = sum(r['total_profit'] for r in reverse_results.values())
    else:
        avg_reverse_win_rate = 0
        total_reverse_profit = 0

    print(f"\n📊 SUMMARY:")
    print(f"   Normal Average: {avg_normal_win_rate:.1f}% win rate, ${total_normal_profit:+.2f} total")
    print(f"   Reverse Average: {avg_reverse_win_rate:.1f}% win rate, ${total_reverse_profit:+.2f} total")

    # The big question!
    print(f"\n🤯 THE BIG QUESTION:")
    if avg_reverse_win_rate > avg_normal_win_rate:
        improvement = avg_reverse_win_rate - avg_normal_win_rate
        print(f"   🎉 YES! Reverse psychology works!")
        print(f"   🔄 Doing OPPOSITE is {improvement:.1f} percentage points better!")
        print(f"   💡 When indicators say BUY, maybe we should SELL!")
    elif abs(avg_reverse_win_rate - avg_normal_win_rate) < 5:
        print(f"   😐 Meh. Reverse psychology doesn't help much.")
        print(f"   📊 Both directions are similarly bad/good.")
        print(f"   💭 Market might be truly random...")
    else:
        print(f"   ❌ Reverse psychology makes it WORSE!")
        print(f"   😅 Sometimes the obvious thing is still better.")

def main():
    """Test the reverse psychology theory!"""

    print("🤯 REVERSE PSYCHOLOGY TEST - TRADE THE OPPOSITE!")
    print("="*60)
    print("🎭 Theory: If confluences fail 76% of time, can we win 76% by doing opposite?")
    print("🚗 Like: If 'car above red line' fails, maybe 'car BELOW red line' wins!")
    print("")

    df = load_data()
    if df is None:
        return

    df = add_indicators(df)
    results = test_reverse_psychology(df)

    if not results:
        print("😞 No results to analyze!")
        return

    analyze_reverse_results(results)

    print(f"\n🎯 FINAL VERDICT:")
    print("="*40)
    print("🧠 You had a brilliant psychological insight!")
    print("🔬 We tested it scientifically with real data!")
    print("📊 Now we know if contrarian trading actually works!")

if __name__ == "__main__":
    main()