#!/usr/bin/env python3
"""
DOUBLE CONFLUENCE TEST - What happens when we need 2 confluences together?
Like you're 5: What if we need TWO signs before the toy car goes fast?
"""

import pandas as pd
import numpy as np
from typing import Dict, List

def load_data():
    """Load our big dataset"""
    try:
        df = pd.read_csv('data/XAUUSD_3YEAR_DATA.csv')
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        print(f"🚗 Loaded {len(df)} toy car races!")
        return df
    except Exception as e:
        print(f"❌ Oops! Can't find the toy car data: {e}")
        return None

def add_simple_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the most important signs we found"""

    print("🔧 Adding the magic signs to predict when car goes FAST...")

    # The magic lines (moving averages)
    df['sma_20'] = df['close'].rolling(window=20).mean()  # Blue line
    df['sma_50'] = df['close'].rolling(window=50).mean()  # Red line
    df['ema_20'] = df['close'].ewm(span=20).mean()       # Green line

    # Is the car above the magic lines?
    df['above_sma20'] = df['close'] > df['sma_20']       # Above blue line
    df['above_sma50'] = df['close'] > df['sma_50']       # Above red line
    df['above_ema20'] = df['close'] > df['ema_20']       # Above green line

    # Is the car going uphill or downhill?
    df['sma20_slope'] = df['sma_20'].diff(5)
    df['uptrend_sma'] = df['sma20_slope'] > 0            # Going uphill

    # Is it a green or red toy car?
    df['is_green'] = df['close'] > df['open']            # Green car

    print("✅ Added all the magic signs!")
    return df

def test_double_confluences(df: pd.DataFrame) -> Dict:
    """Test what happens when we need TWO signs together"""

    print("\n🎪 TESTING DOUBLE CONFLUENCES!")
    print("What happens when we need TWO magic signs together?")
    print("="*60)

    # The best single confluences we found
    top_confluences = [
        ('above_sma50', 'Car above RED line'),
        ('above_sma20', 'Car above BLUE line'),
        ('above_ema20', 'Car above GREEN line'),
        ('uptrend_sma', 'Car going UPHILL'),
        ('is_green', 'GREEN toy car')
    ]

    results = {}

    # Test every pair of confluences together
    for i, (conf1, desc1) in enumerate(top_confluences):
        for j, (conf2, desc2) in enumerate(top_confluences):
            if i >= j:  # Skip duplicates
                continue

            combo_name = f"{conf1} + {conf2}"
            combo_desc = f"{desc1} AND {desc2}"

            print(f"\n🔍 Testing: {combo_desc}")

            # Find when BOTH signs happen together
            both_true = df[conf1] & df[conf2]
            total_occurrences = both_true.sum()

            if total_occurrences < 100:  # Need enough examples
                print(f"   ⏭️  Only {total_occurrences} times - not enough to test")
                continue

            # Simulate what happens next (simple version)
            profits = []

            for idx in range(len(df) - 24):  # Look 24 hours ahead
                if both_true.iloc[idx]:  # Both signs present
                    current_price = df.iloc[idx]['close']
                    future_data = df.iloc[idx+1:idx+25]  # Next 24 hours

                    if len(future_data) < 12:
                        continue

                    # Check if price goes up $40 (win) or down $20 (loss)
                    max_profit = future_data['high'].max() - current_price
                    max_loss = current_price - future_data['low'].min()

                    if max_profit >= 40:  # Car went FAST up!
                        profits.append(40)
                    elif max_loss >= 20:  # Car crashed down
                        profits.append(-20)
                    else:
                        profits.append(0)  # Nothing exciting

            if len(profits) < 50:
                print(f"   ⏭️  Not enough races to test")
                continue

            # Calculate results
            wins = len([p for p in profits if p > 0])
            losses = len([p for p in profits if p < 0])
            total_trades = wins + losses

            if total_trades == 0:
                continue

            win_rate = (wins / total_trades) * 100
            total_profit = sum(profits)
            avg_profit = total_profit / len(profits) if profits else 0

            results[combo_name] = {
                'description': combo_desc,
                'total_occurrences': total_occurrences,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'avg_profit': avg_profit
            }

            print(f"   📊 Happened {total_occurrences} times")
            print(f"   🎯 Win Rate: {win_rate:.1f}% ({wins}/{total_trades})")
            print(f"   💰 Profit: ${total_profit:+.2f}")

    return results

def compare_single_vs_double(df: pd.DataFrame, double_results: Dict):
    """Compare single confluences vs double confluences"""

    print(f"\n🏆 SINGLE vs DOUBLE CONFLUENCE COMPARISON")
    print("="*60)

    # Test single confluences for comparison
    single_confluences = [
        ('above_sma50', 'RED line only'),
        ('above_sma20', 'BLUE line only'),
        ('uptrend_sma', 'Uphill only')
    ]

    print("📊 SINGLE CONFLUENCES:")
    for conf, desc in single_confluences:
        conf_true = df[conf]
        total_occurrences = conf_true.sum()

        # Quick test (simplified)
        profits = []
        for idx in range(len(df) - 24):
            if conf_true.iloc[idx]:
                current_price = df.iloc[idx]['close']
                future_data = df.iloc[idx+1:idx+25]

                if len(future_data) < 12:
                    continue

                max_profit = future_data['high'].max() - current_price
                max_loss = current_price - future_data['low'].min()

                if max_profit >= 40:
                    profits.append(40)
                elif max_loss >= 20:
                    profits.append(-20)

        wins = len([p for p in profits if p > 0])
        losses = len([p for p in profits if p < 0])
        total_trades = wins + losses

        if total_trades > 0:
            win_rate = (wins / total_trades) * 100
            total_profit = sum(profits)
            print(f"   {desc}: {win_rate:.1f}% win rate, ${total_profit:+.2f} profit")

    print(f"\n📊 DOUBLE CONFLUENCES (Best ones):")

    # Sort by win rate
    sorted_doubles = sorted(double_results.items(),
                          key=lambda x: x[1]['win_rate'], reverse=True)

    for combo_name, data in sorted_doubles[:5]:  # Top 5
        print(f"   {data['description']}: {data['win_rate']:.1f}% win rate, ${data['total_profit']:+.2f} profit")

def main():
    """Main function - explain like you're 5!"""

    print("🎪 DOUBLE CONFLUENCE TEST - FOR 5-YEAR-OLDS!")
    print("="*60)
    print("🚗 We're testing if we need TWO signs for the toy car to go fast!")
    print("Like: car must be above RED line AND going uphill")
    print("")

    # Load data
    df = load_data()
    if df is None:
        return

    # Add indicators
    df = add_simple_indicators(df)

    # Test double confluences
    double_results = test_double_confluences(df)

    if not double_results:
        print("😞 No good double confluences found!")
        return

    # Compare single vs double
    compare_single_vs_double(df, double_results)

    # Final explanation for 5-year-old
    print(f"\n🎯 WHAT THIS MEANS (Like You're 5):")
    print("="*40)
    print("🚗 Single sign: 'Car above red line' = sometimes works")
    print("🚗🚗 Double sign: 'Car above red line AND going uphill' = ???")
    print("")

    if double_results:
        best_combo = max(double_results.items(), key=lambda x: x[1]['win_rate'])
        best_name, best_data = best_combo

        print(f"🏆 BEST DOUBLE COMBO:")
        print(f"   {best_data['description']}")
        print(f"   Works {best_data['win_rate']:.1f}% of the time!")
        print(f"   Made ${best_data['total_profit']:+.2f} toy money!")

        if best_data['win_rate'] > 60:
            print("🎉 WOW! Double signs work better than single signs!")
        elif best_data['win_rate'] > 50:
            print("😊 Double signs work a little better!")
        else:
            print("😐 Double signs don't help much...")

    print(f"\n💡 LESSON: Sometimes you need TWO clues to solve the mystery!")

if __name__ == "__main__":
    main()