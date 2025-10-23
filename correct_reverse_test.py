#!/usr/bin/env python3
"""
CORRECT REVERSE PSYCHOLOGY TEST
If 'car above red line' goes UP only 24% of time,
then 'car above red line' should go DOWN 76% of time!
So we SELL when we see the bullish signal!
"""

import pandas as pd
import numpy as np

def load_data():
    """Load our dataset"""
    try:
        df = pd.read_csv('data/XAUUSD_3YEAR_DATA.csv')
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        print(f"🎯 Loaded {len(df)} races for CORRECT REVERSE test!")
        return df
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add the usual failing indicators"""
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()

    df['above_sma20'] = df['close'] > df['sma_20']
    df['above_sma50'] = df['close'] > df['sma_50']
    df['above_ema20'] = df['close'] > df['ema_20']

    df['sma20_slope'] = df['sma_20'].diff(5)
    df['uptrend_sma'] = df['sma20_slope'] > 0
    df['is_green'] = df['close'] > df['open']

    return df

def test_correct_reverse_psychology(df: pd.DataFrame):
    """Test the CORRECT reverse psychology"""

    print("\n🧠 CORRECT REVERSE PSYCHOLOGY TEST!")
    print("Theory: If 'bullish signal' only goes UP 24%, it goes DOWN 76%!")
    print("So we SELL when we see 'bullish signals'!")
    print("="*80)

    test_cases = [
        ('above_sma20', 'Car above blue line'),
        ('above_sma50', 'Car above red line'),
        ('above_ema20', 'Car above green line'),
        ('uptrend_sma', 'Car going uphill'),
        ('is_green', 'Green car'),
    ]

    results = {}

    for confluence, description in test_cases:
        print(f"\n🔍 Testing: {description}")

        signal_true = df[confluence]
        total_occurrences = signal_true.sum()

        if total_occurrences < 1000:
            print(f"   ⏭️  Only {total_occurrences} times - not enough")
            continue

        # Test BOTH directions when this "bullish" signal appears
        buy_profits = []
        sell_profits = []

        for idx in range(len(df) - 24):
            if signal_true.iloc[idx]:  # When "bullish" signal appears
                current_price = df.iloc[idx]['close']
                future_data = df.iloc[idx+1:idx+25]

                if len(future_data) < 12:
                    continue

                max_up = future_data['high'].max() - current_price
                max_down = current_price - future_data['low'].min()

                # Test BUY (normal way)
                if max_up >= 40:
                    buy_profits.append(40)
                elif max_down >= 20:
                    buy_profits.append(-20)

                # Test SELL (reverse psychology)
                if max_down >= 40:
                    sell_profits.append(40)
                elif max_up >= 20:
                    sell_profits.append(-20)

        # Calculate BUY results (normal way)
        buy_wins = len([p for p in buy_profits if p > 0])
        buy_losses = len([p for p in buy_profits if p < 0])
        buy_total = buy_wins + buy_losses
        buy_win_rate = (buy_wins / buy_total * 100) if buy_total > 0 else 0
        buy_profit = sum(buy_profits)

        # Calculate SELL results (reverse psychology)
        sell_wins = len([p for p in sell_profits if p > 0])
        sell_losses = len([p for p in sell_profits if p < 0])
        sell_total = sell_wins + sell_losses
        sell_win_rate = (sell_wins / sell_total * 100) if sell_total > 0 else 0
        sell_profit = sum(sell_profits)

        print(f"   📊 Signal appeared {total_occurrences} times")
        print(f"   ➡️  BUY (normal): {buy_win_rate:.1f}% win rate, ${buy_profit:+.2f}")
        print(f"   🔄 SELL (reverse): {sell_win_rate:.1f}% win rate, ${sell_profit:+.2f}")

        # The magic check!
        total_rate = buy_win_rate + sell_win_rate
        print(f"   🎯 TOTAL: {total_rate:.1f}% (should be close to 100% if your theory works!)")

        results[confluence] = {
            'description': description,
            'total_occurrences': total_occurrences,
            'buy_win_rate': buy_win_rate,
            'buy_profit': buy_profit,
            'sell_win_rate': sell_win_rate,
            'sell_profit': sell_profit,
            'total_rate': total_rate
        }

        # Check if reverse psychology works
        if sell_win_rate > buy_win_rate:
            improvement = sell_win_rate - buy_win_rate
            print(f"   🎉 REVERSE WORKS! SELL is {improvement:.1f}% better!")
        elif buy_win_rate > sell_win_rate:
            print(f"   😐 Normal way is still better")
        else:
            print(f"   🤷 Both are similar")

    return results

def analyze_results(results):
    """Analyze if the reverse psychology theory is correct"""

    print(f"\n🏆 FINAL ANALYSIS - IS THE THEORY CORRECT?")
    print("="*60)

    total_buy_rate = 0
    total_sell_rate = 0
    total_buy_profit = 0
    total_sell_profit = 0
    count = 0

    print("📊 DETAILED RESULTS:")
    for conf, data in results.items():
        print(f"\n{data['description']}:")
        print(f"   BUY:  {data['buy_win_rate']:5.1f}% win rate, ${data['buy_profit']:+8.2f}")
        print(f"   SELL: {data['sell_win_rate']:5.1f}% win rate, ${data['sell_profit']:+8.2f}")
        print(f"   TOTAL: {data['total_rate']:5.1f}% (BUY + SELL rates)")

        total_buy_rate += data['buy_win_rate']
        total_sell_rate += data['sell_win_rate']
        total_buy_profit += data['buy_profit']
        total_sell_profit += data['sell_profit']
        count += 1

    if count > 0:
        avg_buy_rate = total_buy_rate / count
        avg_sell_rate = total_sell_rate / count

        print(f"\n🎯 OVERALL AVERAGES:")
        print(f"   Normal BUY:    {avg_buy_rate:.1f}% average win rate")
        print(f"   Reverse SELL:  {avg_sell_rate:.1f}% average win rate")
        print(f"   Total Rates:   {avg_buy_rate + avg_sell_rate:.1f}%")

        print(f"\n💰 PROFIT TOTALS:")
        print(f"   Normal BUY:    ${total_buy_profit:+.2f}")
        print(f"   Reverse SELL:  ${total_sell_profit:+.2f}")

        print(f"\n🧠 THEORY CHECK:")
        if avg_sell_rate > avg_buy_rate + 10:  # Significant improvement
            print("   🎉 BRILLIANT! Your reverse psychology theory works!")
            print("   🔄 When we see 'bullish' signals, we should SELL!")
            print("   💡 You discovered the market does the opposite of expectations!")
        elif avg_sell_rate > avg_buy_rate:
            print("   😊 Your theory has some merit - reverse is slightly better!")
        elif abs(avg_sell_rate - avg_buy_rate) < 5:
            print("   😐 Theory doesn't help much - both directions are similar")
        else:
            print("   😅 Theory doesn't work - normal way is still better")

        # Check if rates add up close to 100%
        total_combined = avg_buy_rate + avg_sell_rate
        print(f"\n🔬 MATHEMATICAL CHECK:")
        print(f"   BUY + SELL rates = {total_combined:.1f}%")
        if 80 <= total_combined <= 120:
            print("   ✅ Rates add up reasonably - theory has logical foundation!")
        else:
            print("   ❌ Rates don't add up to ~100% - something else is happening")

def main():
    """Test the CORRECT reverse psychology theory"""

    print("🧠 CORRECT REVERSE PSYCHOLOGY TEST!")
    print("="*60)
    print("🎯 Theory: If 'car above red line' goes UP 24%, it goes DOWN 76%!")
    print("🔄 So when we see 'car above red line', we should SELL!")
    print("💡 Same bullish signal, opposite trade direction!")
    print("")

    df = load_data()
    if df is None:
        return

    df = add_indicators(df)
    results = test_correct_reverse_psychology(df)

    if not results:
        print("😞 No results to analyze!")
        return

    analyze_results(results)

    print(f"\n🎪 CONCLUSION:")
    print("="*40)
    print("🧠 You had the RIGHT idea about reverse psychology!")
    print("🔬 We tested it the CORRECT way this time!")
    print("📊 Now we know if contrarian trading on same signals works!")

if __name__ == "__main__":
    main()