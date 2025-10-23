#!/usr/bin/env python3
"""
SINGLE CANDLE PREDICTION ANALYSIS
Testing if we can predict individual hourly candle direction based on confluence factors
Analyzing the predictability of the next 1-hour candle movement
"""

import asyncio
import pandas as pd
from datetime import datetime, timedelta
import pytz
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

async def analyze_single_candle_predictability():
    print("🕯️ SINGLE HOURLY CANDLE PREDICTION ANALYSIS")
    print("=" * 50)
    print("🔍 Testing if confluence can predict next candle direction...")

    try:
        from app.oanda_feed import get_xauusd_candles

        # Get XAUUSD data
        df = await get_xauusd_candles(count=5000)
        if df is None or df.empty:
            print("❌ No XAUUSD data available")
            return

        print(f"✅ Analyzing {len(df)} candles from {df.index[0]} to {df.index[-1]}")

        chicago_tz = pytz.timezone('America/Chicago')

        # Add comprehensive technical indicators
        print("\n🔧 Adding technical indicators for prediction...")

        # Moving averages
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_100'] = df['close'].rolling(window=100).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()

        # MA positions (predictive factors)
        df['above_sma10'] = df['close'] > df['sma_10']
        df['above_sma20'] = df['close'] > df['sma_20']
        df['above_sma50'] = df['close'] > df['sma_50']
        df['above_sma100'] = df['close'] > df['sma_100']
        df['above_ema10'] = df['close'] > df['ema_10']
        df['above_ema20'] = df['close'] > df['ema_20']
        df['above_ema50'] = df['close'] > df['ema_50']

        # Trends and slopes
        df['sma20_slope'] = df['sma_20'].diff(3)
        df['ema20_slope'] = df['ema_20'].diff(3)
        df['price_momentum'] = df['close'].diff(3)
        df['sma20_uptrend'] = df['sma20_slope'] > 0
        df['ema20_uptrend'] = df['ema20_slope'] > 0
        df['price_uptrend'] = df['price_momentum'] > 0

        # Price action patterns
        df['prev_candle_green'] = (df['close'].shift(1) > df['open'].shift(1))
        df['prev_candle_red'] = (df['close'].shift(1) < df['open'].shift(1))
        df['prev_close'] = df['close'].shift(1)
        df['price_above_prev_close'] = df['close'] > df['prev_close']

        # Volatility and ranges
        df['atr_5'] = (df['high'] - df['low']).rolling(window=5).mean()
        df['current_range'] = df['high'] - df['low']
        df['large_range'] = df['current_range'] > df['atr_5'] * 1.5
        df['small_range'] = df['current_range'] < df['atr_5'] * 0.5

        # Support/Resistance
        df['high_10'] = df['high'].rolling(window=10).max()
        df['low_10'] = df['low'].rolling(window=10).min()
        df['near_resistance'] = abs(df['close'] - df['high_10']) < (df['atr_5'] * 0.5)
        df['near_support'] = abs(df['close'] - df['low_10']) < (df['atr_5'] * 0.5)

        # Target variable: Next candle direction
        df['next_candle_bullish'] = (df['close'].shift(-1) > df['close']).astype(int)
        df['next_candle_direction'] = df['next_candle_bullish'].map({1: 'BULLISH', 0: 'BEARISH'})

        # Remove rows with NaN values
        df = df.dropna()

        print(f"✅ Prepared {len(df)} complete candles for analysis")

        # Define prediction factors
        prediction_factors = [
            'above_sma10', 'above_sma20', 'above_sma50', 'above_sma100',
            'above_ema10', 'above_ema20', 'above_ema50',
            'sma20_uptrend', 'ema20_uptrend', 'price_uptrend',
            'prev_candle_green', 'prev_candle_red', 'price_above_prev_close',
            'large_range', 'small_range', 'near_resistance', 'near_support'
        ]

        # Test individual factor predictive power
        print(f"\n🧪 Testing {len(prediction_factors)} individual factors for next candle prediction...")

        factor_results = []

        for factor in prediction_factors:
            # Get predictions when factor is True
            factor_true = df[df[factor] == True]
            factor_false = df[df[factor] == False]

            if len(factor_true) > 50 and len(factor_false) > 50:
                # Bullish prediction rate when factor is True
                bullish_rate_true = factor_true['next_candle_bullish'].mean()
                bullish_rate_false = factor_false['next_candle_bullish'].mean()

                # Overall market bullish rate (baseline)
                overall_bullish_rate = df['next_candle_bullish'].mean()

                # Predictive edge
                edge_true = bullish_rate_true - overall_bullish_rate
                edge_false = bullish_rate_false - overall_bullish_rate

                # Accuracy if we use this factor to predict
                predictions_true = [1 if bullish_rate_true > 0.5 else 0] * len(factor_true)
                predictions_false = [1 if bullish_rate_false > 0.5 else 0] * len(factor_false)

                actual_true = factor_true['next_candle_bullish'].tolist()
                actual_false = factor_false['next_candle_bullish'].tolist()

                if predictions_true and predictions_false:
                    accuracy_true = accuracy_score(actual_true, predictions_true)
                    accuracy_false = accuracy_score(actual_false, predictions_false)
                    overall_accuracy = (accuracy_true * len(factor_true) + accuracy_false * len(factor_false)) / (len(factor_true) + len(factor_false))
                else:
                    overall_accuracy = 0.5

                factor_results.append({
                    'factor': factor,
                    'samples_true': len(factor_true),
                    'samples_false': len(factor_false),
                    'bullish_rate_when_true': bullish_rate_true,
                    'bullish_rate_when_false': bullish_rate_false,
                    'edge_when_true': edge_true,
                    'edge_when_false': edge_false,
                    'overall_accuracy': overall_accuracy,
                    'predictive_power': abs(edge_true) + abs(edge_false)
                })

        # Sort by predictive power
        factor_results.sort(key=lambda x: x['predictive_power'], reverse=True)

        print(f"\n📊 INDIVIDUAL FACTOR PREDICTIVE POWER:")
        print("=" * 80)
        print(f"{'Factor':<25} {'True→Bull%':<11} {'False→Bull%':<12} {'Accuracy%':<10} {'Edge':<8}")
        print("-" * 80)

        baseline_accuracy = max(df['next_candle_bullish'].mean(), 1 - df['next_candle_bullish'].mean())

        for result in factor_results[:12]:  # Top 12 factors
            edge = result['predictive_power']
            print(f"{result['factor']:<25} {result['bullish_rate_when_true']:<11.1%} "
                  f"{result['bullish_rate_when_false']:<12.1%} {result['overall_accuracy']:<10.1%} "
                  f"{edge:<8.3f}")

        # Test combination predictions
        print(f"\n🔬 TESTING FACTOR COMBINATIONS:")
        print("-" * 35)

        # Get top 5 factors
        top_5_factors = [result['factor'] for result in factor_results[:5]]

        combination_results = []

        # Test combinations of 2-5 factors
        from itertools import combinations

        for combo_size in range(2, 6):
            for combo in combinations(top_5_factors, combo_size):
                # Create combined signal
                combined_signal = df[list(combo)].all(axis=1)

                signal_present = df[combined_signal]
                signal_absent = df[~combined_signal]

                if len(signal_present) > 30 and len(signal_absent) > 30:
                    bullish_rate_present = signal_present['next_candle_bullish'].mean()
                    bullish_rate_absent = signal_absent['next_candle_bullish'].mean()

                    # Accuracy if we predict based on this combination
                    if bullish_rate_present > bullish_rate_absent:
                        # Predict bullish when signal present
                        correct_present = sum(signal_present['next_candle_bullish'] == 1)
                        correct_absent = sum(signal_absent['next_candle_bullish'] == 0)
                    else:
                        # Predict bearish when signal present
                        correct_present = sum(signal_present['next_candle_bullish'] == 0)
                        correct_absent = sum(signal_absent['next_candle_bullish'] == 1)

                    total_correct = correct_present + correct_absent
                    total_predictions = len(signal_present) + len(signal_absent)
                    accuracy = total_correct / total_predictions

                    combination_results.append({
                        'combination': ' + '.join(combo),
                        'combo_size': combo_size,
                        'signal_count': len(signal_present),
                        'bullish_when_present': bullish_rate_present,
                        'bullish_when_absent': bullish_rate_absent,
                        'accuracy': accuracy,
                        'edge': abs(bullish_rate_present - bullish_rate_absent)
                    })

        # Sort by accuracy
        combination_results.sort(key=lambda x: x['accuracy'], reverse=True)

        print(f"\n🏆 BEST COMBINATION PREDICTIONS:")
        print("=" * 85)
        print(f"{'Combination':<40} {'Signals':<8} {'Accuracy%':<10} {'Edge':<8}")
        print("-" * 85)

        for result in combination_results[:10]:
            print(f"{result['combination'][:40]:<40} {result['signal_count']:<8} "
                  f"{result['accuracy']:<10.1%} {result['edge']:<8.3f}")

        # Overall market statistics
        total_candles = len(df)
        bullish_candles = sum(df['next_candle_bullish'])
        bearish_candles = total_candles - bullish_candles

        print(f"\n📈 MARKET BASELINE STATISTICS:")
        print("=" * 35)
        print(f"   Total candles analyzed: {total_candles}")
        print(f"   Bullish candles: {bullish_candles} ({bullish_candles/total_candles:.1%})")
        print(f"   Bearish candles: {bearish_candles} ({bearish_candles/total_candles:.1%})")
        print(f"   Baseline accuracy (random): {baseline_accuracy:.1%}")

        # Best achievable accuracy
        if combination_results:
            best_accuracy = combination_results[0]['accuracy']
            improvement = best_accuracy - baseline_accuracy

            print(f"\n🎯 PREDICTABILITY ASSESSMENT:")
            print("=" * 35)
            print(f"   Best combination accuracy: {best_accuracy:.1%}")
            print(f"   Improvement over random: +{improvement:.1%}")
            print(f"   Best edge found: {combination_results[0]['edge']:.3f}")

            if best_accuracy > 0.6:
                print(f"   ✅ STRONG predictive power")
            elif best_accuracy > 0.55:
                print(f"   ⚠️  MODERATE predictive power")
            elif best_accuracy > 0.52:
                print(f"   📊 WEAK predictive power")
            else:
                print(f"   ❌ NO meaningful predictive power")

        # Practical implications
        print(f"\n💡 PRACTICAL IMPLICATIONS:")
        print("=" * 30)

        if combination_results and combination_results[0]['accuracy'] > 0.55:
            best_combo = combination_results[0]
            print(f"   ✅ ANSWER: Limited prediction is possible")
            print(f"   📊 Best approach: {best_combo['combination']}")
            print(f"   🎯 Expected accuracy: {best_combo['accuracy']:.1%}")
            print(f"   📈 Signals per period: {best_combo['signal_count']} out of {total_candles}")

            signal_frequency = best_combo['signal_count'] / total_candles
            print(f"   ⏰ Signal frequency: {signal_frequency:.1%} of candles")

            if signal_frequency < 0.1:
                print(f"   ⚠️  Very rare signals - might miss many opportunities")
            elif signal_frequency > 0.5:
                print(f"   ⚠️  Very frequent signals - might be noise")
            else:
                print(f"   ✅ Good signal frequency for practical use")

        else:
            print(f"   ❌ ANSWER: Single candle direction is NOT reliably predictable")
            print(f"   🎲 Market is too random at 1-hour timeframe")
            print(f"   💡 Focus on multi-candle patterns instead")

        print(f"\n🚨 IMPORTANT LIMITATIONS:")
        print(f"   • Analysis based on only {(df.index[-1] - df.index[0]).days} days of data")
        print(f"   • Market conditions change - past ≠ future")
        print(f"   • Single candle prediction is inherently difficult")
        print(f"   • Better to focus on multi-candle confluence patterns")

    except Exception as e:
        print(f"❌ Single candle prediction analysis error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(analyze_single_candle_predictability())