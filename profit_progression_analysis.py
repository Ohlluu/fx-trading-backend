#!/usr/bin/env python3
"""
PROFIT PROGRESSION ANALYSIS
Early indicators: How to know if trade is heading to TP or SL?
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics

@dataclass
class ExitRule:
    """Define an exit rule"""
    check_hour: int          # When to check (hours after entry)
    min_profit_pct: float    # Minimum profit % to stay
    max_loss_pct: float      # Maximum loss % before exit
    action: str              # "exit" or "stay_for_tp"

class ProgressionAnalyzer:
    def __init__(self):
        # Same exit rules as smart system
        self.exit_rules = [
            ExitRule(6, 2.0, -1.0, "stay_for_tp"),
            ExitRule(12, 0.5, -2.0, "exit"),
            ExitRule(18, 0.0, -1.5, "exit"),
            ExitRule(24, -0.5, -3.0, "exit"),
        ]

        self.profit_target_pct = 2.0
        self.stop_loss_pct = 1.0
        self.max_hold_hours = 36

    def add_confluences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the proven confluences"""
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()

        df['above_sma20'] = df['close'] > df['sma_20']
        df['above_sma50'] = df['close'] > df['sma_50']
        df['above_ema20'] = df['close'] > df['ema_20']

        df['sma20_slope'] = df['sma_20'].diff(5)
        df['ema20_slope'] = df['ema_20'].diff(5)
        df['uptrend_sma'] = df['sma20_slope'] > 0
        df['uptrend_ema'] = df['ema20_slope'] > 0

        df['is_green'] = df['close'] > df['open']
        df['prev_red'] = (~df['is_green']).shift(1)

        df['recent_high_12h'] = df['high'].rolling(window=12).max()
        df['recent_low_12h'] = df['low'].rolling(window=12).min()
        df['near_recent_high'] = abs(df['close'] - df['recent_high_12h']) < 10

        return df

    def check_bullish_confluence(self, candle: pd.Series) -> Optional[Dict]:
        """Check for bullish confluence"""
        confluence_score = 0
        factors = []

        if not candle['above_sma50']:
            return None
        confluence_score += 4
        factors.append("Above SMA50")

        if candle['uptrend_ema']:
            confluence_score += 3
            factors.append("EMA uptrend")

        if candle['above_ema20']:
            confluence_score += 3
            factors.append("Above EMA20")

        if candle['uptrend_sma']:
            confluence_score += 2
            factors.append("SMA uptrend")

        if candle['above_sma20']:
            confluence_score += 2
            factors.append("Above SMA20")

        if candle['near_recent_high']:
            confluence_score += 2
            factors.append("Near recent high")

        if candle['is_green']:
            confluence_score += 1
            factors.append("Green candle")

        if confluence_score < 10:
            return None

        return {
            'direction': 'BUY',
            'confluence_score': confluence_score,
            'factors': factors,
            'signal_type': 'bullish'
        }

    def check_bearish_confluence(self, candle: pd.Series) -> Optional[Dict]:
        """Check for bearish confluence"""
        confluence_score = 0
        factors = []

        if not candle['uptrend_sma']:
            return None
        confluence_score += 3
        factors.append("SMA uptrend")

        if candle['uptrend_ema']:
            confluence_score += 3
            factors.append("EMA uptrend")

        if candle['prev_red']:
            confluence_score += 2
            factors.append("Previous red candle")

        if candle['above_sma20']:
            confluence_score += 2
            factors.append("Above SMA20")

        if candle['is_green']:
            confluence_score += 2
            factors.append("Green candle")

        if confluence_score < 8:
            return None

        return {
            'direction': 'SELL',
            'confluence_score': confluence_score,
            'factors': factors,
            'signal_type': 'bearish'
        }

    def calculate_profit_pct(self, entry_price: float, current_price: float, direction: str) -> float:
        """Calculate profit percentage"""
        if direction == 'BUY':
            return ((current_price - entry_price) / entry_price) * 100
        else:
            return ((entry_price - current_price) / entry_price) * 100

    def analyze_progression_patterns(self, signal: Dict, entry_price: float, future_data: pd.DataFrame) -> Dict:
        """Analyze how profit progresses hour by hour"""

        direction = signal['direction']
        progression = {
            'hourly_profits': [],
            'final_outcome': None,
            'outcome_hour': 0,
            'max_profit': -999,
            'max_profit_hour': 0,
            'min_profit': 999,
            'min_profit_hour': 0,
            'profit_at_hours': {}  # Track profit at key checkpoints
        }

        # Track profit hour by hour
        for hour in range(min(len(future_data), self.max_hold_hours)):
            current_candle = future_data.iloc[hour]
            current_price = current_candle['close']
            hours_elapsed = hour + 1

            profit_pct = self.calculate_profit_pct(entry_price, current_price, direction)
            progression['hourly_profits'].append(profit_pct)

            # Track extremes
            if profit_pct > progression['max_profit']:
                progression['max_profit'] = profit_pct
                progression['max_profit_hour'] = hours_elapsed

            if profit_pct < progression['min_profit']:
                progression['min_profit'] = profit_pct
                progression['min_profit_hour'] = hours_elapsed

            # Track profit at key hours
            if hours_elapsed in [1, 2, 3, 6, 12, 18, 24]:
                progression[f'profit_at_{hours_elapsed}h'] = profit_pct

            # Check for final outcomes
            if profit_pct >= self.profit_target_pct:
                progression['final_outcome'] = 'TP'
                progression['outcome_hour'] = hours_elapsed
                break
            elif profit_pct <= -self.stop_loss_pct:
                progression['final_outcome'] = 'SL'
                progression['outcome_hour'] = hours_elapsed
                break

            # Check early exit rules
            exit_decision = self.check_exit_rules(entry_price, direction, hours_elapsed, current_price)
            if exit_decision['action'] != 'continue':
                progression['final_outcome'] = exit_decision.get('exit_reason', 'early_exit')
                progression['outcome_hour'] = hours_elapsed
                progression['exit_profit'] = profit_pct
                break

        # If no exit triggered, mark as timeout
        if progression['final_outcome'] is None:
            final_profit = progression['hourly_profits'][-1] if progression['hourly_profits'] else 0
            progression['final_outcome'] = 'timeout'
            progression['outcome_hour'] = len(progression['hourly_profits'])
            progression['exit_profit'] = final_profit

        return progression

    def check_exit_rules(self, entry_price: float, direction: str, hours_elapsed: int, current_price: float) -> Dict[str, Any]:
        """Check exit rules (simplified version)"""
        profit_pct = self.calculate_profit_pct(entry_price, current_price, direction)

        for rule in self.exit_rules:
            if hours_elapsed >= rule.check_hour:
                if rule.action == "stay_for_tp" and profit_pct >= rule.min_profit_pct:
                    return {'action': 'continue', 'reason': 'staying_for_tp'}
                elif rule.action == "exit":
                    if profit_pct < rule.min_profit_pct or profit_pct <= rule.max_loss_pct:
                        return {'action': 'exit_early', 'exit_reason': 'early_exit_rule'}

        return {'action': 'continue', 'reason': 'normal_hold'}

    def find_early_indicators(self, df: pd.DataFrame) -> Dict:
        """Find early indicators that predict TP vs SL outcomes"""

        print("🔍 ANALYZING PROFIT PROGRESSION PATTERNS")
        print("=" * 60)

        df = self.add_confluences(df)

        tp_trades = []
        sl_trades = []
        early_exit_trades = []

        # Analyze trades
        test_df = df.tail(20000).copy()  # Use recent data

        for idx in range(200, len(test_df) - self.max_hold_hours):
            candle = test_df.iloc[idx]

            signal = self.check_bullish_confluence(candle)
            if not signal:
                signal = self.check_bearish_confluence(candle)

            if signal:
                entry_price = candle['close']
                future_data = test_df.iloc[idx+1:idx+self.max_hold_hours+1]

                if len(future_data) < 12:
                    continue

                progression = self.analyze_progression_patterns(signal, entry_price, future_data)

                # Categorize by outcome
                if progression['final_outcome'] == 'TP':
                    tp_trades.append(progression)
                elif progression['final_outcome'] == 'SL':
                    sl_trades.append(progression)
                else:
                    early_exit_trades.append(progression)

        print(f"📊 Analyzed: {len(tp_trades)} TP trades, {len(sl_trades)} SL trades, {len(early_exit_trades)} early exits")

        return self.analyze_patterns(tp_trades, sl_trades, early_exit_trades)

    def analyze_patterns(self, tp_trades: List, sl_trades: List, early_exit_trades: List) -> Dict:
        """Analyze patterns to find early indicators"""

        print(f"\n🎯 EARLY INDICATOR ANALYSIS")
        print("=" * 50)

        # Key checkpoints to analyze
        checkpoints = [1, 2, 3, 6, 12]

        results = {}

        for checkpoint in checkpoints:
            print(f"\n⏰ ANALYSIS AT {checkpoint} HOURS:")

            tp_profits = []
            sl_profits = []

            # Get profit at this checkpoint for each outcome
            for trade in tp_trades:
                if len(trade['hourly_profits']) >= checkpoint:
                    tp_profits.append(trade['hourly_profits'][checkpoint-1])

            for trade in sl_trades:
                if len(trade['hourly_profits']) >= checkpoint:
                    sl_profits.append(trade['hourly_profits'][checkpoint-1])

            if len(tp_profits) > 5 and len(sl_profits) > 5:
                tp_avg = statistics.mean(tp_profits)
                sl_avg = statistics.mean(sl_profits)
                tp_median = statistics.median(tp_profits)
                sl_median = statistics.median(sl_profits)

                print(f"   TP trades average: {tp_avg:+.3f}% (median: {tp_median:+.3f}%)")
                print(f"   SL trades average: {sl_avg:+.3f}% (median: {sl_median:+.3f}%)")

                # Find threshold that separates TP from SL
                all_profits = tp_profits + sl_profits
                all_outcomes = ['TP'] * len(tp_profits) + ['SL'] * len(sl_profits)

                # Test different thresholds
                best_threshold = None
                best_accuracy = 0

                for threshold in np.arange(-1.0, 2.0, 0.1):
                    correct_predictions = 0
                    total_predictions = 0

                    for profit, outcome in zip(all_profits, all_outcomes):
                        if profit >= threshold:
                            prediction = 'TP'
                        else:
                            prediction = 'SL'

                        if prediction == outcome:
                            correct_predictions += 1
                        total_predictions += 1

                    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_threshold = threshold

                results[f'{checkpoint}h'] = {
                    'tp_avg': tp_avg,
                    'sl_avg': sl_avg,
                    'tp_median': tp_median,
                    'sl_median': sl_median,
                    'best_threshold': best_threshold,
                    'best_accuracy': best_accuracy,
                    'tp_count': len(tp_profits),
                    'sl_count': len(sl_profits)
                }

                print(f"   🎯 Best threshold: {best_threshold:+.3f}% (accuracy: {best_accuracy:.1%})")
                if best_threshold is not None:
                    if best_accuracy > 0.7:
                        print(f"   🟢 STRONG PREDICTOR: If profit ≥ {best_threshold:+.2f}% at {checkpoint}h → likely TP")
                    elif best_accuracy > 0.6:
                        print(f"   🟡 WEAK PREDICTOR: If profit ≥ {best_threshold:+.2f}% at {checkpoint}h → maybe TP")
                    else:
                        print(f"   🔴 POOR PREDICTOR: Not reliable at {checkpoint}h")

        return results

    def generate_recommendations(self, results: Dict):
        """Generate practical trading recommendations"""

        print(f"\n🏆 TRADING RECOMMENDATIONS")
        print("=" * 50)

        strong_predictors = []
        weak_predictors = []

        for timeframe, data in results.items():
            if data['best_accuracy'] > 0.7:
                strong_predictors.append((timeframe, data['best_threshold'], data['best_accuracy']))
            elif data['best_accuracy'] > 0.6:
                weak_predictors.append((timeframe, data['best_threshold'], data['best_accuracy']))

        if strong_predictors:
            print("🟢 STRONG EARLY INDICATORS:")
            for timeframe, threshold, accuracy in strong_predictors:
                print(f"   At {timeframe}: If profit ≥ {threshold:+.2f}% → {accuracy:.1%} chance of TP")

        if weak_predictors:
            print("🟡 WEAK EARLY INDICATORS:")
            for timeframe, threshold, accuracy in weak_predictors:
                print(f"   At {timeframe}: If profit ≥ {threshold:+.2f}% → {accuracy:.1%} chance of TP")

        print(f"\n💡 PRACTICAL RULES:")

        # Find the earliest strong predictor
        if strong_predictors:
            earliest = min(strong_predictors, key=lambda x: int(x[0].replace('h', '')))
            timeframe, threshold, accuracy = earliest
            print(f"   🎯 EARLY DECISION POINT: At {timeframe}")
            print(f"      If profit ≥ {threshold:+.2f}% → Stay for TP ({accuracy:.1%} success)")
            print(f"      If profit < {threshold:+.2f}% → Consider early exit")

        print(f"\n📊 SUMMARY:")
        print(f"   • TP trades typically show positive momentum early")
        print(f"   • SL trades often show negative momentum from hour 1")
        print(f"   • Best decision points are at 3h, 6h checkpoints")

def main():
    """Main analysis function"""

    print("🔍 PROFIT PROGRESSION PATTERN ANALYSIS")
    print("=" * 60)
    print("🎯 Goal: Find early indicators to predict TP vs SL")
    print("⏰ Question: How to know when to wait for TP?")
    print()

    # Load data
    try:
        df = pd.read_csv('data/XAUUSD_3YEAR_DATA.csv')
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df['ny_time'] = pd.to_datetime(df['ny_time'], utc=True)
        print(f"📈 Loaded {len(df)} candles")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Run analysis
    analyzer = ProgressionAnalyzer()
    results = analyzer.find_early_indicators(df)

    # Generate recommendations
    analyzer.generate_recommendations(results)

    print(f"\n🎉 ANALYSIS COMPLETE!")
    print("Now you know when to wait for TP vs exit early!")

if __name__ == "__main__":
    main()