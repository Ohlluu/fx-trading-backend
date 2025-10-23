#!/usr/bin/env python3
"""
SMART CONFLUENCE SYSTEM WITH DYNAMIC EARLY EXIT MANAGEMENT
Exact confluences from 47% system + intelligent exit rules
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ExitRule:
    """Define an exit rule"""
    check_hour: int          # When to check (hours after entry)
    min_profit_pct: float    # Minimum profit % to stay
    max_loss_pct: float      # Maximum loss % before exit
    action: str              # "exit" or "stay_for_tp"

class SmartConfluenceSystem:
    def __init__(self):
        # Exit rules - customize these!
        self.exit_rules = [
            ExitRule(6, 2.0, -1.0, "stay_for_tp"),   # If +2% in 6h, stay for TP
            ExitRule(12, 0.5, -2.0, "exit"),        # If not +0.5% in 12h, exit
            ExitRule(18, 0.0, -1.5, "exit"),        # If not breakeven in 18h, exit
            ExitRule(24, -0.5, -3.0, "exit"),       # If worse than -0.5% in 24h, exit
        ]

        # Risk management
        self.profit_target_pct = 2.0  # 2% target (about $40 on $2000 gold)
        self.stop_loss_pct = 1.0      # 1% stop (about $20 on $2000 gold)
        self.max_hold_hours = 36

    def add_confluences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the exact confluences from the 47% system"""

        print("🔧 Adding PROVEN confluences...")

        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()

        # Position relative to MAs
        df['above_sma20'] = df['close'] > df['sma_20']
        df['above_sma50'] = df['close'] > df['sma_50']
        df['above_ema20'] = df['close'] > df['ema_20']

        # Trend detection (5-period slope)
        df['sma20_slope'] = df['sma_20'].diff(5)
        df['ema20_slope'] = df['ema_20'].diff(5)
        df['uptrend_sma'] = df['sma20_slope'] > 0
        df['uptrend_ema'] = df['ema20_slope'] > 0

        # Price action
        df['is_green'] = df['close'] > df['open']
        df['prev_red'] = (~df['is_green']).shift(1)

        # Recent high/low analysis
        df['recent_high_12h'] = df['high'].rolling(window=12).max()
        df['recent_low_12h'] = df['low'].rolling(window=12).min()
        df['near_recent_high'] = abs(df['close'] - df['recent_high_12h']) < 10

        print("✅ All confluences added!")
        return df

    def check_bullish_confluence(self, candle: pd.Series) -> Optional[Dict]:
        """Check for bullish confluence - EXACT from 47% system"""

        confluence_score = 0
        factors = []

        # MANDATORY: Must be above SMA50
        if not candle['above_sma50']:
            return None
        confluence_score += 4
        factors.append("Above SMA50 (mandatory)")

        # Add scoring confluences
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

        # Must score at least 10 points
        if confluence_score < 10:
            return None

        return {
            'direction': 'BUY',
            'confluence_score': confluence_score,
            'factors': factors,
            'signal_type': 'bullish'
        }

    def check_bearish_confluence(self, candle: pd.Series) -> Optional[Dict]:
        """Check for bearish confluence - EXACT from 47% system"""

        confluence_score = 0
        factors = []

        # MANDATORY: Must have SMA uptrend (counter-intuitive!)
        if not candle['uptrend_sma']:
            return None
        confluence_score += 3
        factors.append("SMA uptrend (mandatory)")

        # Add scoring confluences
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

        # Must score at least 8 points
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
        else:  # SELL
            return ((entry_price - current_price) / entry_price) * 100

    def check_exit_rules(self, entry_price: float, direction: str, hours_elapsed: int,
                        current_price: float) -> Dict[str, Any]:
        """Check if we should exit based on dynamic rules"""

        profit_pct = self.calculate_profit_pct(entry_price, current_price, direction)

        # Check each exit rule
        for rule in self.exit_rules:
            if hours_elapsed >= rule.check_hour:

                # Check if we should stay for take profit
                if rule.action == "stay_for_tp" and profit_pct >= rule.min_profit_pct:
                    return {
                        'action': 'continue',
                        'reason': f'Profit {profit_pct:.1f}% ≥ {rule.min_profit_pct}% at {rule.check_hour}h - staying for TP',
                        'continue_reason': 'good_progress'
                    }

                # Check if we should exit early
                elif rule.action == "exit":
                    if profit_pct < rule.min_profit_pct or profit_pct <= rule.max_loss_pct:
                        return {
                            'action': 'exit_early',
                            'reason': f'Profit {profit_pct:.1f}% failed rule at {rule.check_hour}h',
                            'exit_price': current_price,
                            'exit_reason': 'early_exit_rule'
                        }

        # Check hard stops
        if profit_pct >= self.profit_target_pct:
            return {
                'action': 'exit_tp',
                'reason': f'Take profit hit: {profit_pct:.1f}%',
                'exit_price': current_price,
                'exit_reason': 'take_profit'
            }

        if profit_pct <= -self.stop_loss_pct:
            return {
                'action': 'exit_sl',
                'reason': f'Stop loss hit: {profit_pct:.1f}%',
                'exit_price': current_price,
                'exit_reason': 'stop_loss'
            }

        # Continue holding
        return {
            'action': 'continue',
            'reason': f'Profit {profit_pct:.1f}% - no exit rules triggered yet',
            'continue_reason': 'normal_hold'
        }

    def simulate_smart_trade(self, signal: Dict, entry_price: float, future_data: pd.DataFrame) -> Dict:
        """Simulate trade with smart exit management"""

        direction = signal['direction']

        # Track the trade hour by hour
        for hour in range(min(len(future_data), self.max_hold_hours)):
            current_candle = future_data.iloc[hour]
            current_price = current_candle['close']
            hours_elapsed = hour + 1

            # Check exit rules
            exit_decision = self.check_exit_rules(entry_price, direction, hours_elapsed, current_price)

            if exit_decision['action'] != 'continue':
                profit_pct = self.calculate_profit_pct(entry_price, current_price, direction)
                profit_dollars = (profit_pct / 100) * entry_price

                return {
                    'result': 'WIN' if profit_pct > 0 else 'LOSS',
                    'exit_price': current_price,
                    'profit_pct': profit_pct,
                    'profit_dollars': profit_dollars,
                    'duration_hours': hours_elapsed,
                    'exit_reason': exit_decision.get('exit_reason', exit_decision['action']),
                    'exit_explanation': exit_decision['reason']
                }

        # Timeout - exit at market
        final_price = future_data.iloc[-1]['close']
        profit_pct = self.calculate_profit_pct(entry_price, final_price, direction)
        profit_dollars = (profit_pct / 100) * entry_price

        return {
            'result': 'WIN' if profit_pct > 0 else 'LOSS',
            'exit_price': final_price,
            'profit_pct': profit_pct,
            'profit_dollars': profit_dollars,
            'duration_hours': self.max_hold_hours,
            'exit_reason': 'timeout',
            'exit_explanation': f'Max hold time reached - exit at {profit_pct:.1f}%'
        }

    def run_backtest(self, df: pd.DataFrame) -> List[Dict]:
        """Run backtest with smart confluence system"""

        print("🚀 RUNNING SMART CONFLUENCE BACKTEST")
        print("="*60)

        df = self.add_confluences(df)
        trades = []

        # Test on recent data (last 2 years)
        test_df = df.tail(17520).copy()
        print(f"📊 Testing on {len(test_df)} candles...")

        daily_trades = {}

        for idx in range(len(test_df) - self.max_hold_hours):
            candle = test_df.iloc[idx]
            date_str = str(candle['ny_time'])[:10]

            # Limit trades per day
            if daily_trades.get(date_str, 0) >= 2:
                continue

            # Need sufficient history
            if idx < 200:
                continue

            # Check for signals
            signal = self.check_bullish_confluence(candle)
            if not signal:
                signal = self.check_bearish_confluence(candle)

            if signal:
                entry_price = candle['close']
                future_data = test_df.iloc[idx+1:idx+self.max_hold_hours+1]

                if len(future_data) < 12:
                    continue

                # Simulate trade with smart exits
                trade_result = self.simulate_smart_trade(signal, entry_price, future_data)

                # Record trade
                trade_record = {
                    'date': date_str,
                    'entry_time': candle['ny_time'],
                    'direction': signal['direction'],
                    'entry_price': entry_price,
                    'confluence_score': signal['confluence_score'],
                    'factors': '; '.join(signal['factors']),
                    **trade_result
                }

                trades.append(trade_record)
                daily_trades[date_str] = daily_trades.get(date_str, 0) + 1

                # Print trade result
                result_emoji = "✅" if trade_result['result'] == 'WIN' else "❌"
                print(f"{result_emoji} Trade {len(trades)}: {signal['direction']} @${entry_price:.2f} "
                      f"→ {trade_result['result']} {trade_result['profit_pct']:+.1f}% "
                      f"({trade_result['duration_hours']}h) - {trade_result['exit_reason']}")

        return trades

def analyze_results(trades: List[Dict]):
    """Analyze the smart system results"""

    if not trades:
        print("❌ No trades found")
        return

    print(f"\n🏆 SMART CONFLUENCE SYSTEM RESULTS")
    print("="*60)

    wins = [t for t in trades if t['result'] == 'WIN']
    total_trades = len(trades)
    win_rate = (len(wins) / total_trades) * 100

    total_profit_pct = sum(t['profit_pct'] for t in trades)
    avg_profit_pct = total_profit_pct / total_trades

    print(f"📊 OVERALL PERFORMANCE:")
    print(f"   Total Trades: {total_trades}")
    print(f"   Win Rate: {win_rate:.1f}% ({len(wins)}/{total_trades})")
    print(f"   Average Profit: {avg_profit_pct:+.2f}% per trade")
    print(f"   Total Profit: {total_profit_pct:+.1f}%")

    # Analyze exit reasons
    print(f"\n🚪 EXIT ANALYSIS:")
    exit_reasons = {}
    for trade in trades:
        reason = trade['exit_reason']
        if reason not in exit_reasons:
            exit_reasons[reason] = {'count': 0, 'wins': 0, 'profit': 0}
        exit_reasons[reason]['count'] += 1
        if trade['result'] == 'WIN':
            exit_reasons[reason]['wins'] += 1
        exit_reasons[reason]['profit'] += trade['profit_pct']

    for reason, data in exit_reasons.items():
        win_rate_reason = (data['wins'] / data['count']) * 100
        avg_profit_reason = data['profit'] / data['count']
        print(f"   {reason}: {win_rate_reason:.1f}% win rate, {avg_profit_reason:+.2f}% avg profit ({data['count']} trades)")

    # Save results
    results_df = pd.DataFrame(trades)
    results_df.to_csv('data/SMART_CONFLUENCE_RESULTS.csv', index=False)
    print(f"\n💾 Results saved to: data/SMART_CONFLUENCE_RESULTS.csv")

def main():
    """Main function"""

    print("🧠 SMART CONFLUENCE SYSTEM WITH DYNAMIC EXITS")
    print("="*60)
    print("✨ Exact confluences from 47% system + intelligent exit rules")
    print("🎯 Rules: Exit early if not meeting profit targets by specific hours")
    print("")

    # Load data
    try:
        df = pd.read_csv('data/XAUUSD_3YEAR_DATA.csv')
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df['ny_time'] = pd.to_datetime(df['ny_time'], utc=True)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Create and run system
    system = SmartConfluenceSystem()
    trades = system.run_backtest(df)

    # Analyze results
    analyze_results(trades)

    print(f"\n🎯 SMART EXIT RULES USED:")
    for rule in system.exit_rules:
        print(f"   Hour {rule.check_hour}: Need {rule.min_profit_pct:+.1f}% profit (max loss {rule.max_loss_pct:.1f}%) → {rule.action}")

if __name__ == "__main__":
    main()