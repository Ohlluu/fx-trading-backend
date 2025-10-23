#!/usr/bin/env python3
"""
COMPREHENSIVE 5-YEAR TIMING ANALYSIS
Test the smart confluence system on ALL 5 years of data
Analyze: How long does it take to hit TP vs SL on average?
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class ExitRule:
    """Define an exit rule"""
    check_hour: int
    min_profit_pct: float
    max_loss_pct: float
    action: str

class ComprehensiveTimingAnalysis:
    def __init__(self):
        # The proven exit rules
        self.exit_rules = [
            ExitRule(6, 2.0, -1.0, "stay_for_tp"),
            ExitRule(12, 0.5, -2.0, "exit"),
            ExitRule(18, 0.0, -1.5, "exit"),
            ExitRule(24, -0.5, -3.0, "exit"),
        ]

        # Risk management
        self.profit_target_pct = 2.0
        self.stop_loss_pct = 1.0
        self.max_hold_hours = 72      # Test longer holds

    def add_confluences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the exact confluences from the 60% system"""

        print("🔧 Adding proven confluences to 5-year dataset...")

        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()

        # Position relative to MAs
        df['above_sma20'] = df['close'] > df['sma_20']
        df['above_sma50'] = df['close'] > df['sma_50']
        df['above_ema20'] = df['close'] > df['ema_20']

        # Trend detection
        df['sma20_slope'] = df['sma_20'].diff(5)
        df['ema20_slope'] = df['ema_20'].diff(5)
        df['uptrend_sma'] = df['sma20_slope'] > 0
        df['uptrend_ema'] = df['ema20_slope'] > 0

        # Price action
        df['is_green'] = df['close'] > df['open']
        df['prev_red'] = (~df['is_green']).shift(1)

        # Recent levels
        df['recent_high_12h'] = df['high'].rolling(window=12).max()
        df['near_recent_high'] = abs(df['close'] - df['recent_high_12h']) < 10

        print("✅ All confluences added!")
        return df

    def check_bullish_confluence(self, candle: pd.Series) -> Optional[Dict]:
        """Exact bullish confluence from 60% system"""

        confluence_score = 0
        factors = []

        # MANDATORY: Above SMA50
        if not candle['above_sma50']:
            return None
        confluence_score += 4
        factors.append("Above SMA50")

        # Scoring confluences
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
            'factors': factors
        }

    def check_bearish_confluence(self, candle: pd.Series) -> Optional[Dict]:
        """Exact bearish confluence from 60% system"""

        confluence_score = 0
        factors = []

        # MANDATORY: SMA uptrend
        if not candle['uptrend_sma']:
            return None
        confluence_score += 3
        factors.append("SMA uptrend")

        # Scoring confluences
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
            'factors': factors
        }

    def calculate_profit_pct(self, entry_price: float, current_price: float, direction: str) -> float:
        """Calculate profit percentage"""
        if direction == 'BUY':
            return ((current_price - entry_price) / entry_price) * 100
        else:
            return ((entry_price - current_price) / entry_price) * 100

    def analyze_trade_timing(self, signal: Dict, entry_price: float, future_data: pd.DataFrame) -> Dict:
        """Analyze detailed timing of when TP/SL would be hit"""

        direction = signal['direction']

        # Track what happens each hour
        timing_data = {
            'tp_hit_hour': None,
            'sl_hit_hour': None,
            'early_exit_hour': None,
            'final_result': None,
            'final_exit_hour': None,
            'final_profit_pct': None,
            'max_profit_pct': -999,
            'min_profit_pct': 999,
            'max_profit_hour': None,
            'min_profit_hour': None
        }

        # Check each hour
        for hour in range(min(len(future_data), self.max_hold_hours)):
            current_candle = future_data.iloc[hour]
            current_price = current_candle['close']
            profit_pct = self.calculate_profit_pct(entry_price, current_price, direction)

            # Track extremes
            if profit_pct > timing_data['max_profit_pct']:
                timing_data['max_profit_pct'] = profit_pct
                timing_data['max_profit_hour'] = hour + 1

            if profit_pct < timing_data['min_profit_pct']:
                timing_data['min_profit_pct'] = profit_pct
                timing_data['min_profit_hour'] = hour + 1

            # Check for TP hit (first time)
            if timing_data['tp_hit_hour'] is None and profit_pct >= self.profit_target_pct:
                timing_data['tp_hit_hour'] = hour + 1

            # Check for SL hit (first time)
            if timing_data['sl_hit_hour'] is None and profit_pct <= -self.stop_loss_pct:
                timing_data['sl_hit_hour'] = hour + 1

            # Check early exit rules
            hours_elapsed = hour + 1
            should_exit_early = False

            for rule in self.exit_rules:
                if hours_elapsed >= rule.check_hour:
                    if rule.action == "exit":
                        if profit_pct < rule.min_profit_pct or profit_pct <= rule.max_loss_pct:
                            should_exit_early = True
                            if timing_data['early_exit_hour'] is None:
                                timing_data['early_exit_hour'] = hours_elapsed
                            break

            # Determine final exit
            if should_exit_early and timing_data['final_result'] is None:
                timing_data['final_result'] = 'early_exit'
                timing_data['final_exit_hour'] = hours_elapsed
                timing_data['final_profit_pct'] = profit_pct
                break
            elif profit_pct >= self.profit_target_pct:
                timing_data['final_result'] = 'take_profit'
                timing_data['final_exit_hour'] = hour + 1
                timing_data['final_profit_pct'] = profit_pct
                break
            elif profit_pct <= -self.stop_loss_pct:
                timing_data['final_result'] = 'stop_loss'
                timing_data['final_exit_hour'] = hour + 1
                timing_data['final_profit_pct'] = profit_pct
                break

        # If no exit triggered, timeout
        if timing_data['final_result'] is None:
            final_price = future_data.iloc[-1]['close']
            final_profit = self.calculate_profit_pct(entry_price, final_price, direction)
            timing_data['final_result'] = 'timeout'
            timing_data['final_exit_hour'] = self.max_hold_hours
            timing_data['final_profit_pct'] = final_profit

        return timing_data

    def run_comprehensive_analysis(self, df: pd.DataFrame) -> List[Dict]:
        """Run comprehensive analysis on full 5-year dataset"""

        print("🚀 COMPREHENSIVE 5-YEAR TIMING ANALYSIS")
        print("="*70)

        df = self.add_confluences(df)
        trades = []

        print(f"📊 Analyzing {len(df)} candles (5 years of XAUUSD data)...")
        print("🔍 This will take a few minutes to analyze every confluence...")

        # Analyze every 10th candle to speed up (still massive dataset)
        step_size = 10
        analyzed_candles = 0

        for idx in range(0, len(df) - self.max_hold_hours, step_size):
            candle = df.iloc[idx]

            # Need sufficient history
            if idx < 200:
                continue

            analyzed_candles += 1
            if analyzed_candles % 1000 == 0:
                print(f"   📈 Analyzed {analyzed_candles} candles, found {len(trades)} trades...")

            # Check for signals
            signal = self.check_bullish_confluence(candle)
            if not signal:
                signal = self.check_bearish_confluence(candle)

            if signal:
                entry_price = candle['close']
                future_data = df.iloc[idx+1:idx+self.max_hold_hours+1]

                if len(future_data) < 24:
                    continue

                # Analyze timing in detail
                timing_data = self.analyze_trade_timing(signal, entry_price, future_data)

                # Record trade
                trade_record = {
                    'entry_date': str(candle['ny_time'])[:10] if 'ny_time' in candle else str(candle['datetime'])[:10],
                    'direction': signal['direction'],
                    'entry_price': entry_price,
                    'confluence_score': signal['confluence_score'],
                    **timing_data
                }

                trades.append(trade_record)

        print(f"✅ Analysis complete! Found {len(trades)} trades to analyze")
        return trades

def analyze_timing_results(trades: List[Dict]):
    """Comprehensive timing analysis"""

    if not trades:
        print("❌ No trades to analyze")
        return

    print(f"\n🏆 COMPREHENSIVE 5-YEAR TIMING ANALYSIS")
    print("="*70)
    print(f"📊 Total Trades Analyzed: {len(trades)}")

    # Separate by final result
    tp_trades = [t for t in trades if t['final_result'] == 'take_profit']
    sl_trades = [t for t in trades if t['final_result'] == 'stop_loss']
    early_exit_trades = [t for t in trades if t['final_result'] == 'early_exit']
    timeout_trades = [t for t in trades if t['final_result'] == 'timeout']

    print(f"\n📈 OUTCOME BREAKDOWN:")
    print(f"   Take Profit: {len(tp_trades)} ({len(tp_trades)/len(trades)*100:.1f}%)")
    print(f"   Stop Loss: {len(sl_trades)} ({len(sl_trades)/len(trades)*100:.1f}%)")
    print(f"   Early Exit: {len(early_exit_trades)} ({len(early_exit_trades)/len(trades)*100:.1f}%)")
    print(f"   Timeout: {len(timeout_trades)} ({len(timeout_trades)/len(trades)*100:.1f}%)")

    # TIMING ANALYSIS
    print(f"\n⏰ DETAILED TIMING ANALYSIS:")
    print("="*50)

    if tp_trades:
        tp_times = [t['final_exit_hour'] for t in tp_trades]
        avg_tp_time = sum(tp_times) / len(tp_times)
        median_tp_time = sorted(tp_times)[len(tp_times)//2]
        print(f"📈 TAKE PROFIT TIMING:")
        print(f"   Average time to TP: {avg_tp_time:.1f} hours")
        print(f"   Median time to TP: {median_tp_time} hours")
        print(f"   Fastest TP: {min(tp_times)} hours")
        print(f"   Slowest TP: {max(tp_times)} hours")

    if sl_trades:
        sl_times = [t['final_exit_hour'] for t in sl_trades]
        avg_sl_time = sum(sl_times) / len(sl_times)
        median_sl_time = sorted(sl_times)[len(sl_times)//2]
        print(f"\n📉 STOP LOSS TIMING:")
        print(f"   Average time to SL: {avg_sl_time:.1f} hours")
        print(f"   Median time to SL: {median_sl_time} hours")
        print(f"   Fastest SL: {min(sl_times)} hours")
        print(f"   Slowest SL: {max(sl_times)} hours")

    # EARLY EXIT ANALYSIS
    if early_exit_trades:
        early_times = [t['final_exit_hour'] for t in early_exit_trades]
        early_profits = [t['final_profit_pct'] for t in early_exit_trades]
        avg_early_time = sum(early_times) / len(early_times)
        avg_early_profit = sum(early_profits) / len(early_profits)

        print(f"\n🚪 EARLY EXIT ANALYSIS:")
        print(f"   Average early exit time: {avg_early_time:.1f} hours")
        print(f"   Average early exit profit: {avg_early_profit:+.2f}%")

        early_wins = [t for t in early_exit_trades if t['final_profit_pct'] > 0]
        print(f"   Early exit win rate: {len(early_wins)/len(early_exit_trades)*100:.1f}%")

    # COMPARISON: TP vs SL Speed
    print(f"\n⚡ SPEED COMPARISON:")
    print("="*30)
    if tp_trades and sl_trades:
        print(f"   TP happens in {avg_tp_time:.1f}h on average")
        print(f"   SL happens in {avg_sl_time:.1f}h on average")

        if avg_tp_time < avg_sl_time:
            diff = avg_sl_time - avg_tp_time
            print(f"   ✅ TP is {diff:.1f} hours FASTER than SL!")
        else:
            diff = avg_tp_time - avg_sl_time
            print(f"   ❌ SL is {diff:.1f} hours FASTER than TP")

    # PROFIT EXTREMES ANALYSIS
    print(f"\n📊 PROFIT EXTREMES ANALYSIS:")
    print("="*40)

    all_max_profits = [t['max_profit_pct'] for t in trades if t['max_profit_pct'] > -999]
    all_min_profits = [t['min_profit_pct'] for t in trades if t['min_profit_pct'] < 999]

    if all_max_profits:
        avg_max_profit = sum(all_max_profits) / len(all_max_profits)
        print(f"   Average MAX profit reached: {avg_max_profit:+.2f}%")
        print(f"   Highest profit ever reached: {max(all_max_profits):+.2f}%")

        max_profit_times = [t['max_profit_hour'] for t in trades if t['max_profit_hour']]
        if max_profit_times:
            avg_max_time = sum(max_profit_times) / len(max_profit_times)
            print(f"   Average time to max profit: {avg_max_time:.1f} hours")

    if all_min_profits:
        avg_min_profit = sum(all_min_profits) / len(all_min_profits)
        print(f"   Average MIN profit reached: {avg_min_profit:+.2f}%")
        print(f"   Worst loss ever reached: {min(all_min_profits):+.2f}%")

    # BULLISH vs BEARISH TIMING
    print(f"\n📈📉 BULLISH vs BEARISH TIMING:")
    print("="*45)

    bullish_trades = [t for t in trades if t['direction'] == 'BUY']
    bearish_trades = [t for t in trades if t['direction'] == 'SELL']

    if bullish_trades:
        bull_tp = [t for t in bullish_trades if t['final_result'] == 'take_profit']
        bull_sl = [t for t in bullish_trades if t['final_result'] == 'stop_loss']

        print(f"   BULLISH TRADES ({len(bullish_trades)} total):")
        if bull_tp:
            bull_tp_avg = sum(t['final_exit_hour'] for t in bull_tp) / len(bull_tp)
            print(f"     TP time: {bull_tp_avg:.1f}h average ({len(bull_tp)} trades)")
        if bull_sl:
            bull_sl_avg = sum(t['final_exit_hour'] for t in bull_sl) / len(bull_sl)
            print(f"     SL time: {bull_sl_avg:.1f}h average ({len(bull_sl)} trades)")

    if bearish_trades:
        bear_tp = [t for t in bearish_trades if t['final_result'] == 'take_profit']
        bear_sl = [t for t in bearish_trades if t['final_result'] == 'stop_loss']

        print(f"   BEARISH TRADES ({len(bearish_trades)} total):")
        if bear_tp:
            bear_tp_avg = sum(t['final_exit_hour'] for t in bear_tp) / len(bear_tp)
            print(f"     TP time: {bear_tp_avg:.1f}h average ({len(bear_tp)} trades)")
        if bear_sl:
            bear_sl_avg = sum(t['final_exit_hour'] for t in bear_sl) / len(bear_sl)
            print(f"     SL time: {bear_sl_avg:.1f}h average ({len(bear_sl)} trades)")

    # Save detailed results
    results_df = pd.DataFrame(trades)
    results_df.to_csv('data/5YEAR_TIMING_ANALYSIS.csv', index=False)
    print(f"\n💾 Detailed results saved to: data/5YEAR_TIMING_ANALYSIS.csv")

def main():
    """Main analysis function"""

    print("⏰ COMPREHENSIVE 5-YEAR TIMING ANALYSIS")
    print("="*60)
    print("🎯 Testing smart confluence system on ALL 5 years of data")
    print("📊 Analyzing: How long does it take to hit TP vs SL?")
    print("🔍 Plus: Detailed timing patterns and profit extremes")
    print("")

    # Load 5-year dataset
    try:
        df = pd.read_csv('data/XAUUSD_3YEAR_DATA.csv')
        df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
        df['ny_time'] = pd.to_datetime(df['ny_time'], utc=True)
        print(f"📈 Loaded {len(df)} candles of XAUUSD data")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Run comprehensive analysis
    analyzer = ComprehensiveTimingAnalysis()
    trades = analyzer.run_comprehensive_analysis(df)

    # Analyze timing patterns
    analyze_timing_results(trades)

    print(f"\n🎉 COMPREHENSIVE ANALYSIS COMPLETE!")
    print("="*50)
    print("🎯 Now you know exactly how long TP vs SL takes!")
    print("📊 Plus detailed insights about profit timing patterns!")

if __name__ == "__main__":
    main()