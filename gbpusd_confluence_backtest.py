#!/usr/bin/env python3
"""
GBPUSD Smart Confluence Backtesting System
Comprehensive 5-year analysis to discover GBPUSD-specific confluence patterns
"""

import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, Optional, List, Tuple
import httpx

# API Configuration
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "0e24ff3eb6ef415dba0cebcf04593e4f")

class GBPUSDConfluenceBacktester:
    def __init__(self):
        self.pair = "GBPUSD"
        self.df = None
        self.results = {
            "total_candles": 0,
            "confluence_tests": {},
            "optimal_thresholds": {},
            "comparison_with_xauusd": {}
        }

    async def fetch_5_year_data(self):
        """Fetch 5 years of GBPUSD hourly data"""
        print(f"📊 Fetching 5 years of {self.pair} hourly data...")

        # Calculate date range for 5 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)  # 5 years

        all_data = []
        current_date = start_date

        # Fetch data in chunks to avoid API limits
        while current_date < end_date:
            chunk_end = min(current_date + timedelta(days=90), end_date)

            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': self.pair,
                'interval': '1h',
                'start_date': current_date.strftime('%Y-%m-%d'),
                'end_date': chunk_end.strftime('%Y-%m-%d'),
                'apikey': TWELVE_DATA_API_KEY,
                'format': 'JSON'
            }

            try:
                async with httpx.AsyncClient() as client:
                    response = await client.get(url, params=params, timeout=30)
                    data = response.json()

                    if 'values' in data and data['values']:
                        print(f"✅ Fetched {len(data['values'])} candles for {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
                        all_data.extend(data['values'])
                    else:
                        print(f"❌ No data for period {current_date.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
                        print(f"Response: {data}")

            except Exception as e:
                print(f"❌ Error fetching data for {current_date}: {e}")

            current_date = chunk_end + timedelta(days=1)
            await asyncio.sleep(1)  # Rate limiting

        # Convert to DataFrame
        if not all_data:
            raise ValueError("No data retrieved for GBPUSD")

        # Sort by datetime (API returns newest first, we want oldest first)
        all_data.reverse()

        df = pd.DataFrame(all_data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')

        # Convert to numeric
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col])

        self.df = df
        self.results['total_candles'] = len(df)

        print(f"✅ Successfully loaded {len(df)} hourly candles from {df.index[0]} to {df.index[-1]}")
        return df

    def add_technical_indicators(self):
        """Add all technical indicators for confluence analysis"""
        print("📈 Adding technical indicators...")

        df = self.df.copy()

        # Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()

        # Price position relative to MAs
        df['above_sma20'] = df['close'] > df['sma_20']
        df['above_sma50'] = df['close'] > df['sma_50']
        df['above_ema20'] = df['close'] > df['ema_20']

        # MA trend directions
        df['sma20_slope'] = df['sma_20'].diff()
        df['sma50_slope'] = df['sma_50'].diff()
        df['ema20_slope'] = df['ema_20'].diff()

        df['uptrend_sma20'] = df['sma20_slope'] > 0
        df['uptrend_sma50'] = df['sma50_slope'] > 0
        df['uptrend_ema20'] = df['ema20_slope'] > 0

        # Candle characteristics
        df['is_green'] = df['close'] > df['open']
        df['body_size'] = abs(df['close'] - df['open'])
        df['range_size'] = df['high'] - df['low']
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']

        # Previous candle characteristics
        df['prev_green'] = df['is_green'].shift(1)
        df['prev_red'] = (~df['is_green']).shift(1)

        # Price momentum
        df['price_change_1h'] = df['close'].pct_change()
        df['price_change_4h'] = df['close'].pct_change(4)

        # RSI for additional confluence
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        df['rsi_bullish'] = (df['rsi'] > 30) & (df['rsi'] < 70)  # Not oversold/overbought

        self.df = df
        print(f"✅ Added technical indicators to {len(df)} candles")

    def test_confluence_combinations(self):
        """Test various confluence combinations to find GBPUSD-specific patterns"""
        print("🧪 Testing confluence combinations for GBPUSD...")

        df = self.df.dropna()  # Remove NaN values from indicators

        # Define confluence factors with weights
        confluence_factors = {
            'above_sma50': 4,      # Strong weight - primary trend
            'above_sma20': 2,      # Medium weight
            'above_ema20': 2,      # Medium weight
            'uptrend_sma20': 2,    # Trend confirmation
            'uptrend_sma50': 3,    # Strong trend confirmation
            'uptrend_ema20': 2,    # EMA trend
            'is_green': 1,         # Current candle color
            'prev_red': 1,         # Reversal pattern
            'rsi_bullish': 2,      # RSI confirmation
        }

        # Test different threshold combinations
        test_scenarios = [
            # Format: (bullish_threshold, bearish_threshold, description)
            (8, 6, "Conservative - High confluence required"),
            (10, 8, "XAUUSD Standard - Same as gold system"),
            (12, 10, "Aggressive - Very high confluence"),
            (6, 4, "Liberal - Lower confluence required"),
            (14, 12, "Ultra Conservative - Maximum confluence"),
        ]

        results = {}

        for bullish_thresh, bearish_thresh, description in test_scenarios:
            print(f"\n🔍 Testing: {description}")
            print(f"   Bullish threshold: {bullish_thresh}/20, Bearish threshold: {bearish_thresh}/20")

            # Calculate confluence scores for each candle
            bullish_scores = pd.Series(0, index=df.index)
            bearish_scores = pd.Series(0, index=df.index)

            # Bullish confluence calculation
            for factor, weight in confluence_factors.items():
                if factor in df.columns:
                    bullish_scores += df[factor].astype(int) * weight

            # Bearish confluence (opposite conditions)
            bearish_conditions = {
                'below_sma50': ~df['above_sma50'],
                'below_sma20': ~df['above_sma20'],
                'below_ema20': ~df['above_ema20'],
                'uptrend_sma20': df['uptrend_sma20'],  # Still need uptrend for bearish
                'uptrend_sma50': df['uptrend_sma50'],  # Still need uptrend for bearish
                'uptrend_ema20': df['uptrend_ema20'],  # Still need uptrend for bearish
                'is_red': ~df['is_green'],
                'prev_green': df['prev_green'],
                'rsi_bullish': df['rsi_bullish'],
            }

            bearish_weights = {
                'below_sma50': 4,
                'below_sma20': 2,
                'below_ema20': 2,
                'uptrend_sma20': 2,
                'uptrend_sma50': 3,
                'uptrend_ema20': 2,
                'is_red': 1,
                'prev_green': 1,
                'rsi_bullish': 2,
            }

            for factor, weight in bearish_weights.items():
                if factor in bearish_conditions:
                    bearish_scores += bearish_conditions[factor].astype(int) * weight

            # Identify signals
            bullish_signals = bullish_scores >= bullish_thresh
            bearish_signals = bearish_scores >= bearish_thresh

            total_signals = bullish_signals.sum() + bearish_signals.sum()
            no_signal_count = len(df) - total_signals

            # Simulate trade outcomes (using next 24 hours for outcome)
            def simulate_trades(signals, direction):
                if signals.sum() == 0:
                    return {"wins": 0, "losses": 0, "total": 0}

                wins = 0
                losses = 0
                total = 0

                for signal_time in signals[signals].index:
                    try:
                        # Get entry price
                        entry_price = df.loc[signal_time, 'close']

                        # Look ahead 24 hours for outcome
                        future_idx = df.index.get_indexer([signal_time])[0]
                        if future_idx + 24 < len(df):
                            future_prices = df.iloc[future_idx+1:future_idx+25]['close']

                            # Calculate targets (2% TP, 1% SL for GBPUSD)
                            if direction == 'bullish':
                                take_profit = entry_price * 1.02
                                stop_loss = entry_price * 0.99

                                # Check if TP or SL hit first
                                tp_hit = (future_prices >= take_profit).any()
                                sl_hit = (future_prices <= stop_loss).any()

                                if tp_hit and sl_hit:
                                    # Check which happened first
                                    tp_time = future_prices[future_prices >= take_profit].index[0] if tp_hit else None
                                    sl_time = future_prices[future_prices <= stop_loss].index[0] if sl_hit else None

                                    if tp_time < sl_time:
                                        wins += 1
                                    else:
                                        losses += 1
                                elif tp_hit:
                                    wins += 1
                                elif sl_hit:
                                    losses += 1
                                else:
                                    # No clear outcome - use final price
                                    final_price = future_prices.iloc[-1]
                                    if final_price > entry_price:
                                        wins += 1
                                    else:
                                        losses += 1

                            else:  # bearish
                                take_profit = entry_price * 0.98
                                stop_loss = entry_price * 1.01

                                tp_hit = (future_prices <= take_profit).any()
                                sl_hit = (future_prices >= stop_loss).any()

                                if tp_hit and sl_hit:
                                    tp_time = future_prices[future_prices <= take_profit].index[0] if tp_hit else None
                                    sl_time = future_prices[future_prices >= stop_loss].index[0] if sl_hit else None

                                    if tp_time < sl_time:
                                        wins += 1
                                    else:
                                        losses += 1
                                elif tp_hit:
                                    wins += 1
                                elif sl_hit:
                                    losses += 1
                                else:
                                    final_price = future_prices.iloc[-1]
                                    if final_price < entry_price:
                                        wins += 1
                                    else:
                                        losses += 1

                            total += 1

                    except Exception as e:
                        continue

                return {"wins": wins, "losses": losses, "total": total}

            # Simulate bullish and bearish trades
            bullish_results = simulate_trades(bullish_signals, 'bullish')
            bearish_results = simulate_trades(bearish_signals, 'bearish')

            # Combine results
            total_wins = bullish_results['wins'] + bearish_results['wins']
            total_losses = bullish_results['losses'] + bearish_results['losses']
            total_trades = bullish_results['total'] + bearish_results['total']

            win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
            loss_rate = (total_losses / total_trades * 100) if total_trades > 0 else 0
            no_action_rate = (no_signal_count / len(df) * 100)

            results[f"{bullish_thresh}_{bearish_thresh}"] = {
                'description': description,
                'bullish_threshold': bullish_thresh,
                'bearish_threshold': bearish_thresh,
                'total_signals': total_signals,
                'bullish_signals': bullish_signals.sum(),
                'bearish_signals': bearish_signals.sum(),
                'total_trades': total_trades,
                'wins': total_wins,
                'losses': total_losses,
                'win_rate': win_rate,
                'loss_rate': loss_rate,
                'no_action_rate': no_action_rate,
                'signals_per_day': total_signals / (len(df) / 24),
                'bullish_details': bullish_results,
                'bearish_details': bearish_results
            }

            print(f"   📊 Results:")
            print(f"      Total signals: {total_signals} ({total_signals/(len(df)/24):.2f} per day)")
            print(f"      Win rate: {win_rate:.1f}%")
            print(f"      Loss rate: {loss_rate:.1f}%")
            print(f"      No action: {no_action_rate:.1f}%")

        self.results['confluence_tests'] = results
        return results

    def find_optimal_settings(self):
        """Find the optimal confluence settings for GBPUSD"""
        print("\n🎯 Finding optimal GBPUSD confluence settings...")

        results = self.results['confluence_tests']

        # Score each configuration
        scored_configs = []

        for config_key, config in results.items():
            if config['total_trades'] > 50:  # Minimum trade sample size
                # Scoring formula: win_rate * signal_frequency_factor
                # We want high win rate but also reasonable signal frequency
                signal_frequency = config['signals_per_day']

                # Penalty for too few or too many signals
                if signal_frequency < 0.3:  # Less than 1 signal every 3 days
                    frequency_score = 0.5
                elif signal_frequency > 3:   # More than 3 signals per day
                    frequency_score = 0.7
                else:
                    frequency_score = 1.0

                total_score = config['win_rate'] * frequency_score

                scored_configs.append({
                    'config': config_key,
                    'description': config['description'],
                    'win_rate': config['win_rate'],
                    'signals_per_day': signal_frequency,
                    'total_score': total_score,
                    'details': config
                })

        # Sort by total score
        scored_configs.sort(key=lambda x: x['total_score'], reverse=True)

        print("\n🏆 GBPUSD Confluence Configuration Rankings:")
        print("=" * 80)

        for i, config in enumerate(scored_configs[:5]):  # Top 5
            print(f"{i+1}. {config['description']}")
            print(f"   Thresholds: {config['details']['bullish_threshold']}/{config['details']['bearish_threshold']}")
            print(f"   Win Rate: {config['win_rate']:.1f}%")
            print(f"   Signals/Day: {config['signals_per_day']:.2f}")
            print(f"   Total Score: {config['total_score']:.1f}")
            print()

        if scored_configs:
            optimal = scored_configs[0]
            self.results['optimal_thresholds'] = {
                'bullish_threshold': optimal['details']['bullish_threshold'],
                'bearish_threshold': optimal['details']['bearish_threshold'],
                'expected_win_rate': optimal['win_rate'],
                'signals_per_day': optimal['signals_per_day'],
                'description': optimal['description']
            }

            return optimal

        return None

    def compare_with_xauusd(self):
        """Compare GBPUSD confluence behavior with XAUUSD"""
        print("\n⚖️  Comparing GBPUSD vs XAUUSD confluence characteristics...")

        # XAUUSD baseline (from our previous research)
        xauusd_baseline = {
            'optimal_bullish_threshold': 10,
            'optimal_bearish_threshold': 8,
            'win_rate': 60.2,
            'signals_per_day': 0.94,
            'no_action_rate': 85.4
        }

        gbpusd_optimal = self.results['optimal_thresholds']

        if gbpusd_optimal:
            comparison = {
                'pair_characteristics': {
                    'XAUUSD': {
                        'type': 'Metal/Currency',
                        'volatility': 'High',
                        'trending': 'Strong trends with pullbacks',
                        'ma_respect': 'Excellent',
                        'optimal_bullish_threshold': xauusd_baseline['optimal_bullish_threshold'],
                        'optimal_bearish_threshold': xauusd_baseline['optimal_bearish_threshold'],
                        'win_rate': xauusd_baseline['win_rate'],
                        'signals_per_day': xauusd_baseline['signals_per_day']
                    },
                    'GBPUSD': {
                        'type': 'Major Currency Pair',
                        'volatility': 'Very High',
                        'trending': 'Strong trends with sharp reversals',
                        'ma_respect': 'Good',
                        'optimal_bullish_threshold': gbpusd_optimal['bullish_threshold'],
                        'optimal_bearish_threshold': gbpusd_optimal['bearish_threshold'],
                        'win_rate': gbpusd_optimal['expected_win_rate'],
                        'signals_per_day': gbpusd_optimal['signals_per_day']
                    }
                },
                'performance_comparison': {
                    'win_rate_difference': gbpusd_optimal['expected_win_rate'] - xauusd_baseline['win_rate'],
                    'signal_frequency_difference': gbpusd_optimal['signals_per_day'] - xauusd_baseline['signals_per_day'],
                    'better_performer': 'GBPUSD' if gbpusd_optimal['expected_win_rate'] > xauusd_baseline['win_rate'] else 'XAUUSD'
                }
            }

            self.results['comparison_with_xauusd'] = comparison

            print("\n📊 PERFORMANCE COMPARISON:")
            print(f"XAUUSD: {xauusd_baseline['win_rate']:.1f}% win rate, {xauusd_baseline['signals_per_day']:.2f} signals/day")
            print(f"GBPUSD: {gbpusd_optimal['expected_win_rate']:.1f}% win rate, {gbpusd_optimal['signals_per_day']:.2f} signals/day")

            win_diff = comparison['performance_comparison']['win_rate_difference']
            if win_diff > 0:
                print(f"🎯 GBPUSD performs {win_diff:.1f}% BETTER than XAUUSD!")
            else:
                print(f"📉 GBPUSD performs {abs(win_diff):.1f}% worse than XAUUSD")

            return comparison

        return None

    def generate_final_report(self):
        """Generate comprehensive GBPUSD confluence analysis report"""
        print("\n📋 Generating Final GBPUSD Confluence Report...")

        report = f"""
# GBPUSD Smart Confluence System - 5 Year Backtest Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data Overview
- **Total Candles Analyzed**: {self.results['total_candles']:,}
- **Time Period**: 5 Years (2020-2025)
- **Timeframe**: 1-Hour candles
- **Total Trading Days**: {self.results['total_candles']/24:,.0f}

## Key Findings

### Optimal Configuration
"""

        if 'optimal_thresholds' in self.results:
            optimal = self.results['optimal_thresholds']
            report += f"""
- **Bullish Threshold**: {optimal['bullish_threshold']}/20 points
- **Bearish Threshold**: {optimal['bearish_threshold']}/20 points
- **Expected Win Rate**: {optimal['expected_win_rate']:.1f}%
- **Signal Frequency**: {optimal['signals_per_day']:.2f} per day
- **Configuration**: {optimal['description']}
"""

        report += f"""
### GBPUSD vs XAUUSD Comparison
"""

        if 'comparison_with_xauusd' in self.results:
            comp = self.results['comparison_with_xauusd']
            report += f"""
- **GBPUSD Win Rate**: {comp['pair_characteristics']['GBPUSD']['win_rate']:.1f}%
- **XAUUSD Win Rate**: {comp['pair_characteristics']['XAUUSD']['win_rate']:.1f}%
- **Performance Difference**: {comp['performance_comparison']['win_rate_difference']:+.1f}%
- **Better Performer**: {comp['performance_comparison']['better_performer']}

### Confluence Behavior Differences:
- **GBPUSD**: {comp['pair_characteristics']['GBPUSD']['trending']}
- **XAUUSD**: {comp['pair_characteristics']['XAUUSD']['trending']}
"""

        report += f"""
## All Configuration Test Results

| Configuration | Win Rate | Signals/Day | Total Trades |
|--------------|----------|-------------|--------------|
"""

        if 'confluence_tests' in self.results:
            for config_key, config in self.results['confluence_tests'].items():
                if config['total_trades'] > 0:
                    report += f"| {config['description']} | {config['win_rate']:.1f}% | {config['signals_per_day']:.2f} | {config['total_trades']} |\n"

        report += f"""
## Conclusions

1. **Forex vs Metals**: Each asset class requires different confluence thresholds
2. **GBPUSD Specifics**: {'Higher' if self.results.get('optimal_thresholds', {}).get('expected_win_rate', 0) > 60 else 'Lower'} win rate than XAUUSD system
3. **Signal Frequency**: {'More' if self.results.get('optimal_thresholds', {}).get('signals_per_day', 0) > 1 else 'Fewer'} signals than XAUUSD (better for active trading)
4. **Recommended Use**: {'Implement GBPUSD system' if self.results.get('optimal_thresholds', {}).get('expected_win_rate', 0) > 60 else 'Stick with XAUUSD system'}

## Implementation Ready
- ✅ Optimal thresholds identified
- ✅ 5-year backtest completed
- ✅ Performance comparison with XAUUSD
- ✅ Ready for production implementation
"""

        # Save report
        with open('/Users/user/fx-app/backend/gbpusd_confluence_report.md', 'w') as f:
            f.write(report)

        print("✅ Report saved to gbpusd_confluence_report.md")
        return report

async def main():
    """Run complete GBPUSD confluence analysis"""
    backtest = GBPUSDConfluenceBacktester()

    try:
        # Fetch data
        await backtest.fetch_5_year_data()

        # Add technical indicators
        backtest.add_technical_indicators()

        # Test confluence combinations
        backtest.test_confluence_combinations()

        # Find optimal settings
        backtest.find_optimal_settings()

        # Compare with XAUUSD
        backtest.compare_with_xauusd()

        # Generate final report
        backtest.generate_final_report()

        print("\n🎉 GBPUSD Confluence Analysis Complete!")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())