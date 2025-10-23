#!/usr/bin/env python3
"""
Forex Technical Confluence Analysis
Analyzes which volatile forex pairs respect technical confluences and provide reliable trading signals.
Focus on identifying pairs that combine high volatility with strong technical reliability.
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ForexConfluenceAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com/time_series"

        # High volatility pairs from our previous analysis
        self.target_pairs = [
            "USD/TRY", "USD/ZAR", "USD/MXN", "USD/BRL",  # Extreme volatility
            "GBP/JPY", "GBP/AUD", "CAD/JPY",             # High volatility cross
            "GBP/USD", "USD/CAD", "EUR/USD"              # Volatile majors
        ]

        self.confluence_results = {}

    def fetch_detailed_data(self, symbol: str, outputsize: int = 1000) -> Optional[pd.DataFrame]:
        """Fetch detailed forex data for technical analysis"""
        params = {
            'symbol': symbol,
            'interval': '1h',  # Hourly data for better technical analysis
            'outputsize': outputsize,
            'apikey': self.api_key
        }

        try:
            print(f"Fetching detailed data for {symbol}...")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'values' not in data:
                print(f"No hourly data available for {symbol}, trying daily...")
                # Fallback to daily data
                params['interval'] = '1day'
                response = requests.get(self.base_url, params=params)
                data = response.json()

                if 'values' not in data:
                    print(f"No data available for {symbol}")
                    return None

            # Convert to DataFrame
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime').sort_index()

            # Convert price columns to float
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove any rows with NaN values
            df = df.dropna()

            print(f"Successfully fetched {len(df)} records for {symbol}")
            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        # Moving Averages
        df['EMA_20'] = df['close'].ewm(span=20).mean()
        df['EMA_50'] = df['close'].ewm(span=50).mean()
        df['SMA_200'] = df['close'].rolling(window=200).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        bb_period = 20
        df['BB_Middle'] = df['close'].rolling(window=bb_period).mean()
        bb_std = df['close'].rolling(window=bb_period).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

        # Support and Resistance levels
        df = self.identify_support_resistance(df)

        # Trend analysis
        df['Trend_EMA'] = np.where(df['EMA_20'] > df['EMA_50'], 1,
                                  np.where(df['EMA_20'] < df['EMA_50'], -1, 0))

        # Psychological levels (round numbers)
        df['Near_Psychological'] = self.identify_psychological_levels(df)

        return df

    def identify_support_resistance(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Identify support and resistance levels using pivot points"""
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()

        # Resistance levels (where high equals the rolling max)
        df['Resistance'] = np.where((df['high'] == highs) & (df['high'].shift(1) != highs), df['high'], np.nan)

        # Support levels (where low equals the rolling min)
        df['Support'] = np.where((df['low'] == lows) & (df['low'].shift(1) != lows), df['low'], np.nan)

        return df

    def identify_psychological_levels(self, df: pd.DataFrame) -> pd.Series:
        """Identify when price is near psychological levels (round numbers)"""
        def is_near_round(price, threshold=0.001):
            # Check if price is within threshold of a round number
            if 'JPY' in df.columns:  # This is a proxy check, we'll improve this
                round_levels = [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
            else:
                round_levels = [0.01, 0.05, 0.1, 0.5, 1.0]

            for level in round_levels:
                if abs(price % level) < threshold or abs((price % level) - level) < threshold:
                    return True
            return False

        return df['close'].apply(lambda x: is_near_round(x))

    def analyze_support_resistance_respect(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Analyze how well the pair respects support and resistance levels"""
        # Get support and resistance levels
        support_levels = df['Support'].dropna()
        resistance_levels = df['Resistance'].dropna()

        if len(support_levels) == 0 or len(resistance_levels) == 0:
            return {'respect_score': 0, 'details': 'Insufficient S/R levels identified'}

        # Test how often price bounces from these levels
        support_tests = 0
        support_holds = 0
        resistance_tests = 0
        resistance_holds = 0

        tolerance = df['close'].std() * 0.1  # Dynamic tolerance based on volatility

        # Check support level respect
        for support in support_levels:
            near_support = df[abs(df['low'] - support) <= tolerance]
            if len(near_support) > 0:
                support_tests += len(near_support)
                # Count how often price bounced (closed higher than the low)
                bounces = near_support[near_support['close'] > near_support['low'] * 1.001]
                support_holds += len(bounces)

        # Check resistance level respect
        for resistance in resistance_levels:
            near_resistance = df[abs(df['high'] - resistance) <= tolerance]
            if len(near_resistance) > 0:
                resistance_tests += len(near_resistance)
                # Count how often price was rejected (closed lower than the high)
                rejections = near_resistance[near_resistance['close'] < near_resistance['high'] * 0.999]
                resistance_holds += len(rejections)

        total_tests = support_tests + resistance_tests
        total_holds = support_holds + resistance_holds

        respect_score = (total_holds / total_tests * 100) if total_tests > 0 else 0

        return {
            'respect_score': respect_score,
            'support_tests': support_tests,
            'support_holds': support_holds,
            'resistance_tests': resistance_tests,
            'resistance_holds': resistance_holds,
            'total_levels': len(support_levels) + len(resistance_levels)
        }

    def analyze_moving_average_confluence(self, df: pd.DataFrame) -> Dict:
        """Analyze EMA crossover reliability and confluence"""
        # EMA crossover signals
        df['EMA_Cross'] = np.where((df['EMA_20'] > df['EMA_50']) & (df['EMA_20'].shift(1) <= df['EMA_50'].shift(1)), 1,
                                  np.where((df['EMA_20'] < df['EMA_50']) & (df['EMA_20'].shift(1) >= df['EMA_50'].shift(1)), -1, 0))

        crossovers = df[df['EMA_Cross'] != 0].copy()

        if len(crossovers) == 0:
            return {'crossover_success_rate': 0, 'total_crossovers': 0}

        # Analyze success rate of crossovers
        successful_signals = 0
        total_signals = len(crossovers)

        for idx, signal in crossovers.iterrows():
            # Look forward 20 periods to check if signal was successful
            future_data = df.loc[idx:].head(20)
            if len(future_data) < 10:
                continue

            signal_direction = signal['EMA_Cross']
            entry_price = signal['close']

            if signal_direction == 1:  # Bullish crossover
                highest_price = future_data['high'].max()
                success = (highest_price - entry_price) / entry_price > 0.01  # 1% profit target
            else:  # Bearish crossover
                lowest_price = future_data['low'].min()
                success = (entry_price - lowest_price) / entry_price > 0.01  # 1% profit target

            if success:
                successful_signals += 1

        success_rate = (successful_signals / total_signals * 100) if total_signals > 0 else 0

        # Check trend alignment with longer MA
        trend_alignment = df[(df['EMA_Cross'] != 0)].copy()
        if len(trend_alignment) > 0:
            aligned_signals = 0
            for idx, signal in trend_alignment.iterrows():
                if not pd.isna(signal['SMA_200']):
                    if (signal['EMA_Cross'] == 1 and signal['close'] > signal['SMA_200']) or \
                       (signal['EMA_Cross'] == -1 and signal['close'] < signal['SMA_200']):
                        aligned_signals += 1

            trend_alignment_rate = (aligned_signals / len(trend_alignment) * 100) if len(trend_alignment) > 0 else 0
        else:
            trend_alignment_rate = 0

        return {
            'crossover_success_rate': success_rate,
            'total_crossovers': total_signals,
            'successful_crossovers': successful_signals,
            'trend_alignment_rate': trend_alignment_rate
        }

    def analyze_rsi_reliability(self, df: pd.DataFrame) -> Dict:
        """Analyze RSI overbought/oversold reliability"""
        # RSI signals
        overbought = df[df['RSI'] >= 70].copy()
        oversold = df[df['RSI'] <= 30].copy()

        overbought_success = 0
        oversold_success = 0

        # Test overbought signals (expect price to go down)
        for idx, signal in overbought.iterrows():
            future_data = df.loc[idx:].head(10)
            if len(future_data) < 5:
                continue
            entry_price = signal['close']
            lowest_price = future_data['low'].min()
            if (entry_price - lowest_price) / entry_price > 0.005:  # 0.5% decline
                overbought_success += 1

        # Test oversold signals (expect price to go up)
        for idx, signal in oversold.iterrows():
            future_data = df.loc[idx:].head(10)
            if len(future_data) < 5:
                continue
            entry_price = signal['close']
            highest_price = future_data['high'].max()
            if (highest_price - entry_price) / entry_price > 0.005:  # 0.5% rise
                oversold_success += 1

        total_signals = len(overbought) + len(oversold)
        successful_signals = overbought_success + oversold_success

        success_rate = (successful_signals / total_signals * 100) if total_signals > 0 else 0

        return {
            'rsi_success_rate': success_rate,
            'overbought_signals': len(overbought),
            'overbought_success': overbought_success,
            'oversold_signals': len(oversold),
            'oversold_success': oversold_success,
            'total_rsi_signals': total_signals
        }

    def analyze_trend_quality(self, df: pd.DataFrame) -> Dict:
        """Analyze trend quality and structure"""
        # Calculate trend duration and strength
        df['Trend_Change'] = df['Trend_EMA'].diff().abs()
        trend_changes = df[df['Trend_Change'] > 0]

        if len(trend_changes) == 0:
            return {'trend_quality_score': 0}

        # Average trend duration
        trend_durations = []
        current_trend = None
        trend_start = None

        for idx, row in df.iterrows():
            if current_trend is None or row['Trend_EMA'] != current_trend:
                if trend_start is not None and current_trend != 0:
                    duration = len(df.loc[trend_start:idx]) - 1
                    if duration > 0:
                        trend_durations.append(duration)
                current_trend = row['Trend_EMA']
                trend_start = idx

        avg_trend_duration = np.mean(trend_durations) if trend_durations else 0

        # Trend strength (how often does price stay above/below EMA during trend)
        uptrend_strength = []
        downtrend_strength = []

        for idx, row in df.iterrows():
            if row['Trend_EMA'] == 1:  # Uptrend
                strength = (row['close'] > row['EMA_20'])
                uptrend_strength.append(strength)
            elif row['Trend_EMA'] == -1:  # Downtrend
                strength = (row['close'] < row['EMA_20'])
                downtrend_strength.append(strength)

        uptrend_quality = np.mean(uptrend_strength) * 100 if uptrend_strength else 0
        downtrend_quality = np.mean(downtrend_strength) * 100 if downtrend_strength else 0
        overall_trend_quality = (uptrend_quality + downtrend_quality) / 2

        return {
            'trend_quality_score': overall_trend_quality,
            'avg_trend_duration': avg_trend_duration,
            'uptrend_strength': uptrend_quality,
            'downtrend_strength': downtrend_quality,
            'trend_changes': len(trend_changes)
        }

    def calculate_signal_noise_ratio(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Calculate signal-to-noise ratio for trading"""
        # Price volatility (noise)
        returns = df['close'].pct_change().dropna()
        daily_volatility = returns.std()

        # Signal strength (trend consistency)
        trend_consistency = abs(df['Trend_EMA'].rolling(window=20).mean()).mean()

        # Range efficiency (how much of the range is captured by trend)
        true_range = np.maximum(df['high'] - df['low'],
                               np.maximum(abs(df['high'] - df['close'].shift(1)),
                                        abs(df['low'] - df['close'].shift(1))))

        directional_movement = abs(df['close'] - df['close'].shift(20))
        range_efficiency = (directional_movement / (true_range.rolling(window=20).sum() + 1e-8)).mean()

        # Signal-to-noise ratio
        snr = (trend_consistency * range_efficiency) / (daily_volatility + 1e-8)

        # False breakout analysis
        breakout_signals = 0
        false_breakouts = 0

        # Simple breakout analysis using Bollinger Bands
        bb_breakouts = df[(df['close'] > df['BB_Upper']) | (df['close'] < df['BB_Lower'])]

        for idx, breakout in bb_breakouts.iterrows():
            breakout_signals += 1
            # Check if price returns to normal range within 5 periods
            future_data = df.loc[idx:].head(5)
            if len(future_data) >= 3:
                returns_to_range = any((future_data['close'] > future_data['BB_Lower']) &
                                     (future_data['close'] < future_data['BB_Upper']))
                if returns_to_range:
                    false_breakouts += 1

        false_breakout_rate = (false_breakouts / breakout_signals * 100) if breakout_signals > 0 else 0

        return {
            'signal_noise_ratio': snr,
            'daily_volatility': daily_volatility * 100,
            'trend_consistency': trend_consistency * 100,
            'range_efficiency': range_efficiency,
            'false_breakout_rate': false_breakout_rate,
            'total_breakouts': breakout_signals,
            'false_breakouts': false_breakouts
        }

    def assess_fundamental_vs_technical(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Assess fundamental vs technical dominance"""
        # Technical dominance indicators

        # 1. How well does technical analysis predict direction
        technical_predictions = 0
        correct_predictions = 0

        for i in range(50, len(df)-10):  # Skip first 50 for indicators to stabilize
            current_data = df.iloc[i]
            future_data = df.iloc[i+1:i+11]  # Next 10 periods

            # Technical signal
            tech_signal = 0
            if current_data['RSI'] < 30 and current_data['close'] < current_data['BB_Lower']:
                tech_signal = 1  # Buy signal
            elif current_data['RSI'] > 70 and current_data['close'] > current_data['BB_Upper']:
                tech_signal = -1  # Sell signal

            if tech_signal != 0:
                technical_predictions += 1
                entry_price = current_data['close']

                if tech_signal == 1:  # Expected up move
                    max_price = future_data['high'].max()
                    if (max_price - entry_price) / entry_price > 0.01:
                        correct_predictions += 1
                else:  # Expected down move
                    min_price = future_data['low'].min()
                    if (entry_price - min_price) / entry_price > 0.01:
                        correct_predictions += 1

        technical_accuracy = (correct_predictions / technical_predictions * 100) if technical_predictions > 0 else 0

        # 2. Price action smoothness (less fundamental interference)
        price_smoothness = 1 - (df['close'].diff().abs() / df['close']).std()

        # 3. Correlation with technical indicators
        tech_correlation = abs(df['close'].corr(df['EMA_20'])) if not df['EMA_20'].isna().all() else 0

        # Overall technical dominance score
        tech_dominance = (technical_accuracy + price_smoothness * 100 + tech_correlation * 100) / 3

        return {
            'technical_dominance_score': tech_dominance,
            'technical_accuracy': technical_accuracy,
            'price_smoothness': price_smoothness * 100,
            'technical_correlation': tech_correlation * 100,
            'total_technical_signals': technical_predictions,
            'correct_technical_signals': correct_predictions
        }

    def analyze_confluence_quality(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Comprehensive confluence analysis for a single pair"""
        print(f"\nAnalyzing confluence for {symbol}...")

        # Calculate technical indicators
        df_with_indicators = self.calculate_technical_indicators(df)

        # Perform all analyses
        sr_analysis = self.analyze_support_resistance_respect(df_with_indicators, symbol)
        ma_analysis = self.analyze_moving_average_confluence(df_with_indicators)
        rsi_analysis = self.analyze_rsi_reliability(df_with_indicators)
        trend_analysis = self.analyze_trend_quality(df_with_indicators)
        snr_analysis = self.calculate_signal_noise_ratio(df_with_indicators, symbol)
        fund_tech_analysis = self.assess_fundamental_vs_technical(df_with_indicators, symbol)

        # Calculate overall confluence score
        confluence_score = (
            sr_analysis.get('respect_score', 0) * 0.25 +
            ma_analysis.get('crossover_success_rate', 0) * 0.20 +
            rsi_analysis.get('rsi_success_rate', 0) * 0.15 +
            trend_analysis.get('trend_quality_score', 0) * 0.20 +
            fund_tech_analysis.get('technical_dominance_score', 0) * 0.20
        )

        return {
            'symbol': symbol,
            'confluence_score': confluence_score,
            'support_resistance': sr_analysis,
            'moving_averages': ma_analysis,
            'rsi_analysis': rsi_analysis,
            'trend_quality': trend_analysis,
            'signal_noise': snr_analysis,
            'fundamental_vs_technical': fund_tech_analysis,
            'data_points': len(df_with_indicators)
        }

    def analyze_all_pairs(self) -> Dict:
        """Analyze all target pairs for confluence quality"""
        results = {}

        for pair in self.target_pairs:
            time.sleep(2)  # Rate limiting

            df = self.fetch_detailed_data(pair)
            if df is not None and len(df) > 200:  # Need sufficient data
                analysis = self.analyze_confluence_quality(df, pair)
                results[pair] = analysis
            else:
                print(f"Insufficient data for {pair}")
                results[pair] = None

        self.confluence_results = results
        return results

    def generate_recommendations(self) -> List[Dict]:
        """Generate final recommendations based on confluence analysis"""
        valid_results = {k: v for k, v in self.confluence_results.items() if v is not None}

        if not valid_results:
            return []

        # Create DataFrame for easier analysis
        summary_data = []
        for pair, data in valid_results.items():
            summary_data.append({
                'pair': pair,
                'confluence_score': data['confluence_score'],
                'sr_respect': data['support_resistance']['respect_score'],
                'ma_success': data['moving_averages']['crossover_success_rate'],
                'rsi_success': data['rsi_analysis']['rsi_success_rate'],
                'trend_quality': data['trend_quality']['trend_quality_score'],
                'snr_ratio': data['signal_noise']['signal_noise_ratio'],
                'technical_dominance': data['fundamental_vs_technical']['technical_dominance_score'],
                'false_breakout_rate': data['signal_noise']['false_breakout_rate']
            })

        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values('confluence_score', ascending=False)

        recommendations = []

        for idx, row in df_summary.iterrows():
            # Categorize the pair
            if row['confluence_score'] >= 60:
                category = "Excellent for confluence trading"
            elif row['confluence_score'] >= 50:
                category = "Good for confluence trading"
            elif row['confluence_score'] >= 40:
                category = "Moderate - requires careful analysis"
            else:
                category = "Not recommended for confluence trading"

            # Generate reasoning
            reasoning = []
            if row['sr_respect'] >= 60:
                reasoning.append(f"Strong S/R respect ({row['sr_respect']:.1f}%)")
            if row['ma_success'] >= 60:
                reasoning.append(f"Reliable MA signals ({row['ma_success']:.1f}%)")
            if row['rsi_success'] >= 60:
                reasoning.append(f"Good RSI reliability ({row['rsi_success']:.1f}%)")
            if row['trend_quality'] >= 70:
                reasoning.append(f"High trend quality ({row['trend_quality']:.1f}%)")
            if row['technical_dominance'] >= 60:
                reasoning.append(f"Technical dominance ({row['technical_dominance']:.1f}%)")
            if row['false_breakout_rate'] <= 30:
                reasoning.append(f"Low false breakouts ({row['false_breakout_rate']:.1f}%)")

            recommendations.append({
                'rank': len(recommendations) + 1,
                'pair': row['pair'],
                'confluence_score': row['confluence_score'],
                'category': category,
                'reasoning': "; ".join(reasoning) if reasoning else "Limited technical confluence",
                'key_metrics': {
                    'sr_respect': row['sr_respect'],
                    'ma_success': row['ma_success'],
                    'rsi_success': row['rsi_success'],
                    'trend_quality': row['trend_quality'],
                    'snr_ratio': row['snr_ratio'],
                    'false_breakout_rate': row['false_breakout_rate']
                }
            })

        return recommendations

    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive confluence analysis report"""
        if not self.confluence_results:
            return "No analysis results available."

        report = []
        report.append("=" * 80)
        report.append("FOREX TECHNICAL CONFLUENCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Pairs Analyzed: {len([k for k, v in self.confluence_results.items() if v is not None])}")
        report.append("")

        # Executive Summary
        recommendations = self.generate_recommendations()
        if recommendations:
            report.append("EXECUTIVE SUMMARY:")
            report.append("=" * 20)
            excellent_pairs = [r for r in recommendations if r['confluence_score'] >= 60]
            good_pairs = [r for r in recommendations if 50 <= r['confluence_score'] < 60]

            report.append(f"✓ {len(excellent_pairs)} pairs excellent for confluence trading")
            report.append(f"✓ {len(good_pairs)} pairs good for confluence trading")
            report.append(f"✓ Top performer: {recommendations[0]['pair']} ({recommendations[0]['confluence_score']:.1f}%)")
            report.append("")

        # Detailed Analysis
        report.append("DETAILED PAIR ANALYSIS:")
        report.append("=" * 30)

        for rec in recommendations:
            pair_data = self.confluence_results[rec['pair']]

            report.append(f"\n{rec['rank']}. {rec['pair']} - {rec['category']}")
            report.append(f"   Overall Confluence Score: {rec['confluence_score']:.1f}%")
            report.append(f"   {rec['reasoning']}")

            report.append("   Technical Metrics:")
            report.append(f"   • Support/Resistance Respect: {pair_data['support_resistance']['respect_score']:.1f}%")
            report.append(f"   • Moving Average Success: {pair_data['moving_averages']['crossover_success_rate']:.1f}%")
            report.append(f"   • RSI Signal Reliability: {pair_data['rsi_analysis']['rsi_success_rate']:.1f}%")
            report.append(f"   • Trend Quality: {pair_data['trend_quality']['trend_quality_score']:.1f}%")
            report.append(f"   • Signal-to-Noise Ratio: {pair_data['signal_noise']['signal_noise_ratio']:.2f}")
            report.append(f"   • False Breakout Rate: {pair_data['signal_noise']['false_breakout_rate']:.1f}%")
            report.append(f"   • Technical Dominance: {pair_data['fundamental_vs_technical']['technical_dominance_score']:.1f}%")

        # Trading Recommendations
        report.append("\n\nTRADING RECOMMENDATIONS:")
        report.append("=" * 30)

        top_3 = recommendations[:3]
        report.append("TOP 3 RECOMMENDED PAIRS FOR CONFLUENCE TRADING:")
        for i, rec in enumerate(top_3, 1):
            report.append(f"{i}. {rec['pair']} - Confluence Score: {rec['confluence_score']:.1f}%")

        report.append("\nTRADING STRATEGY SUGGESTIONS:")
        report.append("• Focus on pairs with confluence scores > 60% for systematic trading")
        report.append("• Use multiple timeframe analysis for confluence confirmation")
        report.append("• Combine S/R levels with moving average signals for higher probability trades")
        report.append("• Monitor RSI divergences at key S/R levels for reversal signals")
        report.append("• Avoid trading during high fundamental impact news for technical pairs")

        report.append("\nRISK MANAGEMENT NOTES:")
        excellent_pairs = [r['pair'] for r in recommendations if r['confluence_score'] >= 60]
        if excellent_pairs:
            report.append(f"• Recommended pairs for confluence trading: {', '.join(excellent_pairs)}")

        moderate_pairs = [r['pair'] for r in recommendations if 40 <= r['confluence_score'] < 50]
        if moderate_pairs:
            report.append(f"• Use smaller position sizes for: {', '.join(moderate_pairs)}")

        report.append("• Always confirm signals across multiple timeframes")
        report.append("• Set stops beyond key technical levels, not at arbitrary distances")

        return "\n".join(report)

    def save_results(self, filename: str = None):
        """Save confluence analysis results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forex_confluence_analysis_{timestamp}.json"

        # Save raw results
        with open(filename, 'w') as f:
            json.dump(self.confluence_results, f, indent=2, default=str)

        # Save report
        report_filename = filename.replace('.json', '_report.txt')
        with open(report_filename, 'w') as f:
            f.write(self.generate_comprehensive_report())

        print(f"Results saved to {filename}")
        print(f"Report saved to {report_filename}")

        return filename, report_filename


def main():
    """Main execution function"""
    API_KEY = "0e24ff3eb6ef415dba0cebcf04593e4f"

    print("Starting Forex Technical Confluence Analysis...")
    print("This will analyze which volatile pairs respect technical confluences...")
    print("Analysis may take 15-20 minutes due to comprehensive calculations...")

    analyzer = ForexConfluenceAnalyzer(API_KEY)

    # Run comprehensive analysis
    results = analyzer.analyze_all_pairs()

    # Generate and display report
    report = analyzer.generate_comprehensive_report()
    print("\n" + report)

    # Save results
    analyzer.save_results()

    return analyzer


if __name__ == "__main__":
    analyzer = main()