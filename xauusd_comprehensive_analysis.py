#!/usr/bin/env python3
"""
Comprehensive XAU/USD Analysis for Confluence Trading
=====================================================

This script performs a detailed analysis of XAU/USD (Gold) using 2 years of historical data
to determine its suitability for confluence-based trading strategies.

Analysis includes:
1. Volatility Analysis (ADR, patterns, comparisons)
2. Technical Confluence Analysis (S/R, MA, indicators)
3. Market Structure Quality Assessment
4. Fundamental vs Technical Balance
5. Trading Characteristics Analysis
6. Risk Assessment

Author: Trading System Analysis
Date: 2025-09-23
"""

import pandas as pd
import numpy as np
import requests
import json
import warnings
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import ta
from scipy import stats
import os

warnings.filterwarnings('ignore')

class XAUUSDAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.symbol = "XAU/USD"
        self.results = {}

        # Trading session times (UTC)
        self.sessions = {
            'Asian': {'start': 0, 'end': 8},
            'London': {'start': 8, 'end': 16},
            'New York': {'start': 13, 'end': 21},
            'Overlap': {'start': 13, 'end': 16}  # London-NY overlap
        }

    def fetch_data(self, interval: str = '1day', outputsize: int = 730) -> pd.DataFrame:
        """Fetch historical data from TwelveData API"""
        try:
            url = f"{self.base_url}/time_series"
            params = {
                'symbol': self.symbol,
                'interval': interval,
                'outputsize': outputsize,
                'apikey': self.api_key,
                'format': 'JSON'
            }

            print(f"Fetching {interval} data for {self.symbol}...")
            response = requests.get(url, params=params)
            data = response.json()

            if 'values' not in data:
                print(f"API Error: {data}")
                return pd.DataFrame()

            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime').sort_index()

            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            print(f"Successfully fetched {len(df)} {interval} candles")
            return df

        except Exception as e:
            print(f"Error fetching data: {e}")
            return pd.DataFrame()

    def load_existing_data(self, filepath: str) -> pd.DataFrame:
        """Load existing CSV data"""
        try:
            df = pd.read_csv(filepath)
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time').sort_index()

            # Rename columns to match API format
            df = df.rename(columns={'time': 'datetime'})

            print(f"Loaded {len(df)} records from {filepath}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    def calculate_adr_pips(self, df: pd.DataFrame) -> Dict:
        """Calculate Average Daily Range in pips"""
        # For XAU/USD, 1 pip = 0.1 (different from major FX pairs)
        pip_value = 0.1

        # Resample hourly data to daily
        daily_data = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        # Calculate daily ranges in pips
        daily_ranges_pips = (daily_data['high'] - daily_data['low']) / pip_value

        adr_metrics = {
            'mean_adr_pips': daily_ranges_pips.mean(),
            'median_adr_pips': daily_ranges_pips.median(),
            'std_adr_pips': daily_ranges_pips.std(),
            'min_range_pips': daily_ranges_pips.min(),
            'max_range_pips': daily_ranges_pips.max(),
            'adr_25_percentile': daily_ranges_pips.quantile(0.25),
            'adr_75_percentile': daily_ranges_pips.quantile(0.75),
            'daily_ranges_pips': daily_ranges_pips.tolist(),
            'total_days': len(daily_ranges_pips)
        }

        # Calculate volatility patterns
        daily_data['weekday'] = daily_data.index.weekday
        daily_data['month'] = daily_data.index.month
        daily_data['range_pips'] = daily_ranges_pips

        # Weekly patterns (0=Monday, 6=Sunday)
        weekly_pattern = daily_data.groupby('weekday')['range_pips'].agg(['mean', 'std']).round(2)
        adr_metrics['weekly_pattern'] = weekly_pattern.to_dict()

        # Monthly patterns
        monthly_pattern = daily_data.groupby('month')['range_pips'].agg(['mean', 'std']).round(2)
        adr_metrics['monthly_pattern'] = monthly_pattern.to_dict()

        return adr_metrics

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for confluence analysis"""
        df = df.copy()

        # Moving Averages
        df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
        df['ema_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
        df['sma_100'] = ta.trend.SMAIndicator(df['close'], window=100).sma_indicator()
        df['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()

        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()

        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()

        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()

        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()

        return df

    def analyze_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Analyze support/resistance levels"""
        # Identify psychological levels (round numbers)
        psychological_levels = []
        price_min = df['close'].min()
        price_max = df['close'].max()

        # Generate round number levels
        for level in range(int(price_min // 100) * 100, int(price_max // 100) * 100 + 200, 100):
            if price_min <= level <= price_max:
                psychological_levels.append(level)

        # Test respect rate for psychological levels
        level_tests = {}
        for level in psychological_levels:
            tolerance = level * 0.005  # 0.5% tolerance

            # Find touches
            touches = 0
            holds = 0

            for i in range(1, len(df)):
                # Check if price touched the level
                if (df.iloc[i-1]['low'] > level + tolerance and
                    df.iloc[i]['low'] <= level + tolerance):
                    touches += 1
                    # Check if it held (didn't close below level significantly)
                    if df.iloc[i]['close'] > level - tolerance:
                        holds += 1

                elif (df.iloc[i-1]['high'] < level - tolerance and
                      df.iloc[i]['high'] >= level - tolerance):
                    touches += 1
                    # Check if it held (didn't close above level significantly)
                    if df.iloc[i]['close'] < level + tolerance:
                        holds += 1

            if touches > 0:
                level_tests[level] = {
                    'touches': touches,
                    'holds': holds,
                    'respect_rate': holds / touches
                }

        # Calculate overall psychological level respect rate
        total_touches = sum([data['touches'] for data in level_tests.values()])
        total_holds = sum([data['holds'] for data in level_tests.values()])
        overall_respect_rate = total_holds / total_touches if total_touches > 0 else 0

        return {
            'psychological_levels': psychological_levels,
            'level_tests': level_tests,
            'overall_respect_rate': overall_respect_rate,
            'total_tests': total_touches,
            'successful_holds': total_holds
        }

    def analyze_moving_average_confluence(self, df: pd.DataFrame) -> Dict:
        """Analyze moving average confluence effectiveness"""
        results = {}

        # EMA 20/50 crossover analysis
        df['ema_20_above_50'] = df['ema_20'] > df['ema_50']
        crossovers = df['ema_20_above_50'].diff().fillna(0)

        bullish_crossovers = (crossovers == 1).sum()
        bearish_crossovers = (crossovers == -1).sum()

        results['ema_crossovers'] = {
            'bullish_crossovers': bullish_crossovers,
            'bearish_crossovers': bearish_crossovers,
            'total_crossovers': bullish_crossovers + bearish_crossovers
        }

        # MA respect analysis
        ma_columns = ['ema_20', 'ema_50', 'sma_100', 'sma_200']
        ma_respect = {}

        for ma_col in ma_columns:
            if ma_col in df.columns:
                # Calculate how often price respects the MA as support/resistance
                touches = 0
                respects = 0

                for i in range(1, len(df)):
                    ma_val = df.iloc[i][ma_col]
                    if pd.isna(ma_val):
                        continue

                    tolerance = ma_val * 0.001  # 0.1% tolerance

                    # Check for touches
                    if (abs(df.iloc[i]['low'] - ma_val) <= tolerance or
                        abs(df.iloc[i]['high'] - ma_val) <= tolerance):
                        touches += 1

                        # Check if it respected (bounced)
                        if i < len(df) - 1:
                            next_candle = df.iloc[i + 1]
                            if (df.iloc[i]['low'] <= ma_val <= df.iloc[i]['high'] and
                                next_candle['close'] > ma_val):
                                respects += 1

                ma_respect[ma_col] = {
                    'touches': touches,
                    'respects': respects,
                    'respect_rate': respects / touches if touches > 0 else 0
                }

        results['ma_respect'] = ma_respect

        return results

    def analyze_technical_indicators(self, df: pd.DataFrame) -> Dict:
        """Analyze technical indicator reliability"""
        results = {}

        # RSI overbought/oversold analysis
        rsi_overbought = df['rsi'] > 70
        rsi_oversold = df['rsi'] < 30

        # Count reversal signals
        overbought_reversals = 0
        oversold_reversals = 0

        for i in range(len(df) - 5):  # Look 5 periods ahead for reversal
            if rsi_overbought.iloc[i]:
                # Check if price declined in next 5 periods
                if df.iloc[i + 5]['close'] < df.iloc[i]['close']:
                    overbought_reversals += 1

            if rsi_oversold.iloc[i]:
                # Check if price increased in next 5 periods
                if df.iloc[i + 5]['close'] > df.iloc[i]['close']:
                    oversold_reversals += 1

        results['rsi_analysis'] = {
            'overbought_signals': rsi_overbought.sum(),
            'oversold_signals': rsi_oversold.sum(),
            'overbought_reversals': overbought_reversals,
            'oversold_reversals': oversold_reversals,
            'overbought_accuracy': overbought_reversals / rsi_overbought.sum() if rsi_overbought.sum() > 0 else 0,
            'oversold_accuracy': oversold_reversals / rsi_oversold.sum() if rsi_oversold.sum() > 0 else 0
        }

        # MACD signal analysis
        macd_bullish = (df['macd'] > df['macd_signal']) & (df['macd'].shift() <= df['macd_signal'].shift())
        macd_bearish = (df['macd'] < df['macd_signal']) & (df['macd'].shift() >= df['macd_signal'].shift())

        results['macd_analysis'] = {
            'bullish_signals': macd_bullish.sum(),
            'bearish_signals': macd_bearish.sum(),
            'total_signals': macd_bullish.sum() + macd_bearish.sum()
        }

        # Bollinger Band analysis
        bb_squeeze = (df['bb_upper'] - df['bb_lower']) < df['bb_middle'] * 0.02  # Tight bands
        price_at_upper = df['close'] >= df['bb_upper']
        price_at_lower = df['close'] <= df['bb_lower']

        results['bollinger_analysis'] = {
            'squeeze_periods': bb_squeeze.sum(),
            'upper_band_touches': price_at_upper.sum(),
            'lower_band_touches': price_at_lower.sum(),
            'squeeze_percentage': (bb_squeeze.sum() / len(df)) * 100
        }

        return results

    def analyze_market_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze market structure and trending behavior"""
        results = {}

        # Calculate trend using EMA slopes
        df['ema_20_slope'] = df['ema_20'].pct_change(periods=5) * 100
        df['ema_50_slope'] = df['ema_50'].pct_change(periods=10) * 100

        # Define trend conditions
        strong_uptrend = (df['ema_20_slope'] > 0.1) & (df['ema_50_slope'] > 0.05)
        strong_downtrend = (df['ema_20_slope'] < -0.1) & (df['ema_50_slope'] < -0.05)
        ranging = (~strong_uptrend) & (~strong_downtrend)

        results['trend_analysis'] = {
            'uptrend_periods': strong_uptrend.sum(),
            'downtrend_periods': strong_downtrend.sum(),
            'ranging_periods': ranging.sum(),
            'uptrend_percentage': (strong_uptrend.sum() / len(df)) * 100,
            'downtrend_percentage': (strong_downtrend.sum() / len(df)) * 100,
            'ranging_percentage': (ranging.sum() / len(df)) * 100
        }

        # Session analysis
        df['hour'] = df.index.hour
        session_analysis = {}

        for session, times in self.sessions.items():
            session_mask = (df['hour'] >= times['start']) & (df['hour'] < times['end'])
            session_data = df[session_mask]

            if len(session_data) > 0:
                session_range = (session_data['high'].max() - session_data['low'].min())
                avg_range = (session_data['high'] - session_data['low']).mean()

                session_analysis[session] = {
                    'periods': len(session_data),
                    'avg_range_pips': avg_range / 0.1,  # Convert to pips
                    'total_range_pips': session_range / 0.1,
                    'volatility': session_data['close'].pct_change().std() * 100
                }

        results['session_analysis'] = session_analysis

        return results

    def analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """Analyze correlations with major indices and DXY"""
        results = {}

        # Note: This would require additional data for SPX, DXY, etc.
        # For now, we'll analyze internal correlations

        # Price vs indicators correlation
        correlations = {
            'price_vs_rsi': df['close'].corr(df['rsi']),
            'price_vs_macd': df['close'].corr(df['macd']),
            'volume_vs_range': df['volume'].corr(df['high'] - df['low'])
        }

        results['internal_correlations'] = correlations

        return results

    def calculate_risk_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate risk assessment metrics"""
        results = {}

        # Daily returns
        daily_returns = df['close'].pct_change().dropna()

        # Basic risk metrics
        results['risk_metrics'] = {
            'daily_volatility': daily_returns.std() * 100,
            'annualized_volatility': daily_returns.std() * np.sqrt(252) * 100,
            'max_daily_gain': daily_returns.max() * 100,
            'max_daily_loss': daily_returns.min() * 100,
            'positive_days': (daily_returns > 0).sum(),
            'negative_days': (daily_returns < 0).sum(),
            'win_rate': (daily_returns > 0).mean() * 100
        }

        # Drawdown analysis
        cumulative_returns = (1 + daily_returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max * 100

        results['drawdown_analysis'] = {
            'max_drawdown': drawdown.min(),
            'avg_drawdown': drawdown[drawdown < 0].mean(),
            'drawdown_periods': (drawdown < -5).sum()  # Periods with >5% drawdown
        }

        # Gap analysis
        gaps = []
        for i in range(1, len(df)):
            # Weekend gaps (Friday close to Monday open)
            if df.index[i].weekday() == 0 and df.index[i-1].weekday() == 4:  # Monday after Friday
                gap = abs(df.iloc[i]['open'] - df.iloc[i-1]['close']) / df.iloc[i-1]['close'] * 100
                gaps.append(gap)

        results['gap_analysis'] = {
            'weekend_gaps': len(gaps),
            'avg_gap_size': np.mean(gaps) if gaps else 0,
            'max_gap_size': np.max(gaps) if gaps else 0
        }

        return results

    def run_comprehensive_analysis(self) -> Dict:
        """Run the complete analysis"""
        print("Starting Comprehensive XAU/USD Analysis...")
        print("=" * 50)

        # Load existing hourly data
        hourly_data = self.load_existing_data('/Users/user/fx-app/backend/data/XAUUSD_H1.csv')

        if hourly_data.empty:
            print("No existing data found, fetching from API...")
            hourly_data = self.fetch_data('1h', 5000)

        if hourly_data.empty:
            print("Failed to get data!")
            return {}

        # Get daily data by resampling or fetching
        try:
            daily_data = self.fetch_data('1day', 730)
            if daily_data.empty:
                # Resample hourly to daily as backup
                daily_data = hourly_data.resample('D').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
        except:
            daily_data = hourly_data.resample('D').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

        print(f"Data period: {hourly_data.index.min()} to {hourly_data.index.max()}")
        print(f"Hourly candles: {len(hourly_data)}")
        print(f"Daily candles: {len(daily_data)}")

        # Add technical indicators
        hourly_data = self.calculate_technical_indicators(hourly_data)

        # Perform all analyses
        print("\n1. Calculating volatility metrics...")
        volatility_results = self.calculate_adr_pips(hourly_data)

        print("2. Analyzing support/resistance levels...")
        sr_results = self.analyze_support_resistance(hourly_data)

        print("3. Analyzing moving average confluence...")
        ma_results = self.analyze_moving_average_confluence(hourly_data)

        print("4. Analyzing technical indicators...")
        indicator_results = self.analyze_technical_indicators(hourly_data)

        print("5. Analyzing market structure...")
        structure_results = self.analyze_market_structure(hourly_data)

        print("6. Analyzing correlations...")
        correlation_results = self.analyze_correlations(hourly_data)

        print("7. Calculating risk metrics...")
        risk_results = self.calculate_risk_metrics(hourly_data)

        # Compile all results
        all_results = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_period': {
                'start': hourly_data.index.min().strftime('%Y-%m-%d'),
                'end': hourly_data.index.max().strftime('%Y-%m-%d'),
                'hourly_candles': len(hourly_data),
                'daily_candles': len(daily_data)
            },
            'volatility_analysis': volatility_results,
            'support_resistance_analysis': sr_results,
            'moving_average_analysis': ma_results,
            'technical_indicator_analysis': indicator_results,
            'market_structure_analysis': structure_results,
            'correlation_analysis': correlation_results,
            'risk_analysis': risk_results
        }

        return all_results

    def generate_trading_assessment(self, results: Dict) -> Dict:
        """Generate trading suitability assessment and recommendations"""
        assessment = {}

        # Calculate overall confluence score
        confluence_factors = []

        # Support/Resistance score (0-100)
        sr_score = results['support_resistance_analysis']['overall_respect_rate'] * 100
        confluence_factors.append(('Support/Resistance', sr_score))

        # Moving Average score
        ma_scores = []
        for ma, data in results['moving_average_analysis']['ma_respect'].items():
            ma_scores.append(data['respect_rate'])
        ma_score = np.mean(ma_scores) * 100 if ma_scores else 0
        confluence_factors.append(('Moving Averages', ma_score))

        # Technical Indicator score
        rsi_score = (results['technical_indicator_analysis']['rsi_analysis']['overbought_accuracy'] +
                    results['technical_indicator_analysis']['rsi_analysis']['oversold_accuracy']) / 2 * 100
        confluence_factors.append(('RSI Signals', rsi_score))

        # Trending behavior score (higher is better for confluence trading)
        trend_score = (results['market_structure_analysis']['trend_analysis']['uptrend_percentage'] +
                      results['market_structure_analysis']['trend_analysis']['downtrend_percentage'])
        confluence_factors.append(('Trending Behavior', trend_score))

        # Calculate weighted overall score
        overall_score = np.mean([score for _, score in confluence_factors])

        # Trading characteristics
        volatility = results['volatility_analysis']['mean_adr_pips']

        # Risk assessment
        risk_level = "HIGH" if volatility > 500 else "MEDIUM" if volatility > 200 else "LOW"

        assessment = {
            'overall_confluence_score': round(overall_score, 2),
            'confluence_breakdown': dict(confluence_factors),
            'volatility_rating': {
                'adr_pips': volatility,
                'risk_level': risk_level,
                'comparison_to_forex': 'Much Higher' if volatility > 150 else 'Higher' if volatility > 100 else 'Similar'
            },
            'trading_recommendation': {
                'suitable_for_confluence': overall_score >= 60,
                'primary_strengths': [],
                'main_weaknesses': [],
                'optimal_timeframes': ['H1', 'H4', 'Daily'],
                'best_sessions': self._get_best_sessions(results),
                'risk_management': {
                    'recommended_risk_per_trade': '0.5-1%' if risk_level == 'HIGH' else '1-2%',
                    'stop_loss_factor': 'ATR-based',
                    'position_sizing': 'Conservative due to high volatility'
                }
            }
        }

        # Add strengths and weaknesses
        if sr_score >= 70:
            assessment['trading_recommendation']['primary_strengths'].append('Strong S/R respect')
        if ma_score >= 70:
            assessment['trading_recommendation']['primary_strengths'].append('Good MA confluence')
        if trend_score >= 60:
            assessment['trading_recommendation']['primary_strengths'].append('Clear trending behavior')
        if volatility >= 300:
            assessment['trading_recommendation']['primary_strengths'].append('High volatility for profits')

        if sr_score < 50:
            assessment['trading_recommendation']['main_weaknesses'].append('Weak S/R levels')
        if ma_score < 50:
            assessment['trading_recommendation']['main_weaknesses'].append('Poor MA respect')
        if volatility >= 400:
            assessment['trading_recommendation']['main_weaknesses'].append('Very high volatility/risk')

        return assessment

    def _get_best_sessions(self, results: Dict) -> List[str]:
        """Determine best trading sessions based on volatility"""
        session_data = results['market_structure_analysis']['session_analysis']

        # Sort by average range
        sorted_sessions = sorted(session_data.items(),
                               key=lambda x: x[1]['avg_range_pips'],
                               reverse=True)

        return [session for session, _ in sorted_sessions[:2]]  # Top 2 sessions

    def save_results(self, results: Dict, assessment: Dict):
        """Save analysis results to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save detailed results as JSON
        json_filename = f'/Users/user/fx-app/backend/xauusd_analysis_{timestamp}.json'
        combined_results = {**results, 'trading_assessment': assessment}

        with open(json_filename, 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)

        # Generate text report
        report_filename = f'/Users/user/fx-app/backend/xauusd_analysis_{timestamp}_report.txt'
        self.generate_text_report(results, assessment, report_filename)

        print(f"\nResults saved to:")
        print(f"JSON: {json_filename}")
        print(f"Report: {report_filename}")

        return json_filename, report_filename

    def generate_text_report(self, results: Dict, assessment: Dict, filename: str):
        """Generate comprehensive text report"""
        with open(filename, 'w') as f:
            f.write("XAU/USD COMPREHENSIVE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Analysis Date: {results['analysis_date']}\n")
            f.write(f"Data Period: {results['data_period']['start']} to {results['data_period']['end']}\n")
            f.write(f"Total Candles: {results['data_period']['hourly_candles']} hourly, {results['data_period']['daily_candles']} daily\n\n")

            # Executive Summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Confluence Score: {assessment['overall_confluence_score']}/100\n")
            f.write(f"Suitable for Confluence Trading: {'YES' if assessment['trading_recommendation']['suitable_for_confluence'] else 'NO'}\n")
            f.write(f"Volatility Level: {assessment['volatility_rating']['risk_level']}\n")
            f.write(f"Average Daily Range: {results['volatility_analysis']['mean_adr_pips']:.1f} pips\n\n")

            # Volatility Analysis
            f.write("1. VOLATILITY ANALYSIS\n")
            f.write("-" * 25 + "\n")
            vol = results['volatility_analysis']
            f.write(f"Mean ADR: {vol['mean_adr_pips']:.1f} pips\n")
            f.write(f"Median ADR: {vol['median_adr_pips']:.1f} pips\n")
            f.write(f"Standard Deviation: {vol['std_adr_pips']:.1f} pips\n")
            f.write(f"Minimum Range: {vol['min_range_pips']:.1f} pips\n")
            f.write(f"Maximum Range: {vol['max_range_pips']:.1f} pips\n")
            f.write(f"25th Percentile: {vol['adr_25_percentile']:.1f} pips\n")
            f.write(f"75th Percentile: {vol['adr_75_percentile']:.1f} pips\n\n")

            # Weekly Pattern
            f.write("Weekly Volatility Pattern:\n")
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for i, day in enumerate(days):
                if str(i) in vol['weekly_pattern']['mean']:
                    f.write(f"  {day}: {vol['weekly_pattern']['mean'][str(i)]:.1f} pips\n")
            f.write("\n")

            # Support/Resistance Analysis
            f.write("2. SUPPORT/RESISTANCE ANALYSIS\n")
            f.write("-" * 35 + "\n")
            sr = results['support_resistance_analysis']
            f.write(f"Overall Respect Rate: {sr['overall_respect_rate']:.1%}\n")
            f.write(f"Total Level Tests: {sr['total_tests']}\n")
            f.write(f"Successful Holds: {sr['successful_holds']}\n")
            f.write(f"Confluence Score: {assessment['confluence_breakdown']['Support/Resistance']:.1f}/100\n\n")

            # Moving Average Analysis
            f.write("3. MOVING AVERAGE ANALYSIS\n")
            f.write("-" * 30 + "\n")
            ma = results['moving_average_analysis']
            f.write(f"EMA Crossovers: {ma['ema_crossovers']['total_crossovers']}\n")
            f.write("MA Respect Rates:\n")
            for ma_name, data in ma['ma_respect'].items():
                f.write(f"  {ma_name}: {data['respect_rate']:.1%} ({data['respects']}/{data['touches']})\n")
            f.write(f"Confluence Score: {assessment['confluence_breakdown']['Moving Averages']:.1f}/100\n\n")

            # Technical Indicators
            f.write("4. TECHNICAL INDICATOR ANALYSIS\n")
            f.write("-" * 35 + "\n")
            ti = results['technical_indicator_analysis']
            rsi = ti['rsi_analysis']
            f.write(f"RSI Overbought Accuracy: {rsi['overbought_accuracy']:.1%}\n")
            f.write(f"RSI Oversold Accuracy: {rsi['oversold_accuracy']:.1%}\n")
            f.write(f"MACD Signals: {ti['macd_analysis']['total_signals']}\n")
            f.write(f"Bollinger Squeeze: {ti['bollinger_analysis']['squeeze_percentage']:.1f}% of time\n")
            f.write(f"Confluence Score: {assessment['confluence_breakdown']['RSI Signals']:.1f}/100\n\n")

            # Market Structure
            f.write("5. MARKET STRUCTURE ANALYSIS\n")
            f.write("-" * 32 + "\n")
            ms = results['market_structure_analysis']
            trend = ms['trend_analysis']
            f.write(f"Uptrending: {trend['uptrend_percentage']:.1f}%\n")
            f.write(f"Downtrending: {trend['downtrend_percentage']:.1f}%\n")
            f.write(f"Ranging: {trend['ranging_percentage']:.1f}%\n")
            f.write(f"Confluence Score: {assessment['confluence_breakdown']['Trending Behavior']:.1f}/100\n\n")

            # Session Analysis
            f.write("Best Trading Sessions:\n")
            for session, data in ms['session_analysis'].items():
                f.write(f"  {session}: {data['avg_range_pips']:.1f} pips avg range\n")
            f.write("\n")

            # Risk Analysis
            f.write("6. RISK ANALYSIS\n")
            f.write("-" * 18 + "\n")
            risk = results['risk_analysis']
            f.write(f"Daily Volatility: {risk['risk_metrics']['daily_volatility']:.2f}%\n")
            f.write(f"Annualized Volatility: {risk['risk_metrics']['annualized_volatility']:.2f}%\n")
            f.write(f"Maximum Drawdown: {risk['drawdown_analysis']['max_drawdown']:.2f}%\n")
            f.write(f"Win Rate: {risk['risk_metrics']['win_rate']:.1f}%\n")
            f.write(f"Weekend Gaps: {risk['gap_analysis']['weekend_gaps']} (avg {risk['gap_analysis']['avg_gap_size']:.2f}%)\n\n")

            # Trading Recommendations
            f.write("7. TRADING RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            rec = assessment['trading_recommendation']
            f.write(f"Suitable for Confluence Trading: {'YES' if rec['suitable_for_confluence'] else 'NO'}\n")
            f.write(f"Risk Level: {assessment['volatility_rating']['risk_level']}\n")
            f.write(f"Recommended Risk per Trade: {rec['risk_management']['recommended_risk_per_trade']}\n")
            f.write(f"Optimal Timeframes: {', '.join(rec['optimal_timeframes'])}\n")
            f.write(f"Best Sessions: {', '.join(rec['best_sessions'])}\n\n")

            f.write("Primary Strengths:\n")
            for strength in rec['primary_strengths']:
                f.write(f"  + {strength}\n")

            f.write("\nMain Weaknesses:\n")
            for weakness in rec['main_weaknesses']:
                f.write(f"  - {weakness}\n")

            f.write("\n" + "=" * 50 + "\n")
            f.write("END OF REPORT\n")

def main():
    """Main execution function"""
    API_KEY = "0e24ff3eb6ef415dba0cebcf04593e4f"

    analyzer = XAUUSDAnalyzer(API_KEY)

    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()

    if not results:
        print("Analysis failed - no data available")
        return

    # Generate trading assessment
    assessment = analyzer.generate_trading_assessment(results)

    # Save results
    json_file, report_file = analyzer.save_results(results, assessment)

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS - XAU/USD CONFLUENCE TRADING ANALYSIS")
    print("=" * 60)
    print(f"Overall Confluence Score: {assessment['overall_confluence_score']}/100")
    print(f"Average Daily Range: {results['volatility_analysis']['mean_adr_pips']:.1f} pips")
    print(f"Suitable for Confluence Trading: {'YES' if assessment['trading_recommendation']['suitable_for_confluence'] else 'NO'}")
    print(f"Risk Level: {assessment['volatility_rating']['risk_level']}")

    print("\nConfluence Breakdown:")
    for factor, score in assessment['confluence_breakdown'].items():
        print(f"  {factor}: {score:.1f}/100")

    if assessment['trading_recommendation']['primary_strengths']:
        print(f"\nPrimary Strengths:")
        for strength in assessment['trading_recommendation']['primary_strengths']:
            print(f"  + {strength}")

    if assessment['trading_recommendation']['main_weaknesses']:
        print(f"\nMain Weaknesses:")
        for weakness in assessment['trading_recommendation']['main_weaknesses']:
            print(f"  - {weakness}")

    print(f"\nBest Trading Sessions: {', '.join(assessment['trading_recommendation']['best_sessions'])}")
    print(f"Recommended Risk per Trade: {assessment['trading_recommendation']['risk_management']['recommended_risk_per_trade']}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()