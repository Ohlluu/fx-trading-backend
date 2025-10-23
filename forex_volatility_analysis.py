#!/usr/bin/env python3
"""
Forex Volatility Analysis using TwelveData API
Analyzes 2 years of historical data to identify most volatile and profitable forex pairs
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

class ForexVolatilityAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com/time_series"
        self.results = {}

        # Define forex pairs by tier (using correct format with /)
        self.tiers = {
            "Tier 1 - Extreme Volatility": ["USD/ZAR", "USD/TRY", "USD/BRL", "USD/MXN"],
            "Tier 2 - High Volatility Cross Pairs": ["GBP/JPY", "AUD/JPY", "NZD/JPY", "GBP/AUD", "CAD/JPY"],
            "Tier 3 - Volatile Majors": ["USD/CAD", "GBP/USD", "EUR/USD"]
        }

        # Pip values for different pair types
        self.pip_values = {
            "JPY": 0.01,    # For pairs with JPY
            "OTHER": 0.0001 # For other pairs
        }

    def get_pip_value(self, pair: str) -> float:
        """Get pip value based on pair type"""
        return self.pip_values["JPY"] if "JPY" in pair else self.pip_values["OTHER"]

    def fetch_forex_data(self, symbol: str, outputsize: int = 730) -> Optional[pd.DataFrame]:
        """Fetch forex data from TwelveData API"""
        params = {
            'symbol': symbol,
            'interval': '1day',
            'outputsize': outputsize,
            'apikey': self.api_key
        }

        try:
            print(f"Fetching data for {symbol}...")
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'values' not in data:
                print(f"No data available for {symbol}")
                if 'message' in data:
                    print(f"API Message: {data['message']}")
                print(f"Response keys: {list(data.keys())}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime').sort_index()

            # Convert price columns to float
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            print(f"Successfully fetched {len(df)} records for {symbol}")
            return df

        except Exception as e:
            print(f"Error fetching data for {symbol}: {str(e)}")
            return None

    def calculate_daily_range_pips(self, df: pd.DataFrame, symbol: str) -> pd.Series:
        """Calculate daily range in pips"""
        pip_value = self.get_pip_value(symbol)
        daily_range = (df['high'] - df['low']) / pip_value
        return daily_range

    def calculate_volatility_metrics(self, df: pd.DataFrame, symbol: str) -> Dict:
        """Calculate comprehensive volatility metrics"""
        if df is None or len(df) == 0:
            return None

        pip_value = self.get_pip_value(symbol)

        # Daily range in pips
        daily_range_pips = self.calculate_daily_range_pips(df, symbol)

        # Daily returns (close to close)
        daily_returns = df['close'].pct_change().dropna()

        # Daily move in pips (absolute change)
        daily_move_pips = (df['close'].diff().abs() / pip_value).dropna()

        # High-Low range statistics
        metrics = {
            'symbol': symbol,
            'data_points': len(df),
            'date_range': f"{df.index[0].date()} to {df.index[-1].date()}",

            # Average Daily Range (ADR) metrics
            'adr_pips': daily_range_pips.mean(),
            'adr_median_pips': daily_range_pips.median(),
            'adr_std_pips': daily_range_pips.std(),

            # Daily move metrics (close-to-close)
            'avg_daily_move_pips': daily_move_pips.mean(),
            'daily_move_std_pips': daily_move_pips.std(),

            # Maximum moves
            'max_daily_range_pips': daily_range_pips.max(),
            'min_daily_range_pips': daily_range_pips.min(),
            'max_positive_move_pips': (df['close'].diff() / pip_value).max(),
            'max_negative_move_pips': (df['close'].diff() / pip_value).min(),

            # Volatility consistency
            'days_above_100_pips': (daily_range_pips > 100).sum(),
            'days_above_200_pips': (daily_range_pips > 200).sum(),
            'days_above_300_pips': (daily_range_pips > 300).sum(),
            'pct_days_above_100_pips': (daily_range_pips > 100).mean() * 100,
            'pct_days_above_200_pips': (daily_range_pips > 200).mean() * 100,

            # Returns-based volatility
            'daily_return_volatility': daily_returns.std() * np.sqrt(252) * 100,  # Annualized
            'max_daily_return': daily_returns.max() * 100,
            'min_daily_return': daily_returns.min() * 100,

            # Trend analysis
            'positive_days': (df['close'].diff() > 0).sum(),
            'negative_days': (df['close'].diff() < 0).sum(),
            'trend_ratio': (df['close'].diff() > 0).mean(),

            # Data quality
            'missing_days': daily_range_pips.isna().sum(),
            'data_quality_score': (1 - daily_range_pips.isna().mean()) * 100
        }

        return metrics

    def analyze_all_pairs(self) -> Dict:
        """Analyze all forex pairs across all tiers"""
        all_results = {}

        for tier_name, pairs in self.tiers.items():
            print(f"\n=== Analyzing {tier_name} ===")
            tier_results = {}

            for pair in pairs:
                # Add delay to respect API rate limits
                time.sleep(1)

                # Fetch data
                df = self.fetch_forex_data(pair)

                if df is not None:
                    # Calculate metrics
                    metrics = self.calculate_volatility_metrics(df, pair)
                    tier_results[pair] = metrics
                else:
                    tier_results[pair] = None

            all_results[tier_name] = tier_results

        self.results = all_results
        return all_results

    def create_ranking_table(self) -> pd.DataFrame:
        """Create a comprehensive ranking table"""
        all_metrics = []

        for tier_name, tier_data in self.results.items():
            for pair, metrics in tier_data.items():
                if metrics is not None:
                    metrics['tier'] = tier_name
                    all_metrics.append(metrics)

        if not all_metrics:
            return pd.DataFrame()

        df = pd.DataFrame(all_metrics)

        # Round numerical columns for better display
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].round(2)

        # Sort by ADR (Average Daily Range)
        df = df.sort_values('adr_pips', ascending=False)

        return df

    def get_top_recommendations(self, n: int = 5) -> List[Dict]:
        """Get top N recommended pairs with reasoning"""
        df = self.create_ranking_table()

        if df.empty:
            return []

        recommendations = []

        for idx, row in df.head(n).iterrows():
            reasoning = []

            # ADR analysis
            if row['adr_pips'] > 200:
                reasoning.append(f"Excellent ADR of {row['adr_pips']:.0f} pips")
            elif row['adr_pips'] > 150:
                reasoning.append(f"Good ADR of {row['adr_pips']:.0f} pips")
            else:
                reasoning.append(f"Moderate ADR of {row['adr_pips']:.0f} pips")

            # Consistency analysis
            if row['pct_days_above_100_pips'] > 70:
                reasoning.append(f"High consistency ({row['pct_days_above_100_pips']:.1f}% days >100 pips)")
            elif row['pct_days_above_100_pips'] > 50:
                reasoning.append(f"Good consistency ({row['pct_days_above_100_pips']:.1f}% days >100 pips)")

            # Volatility analysis
            if row['daily_return_volatility'] > 20:
                reasoning.append(f"High volatility ({row['daily_return_volatility']:.1f}% annualized)")

            # Data quality
            if row['data_quality_score'] > 95:
                reasoning.append("Excellent data quality")
            elif row['data_quality_score'] > 90:
                reasoning.append("Good data quality")
            else:
                reasoning.append(f"Fair data quality ({row['data_quality_score']:.1f}%)")

            # Trend characteristics
            trend_bias = "balanced" if 0.45 <= row['trend_ratio'] <= 0.55 else ("bullish" if row['trend_ratio'] > 0.55 else "bearish")
            reasoning.append(f"Trend profile: {trend_bias} ({row['trend_ratio']*100:.1f}% up days)")

            recommendations.append({
                'rank': len(recommendations) + 1,
                'pair': row['symbol'],
                'tier': row['tier'],
                'adr_pips': row['adr_pips'],
                'consistency': row['pct_days_above_100_pips'],
                'volatility': row['daily_return_volatility'],
                'data_quality': row['data_quality_score'],
                'reasoning': "; ".join(reasoning)
            })

        return recommendations

    def generate_report(self) -> str:
        """Generate comprehensive analysis report"""
        if not self.results:
            return "No analysis results available."

        report = []
        report.append("=" * 80)
        report.append("FOREX VOLATILITY ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data Period: ~2 years (730 trading days)")
        report.append(f"API: TwelveData")
        report.append("")

        # Summary statistics
        df = self.create_ranking_table()
        if not df.empty:
            report.append("SUMMARY STATISTICS:")
            report.append(f"Total pairs analyzed: {len(df)}")
            report.append(f"Average ADR across all pairs: {df['adr_pips'].mean():.1f} pips")
            report.append(f"Highest ADR: {df['adr_pips'].max():.1f} pips ({df.loc[df['adr_pips'].idxmax(), 'symbol']})")
            report.append(f"Average data quality: {df['data_quality_score'].mean():.1f}%")
            report.append("")

        # Ranking table
        report.append("COMPLETE RANKING TABLE:")
        report.append("-" * 120)

        if not df.empty:
            # Select key columns for the ranking table
            display_cols = ['symbol', 'tier', 'adr_pips', 'avg_daily_move_pips', 'daily_return_volatility',
                          'pct_days_above_100_pips', 'max_daily_range_pips', 'data_quality_score', 'data_points']

            display_df = df[display_cols].copy()
            display_df.columns = ['Pair', 'Tier', 'ADR(pips)', 'Avg Move(pips)', 'Vol(%)',
                                '% >100pips', 'Max Range(pips)', 'Data Quality(%)', 'Records']

            report.append(display_df.to_string(index=False))

        report.append("")
        report.append("-" * 120)

        # Top recommendations
        recommendations = self.get_top_recommendations()
        if recommendations:
            report.append("\nTOP 5 RECOMMENDED PAIRS:")
            report.append("=" * 50)

            for rec in recommendations:
                report.append(f"\n{rec['rank']}. {rec['pair']} ({rec['tier']})")
                report.append(f"   ADR: {rec['adr_pips']:.0f} pips | Consistency: {rec['consistency']:.1f}% | Volatility: {rec['volatility']:.1f}%")
                report.append(f"   Data Quality: {rec['data_quality']:.1f}%")
                report.append(f"   Reasoning: {rec['reasoning']}")

        # Data quality issues
        report.append("\n\nDATA QUALITY ASSESSMENT:")
        report.append("=" * 30)

        quality_issues = []
        for tier_name, tier_data in self.results.items():
            for pair, metrics in tier_data.items():
                if metrics is None:
                    quality_issues.append(f"{pair}: No data available")
                elif metrics['data_quality_score'] < 90:
                    quality_issues.append(f"{pair}: {metrics['data_quality_score']:.1f}% quality ({metrics['missing_days']} missing days)")

        if quality_issues:
            report.append("Issues found:")
            for issue in quality_issues:
                report.append(f"- {issue}")
        else:
            report.append("No significant data quality issues found.")

        # Trading recommendations
        report.append("\n\nTRADING RECOMMENDATIONS:")
        report.append("=" * 30)
        report.append("For retail trading, consider:")
        report.append("• Pairs with ADR > 100 pips for sufficient movement")
        report.append("• High consistency (>50% days above 100 pips)")
        report.append("• Good data quality (>90%)")
        report.append("• Be aware that extreme volatility pairs may have wider spreads")
        report.append("• Cross pairs (especially JPY pairs) often show strong trending behavior")

        return "\n".join(report)

    def save_results(self, filename: str = None):
        """Save analysis results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"forex_volatility_analysis_{timestamp}.json"

        # Save raw results
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save report
        report_filename = filename.replace('.json', '_report.txt')
        with open(report_filename, 'w') as f:
            f.write(self.generate_report())

        print(f"Results saved to {filename}")
        print(f"Report saved to {report_filename}")

        return filename, report_filename


def main():
    """Main execution function"""
    API_KEY = "0e24ff3eb6ef415dba0cebcf04593e4f"

    print("Starting Forex Volatility Analysis...")
    print("This will analyze 2 years of data for multiple forex pairs")
    print("Please wait as this may take several minutes due to API rate limits...")

    # Create analyzer
    analyzer = ForexVolatilityAnalyzer(API_KEY)

    # Run analysis
    results = analyzer.analyze_all_pairs()

    # Generate and display report
    report = analyzer.generate_report()
    print("\n" + report)

    # Save results
    analyzer.save_results()

    return analyzer


if __name__ == "__main__":
    analyzer = main()