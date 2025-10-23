#!/usr/bin/env python3
"""
Enhanced Forex Confluence Analysis
Deep dive analysis to find the best confluence-based trading opportunities
from our volatile forex pairs, considering session times and refined metrics.
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime

def load_volatility_data():
    """Load our previous volatility analysis"""
    with open('/Users/user/fx-app/backend/forex_volatility_analysis_20250923_120655.json', 'r') as f:
        return json.load(f)

def load_confluence_data():
    """Load our confluence analysis results"""
    with open('/Users/user/fx-app/backend/forex_confluence_analysis_20250923_123215.json', 'r') as f:
        return json.load(f)

def create_enhanced_summary():
    """Create enhanced analysis combining volatility and confluence data"""

    volatility_data = load_volatility_data()
    confluence_data = load_confluence_data()

    # Extract volatility metrics for our pairs
    volatility_summary = {}
    for tier, pairs in volatility_data.items():
        for pair, metrics in pairs.items():
            if metrics:
                volatility_summary[pair] = {
                    'adr_pips': metrics['adr_pips'],
                    'pct_days_above_100_pips': metrics['pct_days_above_100_pips'],
                    'daily_return_volatility': metrics['daily_return_volatility'],
                    'tier': tier
                }

    # Combine with confluence data
    enhanced_analysis = []

    for pair, confluence_result in confluence_data.items():
        if confluence_result and pair in volatility_summary:
            vol_data = volatility_summary[pair]

            # Calculate composite scores
            volatility_score = min(100, vol_data['adr_pips'] / 30)  # Normalize ADR to 0-100 scale
            confluence_score = confluence_result['confluence_score']

            # Create risk-adjusted opportunity score
            # Higher volatility is good for profits, but needs to be balanced with confluence
            opportunity_score = (volatility_score * 0.4) + (confluence_score * 0.6)

            # Identify key strengths and weaknesses
            strengths = []
            weaknesses = []

            # Volatility strengths
            if vol_data['adr_pips'] > 150:
                strengths.append(f"Excellent volatility ({vol_data['adr_pips']:.0f} pips ADR)")
            elif vol_data['adr_pips'] > 80:
                strengths.append(f"Good volatility ({vol_data['adr_pips']:.0f} pips ADR)")

            if vol_data['pct_days_above_100_pips'] > 70:
                strengths.append(f"Consistent movement ({vol_data['pct_days_above_100_pips']:.1f}% days >100 pips)")

            # Confluence strengths
            if confluence_result['support_resistance']['respect_score'] > 50:
                strengths.append(f"Decent S/R respect ({confluence_result['support_resistance']['respect_score']:.1f}%)")
            else:
                weaknesses.append(f"Poor S/R respect ({confluence_result['support_resistance']['respect_score']:.1f}%)")

            if confluence_result['trend_quality']['trend_quality_score'] > 70:
                strengths.append(f"Strong trend quality ({confluence_result['trend_quality']['trend_quality_score']:.1f}%)")
            elif confluence_result['trend_quality']['trend_quality_score'] < 60:
                weaknesses.append(f"Weak trend quality ({confluence_result['trend_quality']['trend_quality_score']:.1f}%)")

            if confluence_result['signal_noise']['false_breakout_rate'] > 90:
                weaknesses.append(f"Very high false breakouts ({confluence_result['signal_noise']['false_breakout_rate']:.1f}%)")

            if confluence_result['moving_averages']['crossover_success_rate'] < 20:
                weaknesses.append("Poor MA signal reliability")

            # Trading recommendation
            if opportunity_score >= 65:
                recommendation = "EXCELLENT - High profit potential with manageable risk"
            elif opportunity_score >= 55:
                recommendation = "GOOD - Suitable for experienced traders"
            elif opportunity_score >= 45:
                recommendation = "MODERATE - Use smaller positions, focus on best setups"
            elif opportunity_score >= 35:
                recommendation = "CAUTION - Only trade with strong confluences"
            else:
                recommendation = "AVOID - Poor risk/reward profile"

            enhanced_analysis.append({
                'pair': pair,
                'tier': vol_data['tier'],
                'opportunity_score': opportunity_score,
                'volatility_score': volatility_score,
                'confluence_score': confluence_score,
                'adr_pips': vol_data['adr_pips'],
                'consistency': vol_data['pct_days_above_100_pips'],
                'sr_respect': confluence_result['support_resistance']['respect_score'],
                'trend_quality': confluence_result['trend_quality']['trend_quality_score'],
                'false_breakout_rate': confluence_result['signal_noise']['false_breakout_rate'],
                'technical_dominance': confluence_result['fundamental_vs_technical']['technical_dominance_score'],
                'recommendation': recommendation,
                'strengths': strengths,
                'weaknesses': weaknesses
            })

    # Sort by opportunity score
    enhanced_analysis.sort(key=lambda x: x['opportunity_score'], reverse=True)

    return enhanced_analysis

def generate_refined_recommendations():
    """Generate refined trading recommendations"""
    enhanced_data = create_enhanced_summary()

    report = []
    report.append("=" * 100)
    report.append("REFINED FOREX CONFLUENCE TRADING RECOMMENDATIONS")
    report.append("=" * 100)
    report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # Executive Summary
    excellent_pairs = [p for p in enhanced_data if p['opportunity_score'] >= 65]
    good_pairs = [p for p in enhanced_data if 55 <= p['opportunity_score'] < 65]
    moderate_pairs = [p for p in enhanced_data if 45 <= p['opportunity_score'] < 55]

    report.append("EXECUTIVE SUMMARY:")
    report.append("=" * 20)
    report.append(f"• {len(excellent_pairs)} pairs with excellent opportunity scores (≥65)")
    report.append(f"• {len(good_pairs)} pairs with good opportunity scores (55-64)")
    report.append(f"• {len(moderate_pairs)} pairs with moderate opportunity scores (45-54)")
    report.append("")

    # Key Findings
    report.append("KEY FINDINGS:")
    report.append("=" * 15)
    report.append("❌ MAJOR ISSUE: All analyzed pairs show poor technical confluence!")
    report.append("❌ Extremely high false breakout rates (85-100%)")
    report.append("❌ Very poor moving average signal reliability (0-17%)")
    report.append("❌ Weak support/resistance respect (12-54%)")
    report.append("")
    report.append("⚠️  This suggests these volatile pairs are NOT suitable for traditional")
    report.append("    confluence-based trading strategies!")
    report.append("")

    # Alternative Approach
    report.append("RECOMMENDED ALTERNATIVE APPROACH:")
    report.append("=" * 35)
    report.append("Given the poor confluence results, consider these strategies:")
    report.append("")
    report.append("1. NEWS/FUNDAMENTAL TRADING:")
    report.append("   • Extreme volatility pairs (USD/TRY, USD/ZAR) are news-driven")
    report.append("   • Trade economic announcements and geopolitical events")
    report.append("   • Use fundamental analysis rather than technical confluence")
    report.append("")
    report.append("2. BREAKOUT TRADING (with caution):")
    report.append("   • High false breakout rates suggest waiting for confirmed breaks")
    report.append("   • Use larger stop losses to account for volatility")
    report.append("   • Focus on major support/resistance breaks only")
    report.append("")
    report.append("3. MEAN REVERSION ON MAJORS:")
    report.append("   • EUR/USD, GBP/USD show better trend quality")
    report.append("   • Use oversold/overbought conditions for entries")
    report.append("   • Shorter time frames may show better confluence")
    report.append("")

    # Detailed Rankings
    report.append("DETAILED PAIR RANKINGS:")
    report.append("=" * 25)

    for i, pair_data in enumerate(enhanced_data, 1):
        report.append(f"\n{i}. {pair_data['pair']} - {pair_data['recommendation']}")
        report.append(f"   Opportunity Score: {pair_data['opportunity_score']:.1f}/100")
        report.append(f"   ADR: {pair_data['adr_pips']:.0f} pips | S/R Respect: {pair_data['sr_respect']:.1f}% | Trend Quality: {pair_data['trend_quality']:.1f}%")

        if pair_data['strengths']:
            report.append(f"   ✅ Strengths: {'; '.join(pair_data['strengths'])}")

        if pair_data['weaknesses']:
            report.append(f"   ❌ Weaknesses: {'; '.join(pair_data['weaknesses'])}")

    # Strategic Recommendations
    report.append("\n\nSTRATEGIC TRADING RECOMMENDATIONS:")
    report.append("=" * 40)

    # Top 3 pairs for different strategies
    top_3 = enhanced_data[:3]

    report.append("FOR VOLATILITY-BASED STRATEGIES:")
    for i, pair in enumerate(top_3, 1):
        report.append(f"{i}. {pair['pair']} - {pair['adr_pips']:.0f} pips ADR")

    report.append("\nFOR TECHNICAL CONFLUENCE (Limited Options):")
    confluence_sorted = sorted(enhanced_data, key=lambda x: x['confluence_score'], reverse=True)[:3]
    for i, pair in enumerate(confluence_sorted, 1):
        report.append(f"{i}. {pair['pair']} - {pair['confluence_score']:.1f}% confluence score")

    report.append("\nRECOMMENDED TRADING APPROACH:")
    report.append("=" * 30)
    report.append("Given the analysis results, we recommend:")
    report.append("")
    report.append("1. AVOID pure technical confluence trading on these pairs")
    report.append("2. FOCUS on fundamental analysis and news trading")
    report.append("3. IF trading technically:")
    report.append("   • Use multiple timeframe confirmation")
    report.append("   • Wait for extreme oversold/overbought conditions")
    report.append("   • Use wider stops to account for volatility")
    report.append("   • Consider lower position sizes")
    report.append("")
    report.append("4. BEST OPPORTUNITIES:")
    report.append(f"   • {enhanced_data[0]['pair']}: Highest opportunity score but requires news-based approach")
    report.append(f"   • {enhanced_data[1]['pair']}: Good volatility with moderate technical respect")
    report.append(f"   • {enhanced_data[2]['pair']}: Balance of volatility and some technical reliability")
    report.append("")

    # Risk Management
    report.append("CRITICAL RISK MANAGEMENT:")
    report.append("=" * 30)
    report.append("• Position sizes should be 50% smaller than usual due to poor confluence")
    report.append("• Stop losses must account for high volatility (2-3x normal distances)")
    report.append("• Avoid trading during major news events for technical strategies")
    report.append("• Consider using options strategies for high volatility pairs")
    report.append("• Never risk more than 1% per trade on these pairs")

    return "\n".join(report), enhanced_data

def create_alternative_pair_recommendations():
    """Suggest alternative lower-volatility pairs that might have better confluence"""

    alternatives = [
        {
            'pair': 'AUD/USD',
            'reasoning': 'Major pair with typically better technical respect than extreme volatility pairs',
            'expected_adr': '50-70 pips',
            'confluence_expectation': 'Moderate to Good'
        },
        {
            'pair': 'USD/JPY',
            'reasoning': 'Strong trending characteristics and institutional participation',
            'expected_adr': '60-80 pips',
            'confluence_expectation': 'Good'
        },
        {
            'pair': 'EUR/GBP',
            'reasoning': 'Cross pair with good technical characteristics during trending markets',
            'expected_adr': '40-60 pips',
            'confluence_expectation': 'Moderate'
        },
        {
            'pair': 'NZD/USD',
            'reasoning': 'Commodity currency with cleaner technical patterns',
            'expected_adr': '45-65 pips',
            'confluence_expectation': 'Moderate to Good'
        }
    ]

    report = []
    report.append("\n" + "=" * 80)
    report.append("ALTERNATIVE PAIRS FOR CONFLUENCE TRADING")
    report.append("=" * 80)
    report.append("Consider these pairs instead of the high-volatility ones analyzed:")
    report.append("")

    for alt in alternatives:
        report.append(f"• {alt['pair']}")
        report.append(f"  Expected ADR: {alt['expected_adr']}")
        report.append(f"  Confluence Expectation: {alt['confluence_expectation']}")
        report.append(f"  Reasoning: {alt['reasoning']}")
        report.append("")

    report.append("RECOMMENDATION: Test these pairs with the same confluence analysis")
    report.append("to find better technical trading opportunities.")

    return "\n".join(report)

def main():
    """Main execution"""
    print("Creating enhanced confluence analysis...")

    # Generate refined recommendations
    refined_report, enhanced_data = generate_refined_recommendations()

    # Add alternative recommendations
    alternatives = create_alternative_pair_recommendations()

    # Complete report
    complete_report = refined_report + alternatives

    print(complete_report)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save enhanced data
    with open(f'enhanced_forex_analysis_{timestamp}.json', 'w') as f:
        json.dump(enhanced_data, f, indent=2, default=str)

    # Save report
    with open(f'enhanced_forex_analysis_{timestamp}_report.txt', 'w') as f:
        f.write(complete_report)

    print(f"\nEnhanced analysis saved to enhanced_forex_analysis_{timestamp}.json")
    print(f"Detailed report saved to enhanced_forex_analysis_{timestamp}_report.txt")

    return enhanced_data

if __name__ == "__main__":
    main()