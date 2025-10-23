#!/usr/bin/env python3
"""
Advanced XAUUSD Backtesting - Find REAL High Win Rate Confluence
Test professional confluences that actually work in Gold trading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from typing import Dict, List, Any, Optional
import asyncio
import httpx

def load_data():
    """Load existing backtest data"""
    try:
        df = pd.read_csv('data/XAUUSD_BACKTEST_DATA.csv')
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['ny_time'] = pd.to_datetime(df['ny_time'])
        return df
    except:
        print("❌ Run backtest_xauusd.py first to get data")
        return None

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators that professional traders use"""

    # Moving averages
    df['ema_20'] = df['close'].ewm(span=20).mean()
    df['ema_50'] = df['close'].ewm(span=50).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()

    # Bollinger Bands
    rolling_mean = df['close'].rolling(window=20).mean()
    rolling_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = rolling_mean + (rolling_std * 2)
    df['bb_lower'] = rolling_mean - (rolling_std * 2)
    df['bb_middle'] = rolling_mean

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df['atr'] = true_range.rolling(window=14).mean()

    # Volume proxy (using volatility as volume substitute)
    df['volume_proxy'] = df['high'] - df['low']
    df['volume_ma'] = df['volume_proxy'].rolling(window=20).mean()

    # Support/Resistance levels (swing highs/lows)
    df['swing_high'] = df['high'].rolling(window=5, center=True).apply(
        lambda x: x[2] if x[2] == max(x) else np.nan, raw=True)
    df['swing_low'] = df['low'].rolling(window=5, center=True).apply(
        lambda x: x[2] if x[2] == min(x) else np.nan, raw=True)

    return df

def find_key_levels(df: pd.DataFrame, lookback: int = 200) -> Dict[str, List[float]]:
    """Find key psychological and technical levels"""

    recent_data = df.tail(lookback)

    # Round numbers (every $25 for XAUUSD)
    min_price = int(recent_data['low'].min() / 25) * 25
    max_price = int(recent_data['high'].max() / 25) * 25 + 25
    round_numbers = list(range(min_price, max_price + 1, 25))

    # Swing levels
    swing_highs = recent_data['swing_high'].dropna().tolist()
    swing_lows = recent_data['swing_low'].dropna().tolist()

    # Daily highs/lows
    recent_data_copy = recent_data.copy()
    recent_data_copy['ny_date_str'] = recent_data_copy['ny_time'].astype(str).str[:10]
    daily_data = recent_data_copy.groupby('ny_date_str').agg({
        'high': 'max',
        'low': 'min'
    })
    daily_highs = daily_data['high'].tolist()
    daily_lows = daily_data['low'].tolist()

    return {
        'round_numbers': round_numbers,
        'swing_highs': swing_highs,
        'swing_lows': swing_lows,
        'daily_highs': daily_highs,
        'daily_lows': daily_lows
    }

def check_confluence_factors(candle: pd.Series, df: pd.DataFrame, levels: Dict) -> Dict[str, Any]:
    """Check multiple confluence factors for high probability setups"""

    idx = candle.name
    current_price = candle['close']

    confluence_score = 0
    confluence_factors = []

    # 1. EMA Alignment (trend confluence)
    if candle['ema_20'] > candle['ema_50'] > candle['ema_200']:
        if current_price > candle['ema_20']:
            confluence_score += 2
            confluence_factors.append("Bullish EMA alignment")
    elif candle['ema_20'] < candle['ema_50'] < candle['ema_200']:
        if current_price < candle['ema_20']:
            confluence_score += 2
            confluence_factors.append("Bearish EMA alignment")

    # 2. RSI confluence
    rsi = candle['rsi']
    if 30 <= rsi <= 70:  # Avoid overbought/oversold extremes
        confluence_score += 1
        confluence_factors.append("RSI neutral zone")

    # 3. Bollinger Band position
    if candle['bb_lower'] <= current_price <= candle['bb_upper']:
        confluence_score += 1
        confluence_factors.append("Within Bollinger Bands")

    # 4. Volume confluence (high volatility = high volume proxy)
    if candle['volume_proxy'] > candle['volume_ma'] * 1.5:
        confluence_score += 2
        confluence_factors.append("High volume breakout")

    # 5. Key level confluence (most important)
    level_confluence = 0
    tolerance = 10  # $10 tolerance

    for level_type, level_list in levels.items():
        for level in level_list:
            if abs(current_price - level) <= tolerance:
                level_confluence += 1
                confluence_factors.append(f"Near {level_type[:-1]} ${level:.0f}")
                break

    confluence_score += level_confluence * 3  # Weight key levels heavily

    # 6. Session timing confluence
    hour = candle['ny_hour']
    if hour in [5, 10]:  # London Fix
        confluence_score += 2
        confluence_factors.append("London Fix time")
    elif hour in [8, 9, 10, 11]:  # London session
        confluence_score += 1
        confluence_factors.append("London session")

    # 7. ATR confluence (sufficient volatility)
    if candle['atr'] > 15:  # Minimum $15 ATR
        confluence_score += 1
        confluence_factors.append("High ATR environment")

    # 8. Candle pattern confluence
    body_size = abs(candle['close'] - candle['open'])
    total_range = candle['high'] - candle['low']

    if body_size > total_range * 0.7:  # Strong directional candle
        confluence_score += 2
        confluence_factors.append("Strong directional candle")

    return {
        'confluence_score': confluence_score,
        'confluence_factors': confluence_factors,
        'level_confluence': level_confluence
    }

def advanced_signal_detection(candle: pd.Series, df: pd.DataFrame, levels: Dict) -> Optional[Dict]:
    """Detect high-confluence trading signals"""

    # Get confluence analysis
    confluence = check_confluence_factors(candle, df, levels)

    # Require minimum confluence score
    min_score = 8  # High bar for entry

    if confluence['confluence_score'] < min_score:
        return None

    current_price = candle['close']
    direction = None
    entry_price = None

    # Determine direction based on EMA and price action
    if (candle['close'] > candle['open'] and
        candle['close'] > candle['ema_20'] and
        "Bullish EMA alignment" in confluence['confluence_factors']):
        direction = "BUY"
        entry_price = candle['high'] + 2  # Enter on breakout

    elif (candle['close'] < candle['open'] and
          candle['close'] < candle['ema_20'] and
          "Bearish EMA alignment" in confluence['confluence_factors']):
        direction = "SELL"
        entry_price = candle['low'] - 2  # Enter on breakdown

    if not direction:
        return None

    return {
        'direction': direction,
        'entry_price': entry_price,
        'confluence_score': confluence['confluence_score'],
        'confluence_factors': confluence['confluence_factors'],
        'strategy': 'advanced_confluence'
    }

def run_advanced_backtest(df: pd.DataFrame) -> Dict[str, Any]:
    """Run advanced confluence backtesting"""

    print("🎯 RUNNING ADVANCED CONFLUENCE BACKTEST")
    print("=" * 60)

    # Add technical indicators
    print("📊 Calculating technical indicators...")
    df = calculate_technical_indicators(df)

    # Get key levels
    print("🔍 Identifying key levels...")
    levels = find_key_levels(df)

    trades = []
    daily_trades = {}

    print("📈 Scanning for high-confluence signals...")

    # Group by date and scan for signals
    df['ny_date_str'] = df['ny_time'].astype(str).str[:10]  # Get date string YYYY-MM-DD
    for date, day_data in df.groupby('ny_date_str'):
        if date not in daily_trades:
            daily_trades[date] = 0

        day_df = df[df['ny_date_str'] == date].copy()

        for i, (idx, candle) in enumerate(day_df.iterrows()):
            # Skip if daily limit reached
            if daily_trades[date] >= 2:  # Lower daily limit for quality
                continue

            # Need sufficient data for indicators
            if idx < 200:
                continue

            # Get signal
            signal = advanced_signal_detection(candle, df.iloc[:idx], levels)

            if signal:
                # Get future data for trade simulation
                future_data = df.iloc[idx+1:idx+25]
                if len(future_data) < 5:
                    continue

                # Simulate trade with tighter stops
                trade_result = simulate_advanced_trade(signal, future_data, candle)

                if trade_result:
                    trade_record = {
                        'date': date,
                        'entry_time': candle['ny_time'],
                        'direction': signal['direction'],
                        'entry_price': signal['entry_price'],
                        'confluence_score': signal['confluence_score'],
                        'confluence_count': len(signal['confluence_factors']),
                        **trade_result
                    }

                    trades.append(trade_record)
                    daily_trades[date] += 1

                    result_emoji = "✅" if trade_result['result'] == 'WIN' else "❌"
                    print(f"{result_emoji} Trade {len(trades)}: {date} {signal['direction']} @${signal['entry_price']:.2f} "
                          f"[Score: {signal['confluence_score']}] → {trade_result['result']} ${trade_result['profit']:+.2f}")

    return analyze_advanced_results(trades)

def simulate_advanced_trade(signal: Dict, future_data: pd.DataFrame, entry_candle: pd.Series) -> Optional[Dict]:
    """Simulate trade with advanced exit logic"""

    entry_price = signal['entry_price']
    direction = signal['direction']

    # Dynamic stop loss based on ATR
    atr = entry_candle['atr']
    risk_multiplier = 1.5  # 1.5x ATR risk
    reward_multiplier = 3.0  # 3:1 R/R for high confluence

    if direction == "BUY":
        stop_loss = entry_price - (atr * risk_multiplier)
        take_profit = entry_price + (atr * risk_multiplier * reward_multiplier)
    else:
        stop_loss = entry_price + (atr * risk_multiplier)
        take_profit = entry_price - (atr * risk_multiplier * reward_multiplier)

    # Check trade outcome
    for i, (_, candle) in enumerate(future_data.iterrows()):
        hours_elapsed = i + 1

        if direction == "BUY":
            if candle['high'] >= take_profit:
                profit = atr * risk_multiplier * reward_multiplier
                return {
                    'result': 'WIN',
                    'exit_price': take_profit,
                    'profit': profit,
                    'duration_hours': hours_elapsed,
                    'exit_reason': 'take_profit'
                }
            elif candle['low'] <= stop_loss:
                loss = -atr * risk_multiplier
                return {
                    'result': 'LOSS',
                    'exit_price': stop_loss,
                    'profit': loss,
                    'duration_hours': hours_elapsed,
                    'exit_reason': 'stop_loss'
                }
        else:  # SELL
            if candle['low'] <= take_profit:
                profit = atr * risk_multiplier * reward_multiplier
                return {
                    'result': 'WIN',
                    'exit_price': take_profit,
                    'profit': profit,
                    'duration_hours': hours_elapsed,
                    'exit_reason': 'take_profit'
                }
            elif candle['high'] >= stop_loss:
                loss = -atr * risk_multiplier
                return {
                    'result': 'LOSS',
                    'exit_price': stop_loss,
                    'profit': loss,
                    'duration_hours': hours_elapsed,
                    'exit_reason': 'stop_loss'
                }

        # Max 12 hour trade duration
        if hours_elapsed >= 12:
            break

    return None  # No clear outcome

def analyze_advanced_results(trades: List[Dict]) -> Dict[str, Any]:
    """Analyze advanced backtest results"""

    if not trades:
        print("❌ No high-confluence trades found")
        return {}

    print(f"\n🏆 ADVANCED CONFLUENCE RESULTS ({len(trades)} trades)")
    print("=" * 60)

    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']

    win_count = len(wins)
    total_trades = len(trades)
    win_rate = (win_count / total_trades) * 100

    total_profit = sum(t['profit'] for t in trades)
    avg_win = sum(t['profit'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['profit'] for t in losses) / len(losses) if losses else 0

    avg_confluence = sum(t['confluence_score'] for t in trades) / len(trades)

    print(f"🎯 WIN RATE: {win_rate:.1f}% ({win_count}/{total_trades})")
    print(f"💰 TOTAL PROFIT: ${total_profit:+.2f}")
    print(f"📊 AVERAGE WIN: ${avg_win:.2f}")
    print(f"📊 AVERAGE LOSS: ${avg_loss:.2f}")
    print(f"⚖️  PROFIT FACTOR: {abs(avg_win/avg_loss):.2f}:1" if avg_loss != 0 else "N/A")
    print(f"🔗 AVG CONFLUENCE SCORE: {avg_confluence:.1f}")

    # Confluence analysis
    high_confluence_trades = [t for t in trades if t['confluence_score'] >= 10]
    if high_confluence_trades:
        hc_wins = len([t for t in high_confluence_trades if t['result'] == 'WIN'])
        hc_win_rate = (hc_wins / len(high_confluence_trades)) * 100
        print(f"\n⭐ HIGH CONFLUENCE (10+): {hc_win_rate:.1f}% win rate ({hc_wins}/{len(high_confluence_trades)} trades)")

    # Save results
    results_df = pd.DataFrame(trades)
    results_df.to_csv('data/ADVANCED_BACKTEST_RESULTS.csv', index=False)
    print(f"\n💾 Results saved to: data/ADVANCED_BACKTEST_RESULTS.csv")

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'avg_confluence': avg_confluence
    }

def main():
    """Main function"""

    print("🚀 ADVANCED XAUUSD CONFLUENCE BACKTESTING")
    print("=" * 60)

    df = load_data()
    if df is None:
        return

    print(f"📊 Loaded {len(df)} candles for analysis")

    results = run_advanced_backtest(df)

    if results and results['win_rate'] >= 60:
        print(f"\n🎉 SUCCESS! Found {results['win_rate']:.1f}% win rate system!")
        print("✅ This confluence approach works!")
    elif results:
        print(f"\n⚠️  {results['win_rate']:.1f}% win rate - still needs improvement")
        print("💡 Need to refine confluence criteria further")
    else:
        print("❌ No profitable confluence found")

if __name__ == "__main__":
    main()