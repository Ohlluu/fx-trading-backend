#!/usr/bin/env python3
"""
GBPUSD 6-Hour Early Exit Rule Analysis
Calculate exact win/loss percentages with early exit strategy
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import httpx

# OANDA credentials
OANDA_API_KEY = "1c2ee716aac27b425f2fd7a8ffbe9b9a-7a3b3da61668a804b56e2974ff21aaa0"
OANDA_BASE_URL = "https://api-fxpractice.oanda.com/v3"

async def get_gbpusd_data():
    """Get GBPUSD data for analysis"""
    print("📊 Fetching GBPUSD data...")

    url = f"{OANDA_BASE_URL}/instruments/GBP_USD/candles"
    headers = {
        "Authorization": f"Bearer {OANDA_API_KEY}",
        "Accept-Datetime-Format": "RFC3339"
    }

    # Get substantial dataset
    all_data = []
    for chunk in range(5):  # 5 chunks = ~25,000 candles
        if chunk == 0:
            end_time = datetime.now(pytz.UTC)
        else:
            end_time = datetime.now(pytz.UTC) - timedelta(hours=5000 * chunk)

        params = {
            "granularity": "H1",
            "count": 5000,
            "to": end_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.get(url, headers=headers, params=params)

                if response.status_code == 200:
                    data = response.json()
                    candles = data.get("candles", [])

                    if candles:
                        all_data.extend(candles)
                        print(f"✅ Fetched chunk {chunk + 1}: {len(candles)} candles")
                    else:
                        break

            await asyncio.sleep(1)

        except Exception as e:
            print(f"Error fetching chunk {chunk + 1}: {e}")
            break

    # Convert to DataFrame
    processed_data = []
    for candle in all_data:
        if candle.get("complete", True):
            try:
                processed_data.append({
                    'datetime': pd.to_datetime(candle['time']),
                    'open': float(candle['mid']['o']),
                    'high': float(candle['mid']['h']),
                    'low': float(candle['mid']['l']),
                    'close': float(candle['mid']['c']),
                })
            except:
                continue

    df = pd.DataFrame(processed_data)
    df = df.drop_duplicates(subset=['datetime'])
    df = df.set_index('datetime').sort_index()

    print(f"✅ Dataset: {len(df)} hourly candles from {df.index[0]} to {df.index[-1]}")
    return df

def add_indicators(df):
    """Add technical indicators for signal identification"""
    # Moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_20'] = df['close'].ewm(span=20).mean()

    # Price positions
    df['above_sma50'] = (df['close'] > df['sma_50']).fillna(False)
    df['above_sma20'] = (df['close'] > df['sma_20']).fillna(False)
    df['above_ema20'] = (df['close'] > df['ema_20']).fillna(False)

    # Trends
    df['uptrend_sma50'] = (df['sma_50'].diff() > 0).fillna(False)
    df['uptrend_sma20'] = (df['sma_20'].diff() > 0).fillna(False)
    df['uptrend_ema20'] = (df['ema_20'].diff() > 0).fillna(False)

    # Candle characteristics
    df['is_green'] = (df['close'] > df['open']).fillna(False)
    df['body_size'] = abs(df['close'] - df['open'])
    df['range_size'] = df['high'] - df['low']
    df['body_ratio'] = (df['body_size'] / df['range_size']).fillna(0)
    df['large_body'] = (df['body_ratio'] > 0.7).fillna(False)

    # Momentum and session
    df['price_change'] = df['close'].pct_change().fillna(0)
    df['strong_momentum'] = (abs(df['price_change']) > 0.002).fillna(False)
    df['hour_utc'] = df.index.hour
    df['london_session'] = ((df['hour_utc'] >= 7) & (df['hour_utc'] <= 16))
    df['prev_red'] = (~df['is_green']).shift(1).fillna(False)
    df['prev_green'] = df['is_green'].shift(1).fillna(False)

    return df

def identify_signals(df):
    """Identify GBPUSD signals using research-grade confluence"""
    df_clean = df.dropna(subset=['sma_50', 'sma_20', 'ema_20'])

    # GBPUSD confluence weights (from comprehensive research)
    confluence_weights = {
        'above_sma50': 5,
        'uptrend_sma50': 4,
        'above_sma20': 3,
        'uptrend_sma20': 2,
        'above_ema20': 2,
        'uptrend_ema20': 2,
        'london_session': 2,
        'large_body': 2,
        'strong_momentum': 1,
        'is_green': 1,
        'prev_red': 1,
    }

    # Calculate bullish confluence scores
    bullish_scores = pd.Series(0, index=df_clean.index, dtype=int)
    for factor, weight in confluence_weights.items():
        if factor in df_clean.columns:
            bullish_scores += df_clean[factor].astype(int) * weight

    # Calculate bearish confluence scores
    bearish_scores = pd.Series(0, index=df_clean.index, dtype=int)
    bearish_factors = {
        'below_sma50': (~df_clean['above_sma50']).astype(int) * 5,
        'uptrend_sma50': df_clean['uptrend_sma50'].astype(int) * 4,
        'below_sma20': (~df_clean['above_sma20']).astype(int) * 3,
        'uptrend_sma20': df_clean['uptrend_sma20'].astype(int) * 2,
        'below_ema20': (~df_clean['above_ema20']).astype(int) * 2,
        'uptrend_ema20': df_clean['uptrend_ema20'].astype(int) * 2,
        'london_session': df_clean['london_session'].astype(int) * 2,
        'large_body': df_clean['large_body'].astype(int) * 2,
        'strong_momentum': df_clean['strong_momentum'].astype(int) * 1,
        'is_red': (~df_clean['is_green']).astype(int) * 1,
        'prev_green': df_clean['prev_green'].astype(int) * 1,
    }

    for factor, score in bearish_factors.items():
        bearish_scores += score

    # Identify signals using optimal thresholds
    bullish_signals = bullish_scores >= 20
    bearish_signals = bearish_scores >= 16

    df_clean['bullish_signal'] = bullish_signals
    df_clean['bearish_signal'] = bearish_signals
    df_clean['any_signal'] = bullish_signals | bearish_signals

    signal_count = df_clean['any_signal'].sum()
    print(f"✅ Identified {signal_count} GBPUSD signals")
    print(f"   Bullish: {bullish_signals.sum()}, Bearish: {bearish_signals.sum()}")

    return df_clean

def analyze_6h_early_exit_strategy(df):
    """Analyze win/loss with 6-hour early exit strategy"""
    print("\n📊 Analyzing 6-Hour Early Exit Strategy...")

    signals = df[df['any_signal']]

    # Use optimal 4:1 R:R from research
    tp_percent = 4.0  # 4% TP
    sl_percent = 1.0  # 1% SL

    # Track results
    results = {
        'without_early_exit': {'wins': 0, 'losses': 0, 'total': 0},
        'with_early_exit': {'wins': 0, 'losses': 0, 'early_exits': 0, 'total': 0}
    }

    print(f"Testing {len(signals)} signals with 4:1 R:R ratio (4% TP, 1% SL)")

    for signal_time, signal_row in signals.iterrows():
        try:
            entry_price = signal_row['close']
            is_bullish = signal_row['bullish_signal']

            signal_idx = df.index.get_indexer([signal_time])[0]

            # Need enough future data
            if signal_idx + 72 < len(df):
                future_data = df.iloc[signal_idx+1:signal_idx+73]  # 72 hours ahead
                first_6h = df.iloc[signal_idx+1:signal_idx+7] if signal_idx + 7 < len(df) else df.iloc[signal_idx+1:]

                # Set TP/SL levels
                if is_bullish:
                    tp_level = entry_price * (1 + tp_percent/100)
                    sl_level = entry_price * (1 - sl_percent/100)
                else:
                    tp_level = entry_price * (1 - tp_percent/100)
                    sl_level = entry_price * (1 + sl_percent/100)

                # === WITHOUT EARLY EXIT (BASELINE) ===
                if is_bullish:
                    tp_hit = (future_data['high'] >= tp_level).any()
                    sl_hit = (future_data['low'] <= sl_level).any()

                    if tp_hit and sl_hit:
                        tp_times = future_data[future_data['high'] >= tp_level].index
                        sl_times = future_data[future_data['low'] <= sl_level].index
                        baseline_win = tp_times[0] <= sl_times[0]
                    elif tp_hit:
                        baseline_win = True
                    elif sl_hit:
                        baseline_win = False
                    else:
                        final_price = future_data['close'].iloc[-1]
                        baseline_win = final_price > entry_price
                else:  # bearish
                    tp_hit = (future_data['low'] <= tp_level).any()
                    sl_hit = (future_data['high'] >= sl_level).any()

                    if tp_hit and sl_hit:
                        tp_times = future_data[future_data['low'] <= tp_level].index
                        sl_times = future_data[future_data['high'] >= sl_level].index
                        baseline_win = tp_times[0] <= sl_times[0]
                    elif tp_hit:
                        baseline_win = True
                    elif sl_hit:
                        baseline_win = False
                    else:
                        final_price = future_data['close'].iloc[-1]
                        baseline_win = final_price < entry_price

                results['without_early_exit']['total'] += 1
                if baseline_win:
                    results['without_early_exit']['wins'] += 1
                else:
                    results['without_early_exit']['losses'] += 1

                # === WITH EARLY EXIT STRATEGY ===
                # Check for early exit signals in first 6 hours
                early_exit_triggered = False

                if not first_6h.empty:
                    # Early exit conditions:
                    # 1. Momentum reversal: price moves -0.1% against position
                    # 2. MA break: price breaks below SMA20 (for bullish) or above SMA20 (for bearish)

                    if is_bullish:
                        momentum_reversal = (first_6h['close'] < entry_price * 0.999).any()  # -0.1%
                        ma_break = (first_6h['close'] < first_6h['sma_20']).any()
                    else:
                        momentum_reversal = (first_6h['close'] > entry_price * 1.001).any()  # +0.1%
                        ma_break = (first_6h['close'] > first_6h['sma_20']).any()

                    # Combined early exit signal (from research: 55% accuracy)
                    early_exit_triggered = momentum_reversal and ma_break

                results['with_early_exit']['total'] += 1

                if early_exit_triggered:
                    # Early exit - assume small loss (exit at 6-hour price)
                    exit_price = first_6h['close'].iloc[-1] if not first_6h.empty else entry_price
                    results['with_early_exit']['early_exits'] += 1

                    # Count as loss since we're exiting due to negative signals
                    results['with_early_exit']['losses'] += 1
                else:
                    # No early exit - use normal outcome
                    if baseline_win:
                        results['with_early_exit']['wins'] += 1
                    else:
                        results['with_early_exit']['losses'] += 1

        except Exception as e:
            continue

    return results

def calculate_performance_metrics(results):
    """Calculate and display performance metrics"""
    print("\n" + "="*80)
    print("🎯 GBPUSD 6-HOUR EARLY EXIT STRATEGY RESULTS")
    print("="*80)

    baseline = results['without_early_exit']
    early_exit = results['with_early_exit']

    # Baseline performance
    baseline_win_rate = (baseline['wins'] / baseline['total']) * 100 if baseline['total'] > 0 else 0
    baseline_loss_rate = (baseline['losses'] / baseline['total']) * 100 if baseline['total'] > 0 else 0

    print(f"\n📊 WITHOUT Early Exit (Baseline 4:1 R:R):")
    print(f"   Total Trades: {baseline['total']}")
    print(f"   Wins: {baseline['wins']}")
    print(f"   Losses: {baseline['losses']}")
    print(f"   Win Rate: {baseline_win_rate:.1f}%")
    print(f"   Loss Rate: {baseline_loss_rate:.1f}%")

    # Early exit performance
    early_exit_win_rate = (early_exit['wins'] / early_exit['total']) * 100 if early_exit['total'] > 0 else 0
    early_exit_loss_rate = (early_exit['losses'] / early_exit['total']) * 100 if early_exit['total'] > 0 else 0
    early_exit_rate = (early_exit['early_exits'] / early_exit['total']) * 100 if early_exit['total'] > 0 else 0

    print(f"\n📊 WITH 6-Hour Early Exit Strategy:")
    print(f"   Total Trades: {early_exit['total']}")
    print(f"   Wins: {early_exit['wins']}")
    print(f"   Losses: {early_exit['losses']}")
    print(f"   Early Exits: {early_exit['early_exits']} ({early_exit_rate:.1f}% of trades)")
    print(f"   Win Rate: {early_exit_win_rate:.1f}%")
    print(f"   Loss Rate: {early_exit_loss_rate:.1f}%")

    # Performance comparison
    win_rate_improvement = early_exit_win_rate - baseline_win_rate
    loss_rate_change = early_exit_loss_rate - baseline_loss_rate

    print(f"\n🎯 PERFORMANCE COMPARISON:")
    print(f"   Win Rate Change: {win_rate_improvement:+.1f}%")
    print(f"   Loss Rate Change: {loss_rate_change:+.1f}%")

    if win_rate_improvement > 0:
        print(f"   ✅ Early exit IMPROVES performance by {win_rate_improvement:.1f}%!")
    elif win_rate_improvement < -2:
        print(f"   ❌ Early exit HURTS performance by {abs(win_rate_improvement):.1f}%")
    else:
        print(f"   🟡 Early exit has minimal impact ({win_rate_improvement:+.1f}%)")

    # Expected return analysis
    baseline_expected_return = (baseline['wins'] * 4.0 - baseline['losses'] * 1.0) / baseline['total'] if baseline['total'] > 0 else 0

    # For early exit: assume early exits lose 0.2% on average, normal wins/losses use 4:1 ratio
    normal_wins = early_exit['wins']
    normal_losses = early_exit['losses'] - early_exit['early_exits']  # Regular losses
    early_exit_losses = early_exit['early_exits']  # Early exit losses

    early_exit_expected_return = (normal_wins * 4.0 - normal_losses * 1.0 - early_exit_losses * 0.2) / early_exit['total'] if early_exit['total'] > 0 else 0

    print(f"\n💰 EXPECTED RETURN PER TRADE:")
    print(f"   Baseline: {baseline_expected_return:+.2f}%")
    print(f"   With Early Exit: {early_exit_expected_return:+.2f}%")
    print(f"   Improvement: {early_exit_expected_return - baseline_expected_return:+.2f}%")

    print(f"\n🎯 RECOMMENDATION:")
    if early_exit_expected_return > baseline_expected_return:
        print(f"   ✅ USE 6-hour early exit strategy!")
        print(f"   Expected return improves by {(early_exit_expected_return - baseline_expected_return):+.2f}%")
    else:
        print(f"   ❌ Skip 6-hour early exit strategy")
        print(f"   Better to hold trades to normal TP/SL levels")

    return {
        'baseline_win_rate': baseline_win_rate,
        'early_exit_win_rate': early_exit_win_rate,
        'improvement': win_rate_improvement,
        'baseline_expected_return': baseline_expected_return,
        'early_exit_expected_return': early_exit_expected_return
    }

async def main():
    """Execute 6-hour early exit analysis"""
    print("🚨 GBPUSD 6-Hour Early Exit Strategy Analysis")
    print("Testing: Momentum reversal + MA break early exit rule")
    print("="*60)

    try:
        # Get data
        df = await get_gbpusd_data()

        # Add indicators
        df = add_indicators(df)

        # Identify signals
        df = identify_signals(df)

        # Analyze 6-hour early exit strategy
        results = analyze_6h_early_exit_strategy(df)

        # Calculate and display performance
        performance = calculate_performance_metrics(results)

        print("\n" + "="*80)
        print("✅ 6-HOUR EARLY EXIT ANALYSIS COMPLETE!")
        print("="*80)

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())