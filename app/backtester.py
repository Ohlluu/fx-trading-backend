# app/backtester.py - Comprehensive backtesting framework
# v1.0 - Advanced backtesting with walk-forward analysis, Monte Carlo, and performance metrics

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
import os
import json
import warnings
warnings.filterwarnings('ignore')

from .confluence import evaluate_last_closed_bar
from .positions import calculate_optimal_position_size, RISK_CONFIG

class BacktestResults:
    def __init__(self):
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.daily_returns: List[float] = []
        self.performance_metrics: Dict[str, float] = {}
        
class AdvancedBacktester:
    def __init__(self, 
                 initial_balance: float = 10000,
                 risk_per_trade: float = 0.01,
                 commission: float = 0.0001):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.commission = commission
        self.results = BacktestResults()
        
    def load_data(self, pair: str, timeframe: str = "H1") -> pd.DataFrame:
        """Load historical data for backtesting"""
        csv_path = f"data/{pair}_{timeframe}.csv"
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Data file not found: {csv_path}")
            
        df = pd.read_csv(csv_path)
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)
        return df
        
    def walk_forward_analysis(self, 
                            pairs: List[str], 
                            train_period_days: int = 365,
                            test_period_days: int = 90,
                            step_days: int = 30) -> Dict[str, Any]:
        """
        Perform walk-forward analysis to test strategy robustness
        """
        all_results = []
        
        for pair in pairs:
            try:
                df = self.load_data(pair)
                if len(df) < train_period_days * 24:  # Not enough data
                    continue
                    
                start_date = df.index[train_period_days * 24]
                end_date = df.index[-test_period_days * 24]
                
                current_date = start_date
                while current_date < end_date:
                    # Define training period
                    train_start = current_date - timedelta(days=train_period_days)
                    train_end = current_date
                    
                    # Define test period
                    test_start = current_date
                    test_end = current_date + timedelta(days=test_period_days)
                    
                    # Run backtest on test period
                    test_results = self.run_single_pair_backtest(
                        pair, df[test_start:test_end]
                    )
                    
                    if test_results:
                        test_results['pair'] = pair
                        test_results['period_start'] = test_start
                        test_results['period_end'] = test_end
                        all_results.append(test_results)
                    
                    current_date += timedelta(days=step_days)
                    
            except Exception as e:
                print(f"Error processing {pair}: {e}")
                continue
                
        return self.analyze_walk_forward_results(all_results)
        
    def run_single_pair_backtest(self, pair: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest on a single pair"""
        trades = []
        balance = self.initial_balance
        
        # Minimum data requirement
        if len(df) < 300:
            return None
            
        # Iterate through each bar
        for i in range(300, len(df)):
            try:
                # Get historical data up to current point
                hist_data = df.iloc[:i+1].copy()
                
                # Generate signal
                signal = evaluate_last_closed_bar(hist_data, pair=pair)
                
                if signal.get('ok') and not signal.get('is_skip'):
                    # Calculate position size
                    entry = signal.get('entry')
                    sl = signal.get('sl')
                    tp = signal.get('tp')
                    side = signal.get('side')
                    
                    if entry and sl and tp and side:
                        # Position sizing
                        pos_size_info = calculate_optimal_position_size(
                            pair, entry, sl, balance
                        )
                        position_size = pos_size_info['recommended']
                        
                        # Simulate trade
                        trade_result = self.simulate_trade(
                            df.iloc[i+1:], entry, sl, tp, side, position_size
                        )
                        
                        if trade_result:
                            trade_result['pair'] = pair
                            trade_result['signal_time'] = hist_data.index[-1]
                            trade_result['initial_balance'] = balance
                            
                            # Update balance
                            pnl = trade_result.get('pnl', 0)
                            balance += pnl
                            trade_result['final_balance'] = balance
                            
                            trades.append(trade_result)
                            
            except Exception as e:
                continue
                
        return self.calculate_performance_metrics(trades, self.initial_balance)
        
    def simulate_trade(self, 
                      future_data: pd.DataFrame, 
                      entry: float, 
                      sl: float, 
                      tp: float, 
                      side: str, 
                      position_size: float) -> Optional[Dict]:
        """Simulate individual trade execution"""
        
        if len(future_data) == 0 or position_size <= 0:
            return None
            
        side_mult = 1 if side == "BUY" else -1
        
        for i, (timestamp, bar) in enumerate(future_data.iterrows()):
            # Check for stop loss
            if side == "BUY":
                if bar['low'] <= sl:
                    pnl = (sl - entry) * position_size * side_mult
                    pnl -= abs(pnl * self.commission)  # Commission
                    return {
                        'exit_time': timestamp,
                        'exit_price': sl,
                        'exit_reason': 'stop_loss',
                        'bars_held': i + 1,
                        'pnl': pnl,
                        'return_pct': pnl / (entry * position_size),
                        'position_size': position_size
                    }
                    
            else:  # SELL
                if bar['high'] >= sl:
                    pnl = (entry - sl) * position_size * side_mult
                    pnl -= abs(pnl * self.commission)
                    return {
                        'exit_time': timestamp,
                        'exit_price': sl,
                        'exit_reason': 'stop_loss',
                        'bars_held': i + 1,
                        'pnl': pnl,
                        'return_pct': pnl / (entry * position_size),
                        'position_size': position_size
                    }
                    
            # Check for take profit
            if side == "BUY":
                if bar['high'] >= tp:
                    pnl = (tp - entry) * position_size * side_mult
                    pnl -= abs(pnl * self.commission)
                    return {
                        'exit_time': timestamp,
                        'exit_price': tp,
                        'exit_reason': 'take_profit',
                        'bars_held': i + 1,
                        'pnl': pnl,
                        'return_pct': pnl / (entry * position_size),
                        'position_size': position_size
                    }
                    
            else:  # SELL
                if bar['low'] <= tp:
                    pnl = (entry - tp) * position_size * side_mult
                    pnl -= abs(pnl * self.commission)
                    return {
                        'exit_time': timestamp,
                        'exit_price': tp,
                        'exit_reason': 'take_profit',
                        'bars_held': i + 1,
                        'pnl': pnl,
                        'return_pct': pnl / (entry * position_size),
                        'position_size': position_size
                    }
                    
            # Time-based exit after 24 hours (24 bars)
            if i >= 23:
                exit_price = bar['close']
                pnl = (exit_price - entry) * position_size * side_mult
                pnl -= abs(pnl * self.commission)
                return {
                    'exit_time': timestamp,
                    'exit_price': exit_price,
                    'exit_reason': 'time_exit',
                    'bars_held': i + 1,
                    'pnl': pnl,
                    'return_pct': pnl / (entry * position_size),
                    'position_size': position_size
                }
                
        return None
        
    def calculate_performance_metrics(self, trades: List[Dict], initial_balance: float) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        if not trades:
            return {'total_trades': 0}
            
        # Basic statistics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # PnL calculations
        total_pnl = sum(t['pnl'] for t in trades)
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Returns
        returns = [t['return_pct'] for t in trades]
        total_return = total_pnl / initial_balance
        avg_return = np.mean(returns) if returns else 0
        
        # Risk metrics
        return_std = np.std(returns) if len(returns) > 1 else 0
        sharpe_ratio = (avg_return / return_std * np.sqrt(252)) if return_std != 0 else 0
        
        # Drawdown calculation
        equity_curve = [initial_balance]
        for trade in trades:
            equity_curve.append(equity_curve[-1] + trade['pnl'])
            
        peak = equity_curve[0]
        max_drawdown = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        # Additional metrics
        avg_bars_held = np.mean([t['bars_held'] for t in trades]) if trades else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_return_per_trade': avg_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_bars_held': avg_bars_held,
            'equity_curve': equity_curve,
            'trades': trades
        }
        
    def analyze_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze walk-forward results for strategy stability"""
        
        if not results:
            return {'error': 'No walk-forward results to analyze'}
            
        # Aggregate metrics
        total_trades = sum(r.get('total_trades', 0) for r in results)
        weighted_win_rate = np.average([r.get('win_rate', 0) for r in results], 
                                     weights=[r.get('total_trades', 1) for r in results])
        
        returns = [r.get('total_return', 0) for r in results]
        avg_return = np.mean(returns)
        return_std = np.std(returns)
        consistency = len([r for r in returns if r > 0]) / len(returns) if returns else 0
        
        return {
            'periods_tested': len(results),
            'total_trades': total_trades,
            'average_win_rate': weighted_win_rate,
            'average_return_per_period': avg_return,
            'return_consistency': consistency,
            'return_volatility': return_std,
            'profitable_periods': len([r for r in results if r.get('total_return', 0) > 0]),
            'detailed_results': results
        }
        
    def monte_carlo_analysis(self, trades: List[Dict], num_simulations: int = 1000) -> Dict[str, Any]:
        """Perform Monte Carlo analysis on trade sequence"""
        
        if len(trades) < 10:
            return {'error': 'Need at least 10 trades for Monte Carlo analysis'}
            
        returns = [t['return_pct'] for t in trades]
        simulation_results = []
        
        for _ in range(num_simulations):
            # Randomly shuffle trade sequence
            shuffled_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate equity curve
            equity = 1.0
            max_dd = 0
            peak = 1.0
            
            for ret in shuffled_returns:
                equity *= (1 + ret)
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
                    
            final_return = equity - 1
            simulation_results.append({
                'final_return': final_return,
                'max_drawdown': max_dd
            })
            
        # Analyze results
        returns = [s['final_return'] for s in simulation_results]
        drawdowns = [s['max_drawdown'] for s in simulation_results]
        
        return {
            'simulations': num_simulations,
            'mean_return': np.mean(returns),
            'return_std': np.std(returns),
            'return_percentiles': {
                '5%': np.percentile(returns, 5),
                '25%': np.percentile(returns, 25),
                '50%': np.percentile(returns, 50),
                '75%': np.percentile(returns, 75),
                '95%': np.percentile(returns, 95)
            },
            'max_drawdown_percentiles': {
                '5%': np.percentile(drawdowns, 5),
                '25%': np.percentile(drawdowns, 25),
                '50%': np.percentile(drawdowns, 50),
                '75%': np.percentile(drawdowns, 75),
                '95%': np.percentile(drawdowns, 95)
            },
            'probability_of_profit': len([r for r in returns if r > 0]) / len(returns)
        }

def run_comprehensive_backtest(pairs: List[str] = None) -> Dict[str, Any]:
    """Main function to run comprehensive backtesting"""
    
    if pairs is None:
        pairs = ["EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "XAUUSD"]
        
    backtester = AdvancedBacktester()
    
    # Individual pair results
    pair_results = {}
    all_trades = []
    
    for pair in pairs:
        try:
            df = backtester.load_data(pair)
            result = backtester.run_single_pair_backtest(pair, df)
            if result and result.get('total_trades', 0) > 0:
                pair_results[pair] = result
                all_trades.extend(result['trades'])
        except Exception as e:
            print(f"Skipping {pair}: {e}")
            continue
            
    if not all_trades:
        return {'error': 'No successful trades in backtest'}
        
    # Overall performance
    overall_performance = backtester.calculate_performance_metrics(
        all_trades, backtester.initial_balance
    )
    
    # Monte Carlo analysis
    monte_carlo = backtester.monte_carlo_analysis(all_trades)
    
    # Walk-forward analysis
    walk_forward = backtester.walk_forward_analysis(pairs)
    
    return {
        'backtest_summary': {
            'total_pairs_tested': len(pairs),
            'successful_pairs': len(pair_results),
            'total_trades': len(all_trades),
            'test_period': 'Historical data available',
            'initial_balance': backtester.initial_balance
        },
        'overall_performance': overall_performance,
        'pair_breakdown': pair_results,
        'monte_carlo_analysis': monte_carlo,
        'walk_forward_analysis': walk_forward,
        'risk_assessment': {
            'max_concurrent_risk': RISK_CONFIG['max_portfolio_risk'],
            'position_sizing': 'Kelly-optimized with volatility adjustment',
            'commission_impact': backtester.commission
        }
    }