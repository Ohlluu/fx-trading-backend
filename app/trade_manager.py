#!/usr/bin/env python3
"""
Trade State Manager for Pro Trader Gold
Manages active trades, calculates P&L, and provides trade management alerts
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pytz

class TradeManager:
    """Manages active trade state and provides monitoring alerts"""

    def __init__(self):
        self.active_trade = None
        self.trade_history = []

    def enter_trade(self,
                   entry_price: float,
                   position_size: int,  # 50 or 100 (percentage)
                   stop_loss: float,
                   take_profit_1: float,
                   take_profit_2: Optional[float] = None,
                   trade_direction: str = "LONG") -> Dict[str, Any]:
        """
        Record that user entered a trade

        Args:
            entry_price: Price at which trade was entered
            position_size: 50 or 100 (percentage of planned position)
            stop_loss: Stop loss price
            take_profit_1: First take profit target
            take_profit_2: Second take profit target (optional)
            trade_direction: "LONG" or "SHORT"
        """
        entry_time = datetime.now(pytz.UTC)

        # If 50% position already exists, this is adding the second 50%
        if self.active_trade and self.active_trade['position_size'] == 50:
            # Calculate average entry price
            avg_entry = (self.active_trade['entry_price'] + entry_price) / 2

            self.active_trade['entries'].append({
                'price': entry_price,
                'size': position_size,
                'time': entry_time.isoformat()
            })
            self.active_trade['position_size'] = 100
            self.active_trade['average_entry'] = avg_entry

            return {
                "success": True,
                "message": f"Added {position_size}% position at ${entry_price:.2f}",
                "average_entry": avg_entry,
                "total_position_size": 100,
                "trade": self.active_trade
            }

        # New trade
        self.active_trade = {
            'trade_id': f"GOLD_{entry_time.strftime('%Y%m%d_%H%M%S')}",
            'entry_price': entry_price,
            'average_entry': entry_price,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit_1': take_profit_1,
            'take_profit_2': take_profit_2,
            'direction': trade_direction,
            'entry_time': entry_time.isoformat(),
            'status': 'ACTIVE',
            'entries': [{
                'price': entry_price,
                'size': position_size,
                'time': entry_time.isoformat()
            }],
            'exits': [],
            'alerts_shown': []
        }

        return {
            "success": True,
            "message": f"Trade entered: {position_size}% position at ${entry_price:.2f}",
            "trade": self.active_trade
        }

    def exit_trade(self,
                   exit_price: float,
                   position_size: int,  # 50 or 100 (percentage to exit)
                   reason: str = "Manual Exit") -> Dict[str, Any]:
        """
        Record that user exited a trade (fully or partially)

        Args:
            exit_price: Price at which trade was exited
            position_size: Percentage of position to exit (50 or 100)
            reason: Reason for exit
        """
        if not self.active_trade:
            return {
                "success": False,
                "error": "No active trade to exit"
            }

        exit_time = datetime.now(pytz.UTC)

        # Calculate profit/loss for this exit
        entry = self.active_trade['average_entry']
        direction = self.active_trade['direction']

        if direction == "LONG":
            pnl = (exit_price - entry) * (position_size / 100)
            pnl_pct = ((exit_price - entry) / entry) * 100 * (position_size / 100)
        else:  # SHORT
            pnl = (entry - exit_price) * (position_size / 100)
            pnl_pct = ((entry - exit_price) / entry) * 100 * (position_size / 100)

        # Record the exit
        exit_record = {
            'price': exit_price,
            'size': position_size,
            'time': exit_time.isoformat(),
            'reason': reason,
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2)
        }

        self.active_trade['exits'].append(exit_record)

        # Calculate time in trade
        entry_dt = datetime.fromisoformat(self.active_trade['entry_time'])
        time_in_trade = exit_time - entry_dt
        hours = time_in_trade.total_seconds() / 3600

        # Partial exit
        if position_size < 100 and position_size < self.active_trade['position_size']:
            self.active_trade['position_size'] -= position_size

            return {
                "success": True,
                "message": f"Exited {position_size}% at ${exit_price:.2f}",
                "partial_exit": True,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "remaining_position": self.active_trade['position_size'],
                "trade": self.active_trade
            }

        # Full exit - move to history
        self.active_trade['status'] = 'CLOSED'
        self.active_trade['exit_time'] = exit_time.isoformat()
        self.active_trade['exit_price'] = exit_price
        self.active_trade['exit_reason'] = reason
        self.active_trade['final_pnl'] = round(pnl, 2)
        self.active_trade['final_pnl_pct'] = round(pnl_pct, 2)
        self.active_trade['time_in_trade_hours'] = round(hours, 2)

        # Add to history
        self.trade_history.append(self.active_trade.copy())

        # Clear active trade
        closed_trade = self.active_trade.copy()
        self.active_trade = None

        return {
            "success": True,
            "message": f"Trade closed at ${exit_price:.2f}",
            "full_exit": True,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "time_in_trade_hours": round(hours, 2),
            "outcome": "WINNER" if pnl > 0 else "LOSER",
            "trade": closed_trade
        }

    def get_trade_status(self, current_price: float) -> Dict[str, Any]:
        """
        Get current trade status with P&L and alerts

        Args:
            current_price: Current market price
        """
        if not self.active_trade:
            return {
                "status": "NO_ACTIVE_TRADE",
                "in_trade": False
            }

        entry = self.active_trade['average_entry']
        sl = self.active_trade['stop_loss']
        tp1 = self.active_trade['take_profit_1']
        tp2 = self.active_trade.get('take_profit_2')
        direction = self.active_trade['direction']
        position_size = self.active_trade['position_size']

        # Calculate P&L
        if direction == "LONG":
            pnl = (current_price - entry) * (position_size / 100)
            pnl_pct = ((current_price - entry) / entry) * 100
            distance_to_sl = current_price - sl
            distance_to_tp1 = tp1 - current_price
        else:  # SHORT
            pnl = (entry - current_price) * (position_size / 100)
            pnl_pct = ((entry - current_price) / entry) * 100
            distance_to_sl = sl - current_price
            distance_to_tp1 = current_price - tp1

        # Calculate progress to targets
        total_distance_to_tp1 = abs(tp1 - entry)
        current_distance = abs(current_price - entry)
        progress_to_tp1 = min(100, (current_distance / total_distance_to_tp1) * 100)

        # Calculate time in trade
        entry_dt = datetime.fromisoformat(self.active_trade['entry_time'])
        now = datetime.now(pytz.UTC)
        time_in_trade = now - entry_dt
        minutes = int(time_in_trade.total_seconds() / 60)
        hours = minutes // 60
        mins = minutes % 60

        return {
            "status": "ACTIVE_TRADE",
            "in_trade": True,
            "trade_id": self.active_trade['trade_id'],
            "direction": direction,
            "entry_price": entry,
            "current_price": current_price,
            "position_size": position_size,
            "stop_loss": sl,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "pnl_pips": round(abs(current_price - entry), 2),
            "distance_to_sl": round(distance_to_sl, 2),
            "distance_to_tp1": round(distance_to_tp1, 2),
            "progress_to_tp1_pct": round(progress_to_tp1, 1),
            "time_in_trade": f"{hours}h {mins}m" if hours > 0 else f"{mins}m",
            "time_in_trade_minutes": minutes,
            "entry_time": self.active_trade['entry_time'],
            "entries": self.active_trade['entries'],
            "exits": self.active_trade['exits']
        }

    def get_trade_alerts(self,
                        current_price: float,
                        current_candle: Dict[str, Any],
                        momentum_data: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Generate trade management alerts based on current market conditions

        Args:
            current_price: Current market price
            current_candle: Current candle data
            momentum_data: Optional momentum indicators
        """
        if not self.active_trade:
            return []

        alerts = []
        entry = self.active_trade['average_entry']
        sl = self.active_trade['stop_loss']
        tp1 = self.active_trade['take_profit_1']
        direction = self.active_trade['direction']

        # Calculate progress
        total_distance_to_tp1 = abs(tp1 - entry)
        current_distance = abs(current_price - entry)
        progress_pct = (current_distance / total_distance_to_tp1) * 100

        # Time-based alerts
        entry_dt = datetime.fromisoformat(self.active_trade['entry_time'])
        now = datetime.now(pytz.UTC)
        time_in_trade_mins = (now - entry_dt).total_seconds() / 60

        # ALERT 1: Trailing Stop Suggestion
        if progress_pct >= 50:
            alert_key = "trailing_stop_50"
            if alert_key not in self.active_trade['alerts_shown']:
                new_sl = entry if progress_pct < 100 else entry + (tp1 - entry) * 0.5
                alerts.append({
                    "type": "TRAILING_STOP",
                    "priority": "MEDIUM",
                    "title": "✅ Consider Trailing Stop",
                    "message": f"Price is {progress_pct:.0f}% to TP1. Move SL to ${new_sl:.2f} to lock in profit.",
                    "current_sl": sl,
                    "suggested_sl": round(new_sl, 2),
                    "action": "Move stop loss to protect gains"
                })
                self.active_trade['alerts_shown'].append(alert_key)

        # ALERT 2: Partial Exit at 70%+ progress
        if progress_pct >= 70 and self.active_trade['position_size'] == 100:
            alert_key = "partial_exit_70"
            if alert_key not in self.active_trade['alerts_shown']:
                alerts.append({
                    "type": "PARTIAL_EXIT",
                    "priority": "MEDIUM",
                    "title": "🟡 Consider Partial Exit",
                    "message": f"Price is {progress_pct:.0f}% to TP1 (${current_price:.2f}). Consider taking 50% profit here.",
                    "exit_price": current_price,
                    "potential_profit": round((current_price - entry) * 0.5, 2),
                    "action": "Exit 50% to lock profits, let 50% run to TP1"
                })
                self.active_trade['alerts_shown'].append(alert_key)

        # ALERT 3: Price stalling (momentum warning)
        if time_in_trade_mins > 30:
            # Check if price hasn't moved much in last 20+ minutes
            distance_from_entry = abs(current_price - entry)
            if distance_from_entry < total_distance_to_tp1 * 0.3 and progress_pct < 40:
                alert_key = f"stalling_{int(time_in_trade_mins / 30)}"
                if alert_key not in self.active_trade['alerts_shown']:
                    alerts.append({
                        "type": "MOMENTUM_WARNING",
                        "priority": "LOW",
                        "title": "⚠️ Price Stalling",
                        "message": f"Trade open for {int(time_in_trade_mins)}min but only {progress_pct:.0f}% to TP1. Momentum weak.",
                        "suggestion": "Consider exiting at breakeven or small profit if no movement soon.",
                        "action": "Monitor closely - setup may be failing"
                    })
                    self.active_trade['alerts_shown'].append(alert_key)

        # ALERT 4: Close to TP1
        if progress_pct >= 90:
            alert_key = "near_tp1"
            if alert_key not in self.active_trade['alerts_shown']:
                alerts.append({
                    "type": "TARGET_APPROACHING",
                    "priority": "HIGH",
                    "title": "🎯 Close to TP1!",
                    "message": f"TP1 (${tp1:.2f}) is ${abs(tp1 - current_price):.2f} away. Get ready!",
                    "action": "Prepare to take profit or trail stop"
                })
                self.active_trade['alerts_shown'].append(alert_key)

        # ALERT 5: Confirmation entry available (if only 50% in)
        if self.active_trade['position_size'] == 50:
            alerts.append({
                "type": "ADD_POSITION",
                "priority": "HIGH",
                "title": "➕ Add Second 50% Available",
                "message": f"You're in with 50% at ${entry:.2f}. Setup confirmed - add remaining 50%?",
                "action": "Consider adding confirmation entry"
            })

        return alerts

    def get_trade_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trade history"""
        return self.trade_history[-limit:]

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate trading statistics from history"""
        if not self.trade_history:
            return {
                "total_trades": 0,
                "winners": 0,
                "losers": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_winner": 0,
                "avg_loser": 0
            }

        total_trades = len(self.trade_history)
        winners = [t for t in self.trade_history if t['final_pnl'] > 0]
        losers = [t for t in self.trade_history if t['final_pnl'] <= 0]

        total_pnl = sum(t['final_pnl'] for t in self.trade_history)
        avg_winner = sum(t['final_pnl'] for t in winners) / len(winners) if winners else 0
        avg_loser = sum(t['final_pnl'] for t in losers) / len(losers) if losers else 0

        return {
            "total_trades": total_trades,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": round((len(winners) / total_trades) * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_winner": round(avg_winner, 2),
            "avg_loser": round(avg_loser, 2),
            "profit_factor": round(abs(sum(t['final_pnl'] for t in winners) / sum(t['final_pnl'] for t in losers)), 2) if losers and sum(t['final_pnl'] for t in losers) != 0 else 0
        }

# Global trade manager instance
trade_manager = TradeManager()
