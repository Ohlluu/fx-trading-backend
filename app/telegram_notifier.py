#!/usr/bin/env python3
"""
Telegram Notification Service
Sends trading alerts when confluence scores reach threshold
"""

import asyncio
import os
from typing import Optional, Dict, Any
from datetime import datetime
import httpx


class TelegramNotifier:
    """Sends Telegram notifications for high-quality trading setups"""

    def __init__(self):
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"

        # Track last notified score per direction to avoid spam
        # Only notify when score increases above last notified score
        self.last_notified_scores: Dict[str, int] = {
            "BULLISH": 0,
            "BEARISH": 0
        }

    def is_configured(self) -> bool:
        """Check if Telegram is properly configured"""
        return bool(self.bot_token and self.chat_id)

    def _should_send_notification(self, direction: str, current_score: int) -> bool:
        """
        Check if we should send notification based on score changes

        Rules:
        - Notify if score >= 7 (threshold)
        - AND score > last_notified_score for this direction
        - Reset tracking if score drops below 7 (allows re-notification when it comes back up)
        """
        # Below threshold - reset tracking and don't notify
        if current_score < 7:
            self.last_notified_scores[direction] = 0
            return False

        # Score must be higher than last notification
        last_score = self.last_notified_scores.get(direction, 0)
        if current_score > last_score:
            self.last_notified_scores[direction] = current_score
            print(f"✅ {direction} score increased: {last_score} → {current_score} - Sending notification")
            return True

        print(f"⏳ {direction} score unchanged ({current_score}) - Skipping notification")
        return False

    async def send_message(self, message: str, parse_mode: str = "Markdown") -> bool:
        """Send a message via Telegram Bot API"""
        if not self.is_configured():
            print("⚠️ Telegram not configured - skipping notification")
            return False

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": message,
                        "parse_mode": parse_mode,
                        "disable_web_page_preview": True
                    }
                )

                if response.status_code == 200:
                    print(f"✅ Telegram notification sent successfully")
                    return True
                else:
                    print(f"❌ Telegram API error: {response.status_code} - {response.text}")
                    return False

        except Exception as e:
            print(f"❌ Failed to send Telegram message: {e}")
            return False

    async def send_confluence_alert(
        self,
        direction: str,  # "BULLISH" or "BEARISH"
        score: int,
        confidence: str,
        current_price: float,
        confluences: list,
        trade_plan: Optional[Dict] = None
    ) -> bool:
        """Send a formatted confluence alert to Telegram"""

        # Check if we should send notification based on score increase
        if not self._should_send_notification(direction, score):
            return False

        # Format the message
        emoji = "📈" if direction == "BULLISH" else "📉"
        color = "🟢" if direction == "BULLISH" else "🔴"

        message = f"""
{emoji} *{direction} GOLD SETUP READY* {color}

*Confluence Score:* {score} points
*Confidence:* {confidence}
*Current Price:* ${current_price:.2f}

*Detected Patterns:*
"""

        # Add confluence patterns
        for conf in confluences[:5]:  # Show top 5
            pattern_name = conf.get('type', '').replace('_', ' ')
            pattern_score = conf.get('score', 0)
            message += f"• {pattern_name} (+{pattern_score})\n"

        # Add trade plan if available
        if trade_plan and trade_plan.get('entry_price'):
            message += f"\n*Trade Plan:*\n"
            message += f"📍 Entry: {trade_plan.get('entry_price', 'TBD')}\n"

            if trade_plan.get('stop_loss'):
                sl_price = trade_plan['stop_loss'].get('price', 'TBD')
                message += f"🛑 Stop Loss: {sl_price}\n"

            if trade_plan.get('take_profit_1'):
                tp1_price = trade_plan['take_profit_1'].get('price', 'TBD')
                rr_ratio = trade_plan['take_profit_1'].get('rr_ratio', '')
                message += f"🎯 TP1: {tp1_price} {rr_ratio}\n"

        # Add timestamp
        now = datetime.now().strftime("%I:%M %p UTC")
        message += f"\n⏰ {now}"

        # Add link to dashboard
        message += f"\n\n[View Full Analysis](https://fx-trading-web-zcca.vercel.app/pro-trader-gold)"

        # Send the notification
        return await self.send_message(message)

    async def send_test_message(self) -> bool:
        """Send a test message to verify configuration"""
        message = """
🤖 *Telegram Notifications Active*

Your Pro Trader Gold alerts are now configured!

You'll receive notifications when:
• Confluence score ≥ 7 points
• High-quality setups detected
• Both bullish and bearish opportunities

Dashboard: [Pro Trader Gold](https://fx-trading-web-zcca.vercel.app/pro-trader-gold)
"""
        return await self.send_message(message)


# Global instance
notifier = TelegramNotifier()


async def send_setup_notification(
    direction: str,
    setup_data: Dict[str, Any]
) -> bool:
    """
    Send notification for a high-quality setup

    Args:
        direction: "BULLISH" or "BEARISH"
        setup_data: The complete setup data from pro trader analysis
    """
    score = setup_data.get('total_score', 0)
    confidence = setup_data.get('confidence', 'MEDIUM')
    current_price = setup_data.get('current_price', 0)
    confluences = setup_data.get('confluences', [])
    trade_plan = setup_data.get('trade_plan', {})

    return await notifier.send_confluence_alert(
        direction=direction,
        score=score,
        confidence=confidence,
        current_price=current_price,
        confluences=confluences,
        trade_plan=trade_plan
    )
