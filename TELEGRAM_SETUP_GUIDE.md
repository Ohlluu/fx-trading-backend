# 📱 Telegram Notifications Setup Guide

Get real-time alerts when high-quality trading setups (7+ confluence points) are detected!

---

## 🤖 Step 1: Create Your Telegram Bot

1. Open Telegram and search for **@BotFather**
2. Start a chat and send: `/newbot`
3. Follow the prompts:
   - Choose a name: `Pro Trader Gold Bot` (or any name you like)
   - Choose a username: `ProTraderGoldBot` (must end with "bot")
4. **BotFather will give you a TOKEN** - copy and save it!
   - Example: `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz`

---

## 💬 Step 2: Get Your Chat ID

### Option A: Using IDBot (Easiest)
1. Search for **@myidbot** on Telegram
2. Start a chat and click **Start**
3. Send `/getid`
4. Copy the number (your Chat ID)

### Option B: Using Your Bot
1. Start a chat with your new bot (from Step 1)
2. Send any message to it (e.g., "Hello")
3. Open this URL in your browser (replace TOKEN with your bot token):
   ```
   https://api.telegram.org/botYOUR_BOT_TOKEN/getUpdates
   ```
4. Look for `"chat":{"id":123456789}` in the response
5. Copy that number (your Chat ID)

---

## ⚙️ Step 3: Configure Railway Environment Variables

1. Go to your Railway dashboard: https://railway.app
2. Select your **fx-trading-backend** project
3. Go to **Variables** tab
4. Add these two environment variables:

```bash
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
```

5. Click **Deploy** to redeploy with new settings

---

## 🌐 Step 4: Configure Frontend Settings (Alternative)

If you don't want to use Railway environment variables, you can configure it directly from the website:

1. Go to: https://fx-trading-web-zcca.vercel.app/pro-trader-gold
2. Click **⚙️ Settings** button (top right)
3. Enter your **Chat ID** (from Step 2)
4. (Optional) Enter your **Bot Token** if you want to use your own bot
5. Click **Save**
6. You should receive a test notification immediately!

---

## 📊 How It Works

**Automatic Notifications:**
- System checks for setups every hour (at :05 and :35)
- When confluence score reaches **7+ points**, you get notified
- Works for both **BULLISH** and **BEARISH** setups
- **30-minute cooldown** prevents spam

**Notification Includes:**
- Direction (Bullish/Bearish)
- Confluence score and confidence level
- Detected patterns (liquidity grabs, FVGs, order blocks, etc.)
- Trade plan (entry, stop loss, take profits)
- Link to full analysis dashboard

---

## 🧪 Testing Your Setup

### Test from Frontend:
1. Go to Settings modal
2. Enter your Chat ID
3. Click **📤 Test** button
4. Check your Telegram for test message

### Test from Backend:
```bash
curl -X POST https://web-production-8c5ca.up.railway.app/api/settings/telegram/test
```

---

## 🚨 Troubleshooting

**"Failed to send test message"**
- Double-check your Chat ID (no spaces, numbers only)
- Make sure you started a chat with the bot
- Verify Bot Token is correct

**"Telegram not configured"**
- Check Railway environment variables are set
- Redeploy after adding variables

**Not receiving notifications**
- Check that confluence score is 7+ points
- Remember the 30-minute cooldown period
- Verify bot is not blocked on Telegram

---

## 💡 Tips

- **Multiple Users**: Each user should configure their own Chat ID in frontend settings
- **Privacy**: Chat IDs are stored server-side only during session
- **Bot Commands**: You can add commands to your bot via @BotFather later
- **Custom Bot**: Using your own bot gives you full control and privacy

---

## 📝 Example Notification

```
📈 BULLISH GOLD SETUP READY 🟢

Confluence Score: 8 points
Confidence: HIGH CONFIDENCE
Current Price: $2,645.30

Detected Patterns:
• LIQUIDITY GRAB (+4)
• FVG (+2)
• ORDER BLOCK (+2)

Trade Plan:
📍 Entry: $2,646.00
🛑 Stop Loss: $2,640.50
🎯 TP1: $2,655.00 (1:1.5 R:R)

⏰ 3:25 PM UTC

View Full Analysis
```

---

## 🔐 Security Notes

- Never share your Bot Token publicly
- Keep your Chat ID private
- Environment variables on Railway are encrypted
- Frontend settings are session-based (not permanently stored)

---

## 🎯 Ready to Deploy!

Once configured, you'll automatically receive alerts for:
✅ High-quality setups (7+ confluence)
✅ Both bullish and bearish opportunities
✅ Real-time detection as soon as patterns form
✅ Direct links to full trade analysis

Happy Trading! 🚀
