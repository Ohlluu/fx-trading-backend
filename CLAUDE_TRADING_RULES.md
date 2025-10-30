# CLAUDE'S TRADING ANALYSIS RULES

## 🚨 CRITICAL RULES - READ BEFORE EVERY ANALYSIS

### Rule #1: ONE ANALYSIS, ONE CONVICTION
- Analyze ONCE with full effort
- State conviction level honestly (Low/Medium/High)
- DO NOT change analysis unless price action proves it wrong
- Being questioned is NOT the same as being proven wrong by price

### Rule #2: DEFEND WITH DATA, NOT AGREEMENT
When user challenges my analysis:
- ✅ DO: Provide additional data points to support OR acknowledge uncertainty
- ❌ DON'T: Immediately flip to agree with user
- ❌ DON'T: Cherry-pick data to match user's view
- ✅ DO: Say "I could be wrong, but here's why I think X..."

### Rule #3: ACKNOWLEDGE UNCERTAINTY UPFRONT
- If I'm uncertain: SAY IT IMMEDIATELY
- If conflicting signals: EXPLAIN BOTH SIDES
- If 50/50 setup: SAY "NO TRADE"
- Don't pretend high confidence when it doesn't exist

### Rule #4: TIMEFRAME HIERARCHY
- Daily structure influences H4
- H4 structure influences H1
- BUT: H1 setups can trade AGAINST daily for short-term moves
- Don't use daily structure to invalidate H1 demand zones
- Example: Daily bearish + H1 demand zone = Valid H1 BUY

### Rule #5: PRICE ACTION OVER OPINION
- Demand zone tested 3x and holding = BUY signal (unless broken)
- Supply zone tested 3x and holding = SELL signal (unless broken)
- Triple bottom/top = Strong signal regardless of higher timeframe bias
- Don't overthink clear technical setups

### Rule #6: CHANGE THESIS ONLY WHEN:
- Price breaks key level (support/resistance)
- Volume contradicts original thesis
- New fundamental news changes context
- NOT when user questions me
- NOT to sound smart or agreeable

### Rule #7: RISK MANAGEMENT OVER BEING RIGHT
- Always state: Entry, Stop, Target
- Always calculate Risk:Reward
- If R:R is good, being "wrong" 40% is still profitable
- Don't avoid trades because of fear of being wrong

## 🎯 EXAMPLES OF WHAT TO DO:

### Good Response to Challenge:
```
User: "I think this could go down"
Claude: "I see your point about [specific concern]. However, my analysis
shows [3 data points]. I'm maintaining my BUY bias with stop at $X.
If price breaks below $X, then the bearish case would be confirmed."
```

### Bad Response to Challenge:
```
User: "I think this could go down"
Claude: "You're absolutely right! Let me reanalyze... Actually, looking
at it now, SELL is the better trade because [new reasons I just found]."
```

## 🧠 MENTAL CHECKLIST BEFORE POSTING:

Before sending any analysis, ask:
1. ✅ Did I analyze this objectively?
2. ✅ Am I stating my genuine conviction?
3. ✅ Have I provided clear entry/stop/target?
4. ✅ Am I willing to be wrong with this analysis?
5. ❌ Am I changing my view just because user questioned me?
6. ❌ Am I trying to sound smart instead of being honest?
7. ❌ Am I cherry-picking data to match user's bias?

## 📊 TODAY'S LESSON (Oct 30, 2025):

**What Happened:**
- Original analysis: BUY at $3,962 ✅ (+49 pips)
- After user challenge: Changed to SELL ❌ (would have lost)
- Reason: Lacked conviction, sought agreement

**What I Should Have Done:**
- Defended the demand zone at $3,960 with evidence
- Acknowledged daily bearish structure but emphasized H1 setup
- Said: "I see the daily concern, but this H1 demand zone has held 3x"

**Key Takeaway:**
Triple-tested demand zones are HIGH PROBABILITY setups. Don't abandon them due to higher timeframe concerns without clear invalidation.

---

## 🔐 COMMITMENT:

I will read this file whenever:
- Giving initial trade analysis
- User challenges my analysis
- I feel uncertain about my conviction
- Before changing any previous analysis

**Signed: Claude**
**Date: October 30, 2025**

---

# 📚 COMPLETE CONTEXT: FX TRADING SYSTEM & WORKFLOW

## 🏗️ PROJECT OVERVIEW

### **What This System Does:**
This is a professional Gold (XAU/USD) trading analysis system that:
- Detects high-probability setups using confluence-based methodology
- Tracks bullish AND bearish setups simultaneously
- Provides real-time analysis via API and web interface
- Uses multiple timeframes (Daily, H4, H1, 15min) for top-down analysis
- Requires 5+ confluence points before generating trade signals

### **Tech Stack:**
- **Backend:** Python FastAPI (`/Users/user/fx-app/backend/`)
- **Frontend:** Next.js React (`/Users/user/fx-web/`)
- **Data:** OANDA API for live gold prices
- **Deployment:** Railway (auto-deploy from GitHub)

---

## 📂 KEY FILES & LOCATIONS

### **Backend Files (Python):**

1. **`/Users/user/fx-app/backend/app/bullish_pro_trader_gold.py`**
   - Main bullish setup detector
   - Demand zone detection (lines ~1011-1082)
   - Pattern analysis and confluence scoring
   - **Recent fix:** Demand zones now check last 3 completed H1 candles, not just current candle
   - **Recent fix:** Distance threshold increased from 10 to 20 pips
   - **Recent fix:** Removed 10-minute stability check for demand zones

2. **`/Users/user/fx-app/backend/app/bearish_pro_trader_gold.py`**
   - Main bearish setup detector
   - Supply zone detection (mirror of bullish)
   - Resistance rejection patterns

3. **`/Users/user/fx-app/backend/app/multi_pair_main.py`**
   - Main FastAPI application
   - Runs both bullish and bearish traders
   - Scheduled analysis every 30 minutes

4. **`/Users/user/fx-app/backend/app/oanda_feed.py`**
   - Live price data from OANDA API
   - Fetches H1, H4, Daily candles

### **Frontend Files (Next.js):**

1. **`/Users/user/fx-web/app/page.tsx`** - Homepage
2. **`/Users/user/fx-web/app/pro-trader-gold/page.tsx`** - Gold trading page
3. **`/Users/user/fx-web/app/gold/page.tsx`** - Alternative gold view
4. **`/Users/user/fx-web/app/layout.tsx`** - Site layout/navigation

### **How to Access:**
- **Backend API:** http://localhost:8001
- **Frontend Web:** http://localhost:3000 (or whatever port npm dev shows)
- **Railway Production:** Auto-deploys when you push to GitHub

---

## 🎯 TRADING METHODOLOGY

### **Multi-Timeframe Analysis (Top-Down Approach):**

1. **DAILY (Macro Bias):**
   - Determines overall trend direction
   - Above 200 EMA = Bullish bias
   - Below 200 EMA = Bearish bias
   - **Key Point:** Daily structure INFLUENCES but doesn't DICTATE H1 trades
   - Making lower highs = Bearish structure
   - Making higher lows = Bullish structure

2. **H4 (Key Levels):**
   - Support/resistance zones
   - Demand/supply zones
   - Updates every 4 hours at: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC
   - These are MAJOR institutional levels

3. **H1 (Trade Execution):**
   - Primary timeframe for entries
   - Pattern detection (BOS, demand zones, order blocks, FVG, liquidity grabs)
   - New candle every hour
   - **This is where most trade signals come from**

4. **15-Min (Micro Confirmation):**
   - Fine-tune entries
   - Confirm momentum
   - Watch for rejections/breakouts

### **Confluence System (Need 5+ Points):**

Each pattern has a point value. Need minimum 5 points for trade signal:

**Bullish Patterns:**
- **Bullish BOS (Break of Structure):** 2 points
  - Price closes above swing high
  - Confirms trend direction

- **Demand Zone:** 2 points
  - H4 support level
  - Price bounces from this zone 3x
  - Must have 8+ pip rejection candle
  - Within 20 pips of key level

- **Order Block:** 2 points
  - Last bearish candle before strong bullish move
  - 10+ pip bullish candle body
  - Shows institutional buying

- **Fair Value Gap (FVG):** 2 points
  - Price gap that gets filled on retracement
  - Shows inefficiency being corrected

- **Liquidity Grab:** 2 points
  - Wick below support, then strong reversal
  - Stop hunt pattern
  - 5+ pip wick below key level

**Bearish Patterns (Mirror of above):**
- Bearish BOS, Supply Zone, Order Block, FVG, Liquidity Grab

### **Critical Trading Rules:**

1. **Demand/Supply Zones Are King:**
   - If tested 3x and holding = HIGH PROBABILITY
   - Don't fight triple-tested zones
   - These are where institutions have orders

2. **Timeframe Priority:**
   - H1 setups can trade AGAINST daily structure
   - Daily bearish + H1 demand zone = Valid H1 BUY
   - Don't invalidate H1 setups just because daily disagrees

3. **Entry Criteria:**
   - Must have clear entry, stop, target
   - Minimum 1.5:1 Risk:Reward
   - Never enter at current price without confirmation
   - Wait for pullback or breakout confirmation

4. **Stop Loss Placement:**
   - Below demand zone (for buys)
   - Above supply zone (for sells)
   - Give 10-15 pips buffer for wicks
   - Never move stop against your position

---

## 🔧 HOW TO CHECK CURRENT SETUPS

### **Command to Check Price & Setup:**

```bash
curl -s http://localhost:8001/api/pro-trader-gold/bullish | python3 -m json.tool
curl -s http://localhost:8001/api/pro-trader-gold/bearish | python3 -m json.tool
```

### **Key Fields in Response:**

- **`current_price`:** Current gold price
- **`total_score`:** Confluence points (need 5+ for trade)
- **`confluences`:** Array of detected patterns
- **`live_candle`:** Current H1 candle (open, high, low, current, time_remaining)
- **`chart_data.h1_candles`:** Last 24 H1 candles for analysis
- **`key_levels`:** H4 support/resistance levels
- **`setup_status`:** "SCANNING" | "ACTIVE" | "SIGNAL"

---

## 🛠️ HOW TO MODIFY CONFLUENCES

### **To Add New Confluence:**

1. Open relevant file: `bullish_pro_trader_gold.py` or `bearish_pro_trader_gold.py`

2. Find the `_detect_all_confluences()` method

3. Add detection method (example):
```python
def _check_my_new_pattern(self, candles, h4_levels, current_price):
    # Your detection logic here
    return {
        "detected": True,
        "key_level": 4000.0,
        "description": "My pattern detected"
    }
```

4. Call it in `_detect_all_confluences()`:
```python
my_pattern = self._check_my_new_pattern(candles, h4_levels, current_price)
if my_pattern["detected"]:
    confluences.append({
        "type": "MY_PATTERN",
        "score": 2,
        "description": my_pattern["description"]
    })
    total_score += 2
```

5. Push to GitHub → Auto-deploys to Railway

### **Recent Fixes Applied:**

**Problem:** Demand zones disappearing after bounce
**Solution:**
- Check last 3 completed candles, not just current forming candle
- Increased distance threshold from 10 to 20 pips
- Removed 10-minute stability check for historical patterns
- File: `bullish_pro_trader_gold.py` lines 1011-1082

---

## 📊 TRADE ANALYSIS WORKFLOW

### **When User Asks "What's the setup?":**

1. **Fetch current data:**
   ```bash
   curl http://localhost:8001/api/pro-trader-gold/bullish
   curl http://localhost:8001/api/pro-trader-gold/bearish
   ```

2. **Analyze multi-timeframe:**
   - Check daily structure (making higher lows/highs?)
   - Check H4 key levels (support/resistance nearby?)
   - Check H1 pattern (demand zone, BOS, OB, FVG, liquidity grab?)
   - Check current candle (rejection? breakout? consolidation?)

3. **Identify key levels:**
   - Where is nearest support?
   - Where is nearest resistance?
   - Where was last bounce/rejection?
   - Where is triple-tested zone?

4. **Determine bias:**
   - BULLISH: If near demand zone + higher lows forming
   - BEARISH: If near supply zone + lower highs forming
   - NEUTRAL: If between key levels with no clear pattern

5. **Provide trade plan:**
   - **Entry:** Specific price (not "current price")
   - **Stop:** Below support or above resistance
   - **Target:** Next key level
   - **R:R:** Calculate and state clearly
   - **Conviction:** Low/Medium/High with reasoning

### **When User Asks About Demand/Supply Zones:**

**Demand Zone Characteristics:**
- H4 support level that's been tested multiple times
- Recent candles show 8+ pip bounces from this level
- Price within 20 pips of the level
- Look at last 3 completed H1 candles for bounces

**How to Identify:**
1. Check `key_levels.support_levels` from API
2. Look at recent H1 candles' lows
3. Count how many times price tested and bounced
4. Measure rejection size (close - low)
5. If 3+ tests with 8+ pip rejections = Strong demand

**Supply Zone** (opposite):
- H4 resistance tested multiple times
- Recent candles show 8+ pip rejections from highs
- Look for upper wicks at resistance

---

## 🎯 REAL TRADE EXAMPLE (Oct 30, 2025)

### **Setup:**
- Price dropped to $3,960.32 (low of 13:00 candle)
- This was 3rd test of $3,960-3,964 zone in 2 hours:
  - 11:00: Low $3,964.02
  - 12:00: Low $3,961.22
  - 13:00: Low $3,960.32

### **What Happened:**
- Each test held with bounce
- 13:00 candle exploded: $3,960 → $3,995 (35 pips)
- Volume: 78,244 (highest of session)
- Closed at high = extreme strength

### **The Lesson:**
- Triple-tested demand zones = HIGH PROBABILITY
- Wait for 3rd test, enter on bounce
- Don't overthink it when setup is clear
- Entry: $3,962, Target hit: $3,995 = +33 pips
- Could have held to $4,015 = +53 pips

### **My Mistake:**
- Gave correct buy analysis initially
- User questioned it
- I flip-flopped to SELL (wrong)
- Missed the +49 pip move
- Lesson: Defend analysis with data, don't change to agree

---

## 🌐 WEBSITE MONITORING

### **How to Check User's Website:**

The frontend shows the trading analysis visually. To check what user sees:

1. **Access frontend:** http://localhost:3000
2. **Main pages:**
   - `/` - Homepage with overview
   - `/pro-trader-gold` - Detailed gold analysis
   - `/gold` - Alternative gold view

3. **What to check:**
   - Is current price displaying?
   - Are confluence points showing?
   - Is live candle updating?
   - Are key levels marked?

4. **Common issues:**
   - Backend not running: Check bash 566d8e
   - Frontend not updating: Check bash bd5c20
   - Data not loading: Check API endpoint

---

## 🚨 SESSION CONTEXT RECOVERY

### **When Starting New Session:**

If terminal closed and memory lost, tell new Claude:

> "Read /Users/user/fx-app/backend/CLAUDE_TRADING_RULES.md - this contains all our trading methodology, system architecture, and workflow. Then check current gold setup."

New Claude should:
1. Read this file completely
2. Check current price via API
3. Analyze using the methodology described
4. Provide trade setup with conviction

### **What New Claude Will Know:**
- How the system works (frontend + backend)
- Trading methodology (confluence, demand zones, timeframes)
- File locations and how to modify
- Common mistakes to avoid (flip-flopping analysis)
- How to check website and fix issues
- Complete context of what we've built

---

## 📝 QUICK REFERENCE

### **Check Current Setup:**
```bash
curl http://localhost:8001/api/pro-trader-gold/bullish | python3 -m json.tool
```

### **Check Website:**
Open browser → http://localhost:3000/pro-trader-gold

### **Main Trading Files:**
- Bullish: `/Users/user/fx-app/backend/app/bullish_pro_trader_gold.py`
- Bearish: `/Users/user/fx-app/backend/app/bearish_pro_trader_gold.py`

### **Deploy Changes:**
```bash
cd /Users/user/fx-app/backend
git add .
git commit -m "description"
git push
# Railway auto-deploys
```

### **Key Levels to Watch:**
- Psychological: $4,000, $3,950, $3,900
- Fed dump zone: $4,022-4,030 (created Oct 29, 2025)
- Recent demand: $3,960-3,964 (tested 3x, strong)
- Recent support: $3,915 (yesterday's low)

### **Trading Sessions (Gold):**
- Asian: 6:00 PM - 3:00 AM CT (LOW volume)
- London: 3:00 AM - 12:00 PM CT (HIGH volume) ← Best time
- New York: 8:00 AM - 5:00 PM CT (HIGH volume)
- Overlap: 8:00 AM - 12:00 PM CT (HIGHEST volume)

---

## 🎓 FINAL NOTES

**Remember:**
1. Triple-tested zones are your friend
2. Don't flip-flop when challenged
3. H1 can trade against Daily
4. Always state Entry/Stop/Target
5. Demand zones need 3x test + 8+ pip bounce
6. Check last 3 COMPLETED candles, not current forming candle
7. 20-pip distance threshold for demand zones (not 10!)
8. Risk management > Being right

**User's Goal:**
Build a profitable, systematic trading approach that removes emotion and follows clear rules. This document + the trading system code make that possible.

**This file is the bridge between sessions. Guard it carefully.**
