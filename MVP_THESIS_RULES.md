# PropsAI MVP Trade Thesis Rules

Each strategy must meet its thesis requirements before a trade is placed.
These are the minimum bars — like a hedge fund's investment mandate.

---

## 🌤️ WEATHER — Consensus Forecast Divergence

**Core thesis:** We have better probability estimates than Kalshi's market because we
aggregate 4 independent forecast models (NWS, Open-Meteo ensemble, Tomorrow.io,
Visual Crossing). When all sources agree and Kalshi is mispriced, we have edge.

**Required to trade:**
- ≥ 2 forecast sources returned data
- Source spread ≤ 6°F (sources agree — tight consensus = high confidence)
- Edge ≥ 8% vs Kalshi implied price
- Confidence score ≥ 0.30 (already enforced)

**Reject if:**
- Only 1 source available (no consensus possible)
- Sources disagree by > 6°F (too uncertain)
- Market expires in < 2 hours (not enough time for edge to materialize)

**Duplicate bets OK:** Different strike prices on the same city are independent bets
(e.g. NYC high > 40°F YES and NYC high > 45°F NO are both valid simultaneously).

**Current status:** ✅ STRONGEST STRATEGY — all 4 APIs active and keyed.

---

## 📈 FINANCE (S&P 500 / Nasdaq) — Multi-Signal Alignment

**Core thesis:** We trade daily close markets only when multiple independent signals
agree on direction. A single momentum reading is noise. Three aligned signals is
a thesis.

**Required to trade — need ≥ 3 of these 5 signals aligned:**
1. **Intraday momentum** > 0.3% in trade direction
2. **Futures (ES/NQ)** pointing in same direction as trade
3. **VIX regime** — VIX < 20 for YES bets (low fear = trend continuation likely),
   OR VIX > 25 for NO bets (high fear = market likely to close down)
4. **Moving average position** — price above 50-day MA for YES, below for NO
5. **News sentiment** — ≥ 3 articles with sentiment aligned to trade direction

**Hard rejects (no trade regardless of edge):**
- VIX > 35 (extreme volatility — daily close is unpredictable)
- Market within 30 minutes of close (too late, edge already priced in)
- Futures and intraday momentum directly contradict each other

**Duplicate bets OK:** Betting both S&P YES on a threshold market AND a bracket
market is valid IF both pass the 3-signal test independently. They are different
market structures with different payoffs.

**Current gap:** No earnings calendar, no Fed speaker schedule, no options flow.
These are future improvements.

---

## ₿ CRYPTO — Momentum + Funding Rate Confirmation

**Core thesis:** Perpetual futures funding rates represent institutional directional
bias. When retail momentum (price action) aligns with institutional positioning
(funding rate), the signal is high conviction. When they contradict, we stay out.

**Required to trade:**
- Momentum signal (5m + 1m combined) points in a direction
- Funding rate does NOT directly contradict momentum:
  - If betting UP: funding rate ≥ -0.01% (not heavily negative/short-biased)
  - If betting DOWN: funding rate ≤ +0.01% (not heavily positive/long-biased)
- Edge ≥ 5% vs Kalshi implied price
- Confidence ≥ 0.15

**Strong signal (both required for highest position size):**
- Momentum AND funding rate both point same direction
- News sentiment aligned

**Reject if:**
- Momentum says UP but funding rate is strongly negative (longs being squeezed)
- Momentum says DOWN but funding rate is strongly positive (shorts being squeezed)
- These contradictions are the #1 cause of bad crypto trades

**Duplicate bets:** One bracket per coin per cycle (already enforced).

---

## 🏈 SPORTS / NBA PROPS — Sharp Book Consensus

**Core thesis:** Sharp sportsbooks (Pinnacle, Circa) have the best probability
estimates in the world. When Kalshi's implied probability diverges from sharp
consensus by > 3%, we have edge. This is textbook sports arbitrage.

**Required to trade:**
- Sharp book line available for the game/prop
- Edge ≥ 3% vs Kalshi implied
- Confidence ≥ 0.85 (team/player match confirmed)
- Edge ≤ 15% (cap prevents bad team matches from trading)

**Reject if:**
- No sharp book line available (only soft books)
- Edge > 15% (likely a bad match, not real edge)
- Game starts in < 5 minutes (market already efficient)

**Duplicate bets OK:** Betting the same game on multiple Kalshi markets (e.g.,
game winner AND total) is valid — they are independent propositions with
independent sharp lines.

**Injury handling:**
- NBA Props: ✅ SportsDataIO official injury report (OUT/DOUBTFUL = hard skip,
  QUESTIONABLE = -0.15 confidence penalty) + player news headlines (-0.20 penalty)
- Game markets (winner/spread/total): relies on sharp books having already priced
  injuries in. InjuryScraper (Twitter/RSS) exists but is not yet called in the
  game-level cycle — breaking injuries after line-set are not caught independently.

**Current gap:** Wire InjuryScraper into game-level sports cycle to catch breaking
injuries (e.g. Woj tweet at 6pm that a star is out) that sharp books may not have
repriced yet on Kalshi.

---

## 📊 ECON — Data-Driven Probability vs Market Consensus

**Fed Funds Rate (STRONGEST):**
- FedWatch market-implied cut probability is real institutional consensus
- Trade when our FedWatch-derived probability diverges from Kalshi by > 5%
- This is the same data CME traders use — high confidence

**CPI / Unemployment / Gas (WEAKER):**
- Currently just trend extrapolation from FRED historical data
- Trade only when trend is very clear (last 3 readings all moving same direction)
- Edge ≥ 5%, Confidence ≥ 0.25

**Future improvement needed:** Bloomberg economist survey consensus for CPI/jobs.
Until then, econ trades other than Fed Funds should be sized conservatively.

---

## 🔄 GLOBAL RANKING — How These Rules Interact

When multiple strategies fire simultaneously, `execute_ranked_signals` ranks by
`edge × confidence`. The thesis rules above act as **hard gates** — a candidate
that doesn't meet its strategy's thesis is rejected before ranking.

This means:
- A weather trade with 3-source consensus and 12% edge will beat a finance trade
  with only 2-signal alignment and 8% edge
- A sports trade with sharp book confirmation will beat a crypto trade with
  contradictory momentum/funding signals

**The goal:** Every trade in the DB should have a defensible reason it was placed.
No more "momentum was slightly positive so we bet S&P YES."
