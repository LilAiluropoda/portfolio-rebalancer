---
date: 2026-09-01
topic: leaps-lifecycle-leverage
---

# Lifecycle Investing via LEAPS Stock Replacement

## Summary

Extend the portfolio rebalancer to implement lifecycle investing: the user supplies a portfolio-level leverage target, and the tool generates a combined stock + LEAPS trade list that delivers the target exposure through one designated LEAPS sleeve. The tool values options from Alpaca, sizes and rolls contracts by rule, and reports achieved versus target leverage.

## Problem Frame

The rebalancer today handles equities and cash only: it values positions from a CSV, generates integer-share trades toward target weights, applies Futu's equity fee schedule, and reports post-trade ratios. Nothing in the pipeline can express an option position — no contract pricing, no delta semantics, no per-contract fees, no expiry.

The strategy this blocks is lifecycle investing (Ayres/Nalebuff): hold leveraged equity exposure while young, de-lever along a personal glide path. LEAPS calls are the chosen vehicle because the premium at risk is defined — no margin call, no interest drag. Running that strategy by hand means pricing contracts, computing delta-adjusted exposure, sizing whole contracts, tracking decay, and rolling before theta eats the position — every month, by hand. The tool should do all of that arithmetic and leave the user two decisions: the leverage number and a yes/no on any contract it picks.

## Key Decisions

- **Portfolio-level leverage as the direct input.** The user types weights (summing to 100) and one leverage target L; the tool derives all sleeve sizing. Rejected: a per-sleeve multiplier (forces mental math to land a portfolio number) and per-sleeve scaling of L (total lands below target when only one sleeve is leveraged).

- **One designated LEAPS sleeve concentrates the entire excess.** The designated underlying's exposure is delivered through LEAPS (shares may supplement), carrying its base weight plus the portfolio-wide excess (L−1) × MV. Other sleeves stay at 1x. Capital freed by replacement funds the remaining sleeves; the residue is the cash cushion.

- **Delta-adjusted exposure.** One contract counts as 100 × delta shares-equivalent. Notional counting overstates true leverage by 10–20% at typical deep-ITM deltas — material when the strategy is "hold exactly 1.5x."

- **Option data from Alpaca at run time.** Premium = mid of the latest quote; delta = snapshot greeks with a tool-computed Black-Scholes fallback when null. Rejected manual CSV entry once a fetchable source existed; rejected last-trade pricing after a live sample showed it 3 days stale and 17% off mid. Equity valuation is unchanged (yfinance at the CSV timestamp, run same-day on T0 with ~15-minute delayed quotes).

- **Tool picks contracts, user confirms.** Selection follows a stated rule and requires an explicit yes/no before the pick enters the trade list. The user is betting real money on a monthly cadence; a cheap guardrail against a bad pick is worth one prompt.

- **Time-based roll rule instead of modeling theta.** Exit any contract with ≤ 12 months remaining and buy a far-dated replacement. This leaves the decay window before it bites and keeps the tool free of any theta forecast.

- **LEAPS over margin and futures.** Margin financing would be the smaller code change but carries margin-call risk and interest drag — the failure mode lifecycle investing exists to avoid.

## Requirements

### Inputs and valuation

- R1. The input CSV carries current equity positions, held LEAPS positions (OCC symbol and contract quantity, no target weight of their own), cash, and per-underlying target weights summing to 100. A `leapsSleeve` marker on the underlying's row designates the LEAPS sleeve, and the portfolio leverage target L is supplied as a command-line flag.
- R2. LEAPS are valued from Alpaca's option snapshot at run time: premium is the mid of the latest quote, and delta comes from snapshot greeks with a Black-Scholes fallback when null.
- R3. Equities and cash keep the current valuation path (yfinance closing price at the CSV timestamp).
- R4. Exposure is delta-adjusted: one LEAPS contract contributes 100 × delta shares-equivalent of its underlying.

### Target construction

- R5. Total target equity exposure equals L × portfolio market value.
- R6. Every non-designated underlying targets weight × MV in shares.
- R7. The designated LEAPS sleeve targets its base weight plus the entire portfolio excess — (weight + L − 1) × MV of exposure. Whole contracts fill first, sized to the largest count not exceeding the sleeve target, and shares of the designated underlying top up the residual exposure.
- R8. Contract and share quantities are whole numbers: contract counts round to nearest, with the report flagging whether the residual is overshoot or undershoot. Share quantities stay best-effort as today.

### Contract lifecycle

- R9. A held contract with more than 12 months to expiry is kept, and the tool trades only the quantity delta.
- R10. A held contract with 12 months or fewer triggers a roll: sell the held contract and buy a rule-selected replacement.
- R11. When no contract is held, the tool generates an initiation trade for the designated sleeve.
- R12. Contract selection prefers the longest standard expiry at least 18–24 months out, the strike nearest ~0.85 delta, and filters for liquidity (bid-ask spread and open interest).
- R13. Any contract the tool selects is proposed with its reasoning and requires explicit user confirmation before entering the trade list.

### Trades and reporting

- R14. Every trade row carries a human-readable reason (drift rebalance, roll triggered, initiation, cash residual).
- R15. Option trades use Futu HK's US options fee schedule (verified schedule under Dependencies); equity trades keep the existing equity schedule.
- R16. The post-trade report shows target versus achieved exposure per underlying, achieved total leverage, and tracking error from lot rounding, in the same spirit as today's ratio-difference table.

## Key Flows

- F1. Monthly rebalance run
  - **Trigger:** User runs the tool with a T0-dated CSV (positions, weights, L).
  - **Steps:** Value equities (yfinance) and LEAPS (Alpaca snapshot); compute current and target exposures; decide the LEAPS action per R9–R11; confirm any selected contract per R13; generate stock and option trades, sells before buys; apply per-type fees.
  - **Outcome:** A combined trade list with reasons and costs, plus the post-trade exposure report.

- F2. Contract decision (runs inside F1)
  - **Trigger:** The designated sleeve reaches its lifecycle check.
  - **Steps:** Time-to-expiry branches to keep-and-resize, roll, or initiate; selection applies R12 when a contract must be chosen; confirmation gates R13.
  - **Outcome:** Zero or one confirmed contract decision feeding trade generation.

## Acceptance Examples

- AE1. Roll fires before decay
  - **Covers R10, R13, R14.**
  - **Given** the held LEAPS has 9 months to expiry,
  - **When** the run reaches the contract decision,
  - **Then** the trade list contains a sell of the held contract and a proposed replacement per R12, each row carrying a roll reason, and the buy enters only after confirmation.

- AE2. Initiation sizes whole contracts
  - **Covers R5, R7, R8, R16.**
  - **Given** MV 100k, VOO weight 55, L 1.5, and a 0.85-delta contract on a $700 underlying (59.5k exposure per contract),
  - **When** no LEAPS is held and the user confirms the pick,
  - **Then** the sleeve target is 105k exposure ((0.55 + 0.5) × 100k), sizing to 2 contracts (119k, achieved leverage ~1.64x), and the report shows the ~0.14x overshoot as tracking error.

- AE3. Healthy contract is not churned
  - **Covers R9.**
  - **Given** the held LEAPS has 20 months to expiry,
  - **When** the run executes,
  - **Then** no selection or roll occurs and only the quantity delta is traded.

## Success Criteria

- A monthly rebalance requires no manual pricing, exposure math, or contract searching — only the leverage number and contract confirmations.
- Proposed trade lists land achieved leverage within ~0.15x of target at current portfolio size (~100k), with the gap always visible in the report.
- No held contract ever reaches the final 12 months while the tool is in regular use.

## Scope Boundaries

- Deferred for later: multiple LEAPS sleeves, contract ladders, put-based leverage, glide-path automation (L stays manual per rebalance).
- Outside this tool's identity: margin or futures leverage, backtesting with historical option prices, trade execution (the tool suggests; the user executes at Futu).

## Dependencies and Assumptions

- Alpaca Market Data API account and keys (user has or will create). Keys are scoped to read-only market data — no trading or account permissions — and stored in a gitignored `.env` file loaded via `python-dotenv` (already a dependency), never in source or tracked files. The free `indicative` feed is acceptable: delayed trades and modified quotes, greeks occasionally null — covered by the Black-Scholes fallback.
- Futu HK US stock/index options fee schedule, verified by the user (all USD):

| Component | Amount | Applies to |
|---|---|---|
| Commission | $0.65/contract at premium > $0.1; min $1.99/order | all trades |
| Platform fee (fixed package) | $0.30/contract | all trades |
| Options Regulatory Fee (ORF) | $0.013/contract | all trades |
| OCC fee | $0.02/contract, max $55/trade | all trades |
| Settlement fee | $0.18/contract | all trades |
| Consolidated Audit Trail fee | $0.0003/contract | all trades |
| SEC regulatory fee | $0.0000206 × trade value, min $0.01/trade | sells only |
| FINRA trading activity fee | $0.00329/contract, min $0.01/trade | sells only |

  Fee notes: the fixed platform package is chosen — the tiered package only wins above ~6,800 contracts/month. Same-day transactions settle as one order for the commission minimum. Each leg of a multi-leg order is charged separately. The premium ≤ $0.1 commission tier ($0.15/contract) never applies to deep-ITM LEAPS.
- LEAPS liquidity is adequate on the designated underlying for mid-quote valuation to be meaningful.
- All positions and cash are USD.

## Outstanding Questions

### Resolve before planning

- None blocking.

### Deferred to planning

- Exact CSV schema for L and LEAPS rows (new columns versus preamble).
- Black-Scholes fallback parameters: risk-free rate source and IV derivation when snapshot greeks are null.
- Liquidity filter thresholds (minimum open interest, maximum acceptable spread).
- Share-buy priority order when remaining cash cannot fund all sleeves.

## Sources and Research

- Alpaca Market Data API — option chain snapshots (`docs.alpaca.markets`): latest quote/trade, BS greeks, IV; `indicative` vs `opra` feeds.
- Live sample fetched by the user (VOO260904C00680000): mid 24.22 vs last trade 29.30 three days stale; delta 0.9399 present on the free feed.
- Futu HK published US stock/index options fee schedule (user-supplied, current as of 2026-09; full table under Dependencies).
- Existing implementation to extend: `main.py` — price data source abstraction (~line 26), trading platform fee abstraction (~line 62), position enrichment (~line 135), trade generation with sells-first ordering (~line 212).

## Deferred / Open Questions

### From 2026-09-01 review

- **Held LEAPS on a non-designated underlying has no defined behavior** — Contract lifecycle (P1, feasibility + adversarial, confidence 100)

  R9–R11 only define behavior for the designated sleeve's contract, but R1 accepts held LEAPS positions for any underlying. If the user changes the designated sleeve, sets a target weight to zero, or holds legacy contracts elsewhere, the tool has no rule: it will neither size toward them (R6/R7 give no target for them) nor generate an exit, so exposure math silently ignores real positions or the implementer must invent a liquidation rule.

- **No defined behavior when the user declines a contract confirmation** — Contract lifecycle (P1, product-lens, confidence 75)

  The UX promise rests on "a yes/no on any contract it picks," but nothing defines the decline path. The first time the user rejects a pick (illiquid spread, wrong expiry, bad delta), the run has no defined output: does the designated sleeve fall back to shares at base weight with the shortfall shown as tracking error, does the run proceed with the sleeve untouched, or does it abort? Planning will have to invent product behavior mid-flight unless decided.
