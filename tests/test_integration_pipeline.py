from datetime import date, datetime

import pytest

from main import (
    CASH_TYPE,
    OPTION_TYPE,
    TradingPlatformFactory,
    applyTrades,
    buildExposureReport,
    buildSleeveTable,
    enrichPositions,
    executeTrades,
    planTrades,
    printExposureReport,
    Trade,
)
from conftest import (
    CHAIN_LATE_EXPIRY,
    FakeOptionSource,
    FakePriceSource,
    HELD_KEEP_EXPIRY,
    HELD_ROLL_EXPIRY,
    TODAY,
    cash,
    equity,
    frameWith,
    makeSnapshot,
    occ,
    option,
    tradesBySymbol,
)

SPOT = 700.0


def runPipeline(rows, optionSource, leverage=1.5, designated="VOO", prices=None):
    priceSource = FakePriceSource(prices or {"VOO": SPOT, "URA": 30.0, "USD": 1.0})
    frame = frameWith(rows, leverage=leverage)
    enriched = enrichPositions(frame, priceSource, optionSource)
    sleeveTable = buildSleeveTable(enriched, leverage, designated)
    planned = planTrades(enriched, sleeveTable, designated, leverage, optionSource, TODAY)
    platform = TradingPlatformFactory.getTradingPlatform("futubullUS")
    trades = executeTrades(planned, enriched, platform)
    return enriched, sleeveTable, trades


# --- (g) R20 fee invariant: fees live solely in the cash residual row ---


def test_premium_neutral_roll_still_ledgers_fees_and_mv_invariant():
    # Two contract trades whose marketValueChanges net to ~0 but carry fees.
    held = occ("VOO", HELD_ROLL_EXPIRY, 450.0)
    priceSource = FakePriceSource({"VOO": SPOT, "USD": 1.0})
    enriched = enrichPositions(
        frameWith([equity("VOO", 35, 100, sleeve="true"), cash(5000)]), priceSource, None
    )
    planned = [
        Trade(
            tradeId="1", instrumentId=held, instrumentType=OPTION_TYPE, price=20.0,
            sharesChange=-1.0, marketValueChange=-2000.0, timestamp=TODAY,
            quantityKind="contract", exposureChange=-100 * 0.85 * SPOT, reason="roll: exit",
        ),
        Trade(
            tradeId="2", instrumentId=occ("VOO", CHAIN_LATE_EXPIRY, 420.0),
            instrumentType=OPTION_TYPE, price=20.0,
            sharesChange=1.0, marketValueChange=2000.0, timestamp=TODAY,
            quantityKind="contract", exposureChange=100 * 0.85 * SPOT,
            reason="roll: replacement",
        ),
    ]
    trades = executeTrades(planned, enriched, TradingPlatformFactory.getTradingPlatform("futubullUS"))

    cashRows = [t for t in trades if t.quantityKind == "cash"]
    assert len(cashRows) == 1  # net flow ~0, but fees > 0 -> cash row present
    totalFees = sum(t.transactionCost for t in trades if t.quantityKind != "cash")
    assert totalFees > 0
    assert cashRows[0].marketValueChange == pytest.approx(-totalFees)

    # Invariant: post-trade total MV change == -totalFees exactly (R20)
    post = applyTrades(enriched, trades)
    assert post["marketValue"].sum() - enriched["marketValue"].sum() == pytest.approx(-totalFees)


def test_designated_sleeve_no_option_rows_no_source_falls_back_to_shares():
    # Primary initiation flow: sleeve designated, zero held contracts, no option
    # source constructed -> shares fallback, no crash.
    enriched, sleeveTable, trades = runPipeline(
        [
            equity("VOO", 35, 55, sleeve="true"),
            equity("URA", 267, 45),
            cash(29371),
        ],
        None,
    )
    assert all(t.quantityKind in ("share", "cash") for t in trades)
    vooShares = [t for t in trades if t.instrumentId == "VOO" and t.quantityKind == "share"]
    assert len(vooShares) == 1
    assert vooShares[0].reason == "shares fallback: no qualifying contract"
    # Fallback target is the base weight (0.55 x MV), not the leveraged target
    totalMV = 35 * SPOT + 267 * 30.0 + 29371
    expectedShares = int((0.55 * totalMV - 35 * SPOT) / SPOT)
    assert vooShares[0].sharesChange == pytest.approx(float(expectedShares))


def test_duplicate_trades_same_instrument_aggregate_before_join():
    # Duplicate instrumentId trade rows must not fan out into a cartesian join.
    priceSource = FakePriceSource({"VOO": SPOT, "USD": 1.0})
    held = occ("VOO", CHAIN_LATE_EXPIRY, 420.0)
    enriched = enrichPositions(
        frameWith([equity("VOO", 35, 100, sleeve="true"), option(held, 1), cash(5000)]),
        priceSource,
        FakeOptionSource(snapshots={held: makeSnapshot(held, "VOO", 20.0, 0.85, CHAIN_LATE_EXPIRY)}),
    )
    sellThenBuy = [
        Trade(
            tradeId="1", instrumentId=held, instrumentType=OPTION_TYPE, price=20.0,
            sharesChange=-1.0, marketValueChange=-2000.0, timestamp=TODAY,
            quantityKind="contract", underlying="VOO", reason="roll: exit",
        ),
        Trade(
            tradeId="2", instrumentId=held, instrumentType=OPTION_TYPE, price=20.0,
            sharesChange=2.0, marketValueChange=4000.0, timestamp=TODAY,
            quantityKind="contract", underlying="VOO", reason="roll: replacement",
        ),
    ]
    post = applyTrades(enriched, sellThenBuy)
    heldRows = [r for r in post.iter_rows(named=True) if r["instrumentId"] == held]
    assert len(heldRows) == 1  # aggregated once, no fan-out
    assert heldRows[0]["shares"] == pytest.approx(2.0)  # 1 held - 1 + 2
    assert heldRows[0]["marketValue"] == pytest.approx(2 * 20.0 * 100)


# --- (a) mixed frame end-to-end (AE2 initiation arithmetic) ---


def test_mixed_end_to_end_fees_cash_and_leverage():
    candidate = makeSnapshot(occ("VOO", CHAIN_LATE_EXPIRY, 420.0), "VOO", 24.22, 0.85, CHAIN_LATE_EXPIRY)
    source = FakeOptionSource(chainSnapshots=[candidate])

    enriched, sleeveTable, trades = runPipeline(
        [
            equity("VOO", 50, 55, sleeve="true"),
            equity("URA", 2133, 45),
            cash(1000),
        ],
        source,
    )

    bySymbol = tradesBySymbol(trades)
    contractBuy = bySymbol[candidate.symbol]
    assert contractBuy.sharesChange == 2  # AE2: 2 contracts = 119k exposure
    assert contractBuy.quantityKind == "contract"
    assert contractBuy.reason == "initiation"
    assert bySymbol["VOO"].sharesChange == -20  # share residual absorbs -14010.5
    assert bySymbol["VOO"].reason == "initiation share residual"
    assert bySymbol["URA"].sharesChange == -633  # drift rebalance to 45% of 99,990

    # Per-row fee minimums: 2-contract option order hits the 1.99 commission floor
    assert contractBuy.transactionCost == pytest.approx(
        1.99 + 2 * 0.30 + 2 * 0.013 + 2 * 0.02 + 2 * 0.18 + 2 * 0.0003
    )
    # Equity sell fees: per-row minimums (0.99 commission floor, 1.00 platform floor)
    assert bySymbol["VOO"].transactionCost == pytest.approx(0.99 + 1.00 + 20 * 0.003 + 0.01)

    # Cash row = -(net premium + share flows) - fees, exactly
    totalFees = sum(t.transactionCost for t in trades[:-1])
    netFlow = sum(t.marketValueChange for t in trades[:-1])
    cashRow = trades[-1]
    assert cashRow.instrumentId == "USD"
    assert cashRow.instrumentType == CASH_TYPE
    assert cashRow.quantityKind == "cash"
    assert cashRow.marketValueChange == pytest.approx(-netFlow - totalFees)

    # R19: achieved leverage = Σ achieved exposure / pre-trade MV
    report, achievedLeverage, reportFees = buildExposureReport(enriched, sleeveTable, trades)
    assert achievedLeverage == pytest.approx((140000 + 45000) / 99990)
    assert reportFees == pytest.approx(totalFees)

    voo = {r["underlying"]: r for r in report.iter_rows(named=True)}["VOO"]
    assert voo["postShares"] == 30
    assert voo["postContracts"] == 2
    assert voo["achievedExposure"] == pytest.approx(140000)
    assert voo["trackingError"] == pytest.approx(140000 - 104989.5)
    assert voo["isDesignated"]


# --- (b) AE1 / AE3 at pipeline level ---


def test_roll_pipeline_paired_rows_with_fee_minimums():
    held = occ("VOO", HELD_ROLL_EXPIRY, 450.0)
    replacement = makeSnapshot(
        occ("VOO", CHAIN_LATE_EXPIRY, 420.0), "VOO", 20.0, 0.86, CHAIN_LATE_EXPIRY
    )
    source = FakeOptionSource(
        snapshots={held: makeSnapshot(held, "VOO", 24.22, 0.85, HELD_ROLL_EXPIRY)},
        chainSnapshots=[replacement],
    )
    enriched, sleeveTable, trades = runPipeline(
        [
            equity("VOO", 35, 55, sleeve="true"),
            equity("URA", 267, 45),
            option(held, 1),
            cash(29371),
        ],
        source,
    )
    bySymbol = tradesBySymbol(trades)
    sell = bySymbol[held]
    buy = bySymbol[replacement.symbol]
    assert (sell.sharesChange, sell.reason) == (-1, "roll: exit")
    assert (buy.sharesChange, buy.reason) == (1, "roll: replacement")

    # Each contract row pays at least the 1.99 commission floor + per-contract extras
    assert sell.transactionCost >= 1.99 + 0.30 + 0.013 + 0.02 + 0.18 + 0.0003
    assert buy.transactionCost == pytest.approx(1.99 + 0.30 + 0.013 + 0.02 + 0.18 + 0.0003)

    nonCash = [t for t in trades if t.quantityKind != "cash"]
    cashRow = trades[-1]
    expectedCash = -sum(t.marketValueChange for t in nonCash) - sum(t.transactionCost for t in nonCash)
    assert cashRow.marketValueChange == pytest.approx(expectedCash)


def test_ae3_keep_pipeline_no_chain_call():
    held = occ("VOO", HELD_KEEP_EXPIRY, 450.0)
    source = FakeOptionSource(
        snapshots={held: makeSnapshot(held, "VOO", 24.22, 0.85, HELD_KEEP_EXPIRY)},
    )
    enriched, sleeveTable, trades = runPipeline(
        [
            equity("VOO", 35, 55, sleeve="true"),
            equity("URA", 267, 45),
            option(held, 2),
            cash(27646),
        ],
        source,
    )
    assert source.chainCalls == []  # healthy 20-month contract: no selection call
    contractTrades = [t for t in trades if t.quantityKind == "contract"]
    assert len(contractTrades) == 1
    assert contractTrades[0].sharesChange == -1  # resize only
    assert contractTrades[0].reason == "resize"
    _, _, fees = buildExposureReport(enriched, sleeveTable, trades)
    assert fees == pytest.approx(sum(t.transactionCost for t in trades[:-1]))


# --- (c) applyTrades new-instrument regression (join bug) ---


def test_apply_trades_new_instrument_gets_price_and_columns():
    priceSource = FakePriceSource({"VOO": SPOT, "USD": 1.0})
    enriched = enrichPositions(
        frameWith([equity("VOO", 35, 100, sleeve="true"), cash(1000)]), priceSource, None
    )
    newTrade = Trade(
        tradeId="1",
        instrumentId="AAPL",
        instrumentType="Equity",
        price=100.0,
        sharesChange=10.0,
        marketValueChange=1000.0,
        transactionCost=1.0,
        timestamp=TODAY,
    )
    post = applyTrades(enriched, [newTrade])
    byId = {r["instrumentId"]: r for r in post.iter_rows(named=True)}

    aapl = byId["AAPL"]
    assert aapl["closingPrice"] == pytest.approx(100.0)  # trade price survives the join
    assert aapl["shares"] == pytest.approx(10.0)
    # Fees are netted solely via the cash residual row, not per-row (R20)
    assert aapl["marketValue"] == pytest.approx(1000.0)
    assert aapl["kind"] == "equity"
    assert aapl["targetRatioPct"] == 0.0

    # option rows introduced by trades derive kind too
    optTrade = Trade(
        tradeId="2",
        instrumentId=occ("VOO", CHAIN_LATE_EXPIRY, 420.0),
        instrumentType=OPTION_TYPE,
        price=20.0,
        sharesChange=1.0,
        marketValueChange=2000.0,
        timestamp=TODAY,
    )
    post = applyTrades(enriched, [optTrade])
    optRow = {r["instrumentId"]: r for r in post.iter_rows(named=True)}[optTrade.instrumentId]
    assert optRow["kind"] == "option"
    assert optRow["closingPrice"] == pytest.approx(20.0)


# --- (d) zero trades ---


def test_zero_trades_no_fees_no_cash_row_report_still_prints(capsys):
    # Weights exactly at target with no cash row -> planner emits nothing:
    # 24500 = 0.55 * MV forces MV = 24500/0.55, hence URA = (MV - 24500)/30
    priceSource = FakePriceSource({"VOO": SPOT, "URA": 30.0})
    source = FakeOptionSource()
    balancedUra = (24500 / 0.55 - 24500) / 30
    frame = frameWith([equity("VOO", 35, 55), equity("URA", balancedUra, 45)], leverage=1.0)
    enriched = enrichPositions(frame, priceSource, None)
    sleeveTable = buildSleeveTable(enriched, 1.0, None)
    planned = planTrades(enriched, sleeveTable, None, 1.0, source, TODAY)
    assert planned == []

    trades = executeTrades(planned, enriched, TradingPlatformFactory.getTradingPlatform("futubullUS"))
    assert trades == []  # no fee artifacts, no cash row

    post = applyTrades(enriched, trades)
    assert post.height == enriched.height
    report, achievedLeverage, fees = buildExposureReport(enriched, sleeveTable, trades)
    assert fees == 0.0
    assert achievedLeverage == pytest.approx(1.0)  # L = 1: exposure == MV
    printExposureReport(report, achievedLeverage, fees)  # must not raise
    assert "VOO" in capsys.readouterr().out  # report table printed


# --- (e) legacy equivalence: sells first, buys capped by cash ---


def test_legacy_equity_cash_matches_old_share_logic():
    # MV = 35*700 + 267*30 + 1000 = 33,510
    # Old logic: VOO over target (24500 vs 18430.5) -> sell 8; URA under
    # (8010 vs 15079.5) -> want 235 shares, but cash = 1000 + 5600 (VOO sell
    # proceeds) = 6600 -> capped at int(6600/30) = 220. The cap binds.
    _, _, trades = runPipeline(
        [equity("VOO", 35, 55), equity("URA", 267, 45), cash(1000)],
        FakeOptionSource(),
        leverage=1.0,
        designated=None,
    )
    bySymbol = tradesBySymbol(trades)
    assert bySymbol["VOO"].sharesChange == -8
    assert bySymbol["URA"].sharesChange == 220  # capped by cash, not the 235 ideal

    nonCash = [t for t in trades if t.quantityKind != "cash"]
    cashRow = trades[-1]
    assert cashRow.marketValueChange == pytest.approx(
        -sum(t.marketValueChange for t in nonCash) - sum(t.transactionCost for t in nonCash)
    )


# --- (f) equity-buy truncation vs contract-buy immunity ---


def test_equity_buy_truncated_contract_buy_never_truncated():
    priceSource = FakePriceSource({"X": 700.0, "USD": 1.0})
    frame = frameWith([equity("X", 10, 100, sleeve="true"), cash(7000)])
    enriched = enrichPositions(frame, priceSource, None)

    planned = [
        Trade(
            tradeId="1", instrumentId="X", instrumentType="Equity", price=700.0,
            sharesChange=20.0, marketValueChange=14000.0, timestamp=TODAY,
            quantityKind="share", exposureChange=14000.0, reason="wants 20 shares",
        ),
        Trade(
            tradeId="2", instrumentId=occ("X", CHAIN_LATE_EXPIRY, 420.0),
            instrumentType=OPTION_TYPE, price=24.22,
            sharesChange=2.0, marketValueChange=2 * 24.22 * 100, timestamp=TODAY,
            quantityKind="contract", exposureChange=2 * 100 * 0.85 * 700.0,
            reason="initiation",
        ),
    ]
    platform = TradingPlatformFactory.getTradingPlatform("futubullUS")
    trades = executeTrades(planned, enriched, platform)

    contract = next(t for t in trades if t.quantityKind == "contract")
    share = next(t for t in trades if t.quantityKind == "share")
    cashRow = next(t for t in trades if t.quantityKind == "cash")

    # Contract buy executed in full despite insufficient cash
    assert contract.sharesChange == 2
    assert contract.marketValueChange == pytest.approx(4844.0)
    # Equity buy capped at int((7000 - 4844) / 700) = 3 shares
    assert share.sharesChange == 3
    assert share.marketValueChange == pytest.approx(2100.0)
    assert share.exposureChange == pytest.approx(2100.0)
    # Waterfall order: contract buy precedes the equity buy
    assert trades.index(contract) < trades.index(share)
    assert cashRow.marketValueChange == pytest.approx(
        -(4844.0 + 2100.0) - sum(t.transactionCost for t in trades[:-1])
    )


def test_equity_buy_skipped_entirely_when_cash_below_one_share():
    # maxSharesBuyable = int((100 - netCashUsed) / 700) <= 0 -> the buy is
    # entirely absent from executed trades; the sell executes regardless.
    priceSource = FakePriceSource({"X": SPOT, "URA": 30.0, "USD": 1.0})
    frame = frameWith([equity("X", 10, 55, sleeve="true"), equity("URA", 100, 45), cash(100)])
    enriched = enrichPositions(frame, priceSource, None)

    planned = [
        Trade(
            tradeId="1", instrumentId="X", instrumentType="Equity", price=SPOT,
            sharesChange=10.0, marketValueChange=10 * SPOT, timestamp=TODAY,
            quantityKind="share", exposureChange=10 * SPOT, reason="wants 10 shares",
        ),
        Trade(
            tradeId="2", instrumentId="URA", instrumentType="Equity", price=30.0,
            sharesChange=-1.0, marketValueChange=-30.0, timestamp=TODAY,
            quantityKind="share", exposureChange=-30.0, reason="drift rebalance",
        ),
    ]
    trades = executeTrades(planned, enriched, TradingPlatformFactory.getTradingPlatform("futubullUS"))

    # The buy is completely absent (not truncated, not zero-filled)
    assert not any(t.instrumentId == "X" and t.sharesChange > 0 for t in trades)
    # The sell is unaffected by the cash constraint
    sell = next(t for t in trades if t.instrumentId == "URA")
    assert sell.sharesChange == -1.0
    # Cash row still ledgers the sell proceeds minus fees
    cashRow = next(t for t in trades if t.quantityKind == "cash")
    fees = sum(t.transactionCost for t in trades if t.quantityKind != "cash")
    assert cashRow.marketValueChange == pytest.approx(30.0 - fees)
