from datetime import date, datetime
from types import SimpleNamespace

import polars as pl
import pytest

from main import (
    _passesLiquidityFilter,
    buildSleeveTable,
    enrichPositions,
    planTrades,
    sizeSleeve,
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

CHAIN_NEAR_EXPIRY = date(2028, 1, 21)     # within >= 21-month floor
CHAIN_DEFERRED_EXPIRY = date(2027, 12, 17)  # 15 months out -> below floor, above 12

HELD_SYMBOL = "VOO270618C00450000"
SPOT = 700.0
PER_CONTRACT_HELD = 100 * 0.85 * SPOT  # 59,500


def planFor(rows, optionSource, leverage=1.5, designated="VOO", liquidateLeaps=False):
    priceSource = FakePriceSource({"VOO": SPOT, "URA": 30.0, "USD": 1.0})
    frame = frameWith(rows)
    if designated is not None:
        assert frame.filter(pl.col("leapsSleeve")).height > 0 or designated == "VOO"
    enriched = enrichPositions(frame, priceSource, optionSource)
    table = buildSleeveTable(enriched, leverage, designated)
    return planTrades(
        enriched, table, designated, leverage, optionSource, TODAY, liquidateLeaps=liquidateLeaps
    )


# --- AE1: held contract inside the roll window -> sell + rule-selected replacement ---


def test_ae1_roll_sells_held_and_buys_replacement():
    replacement = makeSnapshot(
        occ("VOO", CHAIN_LATE_EXPIRY, 420.0), "VOO", 20.0, 0.86, CHAIN_LATE_EXPIRY
    )
    source = FakeOptionSource(
        snapshots={HELD_SYMBOL: makeSnapshot(HELD_SYMBOL, "VOO", 24.22, 0.85, HELD_ROLL_EXPIRY)},
        chainSnapshots=[replacement],
    )
    trades = planFor(
        [
            equity("VOO", 35, 55, sleeve="true"),
            equity("URA", 267, 45),
            option(HELD_SYMBOL, 1),
            cash(29371),
        ],
        source,
    )
    bySymbol = tradesBySymbol(trades)

    assert len(source.chainCalls) >= 1  # chain consulted for the replacement
    sell = bySymbol[HELD_SYMBOL]
    assert sell.sharesChange == -1
    assert sell.quantityKind == "contract"
    assert sell.reason == "roll: exit"
    assert sell.exposureChange == pytest.approx(-PER_CONTRACT_HELD)

    buy = bySymbol[replacement.symbol]
    assert buy.sharesChange == 1
    assert buy.reason == "roll: replacement"
    assert buy.exposureChange == pytest.approx(100 * 0.86 * SPOT)

    shareTrades = [t for t in trades if t.instrumentId == "VOO"]
    assert any(t.quantityKind == "share" and t.reason == "roll: share residual" for t in shareTrades)


# --- AE3: held contract > 12 months -> keep, no selection call, resize only ---


def rollDeferredFrame(cashAmount=27646):
    heldSymbol = occ("VOO", HELD_KEEP_EXPIRY, 450.0)
    return heldSymbol, [
        equity("VOO", 35, 55, sleeve="true"),
        equity("URA", 267, 45),
        option(heldSymbol, 2),
        cash(cashAmount),
    ]


def test_ae3_keep_never_calls_chain_and_only_resizes():
    heldSymbol, rows = rollDeferredFrame()
    source = FakeOptionSource(
        snapshots={heldSymbol: makeSnapshot(heldSymbol, "VOO", 24.22, 0.85, HELD_KEEP_EXPIRY)},
    )
    trades = planFor(rows, source)

    assert source.chainCalls == []  # no selection call outside the roll window
    contractTrades = [t for t in trades if t.quantityKind == "contract"]
    assert len(contractTrades) == 1
    assert contractTrades[0].instrumentId == heldSymbol
    assert contractTrades[0].sharesChange == -1  # resize sell on the HELD symbol
    assert contractTrades[0].reason == "resize"
    assert any(t.reason == "drift rebalance" for t in trades if t.quantityKind == "share")


def test_hold_two_target_one_resizes_by_one():
    heldSymbol, rows = rollDeferredFrame()
    source = FakeOptionSource(
        snapshots={heldSymbol: makeSnapshot(heldSymbol, "VOO", 24.22, 0.85, HELD_KEEP_EXPIRY)},
    )
    trades = planFor(rows, source)
    contractTrades = [t for t in trades if t.quantityKind == "contract"]
    assert contractTrades[0].sharesChange == -1
    assert contractTrades[0].marketValueChange == pytest.approx(-1 * 24.22 * 100)


# --- Initiation ---


def test_initiation_buys_nearest_target_delta_at_latest_qualifying_expiry():
    late860 = makeSnapshot(occ("VOO", CHAIN_LATE_EXPIRY, 420.0), "VOO", 20.0, 0.86, CHAIN_LATE_EXPIRY)
    late800 = makeSnapshot(occ("VOO", CHAIN_LATE_EXPIRY, 350.0), "VOO", 15.0, 0.80, CHAIN_LATE_EXPIRY)
    near850 = makeSnapshot(occ("VOO", CHAIN_NEAR_EXPIRY, 400.0), "VOO", 18.0, 0.85, CHAIN_NEAR_EXPIRY)
    source = FakeOptionSource(chainSnapshots=[late860, late800, near850])

    trades = planFor(
        [
            equity("VOO", 35, 55, sleeve="true"),
            equity("URA", 267, 45),
            cash(29371),
        ],
        source,
    )
    contractTrades = [t for t in trades if t.quantityKind == "contract"]
    assert len(contractTrades) == 1
    assert contractTrades[0].instrumentId == late860.symbol  # latest expiry, delta nearest 0.85
    assert contractTrades[0].sharesChange == 1
    assert contractTrades[0].reason == "initiation"
    assert any(t.reason == "initiation share residual" for t in trades)


def test_all_candidates_fail_spread_falls_back_to_shares():
    wide = makeSnapshot(
        occ("VOO", CHAIN_LATE_EXPIRY, 420.0), "VOO", 30.0, 0.86,
        CHAIN_LATE_EXPIRY, spread=20.0,  # (40)/30 = 1.33 > MAX_REL_SPREAD
    )
    source = FakeOptionSource(chainSnapshots=[wide])
    trades = planFor(
        [
            equity("VOO", 35, 55, sleeve="true"),
            equity("URA", 267, 45),
            cash(29371),
        ],
        source,
    )
    assert all(t.quantityKind != "contract" for t in trades)
    vooShares = [t for t in trades if t.instrumentId == "VOO"]
    assert len(vooShares) == 1
    assert vooShares[0].reason == "shares fallback: no qualifying contract"
    # fallback target is base weight (0.55 x MV), not the leveraged sleeve target
    totalMV = 35 * SPOT + 267 * 30.0 + 29371
    expectedShares = int((0.55 * totalMV - 35 * SPOT) / SPOT)
    assert vooShares[0].sharesChange == pytest.approx(float(expectedShares))


def test_empty_chain_treated_as_no_candidate_fallback():
    source = FakeOptionSource(chainSnapshots=[])
    trades = planFor(
        [
            equity("VOO", 35, 55, sleeve="true"),
            equity("URA", 267, 45),
            cash(29371),
        ],
        source,
    )
    assert len(source.chainCalls) == 1  # single 12-month-floor query, empty chain
    assert all(t.quantityKind == "share" for t in trades)
    assert any(t.reason == "shares fallback: no qualifying contract" for t in trades)


# --- Roll fallbacks ---


def test_roll_with_illiquid_replacement_sells_held_and_de_levers():
    wide = makeSnapshot(
        occ("VOO", CHAIN_LATE_EXPIRY, 420.0), "VOO", 30.0, 0.86,
        CHAIN_LATE_EXPIRY, spread=20.0,
    )
    source = FakeOptionSource(
        snapshots={HELD_SYMBOL: makeSnapshot(HELD_SYMBOL, "VOO", 24.22, 0.85, HELD_ROLL_EXPIRY)},
        chainSnapshots=[wide],
    )
    trades = planFor(
        [
            equity("VOO", 35, 55, sleeve="true"),
            equity("URA", 267, 45),
            option(HELD_SYMBOL, 1),
            cash(29371),
        ],
        source,
    )
    contractTrades = [t for t in trades if t.quantityKind == "contract"]
    assert len(contractTrades) == 1
    assert contractTrades[0].instrumentId == HELD_SYMBOL
    assert contractTrades[0].sharesChange == -1
    assert contractTrades[0].reason == "roll: exit — no qualifying replacement"

    shareTrades = [t for t in trades if t.instrumentId == "VOO"]
    assert len(shareTrades) == 1
    assert shareTrades[0].reason == "shares fallback: roll de-lever"
    totalMV = 35 * SPOT + 24.22 * 100 + 267 * 30.0 + 29371
    expectedShares = int((0.55 * totalMV - 35 * SPOT) / SPOT)
    assert shareTrades[0].sharesChange == pytest.approx(float(expectedShares))


def test_roll_deferred_when_candidates_only_below_floor():
    deep = makeSnapshot(
        occ("VOO", CHAIN_DEFERRED_EXPIRY, 420.0), "VOO", 20.0, 0.85, CHAIN_DEFERRED_EXPIRY
    )
    heldSymbol = occ("VOO", HELD_ROLL_EXPIRY, 450.0)
    source = FakeOptionSource(
        snapshots={heldSymbol: makeSnapshot(heldSymbol, "VOO", 24.22, 0.85, HELD_ROLL_EXPIRY)},
        chainSnapshots=[deep],
    )
    trades = planFor(
        [
            equity("VOO", 35, 55, sleeve="true"),
            equity("URA", 267, 45),
            option(heldSymbol, 2),
            cash(27646),
        ],
        source,
    )
    assert len(source.chainCalls) == 1  # one 12-month-floor query; depth found only below the 21-month floor
    contractTrades = [t for t in trades if t.quantityKind == "contract"]
    # treated as keep: resize held (2 -> 1), no replacement bought
    assert len(contractTrades) == 1
    assert contractTrades[0].instrumentId == heldSymbol
    assert contractTrades[0].sharesChange == -1
    assert contractTrades[0].reason == "roll deferred — chain depth"
    assert not any(t.instrumentId == deep.symbol for t in trades)


# --- Size guards ---


def test_size_sleeve_rejects_nonpositive_per_contract_exposure():
    with pytest.raises(ValueError, match="(?i)per-contract exposure"):
        sizeSleeve(targetExposure=1000.0, perContractExposure=0.0, heldContracts=0, heldShares=0, spot=700.0)



def test_size_sleeve_rejects_nonpositive_spot():
    with pytest.raises(ValueError, match="spot"):
        sizeSleeve(targetExposure=1000.0, perContractExposure=59500.0, heldContracts=0, heldShares=0, spot=0.0)


# --- --liquidate-leaps full exit ---


def test_liquidate_leaps_flag_forces_full_exit_at_base_weight():
    held = occ("VOO", HELD_KEEP_EXPIRY, 450.0)
    source = FakeOptionSource(
        snapshots={held: makeSnapshot(held, "VOO", 24.22, 0.85, HELD_KEEP_EXPIRY)},
    )
    trades = planFor(
        [
            equity("VOO", 35, 55, sleeve="true"),
            equity("URA", 267, 45),
            option(held, 2),
            cash(27646),
        ],
        source,
        leverage=1.0,
        liquidateLeaps=True,
    )
    contractTrades = [t for t in trades if t.quantityKind == "contract"]
    assert len(contractTrades) == 1
    assert contractTrades[0].instrumentId == held
    assert contractTrades[0].sharesChange == -2  # full exit, not resize
    assert contractTrades[0].reason == "liquidation: --liquidate-leaps"

    shareTrades = [t for t in trades if t.instrumentId == "VOO" and t.quantityKind == "share"]
    assert len(shareTrades) == 1
    assert shareTrades[0].reason == "liquidation: --liquidate-leaps"
    # Shares sized to base weight (0.55 x MV), not the leveraged sleeve target
    totalMV = 35 * SPOT + 2 * 24.22 * 100 + 267 * 30.0 + 27646
    expectedShares = int((0.55 * totalMV - 35 * SPOT) / SPOT)
    assert shareTrades[0].sharesChange == pytest.approx(float(expectedShares))


# --- R17: liquidation ---


def test_option_only_stray_sleeve_gets_liquidation_sell():
    straySymbol = occ("URA", HELD_KEEP_EXPIRY, 30.0)
    source = FakeOptionSource(
        snapshots={straySymbol: makeSnapshot(straySymbol, "URA", 5.0, 0.8, HELD_KEEP_EXPIRY, spread=0.4)},
    )
    trades = planFor(
        [
            equity("VOO", 35, 100, sleeve="true"),
            option(straySymbol, 1),
            cash(1000),
        ],
        source,
    )
    liquidations = [t for t in trades if t.instrumentId == straySymbol]
    assert len(liquidations) == 1
    assert liquidations[0].sharesChange == -1
    assert liquidations[0].quantityKind == "contract"
    assert liquidations[0].reason == "liquidation: non-designated sleeve"
    assert liquidations[0].marketValueChange == pytest.approx(-1 * 5.0 * 100)
    assert liquidations[0].exposureChange == pytest.approx(-1 * 100 * 0.8 * 30.0)
    # URA has no equity row: no share trade for the stray underlying
    assert not any(t.instrumentId == "URA" and t.quantityKind == "share" for t in trades)


# --- Plain equity sleeve ---


def test_plain_equity_sleeve_drift_rebalance():
    priceSource = FakePriceSource({"VOO": SPOT, "URA": 30.0, "USD": 1.0})
    frame = frameWith(
        [
            equity("VOO", 35, 55),
            equity("URA", 267, 45),
            cash(1000),
        ]
    )
    enriched = enrichPositions(frame, priceSource, None)
    table = buildSleeveTable(enriched, leverage=1.0, designatedUnderlying=None)
    source = FakeOptionSource()  # chain would raise if ever called via spy list check below
    trades = planTrades(enriched, table, None, 1.0, source, TODAY)

    assert source.chainCalls == []
    assert all(t.quantityKind == "share" for t in trades)
    assert all(t.reason == "drift rebalance" for t in trades)

    totalMV = 35 * SPOT + 267 * 30.0 + 1000.0
    byId = tradesBySymbol(trades)
    assert byId["URA"].sharesChange == pytest.approx(float(int((0.45 * totalMV - 267 * 30.0) / 30.0)))
    assert byId["VOO"].sharesChange == pytest.approx(float(int((0.55 * totalMV - 35 * SPOT) / SPOT)))


# --- Ladder: two held expiries in the keep window resize on the EARLIEST row ---


def test_ladder_two_expiries_resize_anchored_on_earliest_row_delta():
    # Two option rows, both > 12 months out (keep path), different deltas.
    # Resize sizing must use the EARLIEST-expiry row's own per-contract
    # exposure (anchor), not the sleeve average.
    earliest = occ("VOO", HELD_KEEP_EXPIRY, 450.0)            # 2028-05-18, delta 0.85
    latest = occ("VOO", date(2028, 9, 15), 450.0)             # 2028-09-15, delta 0.70
    source = FakeOptionSource(
        snapshots={
            earliest: makeSnapshot(earliest, "VOO", 24.22, 0.85, HELD_KEEP_EXPIRY),
            latest: makeSnapshot(latest, "VOO", 30.0, 0.70, date(2028, 9, 15)),
        }
    )
    trades = planFor(
        [
            equity("VOO", 35, 55, sleeve="true"),
            equity("URA", 267, 45),
            option(earliest, 1),
            option(latest, 1),
            cash(268068),
        ],
        source,
    )

    assert source.chainCalls == []  # keep window: no selection call

    contractTrades = [t for t in trades if t.quantityKind == "contract"]
    assert len(contractTrades) == 1
    anchorTrade = contractTrades[0]
    assert anchorTrade.instrumentId == earliest  # earliest expiry is the anchor
    assert anchorTrade.reason == "resize"

    totalMV = 35 * SPOT + 24.22 * 100 + 30.0 * 100 + 267 * 30.0 + 268068
    assert totalMV == pytest.approx(306000.0)
    targetExposure = (0.55 + 0.5) * totalMV  # 321,300
    anchorPerContract = 100 * 0.85 * SPOT    # 59,500 (earliest row's own delta)
    sleeveAvgPerContract = 100 * ((0.85 + 0.70) / 2) * SPOT  # 54,250 — the old bug

    desiredAnchor = round(targetExposure / anchorPerContract)  # 5
    desiredSleeveAvg = round(targetExposure / sleeveAvgPerContract)  # 6 — old bug
    assert desiredAnchor != desiredSleeveAvg  # the two rules genuinely diverge here

    assert anchorTrade.sharesChange == float(desiredAnchor - 2)  # buy up to 5
    assert anchorTrade.exposureChange == pytest.approx(
        (desiredAnchor - 2) * anchorPerContract
    )

    # Post-plan contract counts are consistent with the ANCHOR row's delta:
    # 5 contracts x 59,500 + whole-share residual == target exposure
    # (integer share flooring leaves at most one share of tracking error).
    postContracts = 2 + int(anchorTrade.sharesChange)
    assert postContracts == desiredAnchor  # 5 — the sleeve-average rule would give 6
    shareTrades = [t for t in trades if t.instrumentId == "VOO" and t.quantityKind == "share"]
    shareChange = sum(t.sharesChange for t in shareTrades)
    assert postContracts * anchorPerContract + shareChange * SPOT == pytest.approx(
        targetExposure, abs=SPOT
    )
    assert any(t.reason == "drift rebalance" for t in shareTrades)
    # The later-expiry row is untouched
    assert not any(t.instrumentId == latest for t in trades)


# --- Liquidity filter branches (R12) ---


def test_zero_volume_candidate_fails_liquidity_filter_to_shares_fallback():
    # Candidate passes spread and expiry-floor checks but volume = 0 -> filtered
    zeroVolume = makeSnapshot(
        occ("VOO", CHAIN_LATE_EXPIRY, 420.0), "VOO", 20.0, 0.86,
        CHAIN_LATE_EXPIRY, volume=0.0,
    )
    source = FakeOptionSource(chainSnapshots=[zeroVolume])
    trades = planFor(
        [
            equity("VOO", 35, 55, sleeve="true"),
            equity("URA", 267, 45),
            cash(29371),
        ],
        source,
    )
    assert all(t.quantityKind != "contract" for t in trades)
    assert any(t.reason == "shares fallback: no qualifying contract" for t in trades)


def test_passes_liquidity_filter_branches_direct():
    # mid <= 0 is unreachable via _parseSnapshot (bid > 0 guard keeps mid > 0),
    # so exercise the branch directly with a synthetic snapshot.
    assert _passesLiquidityFilter(SimpleNamespace(mid=0.0, bid=-1.0, ask=1.0, volume=100.0)) is False
    # Healthy quote passes
    assert (
        _passesLiquidityFilter(SimpleNamespace(mid=10.0, bid=9.5, ask=10.5, volume=100.0))
        is True
    )
    # Volume below the floor fails
    assert (
        _passesLiquidityFilter(SimpleNamespace(mid=10.0, bid=9.5, ask=10.5, volume=0.0))
        is False
    )
    # Spread above the cap fails
    assert (
        _passesLiquidityFilter(SimpleNamespace(mid=10.0, bid=5.0, ask=15.0, volume=100.0))
        is False
    )
