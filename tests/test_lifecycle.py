from datetime import date, datetime

import polars as pl
import pytest

from main import (
    buildSleeveTable,
    enrichPositions,
    normalizePositions,
    planTrades,
    validateInputs,
    PriceDataSource,
)
from options_data import OptionQuoteSource, OptionSnapshot

TIMESTAMP = "2026-08-04"
TODAY = datetime(2026, 9, 2, 10, 0, 0)  # planTrades reference date

# Held / chain contract dates relative to TODAY (2026-09-02)
HELD_ROLL_EXPIRY = date(2027, 6, 18)      # 9 months out -> roll window
HELD_KEEP_EXPIRY = date(2028, 5, 18)      # 20 months out -> keep
CHAIN_LATE_EXPIRY = date(2028, 6, 19)     # 22 months out -> >= MIN_EXPIRY_MONTHS
CHAIN_NEAR_EXPIRY = date(2028, 1, 21)     # within >= 21-month floor
CHAIN_DEFERRED_EXPIRY = date(2027, 12, 17)  # 15 months out -> below floor, above 12

HELD_SYMBOL = "VOO270618C00450000"
SPOT = 700.0
PER_CONTRACT_HELD = 100 * 0.85 * SPOT  # 59,500


def occ(root: str, expiry: date, strike: float) -> str:
    return f"{root}{expiry:%y%m%d}C{int(strike * 1000):08d}"


class FakePriceSource(PriceDataSource):
    def __init__(self, prices):
        self.prices = prices

    def getClosingPrice(self, ticker, date):
        return self.prices[ticker]


class FakeOptionSource(OptionQuoteSource):
    """Snapshots answer getSnapshots; chainSnapshots answer getChain (spy included)."""

    def __init__(self, snapshots=None, chainSnapshots=None):
        self.snapshots = snapshots or {}
        self.chainSnapshots = chainSnapshots or []
        self.chainCalls: list[str | None] = []

    def getSnapshots(self, symbols):
        return {s: self.snapshots[s] for s in symbols}

    def getChain(self, underlying, expirationDateGte=None, strikePriceGte=None,
                 strikePriceLte=None, optionType="call"):
        self.chainCalls.append(expirationDateGte)
        return {s.symbol: s for s in self.chainSnapshots}


def makeSnapshot(symbol, underlying, mid, delta, expiry, volume=10.0, spread=0.8):
    return OptionSnapshot(
        symbol=symbol,
        underlying=underlying,
        expiry=expiry,
        strike=450.0,
        right="C",
        bid=mid - spread,
        ask=mid + spread,
        mid=mid,
        delta=delta,
        iv=0.21,
        quoteTimestamp=datetime(2026, 9, 2, 16, 0, 0),
        volume=volume,
    )


def frameWith(rows):
    frame = pl.DataFrame(
        rows,
        schema={
            "instrumentId": pl.String,
            "idType": pl.String,
            "instrumentType": pl.String,
            "shares": pl.Float64,
            "targetRatioPct": pl.Float64,
            "timestamp": pl.String,
            "leapsSleeve": pl.String,
        },
    )
    frame = normalizePositions(frame)
    validateInputs(frame, leverage=1.5, liquidateLeaps=True)
    return frame


def equity(ticker, shares, weight, sleeve=None):
    return {
        "instrumentId": ticker,
        "idType": "ticker",
        "instrumentType": "Equity",
        "shares": shares,
        "targetRatioPct": weight,
        "timestamp": TIMESTAMP,
        "leapsSleeve": sleeve,
    }


def option(symbol, contracts, sleeve=None):
    return {
        "instrumentId": symbol,
        "idType": "occ",
        "instrumentType": "LEAPS Call",
        "shares": contracts,
        "targetRatioPct": None,
        "timestamp": TIMESTAMP,
        "leapsSleeve": sleeve,
    }


def cash(amount):
    return {
        "instrumentId": "USD",
        "idType": "name",
        "instrumentType": "Cash and Cash Equivalents",
        "shares": amount,
        "targetRatioPct": 0,
        "timestamp": TIMESTAMP,
        "leapsSleeve": None,
    }


def planFor(rows, optionSource, leverage=1.5, designated="VOO"):
    priceSource = FakePriceSource({"VOO": SPOT, "URA": 30.0, "USD": 1.0})
    frame = frameWith(rows)
    if designated is not None:
        assert frame.filter(pl.col("leapsSleeve")).height > 0 or designated == "VOO"
    enriched = enrichPositions(frame, priceSource, optionSource)
    table = buildSleeveTable(enriched, leverage, designated)
    return planTrades(enriched, table, designated, leverage, optionSource, TODAY)


def tradesBySymbol(trades):
    return {t.instrumentId: t for t in trades}


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
    assert len(source.chainCalls) == 2  # floor query + no-floor probe, both empty
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
    assert len(source.chainCalls) == 2  # floor query empty, 12-month probe found depth
    contractTrades = [t for t in trades if t.quantityKind == "contract"]
    # treated as keep: resize held (2 -> 1), no replacement bought
    assert len(contractTrades) == 1
    assert contractTrades[0].instrumentId == heldSymbol
    assert contractTrades[0].sharesChange == -1
    assert contractTrades[0].reason == "roll deferred — chain depth"
    assert not any(t.instrumentId == deep.symbol for t in trades)


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
