import pytest

from main import buildSleeveTable, enrichPositions, normalizePositions, sizeSleeve
from main import PriceDataSource
from options_data import OptionQuoteSource, OptionSnapshot

from datetime import date, datetime

TIMESTAMP = "2026-08-04"
VOO_OPTION = "VOO270115C00450000"
PER_CONTRACT = 100 * 0.85 * 700.0  # 59,500


class FakePriceSource(PriceDataSource):
    def __init__(self, prices):
        self.prices = prices

    def getClosingPrice(self, ticker, date):
        return self.prices[ticker]


class FakeOptionSource(OptionQuoteSource):
    def __init__(self, snapshots):
        self.snapshots = snapshots

    def getSnapshots(self, symbols):
        return {s: self.snapshots[s] for s in symbols}

    def getChain(self, **kwargs):
        raise NotImplementedError


def makeSnapshot(symbol, underlying, mid, delta):
    return OptionSnapshot(
        symbol=symbol,
        underlying=underlying,
        expiry=date(2027, 1, 15),
        strike=450.0,
        right="C",
        bid=mid - 0.4,
        ask=mid + 0.4,
        mid=mid,
        delta=delta,
        iv=0.21,
        quoteTimestamp=datetime(2026, 9, 1, 16, 0, 0),
        volume=10.0,
    )


# --- sizeSleeve: pure sizing math (AE2 anchor + clamped signed residual) ---


def test_ae2_pure_leaps_sleeve_clamped_at_zero_shares():
    # AE2: target 105k, per-contract 59.5k -> 2 contracts (119k) -> residual -14k
    # cannot be absorbed (0 shares held) -> +14k tracking error, overshoot
    plan = sizeSleeve(
        targetExposure=105000.0,
        perContractExposure=PER_CONTRACT,
        heldContracts=0,
        heldShares=0,
        spot=700.0,
    )
    assert plan.contractChange == 2
    assert plan.shareChange == 0
    assert plan.trackingErrorExposure == pytest.approx(14000.0)
    assert plan.residualDirection == "overshoot"


def test_mixed_delivery_lands_on_target_with_held_shares():
    plan = sizeSleeve(
        targetExposure=105000.0,
        perContractExposure=PER_CONTRACT,
        heldContracts=0,
        heldShares=35,
        spot=700.0,
    )
    assert plan.contractChange == 2
    assert plan.shareChange == pytest.approx(-20.0)  # sell 20 shares, sleeve lands on 105k
    assert plan.trackingErrorExposure == pytest.approx(0.0)


def test_share_sell_never_exceeds_holdings():
    plan = sizeSleeve(
        targetExposure=105000.0,
        perContractExposure=PER_CONTRACT,
        heldContracts=0,
        heldShares=5,
        spot=700.0,
    )
    assert plan.contractChange == 2
    assert plan.shareChange == -5  # clamped: only 5 held
    # sleeve = 119k - 5*700 = 115.5k vs 105k target -> +10.5k error
    assert plan.trackingErrorExposure == pytest.approx(10500.0)
    assert plan.residualDirection == "overshoot"


def test_resize_on_leverage_cut_sells_to_nearest():
    # Held 2 contracts (119k); L cut shrinks target to 65k -> 1.09 -> 1 contract
    plan = sizeSleeve(
        targetExposure=65000.0,
        perContractExposure=PER_CONTRACT,
        heldContracts=2,
        heldShares=10,
        spot=700.0,
    )
    assert plan.contractChange == -1
    assert plan.shareChange == 7  # int(5500/700) = 7 buys
    assert plan.trackingErrorExposure == pytest.approx(-600.0)  # achieved 64.4k vs 65k target
    assert plan.residualDirection == "undershoot"


def test_rounding_boundary_half_rounds_up():
    # 1.5 contracts exactly -> half-up -> 2
    plan = sizeSleeve(
        targetExposure=1.5 * PER_CONTRACT,
        perContractExposure=PER_CONTRACT,
        heldContracts=0,
        heldShares=0,
        spot=700.0,
    )
    assert plan.contractChange == 2
    assert plan.residualDirection == "overshoot"


def test_target_below_half_contract_is_shares_only():
    plan = sizeSleeve(
        targetExposure=25000.0,
        perContractExposure=PER_CONTRACT,
        heldContracts=0,
        heldShares=0,
        spot=700.0,
    )
    assert plan.contractChange == 0
    assert plan.shareChange == 35  # int(25000/700)
    assert plan.trackingErrorExposure == pytest.approx(-500.0)  # achieved 24.5k vs 25k target
    assert plan.residualDirection == "undershoot"


def test_negative_target_rejected():
    with pytest.raises(ValueError):
        sizeSleeve(
            targetExposure=-1000.0,
            perContractExposure=PER_CONTRACT,
            heldContracts=0,
            heldShares=0,
            spot=700.0,
        )


# --- buildSleeveTable: per-underlying aggregation and targets ---


def enrichedMixedFrame():
    priceSource = FakePriceSource({"VOO": 700.0, "URA": 30.0, "USD": 1.0})
    optionSource = FakeOptionSource({VOO_OPTION: makeSnapshot(VOO_OPTION, "VOO", 24.22, 0.85)})

    frame = pl_Frame(
        [
            ("VOO", "ticker", "Equity", 35.0, 55.0, "true"),
            ("URA", "ticker", "Equity", 267.0, 45.0, None),
            (VOO_OPTION, "occ", "LEAPS Call", 2.0, None, "true"),
            ("USD", "name", "Cash and Cash Equivalents", 1000.0, 0.0, None),
        ]
    )
    return enrichPositions(frame, priceSource, optionSource)


def pl_Frame(rows):
    import polars as pl

    columns = ["instrumentId", "idType", "instrumentType", "shares", "targetRatioPct", "leapsSleeve"]
    data = {c: [r[i] for r in rows] for i, c in enumerate(columns)}
    data["timestamp"] = [TIMESTAMP] * len(rows)
    frame = pl.DataFrame(data, schema_overrides={"targetRatioPct": pl.Float64, "leapsSleeve": pl.String})
    return normalizePositions(frame)


def test_sleeve_table_targets_and_holdings():
    enriched = enrichedMixedFrame()
    table = buildSleeveTable(enriched, leverage=1.5, designatedUnderlying="VOO")

    byUnderlying = {row["underlying"]: row for row in table.iter_rows(named=True)}

    totalMV = 35 * 700.0 + 2 * 24.22 * 100 + 267 * 30.0 + 1000.0

    voo = byUnderlying["VOO"]
    assert voo["isDesignated"] is True
    assert voo["currentExposure"] == pytest.approx(35 * 700.0 + 2 * PER_CONTRACT)
    assert voo["targetExposure"] == pytest.approx((0.55 + 0.5) * totalMV)
    assert voo["heldContracts"] == 2
    assert voo["heldShares"] == 35

    ura = byUnderlying["URA"]
    assert ura["isDesignated"] is False
    assert ura["targetExposure"] == pytest.approx(0.45 * totalMV)
    assert ura["heldContracts"] == 0
    assert ura["heldShares"] == 267


def test_sleeve_table_legacy_leverage_one_degenerates_to_weights():
    enriched = enrichedMixedFrame()
    table = buildSleeveTable(enriched, leverage=1.0, designatedUnderlying=None)

    byUnderlying = {row["underlying"]: row for row in table.iter_rows(named=True)}
    totalMV = 35 * 700.0 + 2 * 24.22 * 100 + 267 * 30.0 + 1000.0

    assert byUnderlying["VOO"]["targetExposure"] == pytest.approx(0.55 * totalMV)
    assert byUnderlying["URA"]["targetExposure"] == pytest.approx(0.45 * totalMV)


def test_sleeve_table_option_only_non_designated_targets_zero():
    priceSource = FakePriceSource({"VOO": 700.0, "URA": 30.0, "USD": 1.0})
    optionSource = FakeOptionSource(
        {"URA260619C00030000": makeSnapshot("URA260619C00030000", "URA", 5.0, 0.8)}
    )
    frame = pl_Frame(
        [
            ("VOO", "ticker", "Equity", 35.0, 100.0, None),
            ("URA260619C00030000", "occ", "LEAPS Call", 1.0, None, None),
            ("USD", "name", "Cash and Cash Equivalents", 1000.0, 0.0, None),
        ]
    )
    enriched = enrichPositions(frame, priceSource, optionSource)

    table = buildSleeveTable(enriched, leverage=1.5, designatedUnderlying="VOO")
    byUnderlying = {row["underlying"]: row for row in table.iter_rows(named=True)}

    assert byUnderlying["URA"]["isDesignated"] is False
    assert byUnderlying["URA"]["targetExposure"] == 0.0  # liquidation target
    assert byUnderlying["URA"]["heldContracts"] == 1
