from datetime import date, datetime

import polars as pl
import pytest

from main import (
    CASH_TYPE,
    PriceDataSource,
    enrichPositions,
    normalizePositions,
    validateInputs,
)
from options_data import OptionQuoteSource, OptionSnapshot

TIMESTAMP = "2026-08-04"


class FakePriceSource(PriceDataSource):
    def __init__(self, prices: dict[str, float]):
        self.prices = prices
        self.fetched: list[str] = []

    def getClosingPrice(self, ticker: str, date) -> float:
        self.fetched.append(ticker)
        return self.prices[ticker]


class FakeOptionSource(OptionQuoteSource):
    def __init__(self, snapshots: dict[str, OptionSnapshot], available: set[str] | None = None):
        self.snapshots = snapshots
        self.available = available if available is not None else set(snapshots)

    def getSnapshots(self, symbols: list[str]) -> dict[str, OptionSnapshot]:
        return {s: self.snapshots[s] for s in symbols if s in self.available}

    def getChain(self, **kwargs):
        raise NotImplementedError


def snapshot(symbol, underlying, mid, delta, expiry=date(2027, 1, 15), strike=450.0):
    return OptionSnapshot(
        symbol=symbol,
        underlying=underlying,
        expiry=expiry,
        strike=strike,
        right="C",
        bid=mid - 0.4,
        ask=mid + 0.4,
        mid=mid,
        delta=delta,
        iv=0.21,
        quoteTimestamp=datetime(2026, 9, 1, 16, 0, 0),
        volume=10.0,
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


VOO_OPTION = "VOO270115C00450000"


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


def test_mixed_frame_valuation_and_exposure():
    priceSource = FakePriceSource({"VOO": 700.0, "URA": 30.0, "USD": 1.0})
    optionSource = FakeOptionSource({VOO_OPTION: snapshot(VOO_OPTION, "VOO", 24.22, 0.85)})

    enriched = enrichPositions(
        frameWith(
            [
                equity("VOO", 35, 55, sleeve="true"),
                equity("URA", 267, 45),
                option(VOO_OPTION, 2),
                cash(29371),
            ]
        ),
        priceSource,
        optionSource,
    )

    byId = {row["instrumentId"]: row for row in enriched.iter_rows(named=True)}

    # marketValue = shares x price x multiplier
    assert byId["VOO"]["marketValue"] == pytest.approx(35 * 700.0)
    assert byId[VOO_OPTION]["marketValue"] == pytest.approx(2 * 24.22 * 100)
    assert byId["USD"]["marketValue"] == pytest.approx(29371.0)

    # exposure = shares x multiplier x deltaAdj x underlyingSpot
    assert byId["VOO"]["exposure"] == pytest.approx(35 * 700.0)
    assert byId[VOO_OPTION]["exposure"] == pytest.approx(2 * 100 * 0.85 * 700.0)
    assert byId[VOO_OPTION]["exposure"] == pytest.approx(119000.0)
    assert byId["URA"]["exposure"] == pytest.approx(267 * 30.0)
    assert byId["USD"]["exposure"] == pytest.approx(0.0)

    # VOO sleeve aggregates across share + contract rows
    vooSleeveExposure = sum(
        row["exposure"] for row in enriched.iter_rows(named=True) if row["underlying"] == "VOO"
    )
    assert vooSleeveExposure == pytest.approx(35 * 700.0 + 119000.0)


def test_option_only_sleeve_fetches_spot_via_price_source():
    priceSource = FakePriceSource({"VOO": 700.0, "URA": 30.0, "USD": 1.0})
    optionSource = FakeOptionSource({"URA260619C00030000": snapshot("URA260619C00030000", "URA", 5.0, 0.8, strike=30.0)})

    enriched = enrichPositions(
        frameWith(
            [
                equity("VOO", 35, 100),
                option("URA260619C00030000", 1),
                cash(1000),
            ]
        ),
        priceSource,
        optionSource,
    )

    byId = {row["instrumentId"]: row for row in enriched.iter_rows(named=True)}
    optionRowEnriched = byId["URA260619C00030000"]
    assert optionRowEnriched["underlyingSpot"] == pytest.approx(30.0)
    assert optionRowEnriched["exposure"] == pytest.approx(1 * 100 * 0.8 * 30.0)
    assert "URA" in priceSource.fetched  # spot fetched on demand


def test_snapshot_delta_flows_to_delta_adj():
    priceSource = FakePriceSource({"VOO": 700.0, "USD": 1.0})
    optionSource = FakeOptionSource({VOO_OPTION: snapshot(VOO_OPTION, "VOO", 24.22, 0.9399)})

    enriched = enrichPositions(
        frameWith([equity("VOO", 35, 100, sleeve="true"), option(VOO_OPTION, 2), cash(1000)]),
        priceSource,
        optionSource,
    )

    optionRowEnriched = next(
        row for row in enriched.iter_rows(named=True) if row["kind"] == "option"
    )
    assert optionRowEnriched["deltaAdj"] == pytest.approx(0.9399)
    assert optionRowEnriched["exposure"] == pytest.approx(2 * 100 * 0.9399 * 700.0)


def test_legacy_frame_regression_cell_for_cell():
    priceSource = FakePriceSource({"VOO": 700.0, "URA": 30.0, "USD": 1.0})

    enriched = enrichPositions(
        frameWith([equity("VOO", 35, 55), equity("URA", 267, 45), cash(29371)]),
        priceSource,
        None,
    )

    totalMarketValue = 35 * 700.0 + 267 * 30.0 + 29371.0
    expected = {
        "VOO": (24500.0, 55 / 100 * totalMarketValue),
        "URA": (8010.0, 45 / 100 * totalMarketValue),
        "USD": (29371.0, 0.0),
    }

    for row in enriched.iter_rows(named=True):
        expectedMV, expectedTarget = expected[row["instrumentId"]]
        assert row["marketValue"] == pytest.approx(expectedMV)
        assert row["currentRatio"] == pytest.approx(expectedMV / totalMarketValue * 100)
        assert row["targetMarketValue"] == pytest.approx(expectedTarget)
        assert row["currMinusTargetMarketValue"] == pytest.approx(expectedMV - expectedTarget)
        # L = 1 semantics: exposure mirrors market value for equity and is zero for cash
        if row["instrumentType"] == CASH_TYPE:
            assert row["exposure"] == pytest.approx(0.0)
        else:
            assert row["exposure"] == pytest.approx(expectedMV)


def test_missing_snapshot_aborts_before_trade_generation():
    priceSource = FakePriceSource({"VOO": 700.0, "USD": 1.0})
    optionSource = FakeOptionSource({}, available=set())  # nothing comes back

    with pytest.raises(ValueError, match="aborting before trade generation"):
        enrichPositions(
            frameWith([equity("VOO", 35, 100, sleeve="true"), option(VOO_OPTION, 2), cash(1000)]),
            priceSource,
            optionSource,
        )


def test_option_rows_without_source_raise():
    priceSource = FakePriceSource({"VOO": 700.0, "USD": 1.0})

    with pytest.raises(Exception, match="no option data source"):
        enrichPositions(
            frameWith([equity("VOO", 35, 100, sleeve="true"), option(VOO_OPTION, 2), cash(1000)]),
            priceSource,
            None,
        )
