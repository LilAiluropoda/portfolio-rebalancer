"""Shared test fixtures: fakes, snapshot factory, row builders, frame normalizer.

Superset shape covering test_enrichment / test_sizing / test_lifecycle /
test_integration_pipeline / test_positions_frame.
"""

from datetime import date, datetime

import polars as pl

from main import PriceDataSource, normalizePositions, validateInputs
from options_data import OptionQuoteSource, OptionSnapshot

TIMESTAMP = "2026-08-04"
DEFAULT_EXPIRY = date(2027, 1, 15)

# Shared lifecycle-test reference constants (planTrades reference date and
# held/chain contract dates relative to it)
TODAY = datetime(2026, 9, 2, 10, 0, 0)
HELD_ROLL_EXPIRY = date(2027, 6, 18)   # 9 months out -> roll window
HELD_KEEP_EXPIRY = date(2028, 5, 18)   # 20 months out -> keep
CHAIN_LATE_EXPIRY = date(2028, 6, 19)  # >= MIN_EXPIRY_MONTHS


def occ(root: str, expiry: date, strike: float) -> str:
    return f"{root}{expiry:%y%m%d}C{int(strike * 1000):08d}"


def tradesBySymbol(trades):
    return {t.instrumentId: t for t in trades}


class FakePriceSource(PriceDataSource):
    """Canned prices; records every ticker fetched (spot-on-demand assertions)."""

    def __init__(self, prices: dict[str, float]):
        self.prices = prices
        self.fetched: list[str] = []

    def getClosingPrice(self, ticker: str, date) -> float:
        self.fetched.append(ticker)
        return self.prices[ticker]


class FakeOptionSource(OptionQuoteSource):
    """Recording fake.

    getSnapshots serves `snapshots`, filtered by `available` (defaults to the
    snapshot keys — pass an empty set to simulate a missing snapshot).
    getChain serves `chainSnapshots` — either a canned list of OptionSnapshot
    or a callable(kwargs) -> dict[str, OptionSnapshot] — and records every
    call's expirationDateGte in `chainCalls`.
    """

    def __init__(self, snapshots=None, chainSnapshots=None, available=None):
        self.snapshots = snapshots or {}
        self.available = set(self.snapshots) if available is None else available
        self.chainSnapshots = chainSnapshots
        self.chainCalls: list[str | None] = []

    def getSnapshots(self, symbols: list[str]) -> dict[str, OptionSnapshot]:
        return {s: self.snapshots[s] for s in symbols if s in self.available}

    def getChain(self, underlying, expirationDateGte=None, strikePriceGte=None,
                 strikePriceLte=None, optionType="call"):
        self.chainCalls.append(expirationDateGte)
        if callable(self.chainSnapshots):
            return self.chainSnapshots(
                underlying=underlying,
                expirationDateGte=expirationDateGte,
                strikePriceGte=strikePriceGte,
                strikePriceLte=strikePriceLte,
                optionType=optionType,
            )
        return {s.symbol: s for s in (self.chainSnapshots or [])}


def makeSnapshot(symbol, underlying, mid, delta, expiry=DEFAULT_EXPIRY,
                 strike=450.0, volume=10.0, spread=0.8):
    return OptionSnapshot(
        symbol=symbol,
        underlying=underlying,
        expiry=expiry,
        strike=strike,
        right="C",
        bid=mid - spread,
        ask=mid + spread,
        mid=mid,
        delta=delta,
        iv=0.21,
        quoteTimestamp=datetime(2026, 9, 2, 16, 0, 0),
        volume=volume,
    )


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


def frameWith(rows, leverage=1.5, liquidateLeaps=True, validate=True,
              includeSleeveColumn=True):
    """Normalize a list of row dicts; optionally validate CLI inputs."""
    if not includeSleeveColumn:
        rows = [{k: v for k, v in r.items() if k != "leapsSleeve"} for r in rows]
    frame = pl.DataFrame(
        rows,
        schema={
            "instrumentId": pl.String,
            "idType": pl.String,
            "instrumentType": pl.String,
            "shares": pl.Float64,
            "targetRatioPct": pl.Float64,
            "timestamp": pl.String,
            **({"leapsSleeve": pl.String} if includeSleeveColumn else {}),
        },
    )
    frame = normalizePositions(frame)
    if validate:
        validateInputs(frame, leverage=leverage, liquidateLeaps=liquidateLeaps)
    return frame
