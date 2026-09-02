from datetime import date

import polars as pl
import pytest

from main import normalizePositions, validateInputs
from options_data import OccParseError

TIMESTAMP = "2026-08-04"


def equityRow(ticker, shares, weight, sleeve=None):
    return {
        "instrumentId": ticker,
        "idType": "ticker",
        "instrumentType": "Equity",
        "shares": shares,
        "targetRatioPct": weight,
        "timestamp": TIMESTAMP,
        "leapsSleeve": sleeve,
    }


def optionRow(symbol, contracts, sleeve=None):
    return {
        "instrumentId": symbol,
        "idType": "occ",
        "instrumentType": "LEAPS Call",
        "shares": contracts,
        "targetRatioPct": None,
        "timestamp": TIMESTAMP,
        "leapsSleeve": sleeve,
    }


def cashRow(amount):
    return {
        "instrumentId": "USD",
        "idType": "name",
        "instrumentType": "Cash and Cash Equivalents",
        "shares": amount,
        "targetRatioPct": 0,
        "timestamp": TIMESTAMP,
        "leapsSleeve": None,
    }


def buildFrame(rows, includeSleeveColumn=True):
    if not includeSleeveColumn:
        for row in rows:
            row.pop("leapsSleeve", None)
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
    return normalizePositions(frame)


def test_mixed_csv_normalizes():
    frame = buildFrame(
        [
            equityRow("VOO", 35, 55, sleeve="true"),
            equityRow("URA", 267, 25),
            optionRow("VOO270115C00450000", 2),
            cashRow(29371),
        ]
    )

    byId = {row["instrumentId"]: row for row in frame.iter_rows(named=True)}

    assert byId["VOO"]["kind"] == "equity"
    assert byId["VOO"]["underlying"] == "VOO"
    assert byId["VOO"]["multiplier"] == 1.0
    assert byId["VOO"]["deltaAdj"] == 1.0
    assert byId["VOO"]["leapsSleeve"] is True

    option = byId["VOO270115C00450000"]
    assert option["kind"] == "option"
    assert option["underlying"] == "VOO"
    assert option["multiplier"] == 100.0
    assert option["deltaAdj"] == 1.0
    assert option["leapsSleeve"] is False

    cash = byId["USD"]
    assert cash["kind"] == "cash"
    assert cash["deltaAdj"] == 0.0
    assert cash["leapsSleeve"] is False


def test_legacy_csv_without_sleeve_column():
    frame = buildFrame(
        [equityRow("VOO", 35, 55), equityRow("URA", 267, 45), cashRow(1000)],
        includeSleeveColumn=False,
    )

    assert frame["leapsSleeve"].to_list() == [False, False, False]
    assert set(frame["kind"].to_list()) == {"equity", "cash"}


def test_option_only_sleeve_designated_via_marker():
    frame = buildFrame(
        [
            equityRow("VOO", 35, 100),
            optionRow("URA260619C00030000", 1, sleeve="true"),
            cashRow(1000),
        ]
    )

    designated = validateInputs(frame, leverage=1.5, liquidateLeaps=False)
    assert designated == "URA"  # no URA equity row — option-only sleeve, weight 0


def test_unmarked_stray_option_trips_guard():
    frame = buildFrame(
        [
            equityRow("VOO", 35, 100),
            optionRow("URA260619C00030000", 1),
            cashRow(1000),
        ]
    )

    with pytest.raises(ValueError, match="--liquidate-leaps"):
        validateInputs(frame, leverage=1.5, liquidateLeaps=False)


def test_marker_on_option_row_designates_root():
    frame = buildFrame(
        [
            equityRow("VOO", 35, 100, sleeve=None),
            optionRow("VOO270115C00450000", 2, sleeve="true"),
            cashRow(1000),
        ]
    )

    designated = validateInputs(frame, leverage=1.5, liquidateLeaps=False)
    assert designated == "VOO"


def test_two_markers_rejected():
    frame = buildFrame(
        [
            equityRow("VOO", 35, 55, sleeve="true"),
            equityRow("URA", 267, 45, sleeve="true"),
            optionRow("VOO270115C00450000", 2),
            cashRow(1000),
        ]
    )

    with pytest.raises(ValueError, match="(?i)at most one LEAPS sleeve"):
        validateInputs(frame, leverage=1.5, liquidateLeaps=False)


def test_leverage_below_one_rejected():
    frame = buildFrame([equityRow("VOO", 35, 100), cashRow(1000)])

    with pytest.raises(ValueError, match="--leverage"):
        validateInputs(frame, leverage=0.8, liquidateLeaps=False)


def test_malformed_occ_symbol_rejected():
    with pytest.raises(OccParseError):
        buildFrame([equityRow("VOO", 35, 100), optionRow("VOO270115C00450", 1), cashRow(1000)])


def test_liquidation_guard_no_marker():
    frame = buildFrame(
        [equityRow("VOO", 35, 100), optionRow("VOO270115C00450000", 2), cashRow(1000)]
    )

    with pytest.raises(ValueError, match="--liquidate-leaps"):
        validateInputs(frame, leverage=1.5, liquidateLeaps=False)

    # Explicit opt-in proceeds
    assert validateInputs(frame, leverage=1.5, liquidateLeaps=True) is None


def test_liquidation_guard_default_leverage():
    frame = buildFrame(
        [
            equityRow("VOO", 35, 100, sleeve="true"),
            optionRow("VOO270115C00450000", 2),
            cashRow(1000),
        ]
    )

    with pytest.raises(ValueError, match="--liquidate-leaps"):
        validateInputs(frame, leverage=1.0, liquidateLeaps=False)

    assert validateInputs(frame, leverage=1.0, liquidateLeaps=True) == "VOO"


def test_equity_weights_must_sum_to_100():
    frame = buildFrame(
        [equityRow("VOO", 35, 55), equityRow("URA", 267, 40), cashRow(1000)]
    )

    with pytest.raises(ValueError, match="did not add up to 100"):
        validateInputs(frame, leverage=1.5, liquidateLeaps=False)
