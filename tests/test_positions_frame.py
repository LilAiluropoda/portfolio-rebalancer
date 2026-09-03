import pytest

from main import validateInputs
from options_data import OccParseError
from conftest import cash as cashRow, equity as equityRow, frameWith
from conftest import option as optionRow


def buildFrame(rows, includeSleeveColumn=True):
    # Normalize only — these tests exercise validateInputs themselves.
    return frameWith(rows, validate=False, includeSleeveColumn=includeSleeveColumn)


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


def test_put_option_symbol_rejected():
    # Strategy is call-based stock replacement — puts would corrupt exposure math
    with pytest.raises(ValueError, match="VOO270115P00450000"):
        buildFrame(
            [equityRow("VOO", 35, 100), optionRow("VOO270115P00450000", 1), cashRow(1000)]
        )


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


def test_option_negative_shares_rejected():
    frame = buildFrame(
        [
            equityRow("VOO", 35, 100, sleeve="true"),
            optionRow("VOO270115C00450000", -2),
            cashRow(1000),
        ]
    )

    with pytest.raises(ValueError, match="VOO270115C00450000.*positive whole number"):
        validateInputs(frame, leverage=1.5, liquidateLeaps=False)


def test_option_fractional_shares_rejected():
    frame = buildFrame(
        [
            equityRow("VOO", 35, 100, sleeve="true"),
            optionRow("VOO270115C00450000", 1.5),
            cashRow(1000),
        ]
    )

    with pytest.raises(ValueError, match="VOO270115C00450000.*positive whole number"):
        validateInputs(frame, leverage=1.5, liquidateLeaps=False)
