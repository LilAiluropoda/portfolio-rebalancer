from datetime import date
import pytest

from market_data import OptionSnapshot, OccParseError, parseOccSymbol


class TestParseHappyPath:
    def testCall(self):
        contract = parseOccSymbol("VOO270115C00450000")
        assert contract.underlying == "VOO"
        assert contract.expiry == date(2027, 1, 15)
        assert contract.strike == pytest.approx(450.000)
        assert contract.right == "C"
        assert contract.isCall

    def testPutLowStrike(self):
        contract = parseOccSymbol("TSLA260619P00050000")
        assert contract.underlying == "TSLA"
        assert contract.expiry == date(2026, 6, 19)
        assert contract.strike == pytest.approx(50.000)
        assert contract.right == "P"
        assert not contract.isCall


class TestParseEdge:
    def testShortRootUnpadded(self):
        # AAPL root shorter than 6 chars, no padding — fixed offsets from the right
        contract = parseOccSymbol("AAPL260619C00250000")
        assert contract.underlying == "AAPL"
        assert contract.strike == pytest.approx(250.000)

    def testFullWidthRoot(self):
        contract = parseOccSymbol("GOOGL270115C00300000")
        assert contract.underlying == "GOOGL"

    def testPaddedRootStripped(self):
        contract = parseOccSymbol("VOO   270115C00450000")
        assert contract.underlying == "VOO"

    def testRoundTrip(self):
        symbol = "VOO270115C00450000"
        contract = parseOccSymbol(symbol)
        rebuilt = (
            contract.underlying
            + contract.expiry.strftime("%y%m%d")
            + contract.right
            + f"{int(round(contract.strike * 1000)):08d}"
        )
        assert rebuilt == symbol


class TestParseErrors:
    def testTooShort(self):
        with pytest.raises(OccParseError, match="270115C0045000"):
            parseOccSymbol("VOO270115C0045000")

    def testEmptyString(self):
        with pytest.raises(OccParseError, match="''"):
            parseOccSymbol("")

    def testNonDigitStrike(self):
        with pytest.raises(OccParseError, match="not numeric"):
            parseOccSymbol("VOO270115C0045X000")

    def testInvalidDate(self):
        with pytest.raises(OccParseError, match="invalid expiry date"):
            parseOccSymbol("VOO270230C00450000")

    def testInvalidRight(self):
        with pytest.raises(OccParseError, match="'C' or 'P'"):
            parseOccSymbol("VOO270115X00450000")

class TestOptionSnapshot:
    def testModelFields(self):
        from datetime import datetime

        snapshot = OptionSnapshot(
            symbol="VOO270115C00450000",
            underlying="VOO",
            expiry=date(2027, 1, 15),
            strike=450.0,
            right="C",
            bid=70.10,
            ask=71.50,
            mid=70.80,
            delta=0.852,
            iv=0.246,
            quoteTimestamp=datetime(2026, 9, 1, 10, 30, 0),
            volume=150.0,
        )
        assert snapshot.symbol == "VOO270115C00450000"
        assert snapshot.mid == pytest.approx(70.80)
