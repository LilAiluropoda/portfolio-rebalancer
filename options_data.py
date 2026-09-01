from datetime import date, datetime
from pydantic import BaseModel


class OccParseError(Exception):
    pass


class OptionContract(BaseModel):
    symbol: str
    underlying: str
    expiry: date
    strike: float
    right: str  # "C" or "P"

    @property
    def isCall(self) -> bool:
        return self.right == "C"


def parseOccSymbol(symbol: str) -> OptionContract:
    """
    Parse an OCC option symbol (fixed offsets from the right):
    last 15 chars = YYMMDD (6) + C/P (1) + strike x 1000 zero-padded (8).
    Root is everything left of that, stripped of padding.
    """
    if not isinstance(symbol, str) or len(symbol) < 16:
        raise OccParseError(
            f"Invalid OCC option symbol '{symbol}': too short "
            "(expected root + 15 trailing characters)"
        )

    suffix = symbol[-15:]
    datePart, right, strikePart = suffix[:6], suffix[6], suffix[7:]

    try:
        expiry = datetime.strptime(datePart, "%y%m%d").date()
    except ValueError as e:
        raise OccParseError(
            f"Invalid OCC option symbol '{symbol}': invalid expiry date '{datePart}' ({e})"
        ) from e

    if right not in ("C", "P"):
        raise OccParseError(
            f"Invalid OCC option symbol '{symbol}': right must be 'C' or 'P', got '{right}'"
        )

    if not strikePart.isdigit():
        raise OccParseError(
            f"Invalid OCC option symbol '{symbol}': strike portion '{strikePart}' is not numeric"
        )

    underlying = symbol[:-15].rstrip()
    if not underlying:
        raise OccParseError(f"Invalid OCC option symbol '{symbol}': empty underlying root")

    return OptionContract(
        symbol=symbol,
        underlying=underlying,
        expiry=expiry,
        strike=int(strikePart) / 1000,
        right=right,
    )


class OptionSnapshot(BaseModel):
    symbol: str
    underlying: str
    expiry: date
    strike: float
    right: str
    bid: float
    ask: float
    mid: float
    delta: float
    iv: float
    quoteTimestamp: datetime
    volume: float
