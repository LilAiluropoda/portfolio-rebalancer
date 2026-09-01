import logging
import os
from abc import ABC, abstractmethod
from datetime import date, datetime

import requests
from dotenv import load_dotenv
from pydantic import BaseModel

from black_scholes import call_delta, implied_volatility

_logger = logging.getLogger("Portfolio Rebalancer")


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


class AlpacaConfigError(Exception):
    """Missing Alpaca credential. Names the variable; never a stack trace of it."""


class AlpacaApiError(Exception):
    """HTTP failure from the Alpaca data API. Carries status + short message only."""

    def __init__(self, statusCode: int, message: str):
        self.statusCode = statusCode
        self.message = message
        super().__init__(f"Alpaca API error {statusCode}: {message}")


class OptionQuoteSource(ABC):
    @abstractmethod
    def getSnapshots(self, symbols: list[str]) -> dict[str, OptionSnapshot]:
        pass

    @abstractmethod
    def getChain(
        self,
        underlying: str,
        expirationDateGte: str | None = None,
        strikePriceGte: float | None = None,
        strikePriceLte: float | None = None,
        optionType: str = "call",
    ) -> dict[str, OptionSnapshot]:
        pass


ALPACA_DATA_BASE_URL = "https://data.alpaca.markets"
_MAX_SNAPSHOT_SYMBOLS = 100


class AlpacaOptionData(OptionQuoteSource):
    def __init__(self):
        load_dotenv()
        self.keyId = os.getenv("APCA_API_KEY_ID")
        self.secretKey = os.getenv("APCA_API_SECRET_KEY")
        if not self.keyId:
            raise AlpacaConfigError(
                "Missing environment variable APCA_API_KEY_ID — set it in .env (see .env.example)"
            )
        if not self.secretKey:
            raise AlpacaConfigError(
                "Missing environment variable APCA_API_SECRET_KEY — set it in .env (see .env.example)"
            )

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.keyId,
            "APCA-API-SECRET-KEY": self.secretKey,
        }

    def _get(self, path: str, params: dict) -> dict:
        response = requests.get(
            f"{ALPACA_DATA_BASE_URL}{path}",
            headers=self._headers(),
            params=params,
            timeout=30,
        )
        if response.status_code != 200:
            try:
                detail = response.json().get("message", "")
            except Exception:
                detail = ""
            shortMessage = str(detail)[:200] or response.reason or "request failed"
            raise AlpacaApiError(response.status_code, shortMessage)
        return response.json()

    def getSnapshots(self, symbols: list[str]) -> dict[str, OptionSnapshot]:
        if len(symbols) > _MAX_SNAPSHOT_SYMBOLS:
            raise ValueError(
                f"getSnapshots accepts at most {_MAX_SNAPSHOT_SYMBOLS} symbols per call, got {len(symbols)}"
            )
        payload = self._get(
            "/v1beta1/options/snapshots",
            {"symbols": ",".join(symbols), "feed": "indicative"},
        )
        return self._parseSnapshots(payload)

    def getChain(
        self,
        underlying: str,
        expirationDateGte: str | None = None,
        strikePriceGte: float | None = None,
        strikePriceLte: float | None = None,
        optionType: str = "call",
    ) -> dict[str, OptionSnapshot]:
        params: dict = {"feed": "indicative", "limit": 1000}
        if expirationDateGte is not None:
            params["expiration_date_gte"] = expirationDateGte
        if strikePriceGte is not None:
            params["strike_price_gte"] = strikePriceGte
        if strikePriceLte is not None:
            params["strike_price_lte"] = strikePriceLte
        if optionType is not None:
            params["type"] = optionType

        snapshots: dict[str, OptionSnapshot] = {}
        pageToken: str | None = None
        while True:
            pageParams = dict(params)
            if pageToken is not None:
                pageParams["page_token"] = pageToken
            payload = self._get(f"/v1beta1/options/snapshots/{underlying}", pageParams)
            snapshots.update(self._parseSnapshots(payload))
            pageToken = payload.get("next_page_token")
            if not pageToken:
                break
        return snapshots

    def _parseSnapshots(self, payload: dict) -> dict[str, OptionSnapshot]:
        rawSnapshots = payload.get("snapshots", payload)
        snapshots: dict[str, OptionSnapshot] = {}
        for symbol, raw in rawSnapshots.items():
            if not isinstance(raw, dict):
                continue  # e.g. the chain endpoint's next_page_token sibling
            parsed = self._parseSnapshot(symbol, raw)
            if parsed is not None:
                snapshots[symbol] = parsed
        return snapshots

    def _parseSnapshot(self, symbol: str, raw: dict) -> OptionSnapshot | None:
        contract = parseOccSymbol(symbol)
        quote = raw.get("latestQuote") or {}
        bid, ask = quote.get("bp"), quote.get("ap")
        if not bid or bid <= 0 or not ask or ask <= 0:
            _logger.warning("Skipping %s: non-positive quote (bid=%s)", symbol, bid)
            return None

        quoteTimestamp = datetime.fromisoformat(quote["t"].replace("Z", "+00:00"))
        if quoteTimestamp.date() != datetime.now().astimezone().date():
            _logger.warning(
                "Skipping %s: quote timestamp %s is not same-day", symbol, quote["t"]
            )
            return None

        mid = (bid + ask) / 2
        iv = raw.get("impliedVolatility")
        greeks = raw.get("greeks") or {}
        delta = greeks.get("delta")

        if delta is None:
            delta, iv = self._fallbackDelta(contract, raw, mid, iv)

        return OptionSnapshot(
            symbol=symbol,
            underlying=contract.underlying,
            expiry=contract.expiry,
            strike=contract.strike,
            right=contract.right,
            bid=bid,
            ask=ask,
            mid=mid,
            delta=delta,
            iv=iv if iv is not None else 0.0,
            quoteTimestamp=quoteTimestamp,
            volume=raw.get("dailyBar", {}).get("volume", 0) or 0,
        )

    def _fallbackDelta(
        self, contract: OptionContract, raw: dict, mid: float, iv: float | None
    ) -> tuple[float, float]:
        underlyingAsset = raw.get("underlyingAsset") or {}
        spot = underlyingAsset.get("price") or underlyingAsset.get("close")
        if not spot:
            raise AlpacaApiError(
                200,
                f"Cannot compute Black-Scholes fallback delta for {contract.symbol}: "
                "greeks null and no underlying price in snapshot",
            )
        timeToExpiry = (contract.expiry - datetime.now().astimezone().date()).days / 365.0
        if timeToExpiry <= 0:
            _logger.warning(
                "Contract %s has expired; fallback delta treated as intrinsic",
                contract.symbol,
            )
            timeToExpiry = 1e-6
            delta = 1.0 if contract.strike < spot else 0.0
            return delta, iv if iv is not None else 0.0
        if iv is None:
            iv = implied_volatility(mid, spot, contract.strike, timeToExpiry)
        delta = call_delta(spot, contract.strike, timeToExpiry, iv)
        return delta, iv


class OptionDataSourceFactory:
    instances: dict[str, OptionQuoteSource] = {}

    @classmethod
    def getDataSource(cls, name: str) -> OptionQuoteSource:
        match name:
            case "alpaca":
                if name not in cls.instances:
                    cls.instances[name] = AlpacaOptionData()
                return cls.instances[name]
            case _:
                raise Exception(f"Source {name} not found / supported.")
