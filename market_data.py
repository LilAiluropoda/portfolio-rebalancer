"""Market-data providers: equity prices (yfinance) and option quotes (Alpaca)."""

import logging
import os
import re
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, timezone

import requests
import yfinance as yf
from dotenv import load_dotenv
from pydantic import BaseModel

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
    if not re.fullmatch(r"[A-Z]{1,6}", underlying):
        raise OccParseError(
            f"Invalid OCC option symbol '{symbol}': underlying root must be "
            "1-6 uppercase A-Z characters"
        )

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


class PriceDataSource(ABC):
    @abstractmethod
    def getClosingPrice(self, ticker: str, date: datetime) -> float:
        pass

    def getClosingPrices(self, tickers: list[str], date: datetime) -> dict[str, float]:
        """Batched closing prices; default delegates per-ticker."""
        return {ticker: self.getClosingPrice(ticker, date) for ticker in tickers}


class YFinancePriceData(PriceDataSource):
    def getClosingPrice(self, ticker: str, date: datetime):
        try:
            if ticker == "USD":
                return 1.0

            instrumentQuote = yf.download(
                ticker, start=date, end=date + timedelta(days=1), progress=False
            )

            if instrumentQuote.empty:
                raise Exception(f"No data available for {ticker} on {date}")

            closingPrice: float = instrumentQuote["Close"].squeeze().item()
            return float(closingPrice)

        except Exception as e:
            raise Exception(f"Error fetching data for {ticker} on {date}: {str(e)}")

    def getClosingPrices(self, tickers: list[str], date: datetime) -> dict[str, float]:
        try:
            prices: dict[str, float] = {t: 1.0 for t in tickers if t == "USD"}
            fetch = [t for t in tickers if t != "USD"]
            if not fetch:
                return prices

            instrumentQuote = yf.download(
                tickers=fetch, start=date, end=date + timedelta(days=1), progress=False
            )
            if instrumentQuote.empty:
                raise Exception(f"No data available for {fetch} on {date}")

            close = instrumentQuote["Close"]
            for ticker in fetch:
                # Multi-ticker downloads give a (Price, Ticker) column pair;
                # single-ticker downloads may collapse to a bare Series.
                series = close[ticker] if hasattr(close, "columns") else close
                if series.isna().all():
                    raise Exception(f"No data available for {ticker} on {date}")
                prices[ticker] = float(series.dropna().iloc[-1])
            return prices

        except Exception as e:
            raise Exception(f"Error fetching data for {tickers} on {date}: {str(e)}")


class DataSourceFactory:
    instances: dict[str, PriceDataSource] = {}
    @classmethod
    def getDataSource(cls, name: str)-> PriceDataSource:
        match name:
            case "yFinance":
                if name not in cls.instances:
                    cls.instances[name] = YFinancePriceData()
                return cls.instances["yFinance"]
            case _:
                raise Exception(f"Source {name} not found / supported.")


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
_MAX_CHAIN_PAGES = 50


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
        try:
            response = requests.get(
                f"{ALPACA_DATA_BASE_URL}{path}",
                headers=self._headers(),
                params=params,
                timeout=30,
            )
        except requests.RequestException as e:
            raise AlpacaApiError(
                0, f"{type(e).__name__}: {str(e)[:200]}"
            ) from e
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
        pagesFetched = 0
        while True:
            pagesFetched += 1
            if pagesFetched > _MAX_CHAIN_PAGES:
                raise AlpacaApiError(
                    0,
                    f"getChain for {underlying} exceeded {_MAX_CHAIN_PAGES} pages — "
                    "aborting to avoid an unbounded pagination loop",
                )
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

        rawTs = quote.get("t")
        try:
            quoteTimestamp = datetime.fromisoformat(str(rawTs).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            _logger.warning("Skipping %s: missing or unparseable quote timestamp (%r)", symbol, rawTs)
            return None
        if quoteTimestamp.date() != datetime.now(timezone.utc).date():
            _logger.warning(
                "Skipping %s: quote timestamp %s is not same-day", symbol, quote["t"]
            )
            return None

        mid = (bid + ask) / 2
        greeks = raw.get("greeks") or {}
        delta = greeks.get("delta")
        if delta is None:
            _logger.warning(
                "Skipping %s: greeks.delta is null (no fallback estimation)", symbol
            )
            return None

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
            iv=raw.get("impliedVolatility") or 0.0,
            quoteTimestamp=quoteTimestamp,
            volume=raw.get("dailyBar", {}).get("volume", 0) or 0,
        )


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
