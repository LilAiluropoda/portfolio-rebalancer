from pydantic import BaseModel
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import logging
import math
import yfinance as yf
import polars as pl
import argparse
import calendar

from options_data import (
    OccParseError,
    OptionQuoteSource,
    OptionSnapshot,
    OptionDataSourceFactory,
    parseOccSymbol,
)

APP_NAME = "Portfolio Rebalancer"
logger = logging.Logger(APP_NAME)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

OPTION_MULTIPLIER = 100.0
CASH_TYPE = "Cash and Cash Equivalents"
OPTION_TYPE = "LEAPS Call"

# Contract lifecycle / selection constants (R9-R13, R21)
MIN_EXPIRY_MONTHS = 21
ROLL_MONTHS_THRESHOLD = 12
TARGET_DELTA = 0.85
MAX_REL_SPREAD = 0.10
MIN_DAILY_VOLUME = 1.0

class PriceDataSource(ABC):
    @abstractmethod
    def getClosingPrice(self, ticker: str, date: datetime) -> float:
        pass

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

class TradingPlatform(ABC):
    @abstractmethod
    def calcTransactionCost(self, instrumentType: str, sharesChange: float, marketValueChange: float) -> float:
        pass

    @abstractmethod
    def calcOptionsTransactionCost(self, instrumentType: str, sharesChange: float, marketValueChange: float) -> float:
        pass

class FutuBullUS(TradingPlatform):
    def calcTransactionCost(self, instrumentType: str, sharesChange: float, marketValueChange: float) -> float:
        """
        Calculate US stock transaction cost based on fee schedule.
        """
        if instrumentType == "Cash and Cash Equivalents":
            return 0
        
        sharesChangeGross = abs(sharesChange)
        marketValueChangeGross = abs(marketValueChange)

        # ---- Commission ----
        commissionFee = sharesChangeGross * 0.0049
        commissionFee = max(commissionFee, 0.99)
        commissionFee = min(commissionFee, marketValueChangeGross * 0.005)

        # ---- Platform fee ----
        platformFee = sharesChangeGross * 0.005
        platformFee = max(platformFee, 1.00)
        platformFee = min(platformFee, marketValueChangeGross * 0.005)

        # ---- Clearing fee ----
        clearingFee = sharesChangeGross * 0.003

        # ---- Trading Activity Fee (SELL only) ----
        tradeActivityFee = 0.0
        if sharesChange < 0:
            tradeActivityFee = sharesChangeGross * 0.000195
            tradeActivityFee = max(tradeActivityFee, 0.01)
            tradeActivityFee = min(tradeActivityFee, 9.79)

        transactionCost = commissionFee + platformFee + clearingFee + tradeActivityFee

        return transactionCost

    def calcOptionsTransactionCost(self, instrumentType: str, sharesChange: float, marketValueChange: float) -> float:
        """
        Calculate US option transaction cost based on Futu HK US options fee schedule (USD, per contract).
        sharesChange carries contract counts; marketValueChange carries premium notional (contracts x premium x 100).
        One trade row = one order.
        """
        contractsGross = abs(sharesChange)
        marketValueChangeGross = abs(marketValueChange)

        # ---- Commission (premium > $0.1 assumed for LEAPS) ----
        commissionFee = max(contractsGross * 0.65, 1.99)

        # ---- Platform fee (fixed package) ----
        platformFee = contractsGross * 0.30

        # ---- Options Regulatory Fee (ORF) ----
        orfFee = contractsGross * 0.013

        # ---- OCC clearing fee (capped at 55) ----
        occFee = min(contractsGross * 0.02, 55)

        # ---- Settlement fee ----
        settlementFee = contractsGross * 0.18

        # ---- Consolidated Audit Trail (CAT) fee ----
        catFee = contractsGross * 0.0003

        # ---- SEC fee (SELL only) ----
        secFee = 0.0
        if sharesChange < 0:
            secFee = max(marketValueChangeGross * 0.0000206, 0.01)

        # ---- FINRA Trading Activity Fee (SELL only) ----
        finraActivityFee = 0.0
        if sharesChange < 0:
            finraActivityFee = max(contractsGross * 0.00329, 0.01)

        transactionCost = (
            commissionFee + platformFee + orfFee + occFee
            + settlementFee + catFee + secFee + finraActivityFee
        )

        return transactionCost

class TradingPlatformFactory:
    instances: dict[str, TradingPlatform] = {}
    @classmethod
    def getTradingPlatform(cls, name: str)-> TradingPlatform:
        match name:
            case "futubullUS":
                if name not in cls.instances:
                    cls.instances[name] = FutuBullUS()
                return cls.instances[name]
            case _:
                raise Exception(f"Platform {name} not found / supported.")

class Trade(BaseModel):
    tradeId: str
    instrumentId: str
    instrumentType: str
    price: float
    sharesChange: float
    marketValueChange: float
    transactionCost: float = 0
    timestamp: datetime
    underlying: str = ""
    quantityKind: str = "share"  # "share" | "contract" | "cash"
    exposureChange: float = 0.0
    reason: str = ""

    def calcTransactionCost(self, platform: TradingPlatform)->None:
        if self.instrumentType == "LEAPS Call":
            self.transactionCost = platform.calcOptionsTransactionCost(self.instrumentType, self.sharesChange, self.marketValueChange)
        else:
            self.transactionCost = platform.calcTransactionCost(self.instrumentType, self.sharesChange, self.marketValueChange)

@dataclass
class SleevePlan:
    contractChange: int
    shareChange: float
    trackingErrorExposure: float
    achievedExposure: float
    residualDirection: str  # "overshoot" | "undershoot" | "on-target"


def sizeSleeve(
    targetExposure: float,
    perContractExposure: float,
    heldContracts: float,
    heldShares: float,
    spot: float,
) -> SleevePlan:
    """
    Size a sleeve's delivery: contracts round to nearest (half-up); the signed
    share residual absorbs the difference, clamped so the sleeve never sells
    more shares than it holds; whatever survives integer share rounding is
    tracking error.
    """
    if targetExposure < 0:
        raise ValueError(f"Sleeve target exposure must be >= 0, got {targetExposure}")

    desiredContracts = int(math.floor(targetExposure / perContractExposure + 0.5))
    contractChange = desiredContracts - int(heldContracts)

    residualExposure = targetExposure - desiredContracts * perContractExposure
    desiredShareChange = residualExposure / spot
    shareChange = math.floor(desiredShareChange) if desiredShareChange >= 0 else math.ceil(desiredShareChange)
    shareChange = max(shareChange, -int(heldShares))  # never sell more than held

    achievedExposure = desiredContracts * perContractExposure + shareChange * spot
    # Signed tracking error: positive = overshoot, negative = undershoot
    trackingErrorExposure = achievedExposure - targetExposure
    if trackingErrorExposure > 1e-9:
        residualDirection = "overshoot"
    elif trackingErrorExposure < -1e-9:
        residualDirection = "undershoot"
    else:
        residualDirection = "on-target"

    return SleevePlan(
        contractChange=contractChange,
        shareChange=float(shareChange),
        trackingErrorExposure=trackingErrorExposure,
        achievedExposure=achievedExposure,
        residualDirection=residualDirection,
    )


def buildSleeveTable(
    positionEnrichedDF: pl.DataFrame, leverage: float, designatedUnderlying: str | None
) -> pl.DataFrame:
    """
    Group the enriched frame by underlying into a sleeve table:
      - currentExposure: sum of delta-adjusted exposure across share + contract rows
      - targetExposure: weight x MV for normal sleeves;
        (weight + L - 1) x MV for the designated sleeve;
        0 for non-designated option-only sleeves (liquidation target)
    """
    totalMarketValue = positionEnrichedDF["marketValue"].sum()

    weightByUnderlying = {
        row["underlying"]: row["targetRatioPct"] / 100.0
        for row in positionEnrichedDF.filter(pl.col("kind") == "equity").iter_rows(named=True)
    }

    sleeveTable = (
        positionEnrichedDF.filter(pl.col("kind") != "cash")
        .group_by("underlying")
        .agg(
            pl.col("exposure").sum().alias("currentExposure"),
            pl.when(pl.col("kind") == "option").then(pl.col("shares")).otherwise(0.0).sum().alias("heldContracts"),
            pl.when(pl.col("kind") == "equity").then(pl.col("shares")).otherwise(0.0).sum().alias("heldShares"),
        )
        .with_columns(
            (pl.col("underlying") == (designatedUnderlying or "")).alias("isDesignated"),
            pl.col("underlying").replace_strict(weightByUnderlying, default=0.0).alias("weight"),
        )
        .with_columns(
            pl.when(pl.col("isDesignated"))
            .then((pl.col("weight") + leverage - 1.0) * totalMarketValue)
            .otherwise(pl.col("weight") * totalMarketValue)
            .alias("targetExposure"),
        )
        .with_columns((pl.col("currentExposure") - pl.col("targetExposure")).alias("exposureDiff"))
    )
    return sleeveTable


def addMonths(d: date, months: int) -> date:
    """Shift a date by whole months, clamping the day to the target month's length."""
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def monthsBetween(later: date, earlier: date) -> int:
    """Whole months from `earlier` to `later` (calendar-month difference)."""
    return later.year * 12 + later.month - (earlier.year * 12 + earlier.month)


def _passesLiquidityFilter(snap) -> bool:
    if snap.mid <= 0:
        return False
    relativeSpread = (snap.ask - snap.bid) / snap.mid
    return relativeSpread <= MAX_REL_SPREAD and snap.volume >= MIN_DAILY_VOLUME


def _pickCandidate(candidates) -> OptionSnapshot:
    """Latest expiry first; within it, the strike whose delta is nearest TARGET_DELTA."""
    latestExpiry = max(s.expiry for s in candidates)
    return min(
        (s for s in candidates if s.expiry == latestExpiry),
        key=lambda s: abs(s.delta - TARGET_DELTA),
    )


def selectContract(
    optionSource: OptionQuoteSource, underlying: str, today: date
) -> tuple[OptionSnapshot | None, str]:
    """
    Rule-based contract selection (R12). Returns (snapshot | None, outcome) where
    outcome is "selected", "deferred" (chain-depth: candidates exist only below
    MIN_EXPIRY_MONTHS, R21), or "none" (no candidate passes the liquidity filter).
    """
    expiryFloor = addMonths(today, MIN_EXPIRY_MONTHS)
    chain = optionSource.getChain(
        underlying, expirationDateGte=expiryFloor.isoformat(), optionType="call"
    )
    candidates = [s for s in chain.values() if s.expiry >= expiryFloor and _passesLiquidityFilter(s)]
    if candidates:
        snap = _pickCandidate(candidates)
        logger.info(
            f"[Contract Selection] {underlying}: selected {snap.symbol} "
            f"(expiry {snap.expiry}, delta {snap.delta:.4f}, mid {snap.mid:.2f})"
        )
        return snap, "selected"

    # Chain-depth probe: would candidates exist without the MIN_EXPIRY_MONTHS floor?
    probeFloor = addMonths(today, ROLL_MONTHS_THRESHOLD)
    probeChain = optionSource.getChain(
        underlying, expirationDateGte=probeFloor.isoformat(), optionType="call"
    )
    if any(s.expiry >= probeFloor and _passesLiquidityFilter(s) for s in probeChain.values()):
        logger.info(
            f"[Contract Selection] {underlying}: candidates exist only below "
            f"{MIN_EXPIRY_MONTHS} months — deferring"
        )
        return None, "deferred"

    logger.info(f"[Contract Selection] {underlying}: no qualifying candidate")
    return None, "none"


def _bestEffortInt(shareDelta: float, heldShares: float) -> int:
    """Truncate toward zero; never sell more shares than held."""
    shareChange = int(shareDelta)  # truncates toward zero
    return max(shareChange, -int(heldShares))


def planTrades(
    positionEnrichedDF: pl.DataFrame,
    sleeveTable: pl.DataFrame,
    designatedUnderlying: str | None,
    leverage: float,
    optionSource: OptionQuoteSource,
    tradeTimestamp: datetime,
) -> list[Trade]:
    """
    Sleeve planner: walk the sleeve table and emit per-sleeve order intents
    (contract trades + share trades, each with a reason, R14). Lifecycle per
    R9-R13 and R17: keep/resize, roll, initiate, liquidate; shares fallback
    when no qualifying contract exists. Trade IDs are sequential from 1;
    the executor (U8) owns waterfall ordering and fees.
    """
    today = tradeTimestamp.date() if isinstance(tradeTimestamp, datetime) else tradeTimestamp
    totalMarketValue = positionEnrichedDF["marketValue"].sum()

    spotByUnderlying = {
        row["underlying"]: row["underlyingSpot"]
        for row in positionEnrichedDF.filter(pl.col("kind") != "cash").iter_rows(named=True)
    }
    optionRowsByUnderlying: dict[str, list[dict]] = {}
    for row in positionEnrichedDF.filter(pl.col("kind") == "option").iter_rows(named=True):
        optionRowsByUnderlying.setdefault(row["underlying"], []).append(row)

    trades: list[Trade] = []

    def emitContract(
        underlying: str, symbol: str, contractChange: int, premiumMid: float,
        perContractExposure: float, spot: float, reason: str,
    ) -> None:
        trades.append(
            Trade(
                tradeId=str(len(trades) + 1),
                instrumentId=symbol,
                instrumentType=OPTION_TYPE,
                price=premiumMid,
                sharesChange=float(contractChange),
                marketValueChange=contractChange * premiumMid * OPTION_MULTIPLIER,
                timestamp=tradeTimestamp,
                underlying=underlying,
                quantityKind="contract",
                exposureChange=contractChange * perContractExposure,
                reason=reason,
            )
        )

    def emitShares(underlying: str, shareChange: int, spot: float, reason: str) -> None:
        trades.append(
            Trade(
                tradeId=str(len(trades) + 1),
                instrumentId=underlying,
                instrumentType="Equity",
                price=spot,
                sharesChange=float(shareChange),
                marketValueChange=shareChange * spot,
                timestamp=tradeTimestamp,
                underlying=underlying,
                quantityKind="share",
                exposureChange=shareChange * spot,
                reason=reason,
            )
        )

    def emitResize(
        sleeve: dict, heldPerContractExposure: float, heldContracts: float,
        heldShares: float, spot: float, anchorRow: dict,
        contractReason: str, shareReason: str,
    ) -> None:
        plan = sizeSleeve(
            sleeve["targetExposure"], heldPerContractExposure, heldContracts, heldShares, spot
        )
        if plan.contractChange != 0:
            emitContract(
                sleeve["underlying"], anchorRow["instrumentId"], plan.contractChange,
                anchorRow["closingPrice"], heldPerContractExposure, spot, contractReason,
            )
        if plan.shareChange != 0:
            emitShares(sleeve["underlying"], int(plan.shareChange), spot, shareReason)

    for sleeve in sleeveTable.iter_rows(named=True):
        underlying = sleeve["underlying"]
        spot = spotByUnderlying[underlying]
        heldShares = sleeve["heldShares"]
        heldContracts = sleeve["heldContracts"]
        heldOptionRows = optionRowsByUnderlying.get(underlying, [])
        baseWeightExposure = sleeve["weight"] * totalMarketValue
        equityExposure = heldShares * spot

        if not sleeve["isDesignated"]:
            if heldContracts > 0:
                # R17: stray / option-only sleeve — liquidate all held contracts
                for row in heldOptionRows:
                    emitContract(
                        underlying, row["instrumentId"], -row["shares"], row["closingPrice"],
                        OPTION_MULTIPLIER * row["deltaAdj"] * spot, spot,
                        "liquidation: non-designated sleeve",
                    )
                shareChange = _bestEffortInt(
                    (sleeve["targetExposure"] - equityExposure) / spot, heldShares
                )
                if shareChange != 0:
                    emitShares(underlying, shareChange, spot, "drift rebalance")
            else:
                # Plain equity sleeve — share drift rebalance only
                shareChange = _bestEffortInt(
                    (sleeve["targetExposure"] - sleeve["currentExposure"]) / spot, heldShares
                )
                if shareChange != 0:
                    emitShares(underlying, shareChange, spot, "drift rebalance")
            continue

        # --- Designated sleeve lifecycle ---
        if heldContracts == 0:
            # R11: initiate
            snap, outcome = selectContract(optionSource, underlying, today)
            if outcome == "selected":
                perContract = OPTION_MULTIPLIER * snap.delta * spot
                plan = sizeSleeve(sleeve["targetExposure"], perContract, 0, heldShares, spot)
                if plan.contractChange != 0:
                    emitContract(
                        underlying, snap.symbol, plan.contractChange, snap.mid,
                        perContract, spot, "initiation",
                    )
                if plan.shareChange != 0:
                    emitShares(underlying, int(plan.shareChange), spot, "initiation share residual")
            else:
                # R13: shares fallback at base weight
                shareChange = _bestEffortInt((baseWeightExposure - equityExposure) / spot, heldShares)
                if shareChange != 0:
                    emitShares(
                        underlying, shareChange, spot, "shares fallback: no qualifying contract"
                    )
            continue

        earliestRow = min(
            heldOptionRows, key=lambda r: parseOccSymbol(r["instrumentId"]).expiry
        )
        monthsToExpiry = monthsBetween(parseOccSymbol(earliestRow["instrumentId"]).expiry, today)
        heldOptionExposure = sum(
            row["shares"] * OPTION_MULTIPLIER * row["deltaAdj"] * spot for row in heldOptionRows
        )
        heldPerContract = heldOptionExposure / heldContracts

        if monthsToExpiry > ROLL_MONTHS_THRESHOLD:
            # R9: keep — resize quantity only, no selection call
            emitResize(
                sleeve, heldPerContract, heldContracts, heldShares, spot, earliestRow,
                "resize", "drift rebalance",
            )
            continue

        # R10: roll window
        snap, outcome = selectContract(optionSource, underlying, today)
        if outcome == "selected":
            for row in heldOptionRows:
                emitContract(
                    underlying, row["instrumentId"], -row["shares"], row["closingPrice"],
                    OPTION_MULTIPLIER * row["deltaAdj"] * spot, spot, "roll: exit",
                )
            perContract = OPTION_MULTIPLIER * snap.delta * spot
            plan = sizeSleeve(sleeve["targetExposure"], perContract, 0, heldShares, spot)
            if plan.contractChange != 0:
                emitContract(
                    underlying, snap.symbol, plan.contractChange, snap.mid,
                    perContract, spot, "roll: replacement",
                )
            if plan.shareChange != 0:
                emitShares(underlying, int(plan.shareChange), spot, "roll: share residual")
        elif outcome == "deferred":
            # R21: chain-depth deferral — keep the held contract, resize toward target
            logger.info(
                f"[Planner] {underlying}: roll deferred — chain depth "
                f"(no qualifying expiry >= {MIN_EXPIRY_MONTHS} months)"
            )
            emitResize(
                sleeve, heldPerContract, heldContracts, heldShares, spot, earliestRow,
                "roll deferred — chain depth", "roll deferred — chain depth",
            )
        else:
            # R13: sell held, de-lever to base weight in shares
            for row in heldOptionRows:
                emitContract(
                    underlying, row["instrumentId"], -row["shares"], row["closingPrice"],
                    OPTION_MULTIPLIER * row["deltaAdj"] * spot, spot,
                    "roll: exit — no qualifying replacement",
                )
            shareChange = _bestEffortInt((baseWeightExposure - equityExposure) / spot, heldShares)
            if shareChange != 0:
                emitShares(underlying, shareChange, spot, "shares fallback: roll de-lever")

    return trades


def normalizePositions(positionDF: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize the loaded CSV into the unified positions frame:
    every row (equity, cash, LEAPS option) carries kind, underlying,
    multiplier, and deltaAdj so valuation and exposure math are uniform.
    """
    if "leapsSleeve" not in positionDF.columns:
        positionDF = positionDF.with_columns(pl.lit(None, dtype=pl.String).alias("leapsSleeve"))

    positionDF = positionDF.with_columns(
        pl.when(pl.col("instrumentType") == CASH_TYPE)
        .then(pl.lit("cash"))
        .when(pl.col("instrumentType") == OPTION_TYPE)
        .then(pl.lit("option"))
        .otherwise(pl.lit("equity"))
        .alias("kind")
    )

    optionRows = positionDF.filter(pl.col("kind") == "option")
    rootBySymbol = {
        row["instrumentId"]: parseOccSymbol(row["instrumentId"]).underlying
        for row in optionRows.iter_rows(named=True)
    }

    positionDF = positionDF.with_columns(
        pl.when(pl.col("kind") == "option")
        .then(pl.col("instrumentId").replace_strict(rootBySymbol, default=None))
        .otherwise(pl.col("instrumentId"))
        .alias("underlying"),
        pl.when(pl.col("kind") == "option")
        .then(pl.lit(OPTION_MULTIPLIER))
        .otherwise(pl.lit(1.0))
        .alias("multiplier"),
        pl.when(pl.col("kind") == "cash").then(pl.lit(0.0)).otherwise(pl.lit(1.0)).alias("deltaAdj"),
        pl.col("leapsSleeve")
        .cast(pl.String)
        .fill_null("")
        .str.to_lowercase()
        .is_in(["true", "1", "yes"])
        .alias("leapsSleeve"),
    )
    return positionDF


def validateInputs(positionDF: pl.DataFrame, leverage: float, liquidateLeaps: bool) -> str | None:
    """
    Validate the normalized frame and CLI inputs. Returns the designated
    LEAPS sleeve underlying (or None). Raises on invalid input.
    """
    if leverage < 1.0:
        raise ValueError(f"--leverage must be >= 1.0, got {leverage}")

    designatedUnderlyings = (
        positionDF.filter(pl.col("leapsSleeve")).select(pl.col("underlying")).unique().to_series().to_list()
    )
    if len(designatedUnderlyings) > 1:
        raise ValueError(
            f"At most one LEAPS sleeve underlying may be designated, got {designatedUnderlyings}"
        )
    designated = designatedUnderlyings[0] if designatedUnderlyings else None

    hasOptions = positionDF.filter(pl.col("kind") == "option").height > 0
    if hasOptions and (designated is None or leverage == 1.0) and not liquidateLeaps:
        raise ValueError(
            "Held LEAPS positions present but no sleeve is designated (or --leverage is at "
            "the 1.0 default) — proceeding would liquidate the LEAPS sleeve. Pass "
            "--liquidate-leaps to do that deliberately, or set the leapsSleeve marker and --leverage."
        )

    equityTotalPct = positionDF.filter(pl.col("kind") == "equity").select(pl.col("targetRatioPct").sum()).item()
    if equityTotalPct != 100:
        raise ValueError(
            f"Equity targetRatioPct did not add up to 100 (Actual: {equityTotalPct}), "
            "please check if csv input is correct"
        )

    return designated


def getAvailableCash(positionEnrichedDF: pl.DataFrame) -> tuple[bool, float]:
    cashRows = positionEnrichedDF.filter(pl.col("instrumentType") == CASH_TYPE)
    cashAvailable: float = (
        cashRows.select(pl.col("marketValue").sum()).item() if len(cashRows) > 0 else 0.0
    )
    hasCash = len(cashRows) > 0
    return hasCash, cashAvailable

def enrichPositions(
    positionDF: pl.DataFrame,
    priceDataSource: PriceDataSource,
    optionSource: OptionQuoteSource | None = None,
) -> pl.DataFrame:
    """
    Value every row by kind and derive the uniform exposure columns:
      - equity / cash: closing price via the equity source at the CSV timestamp
      - option: mid premium and delta via the option snapshot source at run time
    marketValue = shares x closingPrice x multiplier (cash impact)
    exposure    = shares x multiplier x deltaAdj x underlyingSpot (delta-adjusted dollars)
    Option valuation failure aborts here — before any trade generation.
    """
    logger.info(f"[Position Enrichment] Position enrichment started")

    positionDF = positionDF.with_row_index("rowIdx")

    equityCashRows = positionDF.filter(pl.col("kind") != "option").with_columns(
        pl.struct(["instrumentId", "timestamp", "kind"])
        .map_elements(
            lambda row: priceDataSource.getClosingPrice(
                ticker=row["instrumentId"],
                date=datetime.strptime(row["timestamp"], "%Y-%m-%d"),
            ),
            return_dtype=pl.Float64,
        )
        .alias("closingPrice"),
    )

    optionRows = positionDF.filter(pl.col("kind") == "option")
    if optionRows.height > 0:
        if optionSource is None:
            raise Exception("Option rows present but no option data source was provided")
        symbols = optionRows["instrumentId"].to_list()
        snapshots = optionSource.getSnapshots(symbols)
        missingSymbols = [s for s in symbols if s not in snapshots]
        if missingSymbols:
            raise ValueError(
                f"No option snapshot available for {missingSymbols} — aborting before trade generation"
            )
        optionRows = optionRows.with_columns(
            pl.col("instrumentId")
            .replace_strict({s: snap.mid for s, snap in snapshots.items()}, default=None)
            .alias("closingPrice"),
            pl.col("instrumentId")
            .replace_strict({s: snap.delta for s, snap in snapshots.items()}, default=None)
            .alias("deltaAdj"),
        )
    else:
        optionRows = optionRows.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("closingPrice")
        )

    positionEnrichedDF = (
        pl.concat([equityCashRows, optionRows], how="vertical")
        .sort("rowIdx")
        .drop("rowIdx")
    )

    positionEnrichedDF = positionEnrichedDF.with_columns(
        ((pl.col("shares") * pl.col("closingPrice") * pl.col("multiplier")).alias("marketValue")),
    )
    logger.info(f"[Position Enrichment] Added columns: closingPrice, marketValue")

    totalMarketValue = positionEnrichedDF["marketValue"].sum()
    logger.info(f"[Position Enrichment] Total market value: {totalMarketValue}")

    # Underlying spots for exposure: equity rows price themselves; option rows
    # price their underlying. Option-only sleeves fetch the spot via the equity
    # source at the CSV timestamp (single fetch, cached).
    spotByUnderlying = {
        row["underlying"]: row["closingPrice"]
        for row in positionEnrichedDF.filter(pl.col("kind") != "option").iter_rows(named=True)
    }
    for optionRow in positionEnrichedDF.filter(pl.col("kind") == "option").iter_rows(named=True):
        underlying = optionRow["underlying"]
        if underlying not in spotByUnderlying:
            spotByUnderlying[underlying] = priceDataSource.getClosingPrice(
                ticker=underlying,
                date=datetime.strptime(optionRow["timestamp"], "%Y-%m-%d"),
            )

    positionEnrichedDF = positionEnrichedDF.with_columns(
        pl.col("underlying").replace_strict(spotByUnderlying, default=None).alias("underlyingSpot"),
    )

    positionEnrichedDF = positionEnrichedDF.with_columns(
        ((pl.col("shares") * pl.col("multiplier") * pl.col("deltaAdj")).alias("deltaShares")),
    )
    positionEnrichedDF = positionEnrichedDF.with_columns(
        ((pl.col("deltaShares") * pl.col("underlyingSpot")).alias("exposure")),
    )
    logger.info(f"[Position Enrichment] Added columns: underlyingSpot, deltaShares, exposure")

    positionEnrichedDF = positionEnrichedDF.with_columns(
        ((pl.col("marketValue") / totalMarketValue * 100).alias("currentRatio")),
        ((pl.col("targetRatioPct") / 100 * totalMarketValue).alias("targetMarketValue")),
    )

    positionEnrichedDF = positionEnrichedDF.with_columns(
        (
            (pl.col("marketValue") - (pl.col("targetRatioPct") / 100 * totalMarketValue)).alias(
                "currMinusTargetMarketValue"
            )
        ),
    )
    logger.info(f"[Position Enrichment] Added columns: currentRatio, targetMarketValue, currMinusTargetMarketValue")

    enrichmentSummary: pl.DataFrame = positionEnrichedDF.select(
        ["instrumentId", "kind", "shares", "marketValue", "exposure", "targetRatioPct", "targetMarketValue"]
    )
    logger.info(f"[Position Enrichment] Enriched Positions:")
    enrichmentSummary.sort("marketValue", descending=True).show(
        limit=None,
        tbl_hide_dataframe_shape=True,
        tbl_column_data_type_inline=True,
        float_precision=2,
    )

    return positionEnrichedDF

def printTradeSummary(trades: list[Trade]) -> None:
    # Convert trades to Polars DataFrame for tabular display
    tradeSummary = pl.DataFrame([
        {
            "tradeId": trade.tradeId,
            "instrumentId": trade.instrumentId,
            "quantityKind": trade.quantityKind,
            "price": trade.price,
            "cost": trade.transactionCost,
            "sharesChange": trade.sharesChange,
            "marketValueChange": trade.marketValueChange,
            "reason": trade.reason,
        }
        for trade in trades if trade.instrumentType != "Cash and Cash Equivalents"
    ])

    tradeSummary.show(
        limit=None, 
        tbl_hide_dataframe_shape=True, 
        tbl_column_data_type_inline=True, 
        float_precision=2
    )

def executeTrades(
    plannedTrades: list[Trade],
    positionEnrichedDF: pl.DataFrame,
    tradingPlatform: TradingPlatform,
) -> list[Trade]:
    """
    Funding-waterfall executor (U8): consumes planner intents in a fixed order —
    contract sells -> equity sells -> contract buys -> equity buys — with the
    cash-residual row appended last (R20).

    Contract buys are NEVER cash-truncated (leverage is the control variable).
    Equity buys are cash-constrained: netCashUsed = Σ marketValueChange of
    emitted rows; each equity buy is capped at int((cashAvailable - netCashUsed)
    / price), skipped (with a log line) when the cap is <= 0, and its
    marketValueChange / exposureChange adjusted proportionally when truncated.
    Fees are computed per row AFTER ordering; the cash row nets the trade flows
    minus total fees (fees are paid from cash).
    """
    logger.info("[Trade Execution] Trade execution started")

    (_, cashAvailable) = getAvailableCash(positionEnrichedDF)
    logger.info(f"[Trade Execution] Available cash before rebalancing: {cashAvailable:.2f}")

    # Waterfall order: contract sells -> equity sells -> contract buys -> equity buys
    def orderKey(t: Trade) -> int:
        if t.quantityKind == "contract":
            return 0 if t.sharesChange < 0 else 2
        return 1 if t.sharesChange < 0 else 3

    netCashUsed: float = 0.0
    executed: list[Trade] = []

    for trade in sorted(plannedTrades, key=orderKey):
        if trade.quantityKind == "share" and trade.sharesChange > 0:
            # Equity buy — constrained by available cash (incl. cash freed from sells)
            maxSharesBuyable = int((cashAvailable - netCashUsed) / trade.price)
            if maxSharesBuyable <= 0:
                logger.info(
                    f"[Trade Execution] [instrumentId={trade.instrumentId}] "
                    f"Insufficient cash for buy — skipped."
                )
                continue
            if trade.sharesChange > maxSharesBuyable:
                ratio = maxSharesBuyable / trade.sharesChange
                logger.info(
                    f"[Trade Execution] [instrumentId={trade.instrumentId}] "
                    f"Buy truncated {trade.sharesChange:.0f} -> {maxSharesBuyable} by cash."
                )
                trade = trade.model_copy(
                    update={
                        "sharesChange": float(maxSharesBuyable),
                        "marketValueChange": trade.marketValueChange * ratio,
                        "exposureChange": trade.exposureChange * ratio,
                    }
                )

        netCashUsed += trade.marketValueChange
        executed.append(trade)
        logger.info(
            f"[Trade Execution] Executed Trade(instrumentId={trade.instrumentId} | "
            f"Price={trade.price:.2f} | Quantity={trade.sharesChange:.2f} "
            f"{trade.quantityKind} | Market Value Change={trade.marketValueChange:.2f})"
        )

    # Transaction costs (per row, computed after ordering)
    for trade in executed:
        trade.calcTransactionCost(platform=tradingPlatform)
    totalFees = sum(t.transactionCost for t in executed)

    # Cash residual row (R20): -(net trade flow) - fees; fees are paid from cash
    if abs(netCashUsed) > 0.01:
        cashMovement = -netCashUsed - totalFees
        executed.append(
            Trade(
                tradeId=str(len(executed) + 1),
                instrumentId="USD",
                instrumentType=CASH_TYPE,
                price=1,  # Assumption: Cash is in USD
                sharesChange=cashMovement,
                marketValueChange=cashMovement,
                timestamp=executed[-1].timestamp if executed else datetime.now(),
                quantityKind="cash",
                reason="cash residual: premium + shares − fees",
            )
        )
        logger.info(f"[Trade Execution] Cash movement: {cashMovement:.2f} (USD), fees: {totalFees:.2f}")

    logger.info(f"[Trade Execution] Trade(s) executed: {len(executed)}")
    return executed


def _kindForInstrumentType(instrumentType: str) -> str:
    if instrumentType == CASH_TYPE:
        return "cash"
    if instrumentType == OPTION_TYPE:
        return "option"
    return "equity"


def applyTrades(positionEnrichedDF: pl.DataFrame, trades: list[Trade]) -> pl.DataFrame:
    logger.info("[Trade Execution] Applying trades to positions")
    tradesSchema = {
        "instrumentId": pl.String,
        "instrumentTypeTrade": pl.String,
        "underlyingTrade": pl.String,
        "kindTrade": pl.String,
        "sharesChange": pl.Float64,
        "marketValueChange": pl.Float64,
        "closingPriceTrade": pl.Float64,
    }
    tradesDF = pl.DataFrame(
        [{
            "instrumentId": t.instrumentId,
            "instrumentTypeTrade": t.instrumentType,
            "underlyingTrade": t.underlying or t.instrumentId,
            "kindTrade": _kindForInstrumentType(t.instrumentType),
            "sharesChange": t.sharesChange,
            "marketValueChange": t.marketValueChange - t.transactionCost,  # Apply transaction cost here
            "closingPriceTrade": t.price,
        } for t in trades],
        schema=tradesSchema,
    )
    logger.info(f"[Trade Execution] Numbers of trades to be executed: {len(tradesDF)}")

    # Full join to include new instruments introduced by trades. Trade-side
    # columns are suffixed so the closingPrice coalesce can prefer the left
    # (position) price but fall back to the trade price for new instruments.
    positionPostTradeDF = (
        positionEnrichedDF
        .join(tradesDF, on="instrumentId", how="full")
        .with_columns(
            # Full join keeps the right-side key separately — coalesce so
            # trade-introduced instruments keep their instrumentId.
            pl.coalesce(pl.col("instrumentId"), pl.col("instrumentId_right")).alias("instrumentId"),
        )
        .with_columns(
            # Post-trade shares (trade-introduced rows start at 0)
            (
                pl.coalesce(pl.col("shares"), pl.lit(0.0)) +
                pl.coalesce(pl.col("sharesChange"), pl.lit(0.0))
            ).alias("shares"),
            # Post-trade market value (fees already deducted in tradesDF)
            (
                pl.coalesce(pl.col("marketValue"), pl.lit(0.0)) +
                pl.coalesce(pl.col("marketValueChange"), pl.lit(0.0))
            ).alias("marketValue"),
            # Instrument type: prefer position side, fall back to trade side
            pl.coalesce(pl.col("instrumentType"), pl.col("instrumentTypeTrade")).alias("instrumentType"),
            # Closing price: prefer position side, coalesce the trade price
            pl.coalesce(pl.col("closingPrice"), pl.col("closingPriceTrade")).alias("closingPrice"),
            # Carry kind / underlying for trade-introduced rows
            pl.coalesce(pl.col("kind"), pl.col("kindTrade")).alias("kind"),
            pl.coalesce(pl.col("underlying"), pl.col("underlyingTrade")).alias("underlying"),
            # Default target for new instruments
            pl.coalesce(pl.col("targetRatioPct"), pl.lit(0.0)).alias("targetRatioPct"),
        )
        .select([
            "instrumentId",
            "instrumentType",
            "underlying",
            "kind",
            "shares",
            "marketValue",
            "targetRatioPct",
            "closingPrice",
        ])
    )

    return positionPostTradeDF


def buildExposureReport(
    positionEnrichedDF: pl.DataFrame, sleeveTable: pl.DataFrame, trades: list[Trade]
) -> tuple[pl.DataFrame, float, float]:
    """
    Per-underlying exposure report (R19). Returns
    (reportDF, achievedLeverage, totalFees):
      - reportDF: underlying, postShares, postContracts, currentExposure,
        exposureChange, achievedExposure, targetExposure, trackingError,
        isDesignated
      - achievedLeverage: Σ achievedExposure / pre-trade market value
      - totalFees: Σ transactionCost across executed trades
    """
    preTradeMV = positionEnrichedDF["marketValue"].sum()

    exposureChangeByUnderlying: dict[str, float] = {}
    shareChangeByUnderlying: dict[str, float] = {}
    contractChangeByUnderlying: dict[str, float] = {}
    for t in trades:
        if not t.underlying:
            continue
        exposureChangeByUnderlying[t.underlying] = (
            exposureChangeByUnderlying.get(t.underlying, 0.0) + t.exposureChange
        )
        if t.quantityKind == "share":
            shareChangeByUnderlying[t.underlying] = (
                shareChangeByUnderlying.get(t.underlying, 0.0) + t.sharesChange
            )
        elif t.quantityKind == "contract":
            contractChangeByUnderlying[t.underlying] = (
                contractChangeByUnderlying.get(t.underlying, 0.0) + t.sharesChange
            )

    totalFees = sum(t.transactionCost for t in trades)

    report = sleeveTable.select(
        [
            "underlying",
            "heldShares",
            "heldContracts",
            "currentExposure",
            "targetExposure",
            "isDesignated",
        ]
    ).with_columns(
        pl.col("underlying").replace_strict(shareChangeByUnderlying, default=0.0).alias("shareChange"),
        pl.col("underlying").replace_strict(contractChangeByUnderlying, default=0.0).alias("contractChange"),
        pl.col("underlying").replace_strict(exposureChangeByUnderlying, default=0.0).alias("exposureChange"),
    ).with_columns(
        (pl.col("heldShares") + pl.col("shareChange")).alias("postShares"),
        (pl.col("heldContracts") + pl.col("contractChange")).alias("postContracts"),
        (pl.col("currentExposure") + pl.col("exposureChange")).alias("achievedExposure"),
    ).with_columns(
        (pl.col("achievedExposure") - pl.col("targetExposure")).alias("trackingError"),
    ).select([
        "underlying",
        "postShares",
        "postContracts",
        "currentExposure",
        "exposureChange",
        "achievedExposure",
        "targetExposure",
        "trackingError",
        "isDesignated",
    ])

    achievedLeverage = report["achievedExposure"].sum() / preTradeMV if preTradeMV else 0.0
    return report, achievedLeverage, totalFees


def printExposureReport(reportDF: pl.DataFrame, achievedLeverage: float, totalFees: float) -> None:
    logger.info(
        f"[Exposure Report] Achieved leverage: {achievedLeverage:.3f}x | Total fees: {totalFees:.2f} USD"
    )
    logger.info("[Exposure Report] Per-underlying exposure after trades:")
    reportDF.sort("achievedExposure", descending=True).show(
        limit=None,
        tbl_hide_dataframe_shape=True,
        tbl_column_data_type_inline=True,
        float_precision=2,
    )

def enrichPostTradePositions(positionPostTradeDF: pl.DataFrame):
    # Compute total portfolio value
    totalMarketValue = positionPostTradeDF["marketValue"].sum()

    # Add current ratio
    positionPostTradeDF = positionPostTradeDF.with_columns(
        pl.when(pl.lit(totalMarketValue) != 0)
        .then(pl.col("marketValue") / totalMarketValue * 100)
        .otherwise(0.0)
        .alias("expectedRatioPct")
    )
    positionPostTradeDF = positionPostTradeDF.with_columns(
        ((pl.col("expectedRatioPct") - pl.col("targetRatioPct")).alias("ratioDiffPct")),
    )
    logger.info(f"[Trade Execution] Added columns: expectedRatioPct, ratioDiffPct")
    printEnrichedPostTradePositions(positionPostTradeDF)
    return positionPostTradeDF

def printEnrichedPostTradePositions(positionPostTradeDF: pl.DataFrame) -> None:
    totalMarketValue = positionPostTradeDF["marketValue"].sum()
    logger.info(
        f"[Post Trade Analysis] Total market value after trades: "
        f"{totalMarketValue:.2f}"
    )

    logger.info("[Post Trade Analysis] Expected portfolio after trades:")

    positionPostTradeDF.select([
        "instrumentId",
        "instrumentType",
        "shares",
        "marketValue",
        "targetRatioPct",
        "expectedRatioPct",
        "ratioDiffPct",
        "closingPrice",
    ]).sort("marketValue", descending=True).show(
        limit=None,
        tbl_hide_dataframe_shape=True,
        tbl_column_data_type_inline=True,
        float_precision=2,
    )

def main():
    parser = argparse.ArgumentParser(description="Portfolio Rebalancer")
    parser.add_argument(
        "--portfolioCSV",
        type=str,
        required=True,
        help="Path to the CSV file containing portfolio data"
    )
    parser.add_argument(
        "--leverage",
        type=float,
        default=1.0,
        help="Portfolio-level leverage target L (total equity exposure = L x portfolio market value)"
    )
    parser.add_argument(
        "--liquidate-leaps",
        dest="liquidateLeaps",
        action="store_true",
        help="Deliberately proceed when held LEAPS would otherwise be liquidated (no sleeve marker, or L at the 1.0 default)"
    )

    args = parser.parse_args()

    FILEPATH = Path(args.portfolioCSV)
    DATASOURCE = "yFinance"
    TRADING_PLATFORM = "futubullUS"

    positionDF = pl.read_csv(FILEPATH)
    positionDF = normalizePositions(positionDF)
    designatedUnderlying = validateInputs(positionDF, args.leverage, args.liquidateLeaps)
    logger.info(f"[Data Loading] Loaded CSV with {len(positionDF)} records")
    logger.info(f"[Data Loading] Leverage target: {args.leverage} | Designated LEAPS sleeve: {designatedUnderlying or 'none'}")

    securityNames = positionDF.select(pl.col("instrumentId")).to_series().to_list()
    logger.info(f"[Data Loading] Names: {securityNames}")

    # Value the positions — the option source is only constructed when option rows exist,
    # so legacy runs need no Alpaca credentials or network access.
    priceDataSource: PriceDataSource = DataSourceFactory.getDataSource(name=DATASOURCE)
    logger.info(f"[Data Loading] Data source selected: {DATASOURCE}")
    optionSource = None
    if positionDF.filter(pl.col("kind") == "option").height > 0:
        optionSource = OptionDataSourceFactory.getDataSource("alpaca")
        logger.info(f"[Data Loading] Option data source selected: alpaca")
    positionEnrichedDF = enrichPositions(positionDF, priceDataSource, optionSource)

    tradeTimestamp = datetime.now()
    tradingPlatform: TradingPlatform = TradingPlatformFactory.getTradingPlatform(TRADING_PLATFORM)
    logger.info(f"[Trade Generation] Trading platform selected: {TRADING_PLATFORM}")

    # Planner -> executor split: planTrades emits intents, executeTrades runs
    # the funding waterfall and prices fees per row.
    sleeveTable = buildSleeveTable(positionEnrichedDF, args.leverage, designatedUnderlying)
    plannedTrades = planTrades(
        positionEnrichedDF, sleeveTable, designatedUnderlying, args.leverage,
        optionSource, tradeTimestamp,
    )
    trades = executeTrades(plannedTrades, positionEnrichedDF, tradingPlatform)

    if trades:
        printTradeSummary(trades)
    else:
        logger.info("[Trade Generation] No trades generated.")

    positionPostTradeDF = applyTrades(positionEnrichedDF, trades)
    _ = enrichPostTradePositions(positionPostTradeDF)

    reportDF, achievedLeverage, totalFees = buildExposureReport(
        positionEnrichedDF, sleeveTable, trades
    )
    printExposureReport(reportDF, achievedLeverage, totalFees)

    return trades


if __name__ == "__main__":
    main()