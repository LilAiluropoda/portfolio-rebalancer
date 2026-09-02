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
            "price": trade.price,
            "cost": trade.transactionCost,
            "sharesChange": trade.sharesChange,
            "marketValueChange": trade.marketValueChange
        }
        for trade in trades if trade.instrumentType != "Cash and Cash Equivalents"
    ])

    tradeSummary.show(
        limit=None, 
        tbl_hide_dataframe_shape=True, 
        tbl_column_data_type_inline=True, 
        float_precision=2
    )

def generateTrades(positionEnrichedDF: pl.DataFrame, tradingPlatform: TradingPlatform, tradeTimestamp: datetime) -> list[Trade]:
    """
    Generate trades to rebalance the portfolio.
    Strategy:
    1. Use cash first (idType == 'cash') to fund purchases.
    2. Two-pass ordering: process sells first to free up cash, then process buys.
    3. For non-cash positions, compute the integer share delta needed to move
       each position toward its target market value.
    4. After share-level rounding, reconcile the residual cash impact so the
       cash position absorbs whatever is left over.
    Returns a list of Trade objects (cash trade last, after netting).
    """
    logger.info(f"[Trade Generation] Trade generation started")
    
    (hasCash, cashAvailable) = getAvailableCash(positionEnrichedDF)
    logger.info(f"[Trade Generation] Available cash before rebalancing: {cashAvailable:.2f}")

    netCashUsed: float = 0.0
    trades: list[Trade] = []
    nonCashRows = positionEnrichedDF.filter(pl.col("instrumentType") != "Cash and Cash Equivalents")

    # Two-pass ordering: sells first (currMinusTargetMarketValue > 0) then buys (< 0)
    sellRows = nonCashRows.filter(pl.col("currMinusTargetMarketValue") > 0)
    buyRows = nonCashRows.filter(pl.col("currMinusTargetMarketValue") < 0)
    orderedRows = pl.concat([sellRows, buyRows])

    for index, row in enumerate(orderedRows.iter_rows(named=True), start=1):
        tradeId: int = index
        instrumentId: str = row["instrumentId"]
        instrumentType: str = row["instrumentType"]
        closingPrice: float = row["closingPrice"]
        currentShares: float = row["shares"]
        targetDifference: float = row["currMinusTargetMarketValue"]  # negative → need to buy

        if closingPrice <= 0:
            logger.warning(f"[Trade Generation] [instrumentId={instrumentId}] No valid closing price, skipping...")
            continue

        requiredSharesChange = -1 * targetDifference / closingPrice  # positive = buy, negative = sell

        if requiredSharesChange >= 0:
            # Buying — constrained by available cash (including cash freed from sells)
            maxSharesBuyable = int((cashAvailable - netCashUsed) / closingPrice)
            sharesChange = min(int(requiredSharesChange), maxSharesBuyable)
        else:
            # Selling — can't sell more shares than we hold
            sharesChange = max(int(requiredSharesChange), -int(currentShares))

        if sharesChange == 0:
            logger.info(f"[Trade Generation] [instrumentId={instrumentId}] No trade needed.")
            continue

        marketValueChange = sharesChange * closingPrice
        netCashUsed += marketValueChange

        trades.append(
            Trade(
                tradeId=str(tradeId),
                instrumentId=instrumentId,
                instrumentType=instrumentType,
                price=closingPrice,
                sharesChange=float(sharesChange),
                marketValueChange=marketValueChange,
                timestamp=tradeTimestamp,
            )
        )
        logger.info(f"[Trade Generation] Generated Trade(instrumentId={instrumentId} | Price={closingPrice:.2f} | Share Change={sharesChange:.2f} | Market Value Change={marketValueChange:.2f})")

    # Cash movement
    isCashUsed = abs(netCashUsed) > 0.01
    if hasCash and isCashUsed:
        cashMovement = -1 * netCashUsed  # cash decreases when we buy, increases when we sell
        trades.append(
            Trade(
                tradeId=str(len(orderedRows)),
                instrumentId="USD",
                instrumentType="Cash and Cash Equivalents",
                price=1,  # Assumption: Cash is in USD
                sharesChange=cashMovement,
                marketValueChange=cashMovement,
                timestamp=tradeTimestamp,
            )
        )
        logger.info(f"[Trade Generation] Cash movement: {cashMovement} (USD).")

    logger.info(f"[Trade Generation] Trade(s) generated: {len(trades)}")

    # Transaction Cost
    for trade in trades:
        trade.calcTransactionCost(platform=tradingPlatform)

    if trades:
        printTradeSummary(trades)
    else:
        logger.info(f"[Trade Generation] No trades generated.")

    return trades

def applyTrades(positionEnrichedDF: pl.DataFrame, trades: list[Trade]) -> pl.DataFrame:
    # Convert trades to DataFrame
    logger.info("[Trade Execution] Trade execution started")
    tradesDF = pl.DataFrame(
        [{
            "instrumentId": t.instrumentId,
            "instrumentType": t.instrumentType,
            "sharesChange": t.sharesChange,
            "marketValueChange": t.marketValueChange - t.transactionCost, # Apply transaction cost here
            "closingPrice": t.price,
        } for t in trades]
    )
    logger.info(f"[Trade Execution] Numbers of trades to be executed: {len(tradesDF)}")

    # Outer join to include new instruments introduced by trades
    positionPostTradeDF = (
        positionEnrichedDF
        .join(tradesDF, on="instrumentId", how="full")
        .with_columns([
            # positionPostTradeDF shares
            (
                pl.coalesce(pl.col("shares"), pl.lit(0)) +
                pl.coalesce(pl.col("sharesChange"), pl.lit(0))
            ).alias("shares"),

            # positionPostTradeDF market value
            (
                pl.coalesce(pl.col("marketValue"), pl.lit(0)) +
                pl.coalesce(pl.col("marketValueChange"), pl.lit(0))
            ).alias("marketValue"),

            # Instrument type (prefer original)
            pl.col("instrumentType"),

            # Closing price (prefer trade price if new instrument)
            pl.col("closingPrice"),
                
            # Default target for new instruments
            pl.coalesce(
                pl.col("targetRatioPct"),
                pl.lit(0.0)
            ).alias("targetRatioPct"),
        ])
        .select([
            "instrumentId",
            "instrumentType",
            "shares",
            "marketValue",
            "targetRatioPct",
            "closingPrice",
        ])
    )

    return positionPostTradeDF

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
    trades = generateTrades(positionEnrichedDF, tradingPlatform, tradeTimestamp)
    positionPostTradeDF = applyTrades(positionEnrichedDF, trades)
    _ = enrichPostTradePositions(positionPostTradeDF)

    return trades


if __name__ == "__main__":
    main()