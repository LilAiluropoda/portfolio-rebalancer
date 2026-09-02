from pydantic import BaseModel
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import logging
import math
import yfinance as yf
import polars as pl
import argparse

from options_data import OccParseError, OptionQuoteSource, OptionDataSourceFactory, parseOccSymbol

APP_NAME = "Portfolio Rebalancer"
logger = logging.Logger(APP_NAME)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

OPTION_MULTIPLIER = 100.0
CASH_TYPE = "Cash and Cash Equivalents"
OPTION_TYPE = "LEAPS Call"

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
        residualDirection = "undershoot"
    elif trackingErrorExposure < -1e-9:
        residualDirection = "overshoot"
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