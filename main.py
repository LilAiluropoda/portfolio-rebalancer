from pydantic import BaseModel
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
import logging
import yfinance as yf
import polars as pl
import argparse

APP_NAME = "Portfolio Rebalancer"
logger = logging.Logger(APP_NAME)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


positionSchema = {
    "instrumentId": pl.String,
    "idType": pl.String,
    "instrumentType": pl.String,
    "marketValue": pl.Float64,
    "shares": pl.Float64,
    "targetRatioPct": pl.Float64,
    "timestamp": pl.Date,
}

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

    def calcTransactionCost(self, platform: TradingPlatform)->None:
        self.transactionCost = platform.calcTransactionCost(self.instrumentType, self.sharesChange, self.marketValueChange)

def getAvailableCash(positionEnrichedDF: pl.DataFrame) -> tuple[bool, float]:
    cashRows = positionEnrichedDF.filter(pl.col("instrumentType") == "Cash and Cash Equivalents")
    cashAvailable: float = (
        cashRows.select(pl.col("marketValue").sum()).item() if len(cashRows) > 0 else 0.0
    )
    hasCash = len(cashRows) > 0
    return hasCash, cashAvailable

def enrichPositions(positionDF: pl.DataFrame, priceDataSource: PriceDataSource):
    logger.info(f"[Position Enrichment] Position enrichment started")
    positionEnrichedDF = positionDF.with_columns(
        (
            pl.struct(["instrumentId", "timestamp", "idType"])
            .map_elements(
                lambda row: priceDataSource.getClosingPrice(
                    ticker=row["instrumentId"],
                    date=datetime.strptime(row["timestamp"], "%Y-%m-%d"),
                ),
                return_dtype=pl.Float64,
            )
            .alias("closingPrice")
        ),
    )

    positionEnrichedDF = positionEnrichedDF.with_columns(
        ((pl.col("shares") * pl.col("closingPrice")).alias("marketValue")),
    )
    logger.info(f"[Position Enrichment] Added columns: closingPrice, marketValue")
    
    overallStats: pl.DataFrame = positionEnrichedDF.sum()
    totalPct = overallStats.select(pl.col("targetRatioPct")).item()
    if totalPct != 100:
        raise Exception(
            f"targetRatioPct did not add up to 100 (Actual: {totalPct}), "
            "please check if csv input is correct"
        )
    totalMarketValue = overallStats.select(pl.col("marketValue")).item()
    logger.info(f"[Position Enrichment] Total market value: {totalMarketValue}")
    
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

    enrichmentSummary: pl.DataFrame = positionEnrichedDF.select(["instrumentId", "shares", "marketValue", "targetRatioPct", "targetMarketValue"])
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
    
    args = parser.parse_args()

    FILEPATH = Path(args.portfolioCSV)
    DATASOURCE = "yFinance"
    TRADING_PLATFORM = "futubullUS"

    positionDF = pl.read_csv(FILEPATH)

    securityNames = positionDF.select(pl.col("instrumentId")).to_series().to_list()
    logger.info(f"[Data Loading] Loaded CSV with {len(positionDF)} records")
    logger.info(f"[Data Loading] Names: {securityNames}")

    # Value the positions
    priceDataSource: PriceDataSource = DataSourceFactory.getDataSource(name=DATASOURCE)
    logger.info(f"[Data Loading] Data source selected: {DATASOURCE}")
    positionEnrichedDF = enrichPositions(positionDF, priceDataSource)

    tradeTimestamp = datetime.now()
    tradingPlatform: TradingPlatform = TradingPlatformFactory.getTradingPlatform(TRADING_PLATFORM)
    logger.info(f"[Trade Generation] Trading platform selected: {TRADING_PLATFORM}")
    trades = generateTrades(positionEnrichedDF, tradingPlatform, tradeTimestamp)
    positionPostTradeDF = applyTrades(positionEnrichedDF, trades)
    _ = enrichPostTradePositions(positionPostTradeDF)

    return trades


if __name__ == "__main__":
    main()