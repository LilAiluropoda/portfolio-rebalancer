"""Trade execution: funding waterfall, application, reporting."""

import logging
from datetime import datetime

import polars as pl

from constants import CASH_TYPE, KIND_CASH, KIND_EQUITY, KIND_OPTION, QTY_CASH, QTY_CONTRACT, QTY_SHARE
from fees import TradingPlatform
from planning import Trade, kindForInstrumentType

logger = logging.getLogger("Portfolio Rebalancer")


def getAvailableCash(positionEnrichedDF: pl.DataFrame) -> float:
    cashRows = positionEnrichedDF.filter(pl.col("kind") == KIND_CASH)
    return cashRows.select(pl.col("marketValue").sum()).item() if len(cashRows) > 0 else 0.0


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
            "exposureChange": trade.exposureChange,
            "reason": trade.reason,
        }
        for trade in trades if trade.instrumentType != CASH_TYPE
    ])

    tradeSummary.show(
        limit=None,
        tbl_hide_dataframe_shape=True,
        tbl_column_data_type_inline=True,
        float_precision=2,
        tbl_cols=50,
        tbl_width_chars=300,
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

    cashAvailable = getAvailableCash(positionEnrichedDF)
    logger.info(f"[Trade Execution] Available cash before rebalancing: {cashAvailable:.2f}")

    # Waterfall order: contract sells -> equity sells -> contract buys -> equity buys
    def orderKey(t: Trade) -> int:
        if t.quantityKind == QTY_CONTRACT:
            return 0 if t.sharesChange < 0 else 2
        return 1 if t.sharesChange < 0 else 3

    netCashUsed: float = 0.0
    executed: list[Trade] = []

    for trade in sorted(plannedTrades, key=orderKey):
        if trade.quantityKind == QTY_SHARE and trade.sharesChange > 0:
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

    # Cash residual row (R20): -(net trade flow) - fees; fees are paid from cash.
    # Emitted whenever there is net flow OR any fees, so premium-neutral rolls
    # still ledger their fees.
    if abs(netCashUsed) > 0.01 or totalFees > 0:
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
                quantityKind=QTY_CASH,
                reason="cash residual: premium + shares − fees",
            )
        )
        logger.info(f"[Trade Execution] Cash movement: {cashMovement:.2f} (USD), fees: {totalFees:.2f}")

    logger.info(f"[Trade Execution] Trade(s) executed: {len(executed)}")
    return executed


def applyTrades(positionEnrichedDF: pl.DataFrame, trades: list[Trade]) -> pl.DataFrame:
    logger.info("[Trade Execution] Applying trades to positions")
    tradesSchema = {
        "instrumentId": pl.String,
        "instrumentTypeTrade": pl.String,
        "underlyingTrade": pl.String,
        "kindTrade": pl.String,
        "sharesChange": pl.Float64,
        "marketValueChange": pl.Float64,
        "exposureChange": pl.Float64,
        "closingPriceTrade": pl.Float64,
    }
    tradesDF = pl.DataFrame(
        [{
            "instrumentId": t.instrumentId,
            "instrumentTypeTrade": t.instrumentType,
            "underlyingTrade": t.underlying or t.instrumentId,
            "kindTrade": kindForInstrumentType(t.instrumentType),
            "sharesChange": t.sharesChange,
            "marketValueChange": t.marketValueChange,
            "exposureChange": t.exposureChange,
            "closingPriceTrade": t.price,
        } for t in trades],
        schema=tradesSchema,
    )
    # Aggregate per instrumentId so duplicate rows (e.g. duplicate CSV option
    # rows) cannot fan out into a cartesian full join. Fees live solely in the
    # cash residual row (R20), not per-row.
    tradesDF = tradesDF.group_by("instrumentId").agg(
        pl.col("sharesChange").sum(),
        pl.col("marketValueChange").sum(),
        pl.col("exposureChange").sum(),
        pl.col("instrumentTypeTrade").first(),
        pl.col("underlyingTrade").first(),
        pl.col("kindTrade").first(),
        pl.col("closingPriceTrade").first(),
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
            # Post-trade market value (fees are netted solely via the cash row)
            (
                pl.coalesce(pl.col("marketValue"), pl.lit(0.0)) +
                pl.coalesce(pl.col("marketValueChange"), pl.lit(0.0))
            ).alias("marketValue"),
            # Post-trade monetized-delta exposure (equity delta = 1; contracts
            # carry their delta via the trade's exposureChange delta)
            (
                pl.coalesce(pl.col("exposure"), pl.lit(0.0)) +
                pl.coalesce(pl.col("exposureChange"), pl.lit(0.0))
            ).alias("exposure"),
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
            "exposure",
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
        if t.quantityKind == QTY_SHARE:
            shareChangeByUnderlying[t.underlying] = (
                shareChangeByUnderlying.get(t.underlying, 0.0) + t.sharesChange
            )
        elif t.quantityKind == QTY_CONTRACT:
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
            "weight",
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
    ).with_columns(
        # achievedRatioPct = monetized-delta share of the whole portfolio's
        # achieved monetized delta. targetRatioPct is the INPUT weight
        # passthrough (equity-row weight x 100; option-only sleeves 0) — NOT
        # a share of total target exposure, so it reads verbatim like the CSV.
        pl.when(pl.col("achievedExposure").sum() != 0)
        .then(pl.col("achievedExposure") / pl.col("achievedExposure").sum() * 100)
        .otherwise(0.0)
        .alias("achievedRatioPct"),
        (pl.col("weight") * 100.0).alias("targetRatioPct"),
    ).select([
        "underlying",
        "postShares",
        "postContracts",
        "currentExposure",
        "exposureChange",
        "achievedExposure",
        "achievedRatioPct",
        "targetExposure",
        "targetRatioPct",
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
        tbl_cols=50,
        tbl_width_chars=300,
    )


def buildExpectedPositions(positionPostTradeDF: pl.DataFrame) -> pl.DataFrame:
    """
    Per-UNDERLYING expected position table: stock and contract rows of the
    same underlying aggregate into one sleeve row after trades.
      - postMarketValue: cash value of the sleeve (sum of marketValue)
      - postExposure: monetized delta (sum of exposure)
      - targetRatioPct: the INPUT weight passthrough from the equity row
        (0 for option-only sleeves / trade-introduced underlyings)
      - achievedRatioPct: sleeve postExposure / total postExposure x 100
    """
    return (
        positionPostTradeDF.filter(pl.col("kind") != KIND_CASH)
        .group_by("underlying")
        .agg(
            pl.when(pl.col("kind") == KIND_EQUITY)
            .then(pl.col("shares")).otherwise(0.0)
            .sum().alias("postShares"),
            pl.when(pl.col("kind") == KIND_OPTION)
            .then(pl.col("shares")).otherwise(0.0)
            .sum().alias("postContracts"),
            pl.col("marketValue").sum().alias("postMarketValue"),
            pl.col("exposure").sum().alias("postExposure"),
            pl.when(pl.col("kind") == KIND_EQUITY)
            .then(pl.col("targetRatioPct")).otherwise(0.0)
            .sum().alias("targetRatioPct"),
        )
        .with_columns(
            pl.when(pl.col("postExposure").sum() != 0)
            .then(pl.col("postExposure") / pl.col("postExposure").sum() * 100)
            .otherwise(0.0)
            .alias("achievedRatioPct"),
        )
        .sort("postMarketValue", descending=True)
    )


def printExpectedPositions(positionPostTradeDF: pl.DataFrame) -> None:
    """Per-underlying expected holdings after trades: market value (cash) and
    exposure (monetized delta) side by side, plus input weights verbatim."""
    totalMarketValue = positionPostTradeDF["marketValue"].sum()
    logger.info(
        f"[Expected Position] Total market value after trades: "
        f"{totalMarketValue:.2f}"
    )

    logger.info("[Expected Position] Expected positions after trades (per underlying):")

    buildExpectedPositions(positionPostTradeDF).select([
        "underlying",
        "postShares",
        "postContracts",
        "postMarketValue",
        "postExposure",
        "targetRatioPct",
        "achievedRatioPct",
    ]).show(
        limit=None,
        tbl_hide_dataframe_shape=True,
        tbl_column_data_type_inline=True,
        float_precision=2,
        tbl_cols=50,
        tbl_width_chars=300,
    )
