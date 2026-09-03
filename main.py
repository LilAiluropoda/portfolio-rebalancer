import argparse
import logging
from datetime import datetime
from pathlib import Path

import polars as pl

from constants import KIND_CASH, KIND_EQUITY, KIND_OPTION, OPTION_MULTIPLIER
from execution import (
    applyTrades,
    buildExposureReport,
    enrichPostTradePositions,
    executeTrades,
    printExposureReport,
    printTradeSummary,
)
from fees import TradingPlatform, TradingPlatformFactory
from market_data import (
    OptionDataSourceFactory,
    OptionQuoteSource,
    PriceDataSource,
    DataSourceFactory,
    parseOccSymbol,
)
from planning import buildSleeveTable, kindForInstrumentType, planTrades

APP_NAME = "Portfolio Rebalancer"
logger = logging.Logger(APP_NAME)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def normalizePositions(positionDF: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize the loaded CSV into the unified positions frame:
    every row (equity, cash, LEAPS option) carries kind, underlying,
    multiplier, and deltaAdj so valuation and exposure math are uniform.
    """
    if "leapsSleeve" not in positionDF.columns:
        positionDF = positionDF.with_columns(pl.lit(None, dtype=pl.String).alias("leapsSleeve"))

    positionDF = positionDF.with_columns(
        pl.col("instrumentType")
        .map_elements(kindForInstrumentType, return_dtype=pl.String)
        .alias("kind")
    )

    optionRows = positionDF.filter(pl.col("kind") == KIND_OPTION)
    rootBySymbol = {}
    for row in optionRows.iter_rows(named=True):
        parsed = parseOccSymbol(row["instrumentId"])
        if parsed.right != "C":
            raise ValueError(
                f"Option {row['instrumentId']} is a {parsed.right} option — the strategy is "
                "call-based stock replacement; puts would corrupt exposure math"
            )
        rootBySymbol[row["instrumentId"]] = parsed.underlying

    positionDF = positionDF.with_columns(
        pl.when(pl.col("kind") == KIND_OPTION)
        .then(pl.col("instrumentId").replace_strict(rootBySymbol, default=None))
        .otherwise(pl.col("instrumentId"))
        .alias("underlying"),
        pl.when(pl.col("kind") == KIND_OPTION)
        .then(pl.lit(OPTION_MULTIPLIER))
        .otherwise(pl.lit(1.0))
        .alias("multiplier"),
        pl.when(pl.col("kind") == KIND_CASH).then(pl.lit(0.0)).otherwise(pl.lit(1.0)).alias("deltaAdj"),
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

    for row in positionDF.filter(pl.col("kind") == KIND_OPTION).iter_rows(named=True):
        shares = row["shares"]
        if shares <= 0 or shares != int(shares):
            raise ValueError(
                f"Option {row['instrumentId']} shares must be a positive whole number "
                f"of contracts, got {shares}"
            )

    hasOptions = positionDF.filter(pl.col("kind") == KIND_OPTION).height > 0
    if hasOptions and (designated is None or leverage == 1.0) and not liquidateLeaps:
        raise ValueError(
            "Held LEAPS positions present but no sleeve is designated (or --leverage is at "
            "the 1.0 default) — proceeding would liquidate the LEAPS sleeve. Pass "
            "--liquidate-leaps to do that deliberately, or set the leapsSleeve marker and --leverage."
        )

    equityTotalPct = positionDF.filter(pl.col("kind") == KIND_EQUITY).select(pl.col("targetRatioPct").sum()).item()
    if equityTotalPct != 100:
        raise ValueError(
            f"Equity targetRatioPct did not add up to 100 (Actual: {equityTotalPct}), "
            "please check if csv input is correct"
        )

    return designated


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

    equityCashRows = positionDF.filter(pl.col("kind") != KIND_OPTION)
    optionRows = positionDF.filter(pl.col("kind") == KIND_OPTION)

    # Batched price fetches: group needed tickers by CSV timestamp (usually one
    # date) — equity/cash rows plus option-only underlyings' spot requests —
    # and fetch each date-group in a single call.
    pricedTickers: dict[str, set[str]] = {}
    for row in equityCashRows.iter_rows(named=True):
        pricedTickers.setdefault(row["timestamp"], set()).add(row["instrumentId"])
    equityUnderlyings = set(equityCashRows["instrumentId"].to_list())
    for row in optionRows.iter_rows(named=True):
        if row["underlying"] not in equityUnderlyings:
            pricedTickers.setdefault(row["timestamp"], set()).add(row["underlying"])
    pricesByTimestamp = {
        ts: priceDataSource.getClosingPrices(sorted(tickers), datetime.strptime(ts, "%Y-%m-%d"))
        for ts, tickers in pricedTickers.items()
    }

    equityCashRows = equityCashRows.with_columns(
        pl.struct(["instrumentId", "timestamp"])
        .map_elements(
            lambda row: pricesByTimestamp[row["timestamp"]][row["instrumentId"]],
            return_dtype=pl.Float64,
        )
        .alias("closingPrice"),
    )

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
    # price their underlying. Option-only sleeves' spots come from the same
    # batched fetch above (cached per underlying).
    spotByUnderlying = {
        row["instrumentId"]: row["closingPrice"]
        for row in equityCashRows.iter_rows(named=True)
    }
    for optionRow in optionRows.iter_rows(named=True):
        underlying = optionRow["underlying"]
        if underlying not in spotByUnderlying:
            spotByUnderlying[underlying] = pricesByTimestamp[optionRow["timestamp"]][underlying]

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
    if (
        positionDF.filter(pl.col("kind") == KIND_OPTION).height > 0
        or designatedUnderlying is not None
    ):
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
        optionSource, tradeTimestamp, liquidateLeaps=args.liquidateLeaps,
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
