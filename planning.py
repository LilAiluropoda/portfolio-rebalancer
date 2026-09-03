"""Trade-planning domain: contract selection, sleeve sizing, lifecycle planning."""

import calendar
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime

import polars as pl
from pydantic import BaseModel

from constants import (
    CASH_TYPE,
    KIND_CASH,
    KIND_EQUITY,
    KIND_OPTION,
    OPTION_MULTIPLIER,
    OPTION_TYPE,
    QTY_CONTRACT,
    QTY_SHARE,
)
from fees import TradingPlatform
from market_data import OptionQuoteSource, OptionSnapshot, parseOccSymbol

logger = logging.getLogger("Portfolio Rebalancer")

# Contract lifecycle / selection constants (R9-R13, R21)
MIN_EXPIRY_MONTHS = 21
ROLL_MONTHS_THRESHOLD = 12
TARGET_DELTA = 0.85
MAX_REL_SPREAD = 0.10
MIN_DAILY_VOLUME = 1.0


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
    quantityKind: str = QTY_SHARE  # "share" | "contract" | "cash"
    exposureChange: float = 0.0
    reason: str = ""

    def calcTransactionCost(self, platform: TradingPlatform)->None:
        if self.quantityKind == QTY_CONTRACT:
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
    if perContractExposure <= 0:
        raise ValueError(
            f"Per-contract exposure must be > 0, got {perContractExposure} "
            "(check spot price and contract delta)"
        )
    if spot <= 0:
        raise ValueError(f"Underlying spot price must be > 0, got {spot}")

    desiredContracts = int(math.floor(targetExposure / perContractExposure + 0.5))
    contractChange = desiredContracts - int(heldContracts)

    residualExposure = targetExposure - desiredContracts * perContractExposure
    desiredShareChange = residualExposure / spot
    shareChange = _bestEffortInt(desiredShareChange, heldShares)

    achievedExposure = desiredContracts * perContractExposure + shareChange * spot
    # Signed tracking error: positive = overshoot, negative = undershoot
    trackingErrorExposure = achievedExposure - targetExposure
    residualDirection = (
        "overshoot" if trackingErrorExposure > 1e-9
        else "undershoot" if trackingErrorExposure < -1e-9
        else "on-target"
    )

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
        for row in positionEnrichedDF.filter(pl.col("kind") == KIND_EQUITY).iter_rows(named=True)
    }

    sleeveTable = (
        positionEnrichedDF.filter(pl.col("kind") != KIND_CASH)
        .group_by("underlying")
        .agg(
            pl.col("exposure").sum().alias("currentExposure"),
            pl.when(pl.col("kind") == KIND_OPTION).then(pl.col("shares")).otherwise(0.0).sum().alias("heldContracts"),
            pl.when(pl.col("kind") == KIND_EQUITY).then(pl.col("shares")).otherwise(0.0).sum().alias("heldShares"),
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


def kindForInstrumentType(instrumentType: str) -> str:
    """Canonical instrumentType -> kind mapping (single source of truth)."""
    if instrumentType == CASH_TYPE:
        return KIND_CASH
    if instrumentType == OPTION_TYPE:
        return KIND_OPTION
    return KIND_EQUITY


def addMonths(d: date, months: int) -> date:
    """Shift a date by whole months, clamping the day to the target month's length."""
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def monthsBetween(later: date, earlier: date) -> int:
    """Whole elapsed months from `earlier` to `later` (day-aware: a partial
    month beyond the calendar difference does not count until the same
    day-of-month is reached)."""
    months = later.year * 12 + later.month - (earlier.year * 12 + earlier.month)
    if later.day < earlier.day:
        months -= 1
    return months


def _bestEffortInt(shareDelta: float, heldShares: float) -> int:
    """Truncate toward zero; never sell more shares than held."""
    shareChange = int(shareDelta)  # truncates toward zero
    return max(shareChange, -int(heldShares))


def perContractExposure(delta: float, spot: float) -> float:
    """Delta-adjusted notional exposure of one option contract."""
    return OPTION_MULTIPLIER * delta * spot


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
    optionSource: OptionQuoteSource | None, underlying: str, today: date
) -> tuple[OptionSnapshot | None, str]:
    """
    Rule-based contract selection (R12). Returns (snapshot | None, outcome) where
    outcome is "selected", "deferred" (chain-depth: candidates exist only below
    MIN_EXPIRY_MONTHS, R21), or "none" (no candidate passes the liquidity filter).
    A missing option source degrades to "none" (shares fallback).
    """
    if optionSource is None:
        logger.warning(
            f"[Contract Selection] {underlying}: no option data source available — "
            "falling back to shares"
        )
        return None, "none"

    # One chain fetch at the roll-window (12-month) floor; the 21-month
    # MIN_EXPIRY_MONTHS qualifying check is applied in memory.
    chainFloor = addMonths(today, ROLL_MONTHS_THRESHOLD)
    chain = optionSource.getChain(
        underlying, expirationDateGte=chainFloor.isoformat(), optionType="call"
    )
    liquid = [s for s in chain.values() if _passesLiquidityFilter(s)]
    candidates = [s for s in liquid if s.expiry >= addMonths(today, MIN_EXPIRY_MONTHS)]
    if candidates:
        snap = _pickCandidate(candidates)
        logger.info(
            f"[Contract Selection] {underlying}: selected {snap.symbol} "
            f"(expiry {snap.expiry}, delta {snap.delta:.4f}, mid {snap.mid:.2f})"
        )
        return snap, "selected"

    # Chain-depth check: candidates exist only below MIN_EXPIRY_MONTHS (R21)
    if liquid:
        logger.info(
            f"[Contract Selection] {underlying}: candidates exist only below "
            f"{MIN_EXPIRY_MONTHS} months — deferring"
        )
        return None, "deferred"

    logger.info(f"[Contract Selection] {underlying}: no qualifying candidate")
    return None, "none"


def planTrades(
    positionEnrichedDF: pl.DataFrame,
    sleeveTable: pl.DataFrame,
    designatedUnderlying: str | None,
    leverage: float,
    optionSource: OptionQuoteSource | None,
    tradeTimestamp: datetime,
    liquidateLeaps: bool = False,
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
        for row in positionEnrichedDF.filter(pl.col("kind") != KIND_CASH).iter_rows(named=True)
    }
    optionRowsByUnderlying: dict[str, list[dict]] = {}
    for row in positionEnrichedDF.filter(pl.col("kind") == KIND_OPTION).iter_rows(named=True):
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
                quantityKind=QTY_CONTRACT,
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
                quantityKind=QTY_SHARE,
                exposureChange=shareChange * spot,
                reason=reason,
            )
        )

    def emitResize(
        sleeve: dict, heldContracts: float,
        heldShares: float, spot: float, anchorRow: dict,
        contractReason: str, shareReason: str,
    ) -> None:
        # Resize sizing uses the ANCHOR row's own per-contract exposure, not the
        # sleeve average — mixed-delta sleeves would otherwise mis-size counts.
        anchorPerContractExposure = perContractExposure(anchorRow["deltaAdj"], spot)
        plan = sizeSleeve(
            sleeve["targetExposure"], anchorPerContractExposure, heldContracts, heldShares, spot
        )
        if plan.contractChange != 0:
            emitContract(
                sleeve["underlying"], anchorRow["instrumentId"], plan.contractChange,
                anchorRow["closingPrice"], anchorPerContractExposure, spot, contractReason,
            )
        if plan.shareChange != 0:
            emitShares(sleeve["underlying"], int(plan.shareChange), spot, shareReason)

    def emitExitAllContracts(
        heldOptionRows: list[dict], underlying: str, spot: float, reason: str,
    ) -> None:
        for row in heldOptionRows:
            emitContract(
                underlying, row["instrumentId"], -row["shares"], row["closingPrice"],
                perContractExposure(row["deltaAdj"], spot), spot, reason,
            )

    def emitSizedPlan(
        sleeve: dict, snap: OptionSnapshot, heldShares: float, spot: float,
        contractReason: str, shareReason: str,
    ) -> None:
        perContract = perContractExposure(snap.delta, spot)
        plan = sizeSleeve(sleeve["targetExposure"], perContract, 0, heldShares, spot)
        if plan.contractChange != 0:
            emitContract(
                sleeve["underlying"], snap.symbol, plan.contractChange, snap.mid,
                perContract, spot, contractReason,
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

        def emitBaseWeightShares(reason: str) -> None:
            """Shares fallback sized to the sleeve's BASE weight (not leveraged)."""
            shareChange = _bestEffortInt((baseWeightExposure - equityExposure) / spot, heldShares)
            if shareChange != 0:
                emitShares(underlying, shareChange, spot, reason)

        if not sleeve["isDesignated"]:
            if heldContracts > 0:
                # R17: stray / option-only sleeve — liquidate all held contracts
                emitExitAllContracts(
                    heldOptionRows, underlying, spot, "liquidation: non-designated sleeve",
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
        if liquidateLeaps:
            # Full-exit liquidation: sell every held contract and re-size the
            # sleeve to its base weight in shares, regardless of L.
            emitExitAllContracts(
                heldOptionRows, underlying, spot, "liquidation: --liquidate-leaps",
            )
            emitBaseWeightShares("liquidation: --liquidate-leaps")
            continue

        if heldContracts == 0:
            # R11: initiate
            snap, outcome = selectContract(optionSource, underlying, today)
            if outcome == "selected":
                emitSizedPlan(
                    sleeve, snap, heldShares, spot, "initiation", "initiation share residual",
                )
            else:
                # R13: shares fallback at base weight
                emitBaseWeightShares("shares fallback: no qualifying contract")
            continue

        earliestRow = min(
            heldOptionRows, key=lambda r: parseOccSymbol(r["instrumentId"]).expiry
        )
        monthsToExpiry = monthsBetween(parseOccSymbol(earliestRow["instrumentId"]).expiry, today)

        if monthsToExpiry > ROLL_MONTHS_THRESHOLD:
            # R9: keep — resize quantity only, no selection call
            emitResize(
                sleeve, heldContracts, heldShares, spot, earliestRow,
                "resize", "drift rebalance",
            )
            continue

        # R10: roll window
        snap, outcome = selectContract(optionSource, underlying, today)
        if outcome == "selected":
            emitExitAllContracts(heldOptionRows, underlying, spot, "roll: exit")
            emitSizedPlan(
                sleeve, snap, heldShares, spot, "roll: replacement", "roll: share residual",
            )
        elif outcome == "deferred":
            # R21: chain-depth deferral — keep the held contract, resize toward target
            logger.info(
                f"[Planner] {underlying}: roll deferred — chain depth "
                f"(no qualifying expiry >= {MIN_EXPIRY_MONTHS} months)"
            )
            emitResize(
                sleeve, heldContracts, heldShares, spot, earliestRow,
                "roll deferred — chain depth", "roll deferred — chain depth",
            )
        else:
            # R13: sell held, de-lever to base weight in shares
            emitExitAllContracts(
                heldOptionRows, underlying, spot,
                "roll: exit — no qualifying replacement",
            )
            emitBaseWeightShares("shares fallback: roll de-lever")

    return trades
