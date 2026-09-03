from datetime import datetime

import pytest

from fees import FutuBullUS
from planning import Trade

platform = FutuBullUS()


def makeOptionTrade(contracts: float, premium: float) -> Trade:
    return Trade(
        tradeId="1",
        instrumentId="VOO270115C00450000",
        instrumentType="LEAPS Call",
        price=premium,
        sharesChange=contracts,
        marketValueChange=contracts * premium * 100,
        timestamp=datetime(2026, 9, 1),
        quantityKind="contract",
    )


def testBuyTwoContractsAtPremium2422():
    # commission max(0.65*2, 1.99) = 1.99
    # platform 0.60 + ORF 0.026 + OCC 0.04 + settlement 0.36 + CAT 0.0006
    cost = platform.calcOptionsTransactionCost("LEAPS Call", 2, 2 * 24.22 * 100)
    expected = 1.99 + 0.60 + 0.026 + 0.04 + 0.36 + 0.0006
    assert cost == expected


def testBuyTwoContractsViaTradeDispatch():
    trade = makeOptionTrade(2, 24.22)
    trade.calcTransactionCost(platform=platform)
    assert trade.transactionCost == pytest.approx(3.0166)


def testSellFiveContractsAddsSecAndFinraFees():
    # notional = 5 * 24.22 * 100 = 12110
    # SEC = max(12110 * 0.0000206, 0.01) = 0.249466
    # FINRA = max(5 * 0.00329, 0.01) = 0.01645
    cost = platform.calcOptionsTransactionCost("LEAPS Call", -5, -(5 * 24.22 * 100))
    expected = (
        5 * 0.65  # commission 3.25
        + 5 * 0.30
        + 5 * 0.013
        + 5 * 0.02
        + 5 * 0.18
        + 5 * 0.0003
        + 12110 * 0.0000206
        + 5 * 0.00329
    )
    assert cost == expected


def testSellOneContractHitsFinraMinimum():
    # FINRA raw = 0.00329 < 0.01 -> floor; SEC raw on 2422 notional = 0.0499 > 0.01
    cost = platform.calcOptionsTransactionCost("LEAPS Call", -1, -(1 * 24.22 * 100))
    expected = (
        1.99 + 0.30 + 0.013 + 0.02 + 0.18 + 0.0003
        + 2422 * 0.0000206
        + 0.01
    )
    assert cost == expected


def testOccFeeCappedAt55():
    # 3000 contracts: OCC raw = 60 -> capped at 55
    cost = platform.calcOptionsTransactionCost("LEAPS Call", 3000, 3000 * 24.22 * 100)
    expected = (
        3000 * 0.65
        + 3000 * 0.30
        + 3000 * 0.013
        + 55  # capped
        + 3000 * 0.18
        + 3000 * 0.0003
    )
    assert cost == expected


def testSingleContractCommissionFloor():
    cost = platform.calcOptionsTransactionCost("LEAPS Call", 1, 1 * 24.22 * 100)
    expected = 1.99 + 0.30 + 0.013 + 0.02 + 0.18 + 0.0003
    assert cost == expected
