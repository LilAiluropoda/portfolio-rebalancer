"""Trading platforms and their fee schedules."""

from abc import ABC, abstractmethod

from constants import CASH_TYPE


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
        if instrumentType == CASH_TYPE:
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
