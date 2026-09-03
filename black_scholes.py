"""Black-Scholes pricing, call delta, and implied-volatility solve.

Pure math, no I/O. Used as the fallback when Alpaca snapshot greeks are null.
"""

import math

RISK_FREE_RATE = 0.04
DIVIDEND_YIELD = 0.0

_MAX_BISECTION_ITERATIONS = 100
_PRICE_TOLERANCE = 1e-10


def _normCdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _validateInputs(spot: float, strike: float, timeToExpiry: float, vol: float | None = None) -> None:
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if timeToExpiry <= 0:
        raise ValueError(f"timeToExpiry must be positive, got {timeToExpiry}")
    if vol is not None and vol < 0:
        raise ValueError(f"vol must be non-negative, got {vol}")


def _d1(spot: float, strike: float, timeToExpiry: float, vol: float, rate: float, dividendYield: float) -> float:
    return (
        math.log(spot / strike)
        + (rate - dividendYield + 0.5 * vol * vol) * timeToExpiry
    ) / (vol * math.sqrt(timeToExpiry))


def callPrice(
    spot: float,
    strike: float,
    timeToExpiry: float,
    vol: float,
    rate: float = RISK_FREE_RATE,
    dividendYield: float = DIVIDEND_YIELD,
) -> float:
    """Black-Scholes call price with continuous dividend yield."""
    _validateInputs(spot, strike, timeToExpiry, vol)
    d1 = _d1(spot, strike, timeToExpiry, vol, rate, dividendYield)
    d2 = d1 - vol * math.sqrt(timeToExpiry)
    return spot * math.exp(-dividendYield * timeToExpiry) * _normCdf(d1) - strike * math.exp(
        -rate * timeToExpiry
    ) * _normCdf(d2)


def callDelta(
    spot: float,
    strike: float,
    timeToExpiry: float,
    vol: float,
    rate: float = RISK_FREE_RATE,
    dividendYield: float = DIVIDEND_YIELD,
) -> float:
    """Call delta N(d1) with continuous dividend yield support."""
    _validateInputs(spot, strike, timeToExpiry, vol)
    return math.exp(-dividendYield * timeToExpiry) * _normCdf(
        _d1(spot, strike, timeToExpiry, vol, rate, dividendYield)
    )


def impliedVolatility(
    mid: float,
    spot: float,
    strike: float,
    timeToExpiry: float,
    rate: float = RISK_FREE_RATE,
    dividendYield: float = DIVIDEND_YIELD,
) -> float:
    """Solve implied volatility from a mid price by bisection.

    Converges well within 100 iterations; raises ValueError when the mid is
    outside the achievable price range for these contract parameters.
    """
    _validateInputs(spot, strike, timeToExpiry)
    if mid <= 0:
        raise ValueError(f"mid must be positive, got {mid}")

    low, high = 1e-8, 10.0
    priceLow = callPrice(spot, strike, timeToExpiry, low, rate, dividendYield)
    priceHigh = callPrice(spot, strike, timeToExpiry, high, rate, dividendYield)
    if not (priceLow - _PRICE_TOLERANCE <= mid <= priceHigh + _PRICE_TOLERANCE):
        raise ValueError(
            f"mid {mid} outside achievable price range [{priceLow:.6f}, {priceHigh:.6f}]"
        )

    for _ in range(_MAX_BISECTION_ITERATIONS):
        guess = 0.5 * (low + high)
        price = callPrice(spot, strike, timeToExpiry, guess, rate, dividendYield)
        if abs(price - mid) < _PRICE_TOLERANCE:
            return guess
        if price < mid:
            low = guess
        else:
            high = guess
    return 0.5 * (low + high)
