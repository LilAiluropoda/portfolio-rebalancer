"""Black-Scholes pricing, call delta, and implied-volatility solve.

Pure math, no I/O. Used as the fallback when Alpaca snapshot greeks are null.
"""

import math

RISK_FREE_RATE = 0.04
DIVIDEND_YIELD = 0.0

_MAX_BISECTION_ITERATIONS = 100
_PRICE_TOLERANCE = 1e-10


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _validate(spot: float, strike: float, time_to_expiry: float, vol: float | None = None) -> None:
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if time_to_expiry <= 0:
        raise ValueError(f"time_to_expiry must be positive, got {time_to_expiry}")
    if vol is not None and vol < 0:
        raise ValueError(f"vol must be non-negative, got {vol}")


def _d1(spot: float, strike: float, time_to_expiry: float, vol: float, rate: float, dividend_yield: float) -> float:
    return (
        math.log(spot / strike)
        + (rate - dividend_yield + 0.5 * vol * vol) * time_to_expiry
    ) / (vol * math.sqrt(time_to_expiry))


def call_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
    rate: float = RISK_FREE_RATE,
    dividend_yield: float = DIVIDEND_YIELD,
) -> float:
    """Black-Scholes call price with continuous dividend yield."""
    _validate(spot, strike, time_to_expiry, vol)
    d1 = _d1(spot, strike, time_to_expiry, vol, rate, dividend_yield)
    d2 = d1 - vol * math.sqrt(time_to_expiry)
    return spot * math.exp(-dividend_yield * time_to_expiry) * _norm_cdf(d1) - strike * math.exp(
        -rate * time_to_expiry
    ) * _norm_cdf(d2)


def call_delta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    vol: float,
    rate: float = RISK_FREE_RATE,
    dividend_yield: float = DIVIDEND_YIELD,
) -> float:
    """Call delta N(d1) with continuous dividend yield support."""
    _validate(spot, strike, time_to_expiry, vol)
    return math.exp(-dividend_yield * time_to_expiry) * _norm_cdf(
        _d1(spot, strike, time_to_expiry, vol, rate, dividend_yield)
    )


def implied_volatility(
    mid: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    rate: float = RISK_FREE_RATE,
    dividend_yield: float = DIVIDEND_YIELD,
) -> float:
    """Solve implied volatility from a mid price by bisection.

    Converges well within 100 iterations; raises ValueError when the mid is
    outside the achievable price range for these contract parameters.
    """
    _validate(spot, strike, time_to_expiry)
    if mid <= 0:
        raise ValueError(f"mid must be positive, got {mid}")

    low, high = 1e-8, 10.0
    price_low = call_price(spot, strike, time_to_expiry, low, rate, dividend_yield)
    price_high = call_price(spot, strike, time_to_expiry, high, rate, dividend_yield)
    if not (price_low - _PRICE_TOLERANCE <= mid <= price_high + _PRICE_TOLERANCE):
        raise ValueError(
            f"mid {mid} outside achievable price range [{price_low:.6f}, {price_high:.6f}]"
        )

    for _ in range(_MAX_BISECTION_ITERATIONS):
        guess = 0.5 * (low + high)
        price = call_price(spot, strike, time_to_expiry, guess, rate, dividend_yield)
        if abs(price - mid) < _PRICE_TOLERANCE:
            return guess
        if price < mid:
            low = guess
        else:
            high = guess
    return 0.5 * (low + high)
