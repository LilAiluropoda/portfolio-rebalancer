import pytest

from black_scholes import (
    DIVIDEND_YIELD,
    RISK_FREE_RATE,
    callDelta,
    callPrice,
    impliedVolatility,
)


def test_constants():
    assert RISK_FREE_RATE == 0.04
    assert DIVIDEND_YIELD == 0.0


def test_deep_itm_call_delta():
    # spot 700, strike 450, 2y, IV 0.25, r=0.04, q=0 -> delta ~0.9+
    delta = callDelta(spot=700, strike=450, timeToExpiry=2.0, vol=0.25)
    assert delta > 0.9
    assert delta <= 1.0


def test_near_expiry_deep_itm_delta_approaches_one():
    delta = callDelta(spot=700, strike=450, timeToExpiry=1.0 / 365, vol=0.25)
    assert delta == pytest.approx(1.0, abs=1e-6)


def test_far_otm_delta_approaches_zero():
    delta = callDelta(spot=200, strike=700, timeToExpiry=2.0, vol=0.15)
    assert delta < 1e-4


def test_dividend_yield_lowers_delta():
    noDiv = callDelta(spot=700, strike=450, timeToExpiry=2.0, vol=0.25)
    withDiv = callDelta(
        spot=700, strike=450, timeToExpiry=2.0, vol=0.25, dividendYield=0.03
    )
    assert withDiv < noDiv


def test_reference_delta_value():
    # Reference BS table: S=100, K=100, T=1, vol=0.2, r=0.05, q=0 -> delta ~0.6368
    delta = callDelta(spot=100, strike=100, timeToExpiry=1.0, vol=0.2, rate=0.05)
    assert delta == pytest.approx(0.6368, abs=1e-3)


def test_iv_round_trip():
    spot, strike, t, vol, rate, q = 700.0, 450.0, 2.0, 0.25, 0.04, 0.0
    mid = callPrice(spot=spot, strike=strike, timeToExpiry=t, vol=vol, rate=rate, dividendYield=q)
    solved = impliedVolatility(mid, spot=spot, strike=strike, timeToExpiry=t, rate=rate, dividendYield=q)
    assert solved == pytest.approx(vol, abs=1e-6)


def test_iv_round_trip_at_the_money():
    spot, strike, t, vol, rate, q = 100.0, 100.0, 1.0, 0.2, 0.05, 0.0
    mid = callPrice(spot=spot, strike=strike, timeToExpiry=t, vol=vol, rate=rate, dividendYield=q)
    solved = impliedVolatility(mid, spot=spot, strike=strike, timeToExpiry=t, rate=rate, dividendYield=q)
    assert solved == pytest.approx(vol, abs=1e-6)


def test_zero_time_to_expiry_raises():
    with pytest.raises(ValueError):
        callDelta(spot=700, strike=450, timeToExpiry=0.0, vol=0.25)


def test_negative_time_to_expiry_raises():
    with pytest.raises(ValueError):
        callDelta(spot=700, strike=450, timeToExpiry=-1.0, vol=0.25)
    with pytest.raises(ValueError):
        impliedVolatility(100.0, spot=700, strike=450, timeToExpiry=-1.0)


def test_negative_volatility_raises():
    with pytest.raises(ValueError):
        callDelta(spot=700, strike=450, timeToExpiry=2.0, vol=-0.1)
    with pytest.raises(ValueError):
        callPrice(spot=700, strike=450, timeToExpiry=2.0, vol=-0.1)


def test_non_positive_spot_or_strike_raises():
    with pytest.raises(ValueError):
        callDelta(spot=0.0, strike=450, timeToExpiry=2.0, vol=0.25)
    with pytest.raises(ValueError):
        callDelta(spot=700, strike=-450, timeToExpiry=2.0, vol=0.25)
