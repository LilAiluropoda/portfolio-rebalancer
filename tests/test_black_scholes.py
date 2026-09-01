import pytest

from black_scholes import (
    DIVIDEND_YIELD,
    RISK_FREE_RATE,
    call_delta,
    call_price,
    implied_volatility,
)


def test_constants():
    assert RISK_FREE_RATE == 0.04
    assert DIVIDEND_YIELD == 0.0


def test_deep_itm_call_delta():
    # spot 700, strike 450, 2y, IV 0.25, r=0.04, q=0 -> delta ~0.9+
    delta = call_delta(spot=700, strike=450, time_to_expiry=2.0, vol=0.25)
    assert delta > 0.9
    assert delta <= 1.0


def test_near_expiry_deep_itm_delta_approaches_one():
    delta = call_delta(spot=700, strike=450, time_to_expiry=1.0 / 365, vol=0.25)
    assert delta == pytest.approx(1.0, abs=1e-6)


def test_far_otm_delta_approaches_zero():
    delta = call_delta(spot=200, strike=700, time_to_expiry=2.0, vol=0.15)
    assert delta < 1e-4


def test_dividend_yield_lowers_delta():
    no_div = call_delta(spot=700, strike=450, time_to_expiry=2.0, vol=0.25)
    with_div = call_delta(
        spot=700, strike=450, time_to_expiry=2.0, vol=0.25, dividend_yield=0.03
    )
    assert with_div < no_div


def test_reference_delta_value():
    # Reference BS table: S=100, K=100, T=1, vol=0.2, r=0.05, q=0 -> delta ~0.6368
    delta = call_delta(spot=100, strike=100, time_to_expiry=1.0, vol=0.2, rate=0.05)
    assert delta == pytest.approx(0.6368, abs=1e-3)


def test_iv_round_trip():
    spot, strike, t, vol, rate, q = 700.0, 450.0, 2.0, 0.25, 0.04, 0.0
    mid = call_price(spot=spot, strike=strike, time_to_expiry=t, vol=vol, rate=rate, dividend_yield=q)
    solved = implied_volatility(mid, spot=spot, strike=strike, time_to_expiry=t, rate=rate, dividend_yield=q)
    assert solved == pytest.approx(vol, abs=1e-6)


def test_iv_round_trip_at_the_money():
    spot, strike, t, vol, rate, q = 100.0, 100.0, 1.0, 0.2, 0.05, 0.0
    mid = call_price(spot=spot, strike=strike, time_to_expiry=t, vol=vol, rate=rate, dividend_yield=q)
    solved = implied_volatility(mid, spot=spot, strike=strike, time_to_expiry=t, rate=rate, dividend_yield=q)
    assert solved == pytest.approx(vol, abs=1e-6)


def test_zero_time_to_expiry_raises():
    with pytest.raises(ValueError):
        call_delta(spot=700, strike=450, time_to_expiry=0.0, vol=0.25)


def test_negative_time_to_expiry_raises():
    with pytest.raises(ValueError):
        call_delta(spot=700, strike=450, time_to_expiry=-1.0, vol=0.25)
    with pytest.raises(ValueError):
        implied_volatility(100.0, spot=700, strike=450, time_to_expiry=-1.0)


def test_negative_volatility_raises():
    with pytest.raises(ValueError):
        call_delta(spot=700, strike=450, time_to_expiry=2.0, vol=-0.1)
    with pytest.raises(ValueError):
        call_price(spot=700, strike=450, time_to_expiry=2.0, vol=-0.1)


def test_non_positive_spot_or_strike_raises():
    with pytest.raises(ValueError):
        call_delta(spot=0.0, strike=450, time_to_expiry=2.0, vol=0.25)
    with pytest.raises(ValueError):
        call_delta(spot=700, strike=-450, time_to_expiry=2.0, vol=0.25)
