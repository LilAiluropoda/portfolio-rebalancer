"""Direct unit tests for the calendar-month helpers behind R9/R10 lifecycle."""

from datetime import date

from main import ROLL_MONTHS_THRESHOLD, addMonths, monthsBetween


# --- monthsBetween: whole calendar months, floor semantics ---


def test_months_between_exact_twelve():
    assert monthsBetween(date(2027, 9, 2), date(2026, 9, 2)) == 12


def test_months_between_eleven_months_plus_days():
    # 2026-09-02 -> 2027-08-31 is 11 whole months + 29 days -> still 11
    assert monthsBetween(date(2027, 8, 31), date(2026, 9, 2)) == 11


def test_months_between_same_month_is_zero():
    assert monthsBetween(date(2026, 9, 30), date(2026, 9, 1)) == 0


def test_months_between_negative_direction():
    assert monthsBetween(date(2026, 9, 2), date(2027, 9, 2)) == -12


# --- Lifecycle boundary: exactly 12 months to expiry -> ROLL, not KEEP ---
# planTrades keeps only when monthsToExpiry > ROLL_MONTHS_THRESHOLD, so an
# expiry at addMonths(today, 12) must land in the roll window.


def test_lifecycle_boundary_exactly_twelve_months_is_roll():
    today = date(2026, 9, 2)
    expiry = addMonths(today, 12)
    assert expiry == date(2027, 9, 2)

    monthsToExpiry = monthsBetween(expiry, today)
    assert monthsToExpiry == 12
    assert monthsToExpiry <= ROLL_MONTHS_THRESHOLD  # roll branch taken...
    assert not monthsToExpiry > ROLL_MONTHS_THRESHOLD  # ...keep branch requires strictly greater


def test_lifecycle_boundary_thirteen_months_is_keep():
    today = date(2026, 9, 2)
    expiry = addMonths(today, 13)
    assert monthsBetween(expiry, today) == 13
    assert monthsBetween(expiry, today) > ROLL_MONTHS_THRESHOLD  # keep branch


# --- addMonths: day clamping and year rollover ---


def test_add_months_clamps_jan_31_to_feb_28_in_non_leap_2026():
    assert addMonths(date(2026, 1, 31), 1) == date(2026, 2, 28)


def test_add_months_clamps_leap_february():
    assert addMonths(date(2028, 1, 31), 1) == date(2028, 2, 29)  # 2028 is leap
    assert addMonths(date(2026, 12, 31), 2) == date(2027, 2, 28)  # via Jan 31 clamp


def test_add_months_jan_31_plus_21_months():
    # Jan 2026 + 21 months = Oct 2027; October has 31 days, so no clamping
    assert addMonths(date(2026, 1, 31), 21) == date(2027, 10, 31)


def test_add_months_clamps_when_target_month_is_shorter():
    # Jan 31 + 22 months = Nov 2027 (30 days) -> clamps to Nov 30
    assert addMonths(date(2026, 1, 31), 22) == date(2027, 11, 30)


def test_add_months_year_rollover():
    assert addMonths(date(2026, 11, 15), 3) == date(2027, 2, 15)
    assert addMonths(date(2026, 12, 15), 1) == date(2027, 1, 15)
    assert addMonths(date(2026, 1, 15), 24) == date(2028, 1, 15)
    assert addMonths(date(2026, 3, 31), -3) == date(2025, 12, 31)
