"""AlpacaOptionData client tests — fixture JSON, zero network calls."""

from datetime import datetime, timedelta, timezone

import pytest

import options_data
import requests
from black_scholes import call_delta, implied_volatility
from options_data import (
    AlpacaApiError,
    AlpacaConfigError,
    AlpacaOptionData,
    OccParseError,
    OptionDataSourceFactory,
    parseOccSymbol,
)

KEY_ID = "test-key-id"
SECRET = "test-secret"


def makeSnapshotPayload(
    bp=23.81,
    ap=24.63,
    ts=None,
    delta=0.9399,
    iv=0.2132,
    underlyingPrice=None,
    symbol="VOO260904C00680000",
):
    return {
        symbol: {
            "latestQuote": {"bp": bp, "ap": ap, "t": ts or freshTimestamp()},
            "greeks": {"delta": delta} if delta is not None else None,
            "impliedVolatility": iv,
            "dailyBar": {"volume": 123},
            **({"underlyingAsset": {"price": underlyingPrice}} if underlyingPrice else {}),
        }
    }


def freshTimestamp():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", KEY_ID)
    monkeypatch.setenv("APCA_API_SECRET_KEY", SECRET)
    return AlpacaOptionData()


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, reason=""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.reason = reason

    def json(self):
        return self._json


def test_snapshotHappyPath(client, monkeypatch):
    payload = makeSnapshotPayload()
    called = {}

    def fakeGet(url, headers=None, params=None, timeout=None):
        called["url"], called["headers"], called["params"] = url, headers, params
        return FakeResponse(json_data={"snapshots": payload})

    monkeypatch.setattr(options_data.requests, "get", fakeGet)
    result = client.getSnapshots(["VOO260904C00680000"])

    assert called["url"].startswith("https://data.alpaca.markets/v1beta1/options/snapshots")
    assert called["params"] == {"symbols": "VOO260904C00680000", "feed": "indicative"}
    assert called["headers"]["APCA-API-KEY-ID"] == KEY_ID
    assert called["headers"]["APCA-API-SECRET-KEY"] == SECRET

    snap = result["VOO260904C00680000"]
    assert snap.mid == pytest.approx((23.81 + 24.63) / 2)
    assert snap.delta == pytest.approx(0.9399)
    assert snap.iv == pytest.approx(0.2132)
    assert snap.underlying == "VOO"
    assert snap.strike == 680.0
    assert snap.right == "C"
    assert snap.volume == 123


def test_chainPagination(client, monkeypatch):
    page1 = dict(makeSnapshotPayload())
    page1["next_page_token"] = "page2"
    page2 = {
        "VOO280121C00690000": makeSnapshotPayload(
            symbol="VOO280121C00690000"
        )["VOO280121C00690000"]
    }
    requestsMade = []

    def fakeGet(url, headers=None, params=None, timeout=None):
        requestsMade.append(params)
        if params.get("page_token") == "page2":
            return FakeResponse(json_data=page2)
        return FakeResponse(json_data=page1)

    monkeypatch.setattr(options_data.requests, "get", fakeGet)
    result = client.getChain("VOO")

    assert set(result) == {"VOO260904C00680000", "VOO280121C00690000"}
    assert len(requestsMade) == 2
    assert requestsMade[1]["page_token"] == "page2"
    assert all(p["feed"] == "indicative" and p["limit"] == 1000 for p in requestsMade)


def test_chainFiltersForwarded(client, monkeypatch):
    seen = {}

    def fakeGet(url, headers=None, params=None, timeout=None):
        seen["url"], seen["params"] = url, params
        return FakeResponse(json_data={})

    monkeypatch.setattr(options_data.requests, "get", fakeGet)
    client.getChain(
        "VOO",
        expirationDateGte="2027-06-01",
        strikePriceGte=400,
        strikePriceLte=800,
        optionType="call",
    )
    assert seen["url"].endswith("/v1beta1/options/snapshots/VOO")
    assert seen["params"]["expiration_date_gte"] == "2027-06-01"
    assert seen["params"]["strike_price_gte"] == 400
    assert seen["params"]["strike_price_lte"] == 800
    assert seen["params"]["type"] == "call"


def test_nullGreeksBsFallback(client, monkeypatch):
    payload = makeSnapshotPayload(
        delta=None,
        iv=None,
        bp=285.10,
        ap=286.90,
        underlyingPrice=700.0,
        symbol="VOO280121C00450000",
    )
    monkeypatch.setattr(
        options_data.requests, "get", lambda *a, **k: FakeResponse(json_data={"snapshots": payload})
    )
    result = client.getSnapshots(["VOO280121C00450000"])
    snap = result["VOO280121C00450000"]

    mid = (285.10 + 286.90) / 2
    days = (snap.expiry - datetime.now(timezone.utc).date()).days
    t = days / 365.0
    expectedIv = implied_volatility(mid, 700.0, 450.0, t)
    expectedDelta = call_delta(700.0, 450.0, t, expectedIv)
    assert snap.iv == pytest.approx(expectedIv, rel=1e-6)
    assert snap.delta == pytest.approx(expectedDelta, abs=1e-4)
    assert snap.delta > 0.85  # deep ITM sanity


def test_bidZeroRejected(client, monkeypatch):
    payload = makeSnapshotPayload(bp=0, ap=1.0)
    monkeypatch.setattr(
        options_data.requests, "get", lambda *a, **k: FakeResponse(json_data={"snapshots": payload})
    )
    assert client.getSnapshots(["VOO260904C00680000"]) == {}


def test_staleQuoteRejected(client, monkeypatch):
    stale = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat().replace("+00:00", "Z")
    payload = makeSnapshotPayload(ts=stale)
    monkeypatch.setattr(
        options_data.requests, "get", lambda *a, **k: FakeResponse(json_data={"snapshots": payload})
    )
    assert client.getSnapshots(["VOO260904C00680000"]) == {}


@pytest.mark.parametrize("status", [401, 429, 500])
def test_httpErrorsSanitized(client, monkeypatch, status):
    monkeypatch.setattr(
        options_data.requests,
        "get",
        lambda *a, **k: FakeResponse(
            status_code=status, json_data={"message": "unauthorized"}, reason="Unauthorized"
        ),
    )
    with pytest.raises(AlpacaApiError) as excInfo:
        client.getSnapshots(["VOO260904C00680000"])
    message = str(excInfo.value)
    assert str(status) in message
    assert KEY_ID not in message
    assert SECRET not in message
    assert "APCA" not in message  # no header names leak either


def test_missingEnvKeyNamesVariable(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(AlpacaConfigError, match="APCA_API_KEY_ID"):
        AlpacaOptionData()


def test_factoryReturnsAlpacaSource(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", KEY_ID)
    monkeypatch.setenv("APCA_API_SECRET_KEY", SECRET)
    OptionDataSourceFactory.instances = {}
    source = OptionDataSourceFactory.getDataSource("alpaca")
    assert isinstance(source, AlpacaOptionData)
    assert OptionDataSourceFactory.getDataSource("alpaca") is source
    with pytest.raises(Exception, match="not found"):
        OptionDataSourceFactory.getDataSource("nope")


def test_tooManySymbolsRejected(client):
    with pytest.raises(ValueError, match="100"):
        client.getSnapshots([f"VOO260904C{i:08d}" for i in range(101)])


def test_transportErrorSanitized(client, monkeypatch):
    def fakeGet(*a, **k):
        raise requests.ConnectionError("DNS failure: data.alpaca.markets unreachable")

    monkeypatch.setattr(options_data.requests, "get", fakeGet)
    with pytest.raises(AlpacaApiError) as excInfo:
        client.getSnapshots(["VOO260904C00680000"])
    assert excInfo.value.statusCode == 0
    message = str(excInfo.value)
    assert "ConnectionError" in message
    assert KEY_ID not in message
    assert SECRET not in message
    assert "APCA" not in message


def test_missingQuoteTimestampSkipped(client, monkeypatch):
    payload = makeSnapshotPayload()
    payload["VOO260904C00680000"]["latestQuote"].pop("t")
    monkeypatch.setattr(
        options_data.requests, "get", lambda *a, **k: FakeResponse(json_data={"snapshots": payload})
    )
    assert client.getSnapshots(["VOO260904C00680000"]) == {}


def test_malformedQuoteTimestampSkipped(client, monkeypatch):
    payload = makeSnapshotPayload(ts="not-a-timestamp")
    monkeypatch.setattr(
        options_data.requests, "get", lambda *a, **k: FakeResponse(json_data={"snapshots": payload})
    )
    assert client.getSnapshots(["VOO260904C00680000"]) == {}


@pytest.mark.parametrize(
    "root",
    ["voo", "GOOG/evil", "AAPL?x=1", "SPXW1234X", ""],
)
def test_invalidUnderlyingRootRejected(root):
    with pytest.raises(OccParseError):
        parseOccSymbol(f"{root}260904C00680000")


def test_sixCharRootAccepted():
    contract = parseOccSymbol("ABCDEF260904C00680000")
    assert contract.underlying == "ABCDEF"


# --- _fallbackDelta branches: expired contract and missing underlying price ---


def test_expiredContractNullGreeksIntrinsicDeltaITM(client, monkeypatch):
    # Expiry in the past, null greeks, strike < spot -> intrinsic delta 1.0
    payload = makeSnapshotPayload(
        delta=None, iv=None, bp=250.0, ap=252.0, underlyingPrice=700.0,
        symbol="VOO250904C00450000",  # expired 2025-09-04
    )
    monkeypatch.setattr(
        options_data.requests, "get", lambda *a, **k: FakeResponse(json_data={"snapshots": payload})
    )
    snap = client.getSnapshots(["VOO250904C00450000"])["VOO250904C00450000"]
    assert snap.delta == 1.0
    assert snap.iv == 0.0  # nothing to solve for


def test_expiredContractNullGreeksIntrinsicDeltaOTM(client, monkeypatch):
    # Same, but strike > spot -> intrinsic delta 0.0; no crash either way
    payload = makeSnapshotPayload(
        delta=None, iv=None, bp=0.05, ap=0.15, underlyingPrice=700.0,
        symbol="VOO250904C00900000",  # expired, strike 900
    )
    monkeypatch.setattr(
        options_data.requests, "get", lambda *a, **k: FakeResponse(json_data={"snapshots": payload})
    )
    snap = client.getSnapshots(["VOO250904C00900000"])["VOO250904C00900000"]
    assert snap.delta == 0.0


def test_nullGreeksNoUnderlyingPriceRaisesNamingSymbol(client, monkeypatch):
    payload = makeSnapshotPayload(
        delta=None, iv=None, underlyingPrice=None, symbol="VOO280121C00450000"
    )
    monkeypatch.setattr(
        options_data.requests, "get", lambda *a, **k: FakeResponse(json_data={"snapshots": payload})
    )
    with pytest.raises(AlpacaApiError, match="VOO280121C00450000"):
        client.getSnapshots(["VOO280121C00450000"])
