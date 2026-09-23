"""Cliente OANDA con una sesión HTTP falsa (sin red)."""
import json

import pytest
import requests

from broker.oanda import OandaBroker, OandaClient, OandaError, parsear_tiempo

TOKEN = "tok-123456789"


class RespuestaFalsa:
    def __init__(self, status, cuerpo):
        self.status_code = status
        self._cuerpo = cuerpo
        self.content = json.dumps(cuerpo).encode()
        self.text = json.dumps(cuerpo)

    def json(self):
        return self._cuerpo


class SesionFalsa:
    def __init__(self, respuestas):
        self.respuestas = list(respuestas)
        self.llamadas = []
        self.headers = {}

    def request(self, metodo, url, params=None, json=None, timeout=None):
        self.llamadas.append((metodo, url, params, json))
        r = self.respuestas.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


VELAS = {"candles": [
    {"complete": True, "time": "2026-09-21T21:00:00.000000000Z", "volume": 10,
     "mid": {"o": "0.6500", "h": "0.6600", "l": "0.6400", "c": "0.6550"},
     "bid": {"o": "0.6499", "h": "0.6599", "l": "0.6399", "c": "0.6549"},
     "ask": {"o": "0.6501", "h": "0.6601", "l": "0.6401", "c": "0.6551"}},
    {"complete": False, "time": "2026-09-22T21:00:00.000000000Z", "volume": 3,
     "mid": {"o": "0.6550", "h": "0.6560", "l": "0.6540", "c": "0.6555"}},
]}


def _cliente(respuestas, cuenta="101-001-1"):
    s = SesionFalsa(respuestas)
    c = OandaClient(TOKEN, cuenta, session=s, reintentos=2)
    return c, s


def test_parsear_tiempo_nanosegundos():
    t = parsear_tiempo("2026-09-21T21:00:00.123456789Z")
    assert t.isoformat() == "2026-09-21T21:00:00.123456+00:00"


def test_velas_parametros_alineacion_17_ny():
    c, s = _cliente([RespuestaFalsa(200, VELAS)])
    velas = c.velas("XCU_USD", n=5)
    metodo, url, params, _ = s.llamadas[0]
    assert url == "https://api-fxpractice.oanda.com/v3/instruments/XCU_USD/candles"
    assert params["dailyAlignment"] == 17
    assert params["alignmentTimezone"] == "America/New_York"
    assert params["price"] == "MBA" and params["granularity"] == "D"
    assert len(velas) == 2 and velas[0].completa and not velas[1].completa
    assert velas[0].c == 0.655 and velas[0].bid_c == 0.6549 and velas[0].ask_c == 0.6551
    assert (velas[0].cierre - velas[0].tiempo).days == 1
    assert s.headers["Authorization"] == f"Bearer {TOKEN}"


def test_get_reintenta_ante_error_de_red(monkeypatch):
    monkeypatch.setattr("broker.oanda.time.sleep", lambda s: None)
    c, s = _cliente([requests.ConnectionError("x"), RespuestaFalsa(200, VELAS)])
    assert len(c.velas("AUD_USD")) == 2
    assert len(s.llamadas) == 2


def test_orden_no_se_reintenta(monkeypatch):
    monkeypatch.setattr("broker.oanda.time.sleep", lambda s: None)
    c, s = _cliente([requests.ConnectionError("x"), RespuestaFalsa(201, {})])
    with pytest.raises(OandaError):
        c.orden_mercado("AUD_USD", 1000)
    assert len(s.llamadas) == 1  # nunca un segundo POST


def test_error_http_no_filtra_token():
    c, _ = _cliente([RespuestaFalsa(401, {"errorMessage": "Insufficient authorization"})])
    with pytest.raises(OandaError) as e:
        c.resumen_cuenta()
    assert TOKEN not in str(e.value)


def test_broker_abrir_y_cerrar_parsea_fills():
    fill_tx = {"id": "55", "time": "2026-09-22T21:20:00.000000000Z", "units": "15000",
               "price": "0.65010", "pl": "0.0", "financing": "0.0", "halfSpreadCost": "0.75",
               "fullPrice": {"bids": [{"price": "0.65000"}], "asks": [{"price": "0.65010"}]}}
    cierre_tx = {"id": "60", "time": "2026-09-23T21:20:00.000000000Z", "units": "-15000",
                 "price": "0.66000", "pl": "148.5", "financing": "-0.42",
                 "halfSpreadCost": "0.75",
                 "fullPrice": {"bids": [{"price": "0.66000"}], "asks": [{"price": "0.66010"}]}}
    pos = {"position": {"long": {"units": "15000", "averagePrice": "0.65010"},
                        "short": {"units": "0"}}}
    c, s = _cliente([
        RespuestaFalsa(201, {"orderFillTransaction": fill_tx}),
        RespuestaFalsa(200, pos),
        RespuestaFalsa(200, {"longOrderFillTransaction": cierre_tx}),
    ])
    b = OandaBroker(c)
    f = b.abrir_mercado("AUD_USD", 15000, "test")
    assert f.precio == 0.6501 and f.ask == 0.6501 and f.costo_spread == 0.75
    assert s.llamadas[0][3]["order"]["units"] == "15000"
    g = b.cerrar_posicion("AUD_USD")
    assert g.pnl == 148.5 and g.financiamiento == -0.42 and g.broker_id == "60"
    assert s.llamadas[2][3] == {"longUnits": "ALL"}


def test_orden_cancelada_lanza_error():
    c, _ = _cliente([RespuestaFalsa(201, {"orderCancelTransaction": {"reason": "MARKET_HALTED"}})])
    with pytest.raises(OandaError, match="MARKET_HALTED"):
        OandaBroker(c).abrir_mercado("AUD_USD", 1000)
