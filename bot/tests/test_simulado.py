from datetime import datetime, timezone

import pandas as pd
import pytest

from broker import crear_broker
from broker.simulado import SimuladoBroker
from broker.yahoo import FUENTE, FuenteYahoo, fecha_ultimo_cierre_ny, mercado_fx_abierto
from estrategias.cobre_aud import CobreAud
from ledger import ErrorLedger, Ledger
from motor import ejecutar_ciclo


def test_broker_simulado_balance_y_equity(cfg, ledger, fuente):
    b = SimuladoBroker(fuente, ledger, cfg.capital_inicial)
    assert b.equity() == b.balance() == 10_000
    f = b.abrir_mercado("AUD_USD", 10_000)
    ledger.abrir_trade("cobre_aud", "simulado", f, None, 10_000)
    # Recién abierto: equity = capital - spread completo (se valoriza al BID)
    assert b.equity() == pytest.approx(10_000 - 10_000 * (fuente.ask - fuente.bid))
    fuente.bid, fuente.ask = 0.66000, 0.66010
    g = b.cerrar_posicion("AUD_USD")
    ledger.cerrar_trade(g, None)
    assert b.posicion_abierta("AUD_USD") is None
    assert b.balance() == pytest.approx(10_000 + 10_000 * (0.66000 - 0.65010))


def test_crear_broker_sin_token_usa_yahoo(cfg, ledger):
    b = crear_broker(cfg, ledger)
    assert isinstance(b, SimuladoBroker) and isinstance(b.fuente, FuenteYahoo)
    assert cfg.fuente_datos == "yahoo"


def test_ledger_no_mezcla_modos(tmp_path):
    ruta = tmp_path / "x.db"
    Ledger(ruta, modo="simulado", capital_inicial=1).cerrar()
    with pytest.raises(ErrorLedger):
        Ledger(ruta, modo="oanda_practice")


def test_no_abre_dos_trades_en_el_mismo_instrumento(cfg, ledger, fuente):
    b = SimuladoBroker(fuente, ledger, cfg.capital_inicial)
    f = b.abrir_mercado("AUD_USD", 1000)
    ledger.abrir_trade("cobre_aud", "simulado", f, None, 10_000)
    with pytest.raises(ErrorLedger):
        ledger.abrir_trade("cobre_aud", "simulado", f, None, 10_000)


# ---------------------------------------------------------------- fallback Yahoo
def _df_diario(fechas, closes, tz):
    idx = pd.DatetimeIndex(pd.to_datetime(fechas)).tz_localize(tz)
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes,
                         "Volume": [0] * len(closes)}, index=idx)


def _descargador(ahora):
    def descargar(ticker, period, interval):
        if interval == "1d":
            fechas = ["2026-09-18", "2026-09-21", "2026-09-22", "2026-09-23"]
            if ticker == "HG=F":
                return _df_diario(fechas, [4.0, 4.05, 4.10, 3.90], "America/New_York")
            return _df_diario(fechas, [0.65, 0.651, 0.652, 0.653], "Europe/London")
        return _df_diario([ahora.strftime("%Y-%m-%d %H:%M")], [0.6520], "UTC")
    return descargar


def test_fallback_yahoo_etiqueta_y_completitud(cfg, ledger):
    ahora = datetime(2026, 9, 22, 21, 30, tzinfo=timezone.utc)  # 17:30 NY martes 22
    fy = FuenteYahoo(spread_pips=1.4, reloj=lambda: ahora, descargador=_descargador(ahora))
    velas = fy.velas_diarias("XCU_USD")
    assert all(v.fuente == FUENTE for v in velas)
    # La barra del 23 todavía no está completa a las 17:30 NY del 22.
    assert [v.completa for v in velas] == [True, True, True, False]
    cot = fy.precio("AUD_USD")
    assert cot.fuente == FUENTE and cot.ask - cot.bid == pytest.approx(0.00014)

    b = SimuladoBroker(fy, ledger, cfg.capital_inicial)
    r = ejecutar_ciclo(cfg, b, CobreAud(), ledger, ahora=ahora)
    assert r.estado == "ejecutada" and r.senal.direccion == 1  # 4.05 -> 4.10
    t = ledger.df("trades")
    assert t.fuente.tolist() == [FUENTE]
    assert ledger.df("senales").fuente.tolist() == [FUENTE]


def test_horario_mercado_fx():
    assert not mercado_fx_abierto(datetime(2026, 9, 26, 15, 0, tzinfo=timezone.utc))  # sábado
    assert not mercado_fx_abierto(datetime(2026, 9, 25, 21, 30, tzinfo=timezone.utc))  # vie 17:30 NY
    assert mercado_fx_abierto(datetime(2026, 9, 27, 21, 30, tzinfo=timezone.utc))  # dom 17:30 NY
    assert mercado_fx_abierto(datetime(2026, 9, 23, 14, 0, tzinfo=timezone.utc))  # miércoles
    assert fecha_ultimo_cierre_ny(datetime(2026, 9, 22, 20, 0, tzinfo=timezone.utc)).isoformat() \
        == "2026-09-21"
    assert fecha_ultimo_cierre_ny(datetime(2026, 9, 22, 21, 0, tzinfo=timezone.utc)).isoformat() \
        == "2026-09-22"
