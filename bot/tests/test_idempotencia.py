from dataclasses import replace
from datetime import datetime, timezone

from broker.simulado import SimuladoBroker
from conftest import hacer_velas
from estrategias.cobre_aud import CobreAud
from motor import ejecutar_ciclo

AHORA = datetime(2026, 9, 22, 21, 30, tzinfo=timezone.utc)  # 17:30 NY del 22-sep


def _preparar(fuente, cobre, aud=(0.65, 0.66), fecha="2026-09-22"):
    fuente.velas["XCU_USD"] = hacer_velas("XCU_USD", list(cobre), fecha)
    fuente.velas["AUD_USD"] = hacer_velas("AUD_USD", list(aud), fecha)


def _n_trades(ledger):
    return ledger.conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]


def test_correr_dos_veces_el_mismo_dia_no_duplica(cfg, ledger, fuente):
    _preparar(fuente, [4.0, 4.1])
    broker = SimuladoBroker(fuente, ledger, cfg.capital_inicial)
    r1 = ejecutar_ciclo(cfg, broker, CobreAud(), ledger, ahora=AHORA)
    r2 = ejecutar_ciclo(cfg, broker, CobreAud(), ledger, ahora=AHORA)
    r3 = ejecutar_ciclo(cfg, broker, CobreAud(), ledger, ahora=AHORA)
    assert r1.estado == "ejecutada" and len(r1.trades_abiertos) == 1
    assert r2.estado == r3.estado == "ya_procesada"
    assert _n_trades(ledger) == 1
    t = ledger.trade_abierto("AUD_USD")
    assert t["unidades"] > 0 and t["entrada_precio"] == fuente.ask
    assert ledger.conn.execute("SELECT COUNT(*) FROM senales").fetchone()[0] == 1
    assert ledger.conn.execute("SELECT COUNT(*) FROM equity_diaria").fetchone()[0] == 1


def test_senal_pendiente_tras_falla_no_duplica(cfg, ledger, fuente):
    """Si la señal quedó 'pendiente' pero la posición ya es la objetivo, no reabre."""
    _preparar(fuente, [4.0, 4.1])
    broker = SimuladoBroker(fuente, ledger, cfg.capital_inicial)
    ejecutar_ciclo(cfg, broker, CobreAud(), ledger, ahora=AHORA)
    ledger.conn.execute("UPDATE senales SET estado = 'pendiente'")
    ledger.conn.commit()
    r = ejecutar_ciclo(cfg, broker, CobreAud(), ledger, ahora=AHORA)
    assert r.estado == "sin_cambio"
    assert _n_trades(ledger) == 1


def test_cambio_de_senal_cierra_y_abre(cfg, ledger, fuente):
    broker = SimuladoBroker(fuente, ledger, cfg.capital_inicial)
    _preparar(fuente, [4.0, 4.1], fecha="2026-09-22")
    ejecutar_ciclo(cfg, broker, CobreAud(), ledger, ahora=AHORA)
    # Día siguiente: cobre baja -> corto. El AUD subió: el largo cierra con ganancia.
    fuente.bid, fuente.ask = 0.66000, 0.66010
    _preparar(fuente, [4.1, 4.0], fecha="2026-09-23")
    r = ejecutar_ciclo(cfg, broker, CobreAud(), ledger,
                       ahora=datetime(2026, 9, 23, 21, 30, tzinfo=timezone.utc))
    assert r.estado == "ejecutada"
    assert len(r.trades_cerrados) == 1 and len(r.trades_abiertos) == 1
    df = ledger.df("trades")
    cerrado = df[df.estado == "cerrada"].iloc[0]
    assert cerrado.salida_precio == 0.66000  # largo se cierra al BID
    assert cerrado.pnl > 0
    abierto = ledger.trade_abierto("AUD_USD")
    assert abierto["unidades"] < 0 and abierto["entrada_precio"] == 0.66000  # corto al BID


def test_misma_senal_mantiene_posicion(cfg, ledger, fuente):
    broker = SimuladoBroker(fuente, ledger, cfg.capital_inicial)
    _preparar(fuente, [4.0, 4.1], fecha="2026-09-22")
    ejecutar_ciclo(cfg, broker, CobreAud(), ledger, ahora=AHORA)
    _preparar(fuente, [4.1, 4.2], fecha="2026-09-23")
    r = ejecutar_ciclo(cfg, broker, CobreAud(), ledger,
                       ahora=datetime(2026, 9, 23, 21, 30, tzinfo=timezone.utc))
    assert r.estado == "sin_cambio"
    assert _n_trades(ledger) == 1


def test_reabrir_si_misma_senal(cfg, ledger, fuente):
    cfg2 = replace(cfg, reabrir_si_misma_senal=True)
    broker = SimuladoBroker(fuente, ledger, cfg.capital_inicial)
    _preparar(fuente, [4.0, 4.1], fecha="2026-09-22")
    ejecutar_ciclo(cfg2, broker, CobreAud(), ledger, ahora=AHORA)
    _preparar(fuente, [4.1, 4.2], fecha="2026-09-23")
    ejecutar_ciclo(cfg2, broker, CobreAud(), ledger,
                   ahora=datetime(2026, 9, 23, 21, 30, tzinfo=timezone.utc))
    assert _n_trades(ledger) == 2


def test_espera_tras_cierre(cfg, ledger, fuente):
    _preparar(fuente, [4.0, 4.1])
    broker = SimuladoBroker(fuente, ledger, cfg.capital_inicial)
    r = ejecutar_ciclo(cfg, broker, CobreAud(), ledger,
                       ahora=datetime(2026, 9, 22, 21, 5, tzinfo=timezone.utc))  # 17:05 NY
    assert r.estado == "esperando" and _n_trades(ledger) == 0


def test_mercado_cerrado_deja_pendiente_y_luego_ejecuta(cfg, ledger, fuente):
    _preparar(fuente, [4.0, 4.1])
    broker = SimuladoBroker(fuente, ledger, cfg.capital_inicial)
    fuente.tradeable = False
    r = ejecutar_ciclo(cfg, broker, CobreAud(), ledger, ahora=AHORA)
    assert r.estado == "mercado_cerrado" and _n_trades(ledger) == 0
    assert ledger.df("senales").estado.tolist() == ["pendiente"]
    fuente.tradeable = True
    r = ejecutar_ciclo(cfg, broker, CobreAud(), ledger, ahora=AHORA)
    assert r.estado == "ejecutada" and _n_trades(ledger) == 1


def test_dry_run_no_escribe(cfg, ledger, fuente):
    _preparar(fuente, [4.0, 4.1])
    broker = SimuladoBroker(fuente, ledger, cfg.capital_inicial)
    r = ejecutar_ciclo(cfg, broker, CobreAud(), ledger, ahora=AHORA, dry_run=True)
    assert r.estado == "dry_run" and r.senal.direccion == 1
    assert _n_trades(ledger) == 0
    assert ledger.conn.execute("SELECT COUNT(*) FROM senales").fetchone()[0] == 0
