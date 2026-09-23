"""Punto de entrada: códigos de salida y registro en ejecuciones_log."""
import logging

import run_diario
from conftest import FuenteFalsa, hacer_velas
from ledger import Ledger


def _env(tmp_path, extra=""):
    p = tmp_path / ".env"
    p.write_text(f"MODO=simulado\nDB_PATH={tmp_path / 'bot.db'}\nLOG_DIR={tmp_path / 'logs'}\n"
                 f"MINUTOS_ESPERA_CIERRE=0\n{extra}", encoding="utf-8")
    return str(p)


def test_host_real_sale_con_codigo_2(tmp_path):
    assert run_diario.main(["--env", _env(tmp_path, "OANDA_HOST=api-fxtrade.oanda.com\n")]) == 2


def test_ok_dos_corridas_y_error_no_cero(tmp_path, monkeypatch):
    fuente = FuenteFalsa()
    fuente.velas["XCU_USD"] = hacer_velas("XCU_USD", [4.0, 4.1], "2026-09-21")
    fuente.velas["AUD_USD"] = hacer_velas("AUD_USD", [0.65, 0.66], "2026-09-21")
    monkeypatch.setattr("broker.crear_fuente_datos", lambda cfg: fuente)
    env = _env(tmp_path)
    try:
        assert run_diario.main(["--env", env]) == 0
        assert run_diario.main(["--env", env]) == 0

        def rompe(*a, **k):
            raise RuntimeError("sin red")
        monkeypatch.setattr(fuente, "velas_diarias", rompe)
        assert run_diario.main(["--env", env]) == 1
    finally:
        # soltar el FileHandler para poder borrar tmp_path en Windows
        for h in list(logging.getLogger().handlers):
            h.close()
            logging.getLogger().removeHandler(h)

    lg = Ledger(tmp_path / "bot.db")
    estados = lg.df("ejecuciones_log").estado.tolist()
    assert estados == ["ejecutada", "ya_procesada", "error"]
    assert len(lg.df("trades")) == 1
    assert lg.capital_inicial == 10_000
    lg.cerrar()
    assert not (tmp_path / "bot.lock").exists()
