"""Fixtures comunes. Ningún test toca la red: las fuentes de datos son falsas."""
from __future__ import annotations

import os
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

from config import Config  # noqa: E402
from ledger import Ledger  # noqa: E402
from modelos import Cotizacion, Vela  # noqa: E402

NY = ZoneInfo("America/New_York")


def cierre_ny(fecha: str) -> datetime:
    """17:00 NY de `fecha` en UTC."""
    d = datetime.fromisoformat(fecha).date()
    return datetime.combine(d, time(17, 0), tzinfo=NY).astimezone(timezone.utc)


def hacer_velas(instrumento: str, cierres: list[float], ultima_fecha: str,
                ultima_completa: bool = True, fuente: str = "oanda") -> list[Vela]:
    """Velas diarias consecutivas (días hábiles simplificados = días corridos)."""
    fin = cierre_ny(ultima_fecha)
    n = len(cierres)
    velas = []
    for i, c in enumerate(cierres):
        cierre = fin - timedelta(days=n - 1 - i)
        velas.append(Vela(instrumento, cierre - timedelta(days=1), cierre,
                          completa=True if i < n - 1 else ultima_completa,
                          o=c, h=c, l=c, c=c, bid_c=c - 0.00005, ask_c=c + 0.00005,
                          fuente=fuente))
    return velas


class FuenteFalsa:
    """Imita OandaClient/FuenteYahoo: velas y precio controlados por el test."""
    nombre = "falsa"

    def __init__(self):
        self.velas: dict[str, list[Vela]] = {}
        self.bid = 0.65000
        self.ask = 0.65010
        self.tradeable = True
        self.ahora = datetime(2026, 9, 22, 21, 30, tzinfo=timezone.utc)
        self.llamadas_precio = 0

    def velas_diarias(self, instrumento: str, n: int = 10):
        return self.velas.get(instrumento, [])[-n:]

    def precio(self, instrumento: str) -> Cotizacion:
        self.llamadas_precio += 1
        return Cotizacion(instrumento, self.bid, self.ask, self.ahora, self.tradeable,
                          fuente="falsa")


@pytest.fixture
def cfg(tmp_path) -> Config:
    return Config(modo="simulado", capital_inicial=10_000.0, fraccion_notional=1.0,
                  apalancamiento_max=2.0, minutos_espera_cierre=15,
                  db_path=tmp_path / "bot.db", log_dir=tmp_path / "logs").validar()


@pytest.fixture
def ledger(cfg):
    lg = Ledger(cfg.db_path, modo="simulado", capital_inicial=cfg.capital_inicial)
    yield lg
    lg.cerrar()


@pytest.fixture
def fuente():
    return FuenteFalsa()


@pytest.fixture(autouse=True)
def entorno_limpio(monkeypatch):
    """Que un bot/.env real del usuario no contamine los tests."""
    variables = ("MODO", "DB_PATH", "LOG_DIR", "CAPITAL_INICIAL", "FRACCION_NOTIONAL",
                 "APALANCAMIENTO_MAX", "ESTRATEGIA", "INSTRUMENTO", "INSTRUMENTO_SENAL",
                 "MINUTOS_ESPERA_CIERRE", "REABRIR_SI_MISMA_SENAL", "FUENTE_DATOS_SIMULADO",
                 "SPREAD_FALLBACK_PIPS")
    for k in list(os.environ):
        if k.startswith("OANDA_") or k in variables:
            monkeypatch.delenv(k, raising=False)
    yield
    # load_dotenv escribe en os.environ: limpiar también lo que dejó el test.
    for k in list(os.environ):
        if k.startswith("OANDA_") or k in variables:
            os.environ.pop(k, None)
