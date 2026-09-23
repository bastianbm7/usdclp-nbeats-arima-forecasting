"""Crea una base de DEMOSTRACIÓN (bot/data/demo.db) con historia SINTÉTICA para ver el dashboard.

- Precios inventados (caminatas aleatorias sin edge), aplicando la misma regla y la misma
  contabilidad bid/ask del bot. Todo queda etiquetado fuente='demo_sintetico' y la base
  marcada con meta demo=1 (el dashboard muestra un aviso grande).
- Se niega a escribir sobre la base real configurada en DB_PATH.

    python scripts/sembrar_demo.py [--dias 320] [--salida data/demo.db] [--semilla 7]
"""
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

BOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT_DIR))

from broker.simulado import fill_apertura, fill_cierre, pnl_no_realizado  # noqa: E402
from config import cargar_config  # noqa: E402
from ledger import Ledger  # noqa: E402
from modelos import Cotizacion, Senal  # noqa: E402
from motor import calcular_unidades  # noqa: E402

NY = ZoneInfo("America/New_York")
FUENTE = "demo_sintetico"


def dias_habiles(n: int, hasta: datetime) -> list[datetime]:
    dias, d = [], hasta.date()
    while len(dias) < n:
        if d.weekday() < 5:
            dias.append(datetime.combine(d, time(17, 0), tzinfo=NY).astimezone(timezone.utc))
        d -= timedelta(days=1)
    return dias[::-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dias", type=int, default=320)
    ap.add_argument("--salida", default=str(BOT_DIR / "data" / "demo.db"))
    ap.add_argument("--semilla", type=int, default=7)
    ap.add_argument("--capital", type=float, default=10_000.0)
    a = ap.parse_args(argv)

    salida = Path(a.salida).resolve()
    real = cargar_config(validar=False).db_path.resolve()
    if salida == real:
        print(f"ERROR: {salida} es la base REAL configurada (DB_PATH). Usa otra ruta.")
        return 2
    if salida.exists():
        salida.unlink()

    rnd = random.Random(a.semilla)
    lg = Ledger(salida, modo="simulado", capital_inicial=a.capital)
    lg.conn.execute("INSERT OR REPLACE INTO meta VALUES ('demo', '1')")
    lg.conn.commit()

    ayer = datetime.now(NY) - timedelta(days=1)
    cierres = dias_habiles(a.dias, ayer)
    aud, cu, medio_spread = 0.66, 4.30, 0.00007
    balance = a.capital
    pos_u, pos_px = 0, 0.0

    for i, cierre in enumerate(cierres):
        r_cu = rnd.gauss(0, 0.014)
        cu *= 1 + r_cu
        aud *= 1 + rnd.gauss(0, 0.006) + 0.15 * r_cu * 0.4  # co-movimiento, sin predicción
        ejec = cierre + timedelta(minutes=20)
        cot = Cotizacion("AUD_USD", aud - medio_spread, aud + medio_spread, ejec, True, FUENTE)
        direccion = (r_cu > 0) - (r_cu < 0)
        s = Senal("cobre_aud", "AUD_USD", direccion, cierre - timedelta(days=1), cierre, r_cu,
                  "demo", FUENTE, {"sintetico": True})
        fecha = cierre.astimezone(NY).date().isoformat()
        sid = lg.registrar_senal(s, fecha)
        actual = (pos_u > 0) - (pos_u < 0)
        if actual == direccion:
            lg.marcar_senal(sid, "sin_cambio")
        else:
            if pos_u:
                f = fill_cierre("AUD_USD", pos_u, pos_px, cot)
                lg.cerrar_trade(f, sid)
                balance += f.pnl
                pos_u = 0
            if direccion:
                u = calcular_unidades(balance, cot.ask, 1.0, 2.0) * direccion
                f = fill_apertura("AUD_USD", u, cot)
                lg.abrir_trade("cobre_aud", "simulado", f, sid, balance)
                pos_u, pos_px = u, f.precio
            lg.marcar_senal(sid, "ejecutada")
        nr = pnl_no_realizado(pos_u, pos_px, cot) if pos_u else 0.0
        lg.registrar_equity(fecha, balance + nr, balance, nr, pos_u, FUENTE)

    n = lg.conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    lg.cerrar()
    print(f"Demo creada: {salida} ({len(cierres)} días, {n} trades, fuente={FUENTE})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
