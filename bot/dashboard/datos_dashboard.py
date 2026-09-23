"""Carga de datos del ledger para el dashboard (Streamlit) y el HTML estático."""
from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BOT_DIR = Path(__file__).resolve().parents[1]
if str(BOT_DIR) not in sys.path:
    sys.path.insert(0, str(BOT_DIR))

import metricas  # noqa: E402

FUENTES_NO_VALIDAS = {"fallback_yahoo": "Yahoo (timestamps ambiguos, spread sintético)",
                      "demo_sintetico": "datos SINTÉTICOS de demostración"}


@dataclass
class DatosDashboard:
    ruta: Path
    modo: str
    capital_inicial: float
    equity: pd.DataFrame
    trades: pd.DataFrame
    senales: pd.DataFrame
    ejecuciones: pd.DataFrame
    kpis: dict
    diaria: pd.DataFrame
    mensual: pd.DataFrame
    anual: pd.DataFrame
    avisos: list[str]


def db_por_defecto() -> Path:
    try:
        from config import cargar_config
        return cargar_config(validar=False).db_path
    except Exception:  # noqa: BLE001
        return BOT_DIR / "data" / "bot.db"


def cargar(ruta: Path | str) -> DatosDashboard:
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No existe la base {ruta}. Corre primero run_diario.py.")
    # solo lectura: el dashboard nunca escribe en el ledger
    conn = sqlite3.connect(f"file:{ruta.as_posix()}?mode=ro", uri=True)
    try:
        meta = dict(conn.execute("SELECT clave, valor FROM meta").fetchall())
        leer = lambda t, o: pd.read_sql_query(f"SELECT * FROM {t} ORDER BY {o}", conn)  # noqa: E731
        equity = leer("equity_diaria", "fecha")
        trades = leer("trades", "id")
        senales = leer("senales", "id")
        ejecuciones = leer("ejecuciones_log", "id")
    finally:
        conn.close()

    capital = float(meta.get("capital_inicial", 10_000.0))
    eq = metricas.serie_equity(equity)
    avisos = []
    fuentes = set(trades.get("fuente", pd.Series(dtype=str)).dropna()) | \
        set(senales.get("fuente", pd.Series(dtype=str)).dropna())
    for f, desc in FUENTES_NO_VALIDAS.items():
        if f in fuentes:
            avisos.append(f"Hay registros con fuente '{f}': {desc}. No sirven para evaluar la señal.")
    if meta.get("demo") == "1":
        avisos.insert(0, "BASE DE DEMOSTRACIÓN con historia sintética: no son operaciones reales.")

    return DatosDashboard(
        ruta=ruta, modo=meta.get("modo", "?"), capital_inicial=capital, equity=equity,
        trades=trades, senales=senales, ejecuciones=ejecuciones,
        kpis=metricas.kpis(equity, trades, capital),
        diaria=metricas.tabla_diaria(equity, capital),
        mensual=metricas.retornos_periodo(eq, capital, "ME"),
        anual=metricas.retornos_periodo(eq, capital, "YE"),
        avisos=avisos,
    )


def tabla_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Vista legible de la tabla de trades."""
    if len(trades) == 0:
        return pd.DataFrame()
    t = trades.copy()
    t["lado"] = t["direccion"].map({1: "largo", -1: "corto"})
    t["costo_spread"] = t["entrada_costo_spread"].fillna(0) + t["salida_costo_spread"].fillna(0)
    t["pnl_neto"] = t["pnl"].fillna(0) + t["financiamiento"].fillna(0)
    t.loc[t["estado"] == "abierta", "pnl_neto"] = None
    for c in ("entrada_tiempo", "salida_tiempo"):
        t[c] = pd.to_datetime(t[c], utc=True, format="ISO8601").dt.strftime("%Y-%m-%d %H:%M")
    cols = ["id", "estado", "instrumento", "lado", "unidades", "entrada_tiempo", "entrada_precio",
            "salida_tiempo", "salida_precio", "costo_spread", "financiamiento", "pnl_neto",
            "retorno_pct", "fuente"]
    return t[cols].sort_values("id", ascending=False).reset_index(drop=True)


def fmt_pct(x, dec: int = 2) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:+.{dec}f}%"


def fmt_tasa(x) -> str:
    """Proporción sin signo (p.ej. tasa de acierto)."""
    return "—" if x is None or pd.isna(x) else f"{x * 100:.1f}%"


def fmt_num(x, dec: int = 2) -> str:
    return "—" if x is None or pd.isna(x) else f"{x:,.{dec}f}"
