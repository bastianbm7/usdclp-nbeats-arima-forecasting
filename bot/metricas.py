"""Métricas de desempeño a partir de la tabla equity_diaria (compartido por dashboard y HTML).

Convenciones:
- Retorno diario = equity_t / equity_{t-1} - 1; el primer día se compara contra el
  capital inicial.
- Retornos mensuales/anuales = compuestos: último equity del período / último equity
  del período anterior - 1 (el primer período, contra el capital inicial).
- Sharpe anualizado con 252 días y tasa libre de riesgo 0 (simple y explícito).
- Max drawdown sobre la curva de equity incluyendo el capital inicial.
"""
from __future__ import annotations

import math

import pandas as pd

DIAS_ANIO = 252


def serie_equity(df_equity: pd.DataFrame) -> pd.Series:
    if df_equity is None or len(df_equity) == 0:
        return pd.Series(dtype=float)
    s = pd.Series(df_equity["equity"].astype(float).values,
                  index=pd.to_datetime(df_equity["fecha"]), name="equity")
    return s.sort_index()


def retornos_diarios(eq: pd.Series, capital_inicial: float) -> pd.Series:
    if len(eq) == 0:
        return pd.Series(dtype=float)
    previo = eq.shift(1)
    previo.iloc[0] = capital_inicial
    return (eq / previo - 1.0).rename("retorno")


def retornos_periodo(eq: pd.Series, capital_inicial: float, freq: str) -> pd.DataFrame:
    """freq: 'ME' (mensual) o 'YE' (anual). Devuelve periodo, equity_final, retorno."""
    if len(eq) == 0:
        return pd.DataFrame(columns=["periodo", "equity_final", "retorno"])
    finales = eq.resample(freq).last().dropna()
    previo = finales.shift(1)
    previo.iloc[0] = capital_inicial
    ret = finales / previo - 1.0
    fmt = "%Y-%m" if freq.startswith("M") else "%Y"
    return pd.DataFrame({
        "periodo": finales.index.strftime(fmt),
        "equity_final": finales.values,
        "retorno": ret.values,
    })


def sharpe(ret_diarios: pd.Series) -> float | None:
    r = ret_diarios.dropna()
    if len(r) < 2:
        return None
    sd = r.std(ddof=1)
    if not sd or math.isnan(sd):
        return None
    return float(r.mean() / sd * math.sqrt(DIAS_ANIO))


def max_drawdown(eq: pd.Series, capital_inicial: float) -> float:
    if len(eq) == 0:
        return 0.0
    curva = pd.concat([pd.Series([capital_inicial]), eq.reset_index(drop=True)])
    return float((curva / curva.cummax() - 1.0).min())


def _retorno_desde_corte(eq: pd.Series, capital_inicial: float, corte: pd.Timestamp) -> float:
    """Retorno desde el último equity ANTERIOR a `corte` (o capital inicial) hasta hoy."""
    antes = eq[eq.index < corte]
    base = float(antes.iloc[-1]) if len(antes) else capital_inicial
    return float(eq.iloc[-1] / base - 1.0)


def kpis(df_equity: pd.DataFrame, df_trades: pd.DataFrame | None,
         capital_inicial: float) -> dict:
    eq = serie_equity(df_equity)
    trades = df_trades if df_trades is not None else pd.DataFrame()
    n_cerradas = int((trades["estado"] == "cerrada").sum()) if len(trades) else 0
    n_abiertas = int((trades["estado"] == "abierta").sum()) if len(trades) else 0
    costo_spread = 0.0
    financiamiento = 0.0
    ganadoras = 0
    if len(trades):
        costo_spread = float(trades[["entrada_costo_spread", "salida_costo_spread"]]
                             .fillna(0).to_numpy().sum())
        financiamiento = float(trades["financiamiento"].fillna(0).sum())
        cerr = trades[trades["estado"] == "cerrada"]
        ganadoras = int(((cerr["pnl"].fillna(0) + cerr["financiamiento"].fillna(0)) > 0).sum())
    base = {
        "capital_inicial": capital_inicial,
        "n_operaciones": n_cerradas + n_abiertas,
        "n_cerradas": n_cerradas,
        "n_abiertas": n_abiertas,
        "tasa_acierto": (ganadoras / n_cerradas) if n_cerradas else None,
        "costo_spread_total": costo_spread,
        "financiamiento_total": financiamiento,
    }
    if len(eq) == 0:
        return {**base, "capital_actual": capital_inicial, "retorno_total": 0.0,
                "retorno_hoy": None, "fecha_ultima": None, "retorno_mes": None,
                "retorno_anio": None, "sharpe": None, "max_drawdown": 0.0, "n_dias": 0}
    rd = retornos_diarios(eq, capital_inicial)
    ultima = eq.index[-1]
    return {
        **base,
        "capital_actual": float(eq.iloc[-1]),
        "retorno_total": float(eq.iloc[-1] / capital_inicial - 1.0),
        "retorno_hoy": float(rd.iloc[-1]),
        "fecha_ultima": ultima.date().isoformat(),
        "retorno_mes": _retorno_desde_corte(eq, capital_inicial, ultima.replace(day=1)),
        "retorno_anio": _retorno_desde_corte(eq, capital_inicial,
                                             ultima.replace(month=1, day=1)),
        "sharpe": sharpe(rd),
        "max_drawdown": max_drawdown(eq, capital_inicial),
        "n_dias": int(len(eq)),
    }


def tabla_diaria(df_equity: pd.DataFrame, capital_inicial: float) -> pd.DataFrame:
    eq = serie_equity(df_equity)
    if len(eq) == 0:
        return pd.DataFrame(columns=["fecha", "equity", "retorno"])
    rd = retornos_diarios(eq, capital_inicial)
    return pd.DataFrame({"fecha": eq.index.strftime("%Y-%m-%d"), "equity": eq.values,
                         "retorno": rd.values})
