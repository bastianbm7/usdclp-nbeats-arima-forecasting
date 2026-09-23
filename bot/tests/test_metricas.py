import math

import pandas as pd
import pytest

from metricas import (kpis, max_drawdown, retornos_diarios, retornos_periodo, serie_equity,
                      sharpe, tabla_diaria)

CAP = 10_000.0


def _df(pares):
    return pd.DataFrame(pares, columns=["fecha", "equity"])


def test_retornos_diarios_primer_dia_contra_capital():
    eq = serie_equity(_df([("2026-01-02", 10_100), ("2026-01-05", 9_999)]))
    rd = retornos_diarios(eq, CAP)
    assert rd.iloc[0] == pytest.approx(0.01)
    assert rd.iloc[1] == pytest.approx(9_999 / 10_100 - 1)


def test_agregacion_mensual_y_anual_compuesta():
    eq = serie_equity(_df([
        ("2025-12-30", 10_500),   # dic-25: +5%
        ("2026-01-15", 11_000),
        ("2026-01-30", 10_290),   # ene-26: 10290/10500 - 1 = -2%
        ("2026-02-27", 11_319),   # feb-26: +10%
    ]))
    m = retornos_periodo(eq, CAP, "ME")
    assert m.periodo.tolist() == ["2025-12", "2026-01", "2026-02"]
    assert m.retorno.tolist() == pytest.approx([0.05, -0.02, 0.10])
    a = retornos_periodo(eq, CAP, "YE")
    assert a.periodo.tolist() == ["2025", "2026"]
    assert a.retorno.tolist() == pytest.approx([0.05, 11_319 / 10_500 - 1])
    # compuesto de meses de 2026 == retorno anual 2026
    assert (1 - 0.02) * 1.10 - 1 == pytest.approx(a.retorno.iloc[1])


def test_max_drawdown_incluye_capital_inicial():
    eq = serie_equity(_df([("2026-01-02", 9_000), ("2026-01-05", 12_000),
                           ("2026-01-06", 9_600)]))
    assert max_drawdown(eq, CAP) == pytest.approx(-0.2)
    assert max_drawdown(serie_equity(_df([("2026-01-02", 9_000)])), CAP) == pytest.approx(-0.1)


def test_sharpe():
    r = pd.Series([0.01, -0.005, 0.002, 0.004])
    esperado = r.mean() / r.std(ddof=1) * math.sqrt(252)
    assert sharpe(r) == pytest.approx(esperado)
    assert sharpe(pd.Series([0.01])) is None
    assert sharpe(pd.Series([0.0, 0.0, 0.0])) is None


def test_kpis_vacio_y_con_datos():
    k0 = kpis(_df([]), None, CAP)
    assert k0["capital_actual"] == CAP and k0["retorno_total"] == 0 and k0["n_operaciones"] == 0
    eq = _df([("2025-12-30", 10_500), ("2026-01-30", 10_290), ("2026-02-26", 11_000),
              ("2026-02-27", 11_319)])
    trades = pd.DataFrame({
        "estado": ["cerrada", "cerrada", "abierta"],
        "pnl": [10.0, -5.0, None],
        "financiamiento": [-1.0, 0.0, 0.0],
        "entrada_costo_spread": [0.7, 0.7, 0.7],
        "salida_costo_spread": [0.7, 0.7, None],
    })
    k = kpis(eq, trades, CAP)
    assert k["capital_actual"] == 11_319
    assert k["retorno_total"] == pytest.approx(0.1319)
    assert k["retorno_hoy"] == pytest.approx(11_319 / 11_000 - 1)
    assert k["retorno_mes"] == pytest.approx(11_319 / 10_290 - 1)
    assert k["retorno_anio"] == pytest.approx(11_319 / 10_500 - 1)
    assert k["n_operaciones"] == 3 and k["n_cerradas"] == 2 and k["n_abiertas"] == 1
    assert k["tasa_acierto"] == pytest.approx(0.5)
    assert k["costo_spread_total"] == pytest.approx(3.5)
    assert tabla_diaria(eq, CAP).shape[0] == 4
