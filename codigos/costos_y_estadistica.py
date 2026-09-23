# CORRECCION (2026-09-23): costos de transaccion realistas + incertidumbre
# estadistica del Sharpe, compartidos por todos los scripts corregidos (58+).
#
# BUG DE COSTOS QUE ESTO REEMPLAZA: en 22/27/28/32/33/40/42/48-50 el costo era
# un "slippage" de 0.05% del notional cobrado UNA sola vez, solo al entrar y
# solo cuando la posicion CAMBIABA respecto al dia anterior. Pero en los
# entornos diarios cada decision es una operacion completa (se abre en el
# precio de la fila y se cierra al dia siguiente o al tocar TP/SL): son
# ida+vuelta todos los dias. Con apalancamientos de 3.8-5.9x (notional/capital
# del risk sizing de 27) el costo pesa enormemente: para la regla del cobre en
# CLP el spread ida+vuelta de breakeven era ~0.20% del notional aun con los
# datos contaminados (auditoria, costos.py).
#
# SUPUESTOS DE SPREAD (bid-ask completo = costo de una ida+vuelta, como
# fraccion del notional). Son supuestos, no datos medidos: rangos tipicos de
# spread institucional/"prime" pequeno en horario liquido, mas amplios en pares
# emergentes poco liquidos y fuera de horario (la entrada corregida es ~20:00
# NY, hora de liquidez baja para CLP/BRL/ZAR). Un cliente minorista (CFD) suele
# pagar bastante mas. Por eso TODOS los resultados se reportan ademas con
# sensibilidad a un rango de spreads, incluido spread 0 (bruto).
SPREAD_IDA_VUELTA = {
    "CLP": 0.0015,  # rango 0.10-0.20%
    "AUD": 0.0002,
    "NZD": 0.0003,
    "CAD": 0.0002,
    "NOK": 0.0008,  # rango 0.05-0.10%
    "ZAR": 0.0010,  # rango 0.07-0.15%
    "BRL": 0.0015,
    "MXN": 0.0005,
    "COP": 0.0020,
    "PEN": 0.0020,
    "JPY": 0.0002,
    "CHF": 0.0003,
    "EUR": 0.0001,
    "GBP": 0.0002,
}
TICKER_A_CODIGO = {
    "CLP=X": "CLP", "MXN=X": "MXN", "BRL=X": "BRL", "COP=X": "COP", "PEN=X": "PEN", "ZAR=X": "ZAR",
    "CAD=X": "CAD", "AUDUSD=X": "AUD", "NZDUSD=X": "NZD", "JPY=X": "JPY", "CHF=X": "CHF",
    "EURUSD=X": "EUR", "GBPUSD=X": "GBP", "USDNOK=X": "NOK", "NOK=X": "NOK",
}
GRILLA_SPREADS = [0.0, 0.0002, 0.0005, 0.0010, 0.0015, 0.0020, 0.0030]

import numpy as np
import pandas as pd
from scipy.stats import norm


def spread_de(par):
    codigo = TICKER_A_CODIGO.get(par, par)
    return SPREAD_IDA_VUELTA[codigo]


# ---------------------------------------------------------------------------
# Sharpe y su incertidumbre
# ---------------------------------------------------------------------------
def sharpe(r, periodos=252):
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 3 or r.std() == 0:
        return np.nan
    return r.mean() / r.std() * np.sqrt(periodos)


def se_sharpe(r, periodos=252):
    """Error estandar asintotico del Sharpe (Lo 2002, retornos iid)."""
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 3 or r.std() == 0:
        return np.nan
    sr = r.mean() / r.std()
    return np.sqrt((1 + 0.5 * sr ** 2) / T) * np.sqrt(periodos)


def ic_sharpe_bootstrap(r, periodos=252, n_boot=5000, bloque=10, seed=0, nivel=0.95):
    """IC del Sharpe por bootstrap circular de bloques (preserva algo de
    autocorrelacion / clustering de volatilidad)."""
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    T = len(r)
    if T < 10 or r.std() == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n_bloques = int(np.ceil(T / bloque))
    sims = np.empty(n_boot)
    for b in range(n_boot):
        inicios = rng.integers(0, T, n_bloques)
        idx = (inicios[:, None] + np.arange(bloque)[None, :]).ravel()[:T] % T
        m = r[idx]
        s = m.std()
        sims[b] = m.mean() / s * np.sqrt(periodos) if s > 0 else np.nan
    a = (1 - nivel) / 2
    return tuple(np.nanpercentile(sims, [100 * a, 100 * (1 - a)]))


def sharpe_maximo_esperado_nulo(n_pruebas, T, periodos=252):
    """E[max Sharpe] de n_pruebas estrategias SIN senal (Bailey & Lopez de
    Prado 2014), anualizado - el umbral que un Sharpe elegido entre muchas
    configuraciones tiene que superar para no ser explicable por azar."""
    if n_pruebas < 2:
        return 0.0
    gamma = 0.5772156649
    se_d = np.sqrt(1.0 / T)
    emax = se_d * ((1 - gamma) * norm.ppf(1 - 1.0 / n_pruebas) + gamma * norm.ppf(1 - 1.0 / (n_pruebas * np.e)))
    return emax * np.sqrt(periodos)


def resumen_sharpe(r, periodos=252, n_pruebas=None, seed=0):
    sr = sharpe(r, periodos)
    lo, hi = ic_sharpe_bootstrap(r, periodos, seed=seed)
    out = {"sharpe": sr, "se_sharpe": se_sharpe(r, periodos), "ic95_bajo": lo, "ic95_alto": hi,
           "n_obs": int(np.sum(~np.isnan(np.asarray(r, dtype=float))))}
    if n_pruebas is not None:
        out["sharpe_max_nulo_esperado"] = sharpe_maximo_esperado_nulo(n_pruebas, out["n_obs"], periodos)
    return out


# ---------------------------------------------------------------------------
# Retornos netos de costos
# ---------------------------------------------------------------------------
def retornos_posicion_fija(pos, ret, spread_ida_vuelta, modo="ida_vuelta_diaria"):
    """Retorno diario (fraccion del capital) de una posicion f in [-1,1]
    (fraccion del capital) aplicada al retorno simple 'ret'.

    modo="ida_vuelta_diaria": cada dia con posicion paga el spread completo
        sobre |f| (la operacion se abre y se cierra en el dia) - criterio
        principal, consistente con como estan construidos los entornos.
    modo="rotacion": cota inferior - solo se paga medio spread por unidad de
        cambio de posicion |f_t - f_{t-1}| (como si se mantuviera la posicion
        entre dias sin cerrarla)."""
    pos = np.asarray(pos, dtype=float)
    ret = np.asarray(ret, dtype=float)
    bruto = pos * ret
    if modo == "ida_vuelta_diaria":
        costo = spread_ida_vuelta * np.abs(pos)
    elif modo == "rotacion":
        prev = np.r_[0.0, pos[:-1]]
        costo = 0.5 * spread_ida_vuelta * np.abs(pos - prev)
    else:
        raise ValueError(modo)
    return bruto - costo


def metricas_desde_retornos(r, nombre, periodos=252, capital_inicial=100.0, extra=None, seed=0):
    r = np.asarray(r, dtype=float)
    r = np.where(np.isnan(r), 0.0, r)
    capital = capital_inicial * np.cumprod(1 + r)
    dd = capital / np.maximum.accumulate(capital) - 1
    fila = {"estrategia": nombre, "retorno_total_%": 100 * (capital[-1] / capital_inicial - 1) if len(r) else np.nan,
            "max_drawdown_%": 100 * dd.min() if len(r) else np.nan}
    fila.update(resumen_sharpe(r, periodos, seed=seed))
    if extra:
        fila.update(extra)
    return fila


def breakeven_spread(r_bruto, exposicion):
    """Spread ida+vuelta (fraccion del notional) que lleva el retorno medio a 0,
    dado r_bruto (fraccion del capital) y exposicion = notional/capital por dia."""
    exp_media = np.nanmean(np.asarray(exposicion, dtype=float))
    if exp_media <= 0:
        return np.nan
    return np.nanmean(np.asarray(r_bruto, dtype=float)) / exp_media
