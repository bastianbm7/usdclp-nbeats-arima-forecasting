# Issue #14: descarga las variables candidatas nuevas para USD/CLP (semanal/
# mensual) y les asigna el TIMESTAMP REAL en que se conoce cada dato - la
# leccion de 9.35: una variable solo sirve si su hora de publicacion es
# estrictamente anterior al precio FX al que se opera.
#
# Candidatas (ancla economica, no mineria de datos):
#   - Regimen de riesgo global: VIX (FRED VIXCLS, cierre 16:15 ET), S&P 500
#     (^GSPC, 16:00 ET), mercados emergentes (EEM, 16:00 ET). Las monedas
#     emergentes se deprecian en episodios risk-off.
#   - Dolar global: indice amplio del dolar (FRED DTWEXBGS). Se calcula con
#     tipos de mediodia de H.10, que la Fed publica SEMANALMENTE (lunes): se
#     usa un rezago conservador de 7 dias calendario.
#   - Tasas EE.UU.: bono a 2 anos (FRED DGS2, se toma como conocido a las
#     17:00 ET del mismo dia) y letra a 3 meses (DTB3, para el carry).
#   - Chile: acciones chilenas en USD (ECH, ETF MSCI Chile, 16:00 ET - el IPSA
#     no esta disponible en Yahoo); tasa interbancaria 3m (OECD via FRED,
#     IR3TIB01CLM156N, mensual, con rezago de publicacion de 3 meses, igual
#     que en 68).
#   - Petroleo (WTI, CL=F, settlement 14:30 ET): Chile es importador neto.
#
# No incluidas por falta de datos accesibles sin credenciales (quedan para el
# Issue #16, API del BCCh): posiciones en forwards de las AFP, sorpresas de TPM
# vs. Encuesta de Expectativas, EMBI Chile.

import numpy as np
import pandas as pd
import yfinance as yf

SALIDA = "../datos/bases/issue14_variables_externas.csv"
INICIO = "2009-01-01"
ZONA_NY = "America/New_York"

FRED = {
    # serie: (hora ET a la que se conoce el dato del dia D, rezago extra en dias)
    "VIXCLS": ((16, 15), 0),
    "DGS2": ((17, 0), 0),
    "DTB3": ((17, 0), 0),
    "DTWEXBGS": ((12, 0), 7),
}
FRED_MENSUAL_REZAGO_MESES = {"IR3TIB01CLM156N": 3}
YAHOO = {"^GSPC": (16, 0), "EEM": (16, 0), "ECH": (16, 0), "CL=F": (14, 30)}


def ts_et(fechas, hora, rezago_dias=0):
    hh, mm = hora
    local = pd.DatetimeIndex(pd.to_datetime(fechas)) + pd.Timedelta(days=rezago_dias, hours=hh, minutes=mm)
    return local.tz_localize(ZONA_NY, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")


def fred(serie):
    d = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={serie}")
    d.columns = ["ds", "valor"]
    d["ds"] = pd.to_datetime(d["ds"])
    d["valor"] = pd.to_numeric(d["valor"], errors="coerce")
    return d[d["ds"] >= INICIO].dropna()


def main():
    partes = []
    hoy = pd.Timestamp.today().normalize()
    for serie, (hora, rezago) in FRED.items():
        d = fred(serie)
        d["ts_conocido_utc"] = ts_et(d["ds"], hora, rezago)
        d["serie"] = serie
        partes.append(d)
    for serie, meses in FRED_MENSUAL_REZAGO_MESES.items():
        d = fred(serie)
        # dato del mes M se da por conocido al inicio del mes M+meses
        d["ts_conocido_utc"] = ts_et(d["ds"] + pd.DateOffset(months=meses), (0, 0))
        d["serie"] = serie
        partes.append(d)
    for ticker, hora in YAHOO.items():
        y = yf.download(ticker, start=INICIO, progress=False, auto_adjust=True)
        if isinstance(y.columns, pd.MultiIndex):
            y.columns = y.columns.get_level_values(0)
        d = y[["Close"]].reset_index()
        d.columns = ["ds", "valor"]
        d = d[d["ds"] < hoy].dropna()  # la barra de hoy puede estar incompleta
        d["ts_conocido_utc"] = ts_et(d["ds"], hora)
        d["serie"] = ticker
        partes.append(d)

    out = pd.concat(partes, ignore_index=True)[["serie", "ds", "valor", "ts_conocido_utc"]]
    out.to_csv(SALIDA, index=False)
    print(out.groupby("serie").agg(n=("valor", "size"), desde=("ds", "min"), hasta=("ds", "max")).to_string())


if __name__ == "__main__":
    main()
