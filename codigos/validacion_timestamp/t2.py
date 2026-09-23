# EVIDENCIA DE LA ERRATA (2026-09-23) - copia (rutas adaptadas) del script de
# validacion de timestamp de la auditoria. Se deja tal cual como registro; la
# version consolidada y reproducible que genera las tablas del paper es
# codigos/58_validacion_timestamp.py.
# Que hace: descarga barras diarias y horarias (730 dias) de CLP=X, HG=F, AUDUSD=X, NOK=X desde Yahoo y muestra metadatos de zona horaria.
# Datos: datos/bases/validacion_timestamp/ (fetch.py/t2.py/t3.py descargan; el resto lee cache).
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'datos', 'bases', 'validacion_timestamp'))
import yfinance as yf, pandas as pd, numpy as np
out={}
for t in ["CLP=X","HG=F","AUDUSD=X","NOK=X"]:
    tk=yf.Ticker(t)
    d=tk.history(period="2y",interval="1d",auto_adjust=False)
    h=tk.history(period="730d",interval="60m",auto_adjust=False)
    d.to_csv(f"daily_{t}.csv"); h.to_csv(f"hourly_{t}.csv")
    md=tk.history_metadata
    print(t,"tz",md.get("exchangeTimezoneName"),md.get("timezone"),"gmtoff",md.get("gmtoffset"),"tradingPeriod",md.get("currentTradingPeriod"))
    print(" daily idx sample",d.index[-3:].tolist())
    print(" hourly idx sample",h.index[-3:].tolist(), len(h))
