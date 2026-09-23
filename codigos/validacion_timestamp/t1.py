# EVIDENCIA DE LA ERRATA (2026-09-23) - copia (rutas adaptadas) del script de
# validacion de timestamp de la auditoria. Se deja tal cual como registro; la
# version consolidada y reproducible que genera las tablas del paper es
# codigos/58_validacion_timestamp.py.
# Que hace: reproduce el escaneo de rezagos original de 9.14 (usdclp_long + macro) y el Sharpe de la regla signo-del-cobre con la alineacion ORIGINAL.
# Datos: datos/bases/validacion_timestamp/ (fetch.py/t2.py/t3.py descargan; el resto lee cache).
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'datos', 'bases', 'validacion_timestamp'))
import pandas as pd, numpy as np
R = "../"
d=pd.read_csv(R+"usdclp_long.csv",parse_dates=["ds"]).sort_values("ds")
m=pd.read_csv(R+"macro_tasas_cobre.csv",parse_dates=["ds"])
d=pd.merge_asof(d,m[["ds","copper"]],on="ds",direction="backward")
d["r"]=np.log(d.y/d.y.shift(1)); d["c"]=np.log(d.copper/d.copper.shift(1))
print("n",len(d), d.ds.min(), d.ds.max())
for k in range(-2,4): print("lag",k, round(d.c.corr(d.r.shift(-k)),3))
# weekday of CLP dates
print(d.ds.dt.dayofweek.value_counts().sort_index())
# strategy sign(c_t) short USD when copper up, hold t->t+1
pos=-np.sign(d.c); pnl=pos*d.r.shift(-1); pnl=pnl.dropna()
print("Sharpe full", pnl.mean()/pnl.std()*np.sqrt(252))
