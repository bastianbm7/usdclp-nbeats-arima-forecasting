# EVIDENCIA DE LA ERRATA (2026-09-23) - copia (rutas adaptadas) del script de
# validacion de timestamp de la auditoria. Se deja tal cual como registro; la
# version consolidada y reproducible que genera las tablas del paper es
# codigos/58_validacion_timestamp.py.
# Que hace: para cada ticker, busca que hora UTC de las barras horarias reproduce mejor el retorno de la barra DIARIA -> AUDUSD/NOK: 00:00 UTC de D (corr 0.97); HG=F: ~17:00 UTC (settlement ~13:00 ET).
# Datos: datos/bases/validacion_timestamp/ (fetch.py/t2.py/t3.py descargan; el resto lee cache).
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'datos', 'bases', 'validacion_timestamp'))
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
def load(f):
    x=pd.read_csv(f,index_col=0); x.index=pd.to_datetime(x.index,utc=True); return x
for t in ["CLP=X","AUDUSD=X","NOK=X","HG=F"]:
    d=pd.read_csv(f"daily_{t}.csv",index_col=0); d.index=pd.to_datetime(pd.Series(d.index).str[:10].values)
    d=d.iloc[:-1]
    h=load(f"hourly_{t}.csv")["Close"]
    # price as-of time: last hourly close whose bar END (start+1h) <= T
    ends=h.copy(); ends.index=ends.index+pd.Timedelta(hours=1)
    dr=np.log(d.Close).diff()
    d=d[d.index>=ends.index.min().tz_localize(None)+pd.Timedelta(days=3)]
    out=[]
    for off in range(-30,31,1):
        T=pd.DatetimeIndex(d.index).tz_localize("UTC")+pd.Timedelta(hours=off)
        p=ends.reindex(ends.index.union(T)).ffill().reindex(T)
        s=pd.Series(np.log(p.values),index=d.index).diff()
        lev=np.nanmedian(np.abs(p.values-d.Close.values)/d.Close.values)
        out.append((off,dr.reindex(d.index).corr(s),lev))
    o=pd.DataFrame(out,columns=["off_h_from_D00UTC","corr_ret","med_level_err"])
    b=o.loc[o.corr_ret.idxmax()]
    print("=====",t," best:",b.to_dict())
    print(o.iloc[::3].round(4).to_string(index=False))
