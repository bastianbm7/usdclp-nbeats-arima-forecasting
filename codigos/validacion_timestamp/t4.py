# EVIDENCIA DE LA ERRATA (2026-09-23) - copia (rutas adaptadas) del script de
# validacion de timestamp de la auditoria. Se deja tal cual como registro; la
# version consolidada y reproducible que genera las tablas del paper es
# codigos/58_validacion_timestamp.py.
# Que hace: alineacion con barras HORARIAS a la misma hora de reloj para cobre y FX (14-21 UTC): la correlacion se mueve al rezago 0 y el Sharpe de la regla cae a ~0.
# Datos: datos/bases/validacion_timestamp/ (fetch.py/t2.py/t3.py descargan; el resto lee cache).
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'datos', 'bases', 'validacion_timestamp'))
import pandas as pd, numpy as np, warnings; warnings.filterwarnings("ignore")
def load(f):
    x=pd.read_csv(f,index_col=0); x.index=pd.to_datetime(x.index,utc=True); x=x["Close"]; x.index=x.index+pd.Timedelta(hours=1); return x  # index = bar end
def daily(t):
    d=pd.read_csv(f"daily_{t}.csv",index_col=0); d.index=pd.to_datetime(pd.Series(d.index).str[:10].values); return d.Close.iloc[:-1]
hg=load("hourly_HG=F.csv")
def asof(s,dates,hour,maxgap=3):
    T=pd.DatetimeIndex(dates).tz_localize("UTC")+pd.Timedelta(hours=hour)
    idx=s.index.searchsorted(T,side="right")-1
    ok=idx>=0; v=np.full(len(T),np.nan); tt=s.index[np.clip(idx,0,None)]
    good=ok&((T-tt)<=pd.Timedelta(hours=maxgap))
    v[good]=s.values[idx[good]]; return pd.Series(v,index=dates)
def scan(c,f):
    return {k: round(c.corr(f.shift(-k)),3) for k in range(-2,4)}
def sharpe(c,f,sign=-1):
    pnl=(sign*np.sign(c)*f.shift(-1)).dropna(); return round(pnl.mean()/pnl.std()*np.sqrt(252),2), len(pnl)
rows=[]
for t,sign in [("CLP=X",-1),("AUDUSD=X",1),("NOK=X",-1)]:
    fx=load(f"hourly_{t}.csv"); fd=daily(t); hd=daily("HG=F")
    dates=pd.bdate_range(max(fx.index.min(),hg.index.min()).tz_localize(None).normalize()+pd.Timedelta(days=2), fd.index.max())
    # baseline: yahoo daily closes, same window
    b=pd.concat([np.log(hd).diff().rename("c"),np.log(fd).diff().rename("f")],axis=1).dropna()
    b=b[b.index>=dates[0]]
    print("=====",t,"| baseline Yahoo daily (same 730d window) n=",len(b), scan(b.c,b.f), "Sharpe(sign copper)",sharpe(b.c,b.f,sign))
    for H in [14,16,17,18,19,20,21]:
        c=np.log(asof(hg,dates,H)); f=np.log(asof(fx,dates,H))
        df=pd.concat([c.rename("c"),f.rename("f")],axis=1).dropna()
        # returns between consecutive valid days
        r=df.diff().dropna()
        print(f"  hourly-aligned @{H:02d}UTC n={len(r)}",scan(r.c,r.f),"Sharpe",sharpe(r.c,r.f,sign))
    # mixed: copper settle(yahoo daily) vs FX at 18:00 UTC same date
    f18=np.log(asof(fx,dates,18)); m=pd.concat([np.log(hd).diff().rename("c"),f18.diff().rename("f")],axis=1).dropna()
    print("  Yahoo HG settle vs FX@18UTC",scan(m.c,m.f),"Sharpe",sharpe(m.c,m.f,sign))
    # Yahoo daily FX but shifted label back one day (Close_D -> represents D-1 evening)
    fs=np.log(fd).diff().shift(-1)  # return labeled D uses Close_{D+1}/Close_D = price at D+1 00UTC / D 00UTC -> day D move
    s=pd.concat([np.log(hd).diff().rename("c"),fs.rename("f")],axis=1).dropna(); s=s[s.index>=dates[0]]
    print("  Yahoo daily FX relabeled (-1 day)",scan(s.c,s.f),"Sharpe",sharpe(s.c,s.f,sign))
