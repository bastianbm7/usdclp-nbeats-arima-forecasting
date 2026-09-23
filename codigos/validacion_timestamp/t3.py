# EVIDENCIA DE LA ERRATA (2026-09-23) - copia (rutas adaptadas) del script de
# validacion de timestamp de la auditoria. Se deja tal cual como registro; la
# version consolidada y reproducible que genera las tablas del paper es
# codigos/58_validacion_timestamp.py.
# Que hace: escaneo de rezagos commodity vs FX con Yahoo diario vs FRED H.10 (mediodia NY) y vs dolar observado BCCh (fecha de publicacion y reetiquetado a dia de transaccion).
# Datos: datos/bases/validacion_timestamp/ (fetch.py/t2.py/t3.py descargan; el resto lee cache).
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'datos', 'bases', 'validacion_timestamp'))
import pandas as pd, numpy as np, json, yfinance as yf, warnings; warnings.filterwarnings("ignore")
tick=["HG=F","CL=F","GC=F","PL=F","ZS=F","CLP=X","AUDUSD=X","NOK=X","ZAR=X","BRL=X","CAD=X","MXN=X"]
px=yf.download(tick,start="2010-01-01",end="2026-09-22",progress=False,auto_adjust=False)["Close"]
px.index=pd.to_datetime(px.index).tz_localize(None); px.to_csv("yahoo_daily_long.csv")
def fred(s):
    x=pd.read_csv(f"fred_{s}.csv"); x.columns=["ds","v"]; x["ds"]=pd.to_datetime(x.ds); x["v"]=pd.to_numeric(x.v,errors="coerce"); return x.dropna().set_index("ds").v
m=json.load(open("mindicador_dolar.json")); ob=pd.Series({pd.Timestamp(e["fecha"][:10]):e["valor"] for e in m}).sort_index()
ob=ob[~ob.index.duplicated()]
ob_rel=pd.Series(ob.values[1:],index=ob.index[:-1])  # valor publicado en t -> transacciones del dia habil anterior
def scan(cm,fx,start="2010-01-01"):
    df=pd.concat([cm.rename("c"),fx.rename("f")],axis=1).dropna(); df=df[df.index>=start]
    r=np.log(df).diff().dropna()
    return {k: round(r.c.corr(r.f.shift(-k)),3) for k in range(-2,4)}, len(r), r
def sh(r,sgn):
    p=(sgn*np.sign(r.c)*r.f.shift(-1)).dropna(); return round(p.mean()/p.std()*np.sqrt(252),2)
pairs=[("HG=F","AUD","DEXUSAL","AUDUSD=X"),("HG=F","CAD","DEXCAUS","CAD=X"),("HG=F","MXN","DEXMXUS","MXN=X"),
       ("CL=F","NOK","DEXNOUS","NOK=X"),("PL=F","ZAR","DEXSFUS","ZAR=X"),("GC=F","ZAR","DEXSFUS","ZAR=X"),("ZS=F","BRL","DEXBZUS","BRL=X"),("HG=F","BRL","DEXBZUS","BRL=X")]
for cm,name,fs,yt in pairs:
    f=fred(fs); ya=px[yt]
    s1,n1,r1=scan(px[cm],ya); s2,n2,r2=scan(px[cm],f)
    sg=np.sign(s2[0]) if abs(s2[0])>0.05 else np.sign(s1[1])
    print(f"{cm} vs {name}: Yahoo {yt} n={n1} {s1} Sharpe={sh(r1,np.sign(s1[1]))} | FRED {fs} (noon NY) n={n2} {s2} Sharpe(lag+1 strat)={sh(r2,sg)}")
cl=px["CLP=X"]
for lab,ser in [("Yahoo CLP=X",cl),("Observado fecha publicacion (raw)",ob),("Observado reetiquetado a dia de transaccion",ob_rel)]:
    s,n,r=scan(px["HG=F"],ser); print(f"HG=F vs {lab}: n={n} {s} Sharpe(-sign)={sh(r,-1)}")
# subperiod for yahoo CLP matching paper test window
s,n,r=scan(px["HG=F"],ob_rel,"2025-07-01"); print("Observado reetiq. 2025-07..:",n,s,sh(r,-1))
s,n,r=scan(px["HG=F"],cl,"2025-07-01"); print("Yahoo CLP 2025-07..:",n,s,sh(r,-1))
