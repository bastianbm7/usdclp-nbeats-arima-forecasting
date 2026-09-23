# EVIDENCIA DE LA ERRATA (2026-09-23) - copia (rutas adaptadas) del script de
# validacion de timestamp de la auditoria. Se deja tal cual como registro; la
# version consolidada y reproducible que genera las tablas del paper es
# codigos/58_validacion_timestamp.py.
# Que hace: idem t3 sin red (usa cache): Yahoo CLP vs dolar observado reetiquetado, muestra completa y tramo de test 2025-07+.
# Datos: datos/bases/validacion_timestamp/ (fetch.py/t2.py/t3.py descargan; el resto lee cache).
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'datos', 'bases', 'validacion_timestamp'))
import pandas as pd, numpy as np, json
px=pd.read_csv("yahoo_daily_long.csv",index_col=0,parse_dates=True)
m=json.load(open("mindicador_dolar.json")); ob=pd.Series({pd.Timestamp(e["fecha"][:10]):e["valor"] for e in m}).sort_index()
ob_rel=pd.Series(ob.values[1:],index=ob.index[:-1])
cl=px["CLP=X"].copy(); cl[(cl<300)|(cl>1500)]=np.nan
def scan(cm,fx,start="2010-01-01",end="2100"):
    df=pd.concat([cm.rename("c"),fx.rename("f")],axis=1).dropna(); df=df[(df.index>=start)&(df.index<end)]
    r=np.log(df).diff().dropna(); p=(-np.sign(r.c)*r.f.shift(-1)).dropna()
    return {k: round(r.c.corr(r.f.shift(-k)),3) for k in range(-2,4)}, len(r), round(p.mean()/p.std()*np.sqrt(252),2)
for lab,ser in [("Yahoo CLP=X (limpio)",cl),("Observado raw",ob),("Observado reetiq.",ob_rel)]:
    for a,b in [("2010","2100"),("2025-07-01","2100")]:
        print(lab,a,scan(px["HG=F"],ser,a,b))
# Yahoo CLP vs observado: which lag of Yahoo matches observado-reetiq best
df=pd.concat([np.log(cl).diff().rename("y"),np.log(ob_rel).diff().rename("o")],axis=1).dropna()
print("corr Yahoo CLP ret(t+k) vs observado-reetiq ret(t):",{k:round(df.o.corr(df.y.shift(-k)),3) for k in range(-2,3)})
