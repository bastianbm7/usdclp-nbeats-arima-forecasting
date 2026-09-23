# EVIDENCIA DE LA ERRATA (2026-09-23) - copia (rutas adaptadas) del script de
# validacion de timestamp de la auditoria. Se deja tal cual como registro; la
# version consolidada y reproducible que genera las tablas del paper es
# codigos/58_validacion_timestamp.py.
# Que hace: descarga FRED H.10 (DEXUSAL, DEXNOUS, DEXSFUS, DEXBZUS, DEXCAUS, DEXMXUS) y el dolar observado del BCCh via mindicador.cl.
# Datos: datos/bases/validacion_timestamp/ (fetch.py/t2.py/t3.py descargan; el resto lee cache).
import os as _os
_os.chdir(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'datos', 'bases', 'validacion_timestamp'))
import requests, json, time
for s in ["DEXUSAL","DEXNOUS","DEXSFUS","DEXBZUS","DEXCAUS","DEXMXUS"]:
    r=requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={s}",timeout=60); open(f"fred_{s}.csv","w").write(r.text); print(s,r.status_code,len(r.text))
allv=[]
for y in range(2010,2027):
    r=requests.get(f"https://mindicador.cl/api/dolar/{y}",timeout=60)
    try: allv+=r.json()["serie"]; print(y,r.status_code,len(r.json()["serie"]))
    except Exception as e: print(y,r.status_code,e)
    time.sleep(0.5)
json.dump(allv,open("mindicador_dolar.json","w"))
