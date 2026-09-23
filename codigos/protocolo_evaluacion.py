# Protocolo de evaluacion compartido (Issue #17, parte minima) para los
# scripts posteriores a la errata de 9.35 (Issues #14 y #15 en adelante).
#
# Por que existe: la errata mostro que >35 configuraciones se compararon sobre
# la misma ventana de test sin contar cuantas pruebas se hacian (9.35.5). Desde
# aca, dos reglas comunes:
#   1. HOLD-OUT INTOCABLE: nada con fecha >= HOLDOUT_INICIO se usa para
#      elegir variables, signos, umbrales ni hiperparametros. Se abre UNA sola
#      vez por estrategia candidata final, y esa apertura queda anotada.
#   2. REGISTRO DE PRUEBAS: cada variable/estrategia evaluada (pase o no) se
#      anota en datos/resultados/registro_pruebas.csv. El numero total de
#      pruebas alimenta la correccion por pruebas multiples
#      (costos_y_estadistica.sharpe_maximo_esperado_nulo; Sharpe deflactado en
#      el Issue #17).
import os
from datetime import datetime

import pandas as pd

HOLDOUT_INICIO = pd.Timestamp("2025-01-01")

_AQUI = os.path.dirname(os.path.abspath(__file__))
RUTA_REGISTRO = os.path.join(_AQUI, "..", "datos", "resultados", "registro_pruebas.csv")
COLUMNAS_REGISTRO = ["fecha_registro", "issue", "script", "prueba", "frecuencia",
                     "metrica", "valor", "n", "usa_holdout", "nota"]


def antes_del_holdout(df, col_fecha="ds"):
    """Filas estrictamente anteriores al hold-out (lo unico que se puede mirar
    para seleccionar)."""
    return df[pd.to_datetime(df[col_fecha]) < HOLDOUT_INICIO].copy()


def registrar_pruebas(filas, issue, script, reemplazar_script=True):
    """Anota pruebas en el registro comun. `filas`: lista de dicts con las
    claves de COLUMNAS_REGISTRO (las faltantes quedan vacias). Si
    reemplazar_script=True, borra antes las filas previas del mismo script,
    para que re-correr un script no duplique su conteo de pruebas."""
    nuevo = pd.DataFrame(filas)
    nuevo["fecha_registro"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    nuevo["issue"] = issue
    nuevo["script"] = script
    for c in COLUMNAS_REGISTRO:
        if c not in nuevo:
            nuevo[c] = None
    nuevo = nuevo[COLUMNAS_REGISTRO]
    if os.path.exists(RUTA_REGISTRO):
        previo = pd.read_csv(RUTA_REGISTRO)
        if reemplazar_script:
            previo = previo[previo["script"] != script]
        nuevo = pd.concat([previo, nuevo], ignore_index=True)
    nuevo.to_csv(RUTA_REGISTRO, index=False)
    return nuevo


def n_pruebas_registradas(issue=None):
    if not os.path.exists(RUTA_REGISTRO):
        return 0
    reg = pd.read_csv(RUTA_REGISTRO)
    if issue is not None:
        reg = reg[reg["issue"] == issue]
    return len(reg)
