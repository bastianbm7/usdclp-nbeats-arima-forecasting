# Chequeo de correlacion (sin RL, mismo patron que 19/21/24) para la idea de
# Bastian de agregar precios de la SEMANA CALENDARIO anterior (apertura,
# cierre, minimo, maximo) como features del agente diario - la hipotesis es
# que dan informacion de "tendencia semanal" que las features actuales
# (basadas en ventanas de dias, no semanas calendario) no capturan.
#
# Prior de este proyecto sobre features derivadas puramente del propio precio
# de USD/CLP: NINGUNA ha superado |r|=0.11 hasta ahora (MACD/RSI/momentum en
# 9.16, |r|<0.042; SMA/EMA/Bollinger/CCI/ADX en el Issue #4, |r|<0.076) - solo
# el cobre (una variable EXTERNA) lo supera. Cierre/apertura/min/max semanal
# es la misma familia de idea (transformacion del propio precio), pero nunca
# se probo en esta forma especifica (agregacion semanal vista desde el agente
# DIARIO) - se verifica en vez de asumir el resultado por analogia.
#
# Sin look-ahead: cada dia ve la semana calendario ANTERIOR completa (nunca
# la semana en curso, ni siquiera parcialmente) - se logra con un merge por
# "numero de semana + 1" en vez de un merge_asof por fecha mas cercana, que
# podria filtrar datos de la semana en curso para dias ya avanzados de esa
# semana.

import importlib

import numpy as np
import pandas as pd

features21_mod = importlib.import_module("21_features_nuevas_frecuencias")  # reusa calcular_correlaciones_generico()

DATOS_LARGO = "../datos/bases/usdclp_long.csv"
RESULTADOS_DIR = "../datos/resultados"


def preparar_dataset(path=DATOS_LARGO):
    diario = pd.read_csv(path, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)
    diario["retorno_futuro"] = (diario["y"].shift(-1) - diario["y"]) / diario["y"]

    # Semana calendario terminando en viernes (dia habil tipico de cierre FX).
    diario["semana"] = diario["ds"].dt.to_period("W-FRI")
    semanal = diario.groupby("semana")["y"].agg(apertura="first", cierre="last", minimo="min", maximo="max").reset_index()
    semanal["semana_objetivo"] = semanal["semana"] + 1  # la semana que va a VER estos valores como "semana anterior"

    diario = diario.merge(
        semanal[["semana_objetivo", "apertura", "cierre", "minimo", "maximo"]],
        left_on="semana", right_on="semana_objetivo", how="left",
    ).drop(columns=["semana_objetivo"])

    diario["sem_ant_cierre_rel"] = (diario["cierre"] - diario["y"]) / diario["y"]
    diario["sem_ant_apertura_rel"] = (diario["apertura"] - diario["y"]) / diario["y"]
    diario["sem_ant_min_rel"] = (diario["minimo"] - diario["y"]) / diario["y"]
    diario["sem_ant_max_rel"] = (diario["maximo"] - diario["y"]) / diario["y"]
    diario["sem_ant_retorno"] = (diario["cierre"] - diario["apertura"]) / diario["apertura"]  # tendencia de la semana anterior (sube/baja)
    diario["sem_ant_posicion_rango"] = (diario["y"] - diario["minimo"]) / (diario["maximo"] - diario["minimo"]).replace(0, np.nan)  # breakout: hoy, ¿arriba o abajo del rango de la semana pasada?

    return diario.dropna().reset_index(drop=True)


COLS_SEMANALES = ["sem_ant_cierre_rel", "sem_ant_apertura_rel", "sem_ant_min_rel", "sem_ant_max_rel", "sem_ant_retorno", "sem_ant_posicion_rango"]


if __name__ == "__main__":
    df = preparar_dataset()
    print(f"Dataset diario con features semanales: {len(df)} obs, {df['ds'].min().date()} a {df['ds'].max().date()}")

    tabla = features21_mod.calcular_correlaciones_generico(df, COLS_SEMANALES)
    tabla.to_csv(f"{RESULTADOS_DIR}/features_semanales_validacion_correlaciones.csv", index=False)

    print(f"\n=== Correlacion de precios de la semana calendario anterior con el retorno futuro diario ===")
    print(f"(referencia: copper_ret_1d=-0.256 en 9.13; umbral que nada derivado del propio precio ha superado hasta ahora: |r|=0.11)\n")
    print(tabla.to_string(index=False))

    supera_umbral = tabla[tabla["correlacion_con_retorno_futuro"].abs() > 0.11]
    if len(supera_umbral) > 0:
        print(f"\n{len(supera_umbral)} feature(s) SUPERAN |r|=0.11 - candidatas a agregar al estado del agente:")
        print(supera_umbral.to_string(index=False))
    else:
        print(f"\nNinguna supera |r|=0.11 - consistente con el patron de features derivadas del propio precio en este proyecto (9.16, Issue #4).")
