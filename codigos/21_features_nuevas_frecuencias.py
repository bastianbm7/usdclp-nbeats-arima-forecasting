# Hoja de ruta del radar-baseline (2026-09-13), seguimiento a 9.11: el chequeo
# de correlacion de 19_features_nuevas_validacion.py se hizo a frecuencia
# SEMANAL (la que usa el agente RL) y ninguna de las 5 candidatas supero
# |r|=0.11. Pero el propio paper que ancla la hipotesis del cobre (Ferraro,
# Rogoff & Rossi 2015) encuentra el efecto a frecuencia DIARIA, no mensual -
# hay un mismatch de frecuencia sin resolver. Este script repite el MISMO
# chequeo (correlacion con el retorno futuro real) a diario y a mensual, para
# ver si el efecto aparece en alguna de esas dos escalas.
#
# Puramente exploratorio - no toca el agente RL ni el dataset semanal que usa.
# Reusa el cache de macro_tasas_cobre.csv de 19 (mismo cobre/tasas, ya tienen
# resolucion diaria/mensual real, no hace falta re-descargar).

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

features_mod = importlib.import_module("19_features_nuevas_validacion")

RESULTADOS_DIR = "../datos/resultados"
DATOS_LARGO = "../datos/bases/usdclp_long.csv"


def calcular_correlaciones_generico(df, cols_features, nombre_retorno_futuro="retorno_futuro"):
    filas = []
    for feature in cols_features:
        corr = df[feature].corr(df[nombre_retorno_futuro])
        acierto = (np.sign(df[feature]) == np.sign(df[nombre_retorno_futuro])).mean()
        filas.append({"feature": feature, "correlacion_con_retorno_futuro": corr, "acierto_direccion_%": 100 * acierto})
    return pd.DataFrame(filas).sort_values("correlacion_con_retorno_futuro", key=abs, ascending=False).reset_index(drop=True)


def preparar_diario(macro):
    diario = pd.read_csv(DATOS_LARGO, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)
    diario["retorno_futuro"] = (diario["y"].shift(-1) - diario["y"]) / diario["y"]
    diario = pd.merge_asof(diario, macro[["ds", "copper", "tasa_chile", "tasa_us"]], on="ds", direction="backward")
    diario["rate_diff"] = diario["tasa_chile"] - diario["tasa_us"]
    diario["copper_ret_1d"] = np.log(diario["copper"] / diario["copper"].shift(1))
    diario["copper_mom_5d"] = (diario["copper"] - diario["copper"].shift(5)) / diario["copper"].shift(5)
    diario["copper_mom_20d"] = (diario["copper"] - diario["copper"].shift(20)) / diario["copper"].shift(20)
    return diario.dropna().reset_index(drop=True)


def preparar_mensual(macro):
    diario = pd.read_csv(DATOS_LARGO, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)
    mensual = diario.set_index("ds")["y"].resample("ME").last().reset_index()
    mensual["retorno_futuro"] = (mensual["y"].shift(-1) - mensual["y"]) / mensual["y"]
    mensual = pd.merge_asof(mensual, macro[["ds", "copper", "tasa_chile", "tasa_us"]], on="ds", direction="backward")
    mensual["rate_diff"] = mensual["tasa_chile"] - mensual["tasa_us"]
    mensual["copper_ret_1m"] = np.log(mensual["copper"] / mensual["copper"].shift(1))
    mensual["copper_mom_3m"] = (mensual["copper"] - mensual["copper"].shift(3)) / mensual["copper"].shift(3)
    return mensual.dropna().reset_index(drop=True)


def graficar(tabla_semanal, tabla_diaria, tabla_mensual, path_salida):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    for ax, tabla, titulo in [(axes[0], tabla_semanal, "Semanal (la que usa el agente)"),
                               (axes[1], tabla_diaria, "Diaria"), (axes[2], tabla_mensual, "Mensual")]:
        colores = ["seagreen" if v > 0 else "firebrick" for v in tabla["correlacion_con_retorno_futuro"]]
        ax.barh(tabla["feature"], tabla["correlacion_con_retorno_futuro"], color=colores)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.axvline(0.11, color="gray", linewidth=0.6, linestyle="--")
        ax.axvline(-0.11, color="gray", linewidth=0.6, linestyle="--")
        ax.set_title(titulo)
        ax.invert_yaxis()
    fig.suptitle("Correlacion de cobre/tasas con el retorno futuro, por frecuencia (lineas grises = |r|=0.11)")
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    macro = features_mod.cargar_macro()

    diario = preparar_diario(macro)
    cols_diario = ["rate_diff", "copper_ret_1d", "copper_mom_5d", "copper_mom_20d"]
    tabla_diaria = calcular_correlaciones_generico(diario, cols_diario)
    print(f"=== Diario ({len(diario)} obs, {diario['ds'].min().date()} a {diario['ds'].max().date()}) ===")
    print(tabla_diaria.to_string(index=False))

    mensual = preparar_mensual(macro)
    cols_mensual = ["rate_diff", "copper_ret_1m", "copper_mom_3m"]
    tabla_mensual = calcular_correlaciones_generico(mensual, cols_mensual)
    print(f"\n=== Mensual ({len(mensual)} obs, {mensual['ds'].min().date()} a {mensual['ds'].max().date()}) ===")
    print(tabla_mensual.to_string(index=False))

    tabla_semanal = pd.read_csv(f"{RESULTADOS_DIR}/analisis_features_nuevas_correlacion.csv")
    tabla_semanal = tabla_semanal[tabla_semanal["feature"].isin(["rate_diff", "copper_ret_1s", "copper_mom_4s"])]

    tabla_diaria.to_csv(f"{RESULTADOS_DIR}/analisis_features_nuevas_correlacion_diaria.csv", index=False)
    tabla_mensual.to_csv(f"{RESULTADOS_DIR}/analisis_features_nuevas_correlacion_mensual.csv", index=False)
    graficar(tabla_semanal, tabla_diaria, tabla_mensual, f"{RESULTADOS_DIR}/analisis_features_nuevas_correlacion_frecuencias.png")

    print(f"\n=== Semanal (referencia, ya en 9.11) ===")
    print(tabla_semanal.to_string(index=False))
