# Cuanto aporta cada variable del estado del agente (Issue #1, Fase 4 -
# documentacion). No es un ablation de reentrenar N veces sin cada feature
# (caro, y el agente ya convergio a una politica constante en el walk-forward
# de riesgo real - un ablation ahi no diria nada nuevo). En cambio, se mide
# la correlacion de cada feature con el retorno semanal REAL siguiente -
# responde la pregunta mas basica y honesta: de las senales que le dimos al
# agente, ¿cuales efectivamente se relacionan con lo que pasa despues?

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

entorno_mod = importlib.import_module("11_entorno_trading_rl")

RESULTADOS_DIR = "../datos/resultados"


def calcular_correlaciones(df):
    retorno_futuro = (df["y_next"] - df["y"]) / df["y"]
    filas = []
    for feature in entorno_mod.FEATURES_ESTADO:
        corr = df[feature].corr(retorno_futuro)
        acierto_direccion = (np.sign(df[feature]) == np.sign(retorno_futuro)).mean()
        filas.append({"feature": feature, "correlacion_con_retorno_futuro": corr, "acierto_direccion_%": 100 * acierto_direccion})
    tabla = pd.DataFrame(filas).sort_values("correlacion_con_retorno_futuro", key=abs, ascending=False).reset_index(drop=True)
    return tabla


def graficar_correlaciones(tabla, path_salida):
    fig, ax = plt.subplots(figsize=(9, 5))
    colores = ["seagreen" if v > 0 else "firebrick" for v in tabla["correlacion_con_retorno_futuro"]]
    ax.barh(tabla["feature"], tabla["correlacion_con_retorno_futuro"], color=colores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlacion con el retorno semanal siguiente (y_next vs y)")
    ax.set_title("Cuanto se relaciona cada feature del estado con lo que pasa despues")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = entorno_mod.cargar_dataset()
    tabla = calcular_correlaciones(df)
    tabla.to_csv(f"{RESULTADOS_DIR}/analisis_features_correlacion.csv", index=False)
    graficar_correlaciones(tabla, f"{RESULTADOS_DIR}/analisis_features_correlacion.png")

    print(f"Analisis sobre {len(df)} semanas completas del dataset\n")
    print(tabla.to_string(index=False))
    print("\nNota: el agente de RL convergio a una politica constante (0 operaciones en las 5")
    print("ventanas del walk-forward de riesgo real) - en la practica, le asigno importancia")
    print("~nula a TODAS las features del estado, no solo a algunas. Esta tabla mide la senal")
    print("disponible en los datos, no lo que el agente termino usando.")
