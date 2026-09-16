# Graficos pedidos por Bastian sobre la grilla del Issue #10, fijando h4 y
# comparando los 3 valores de N (3, 5, 7):
# 1. Curva de retorno % en el tiempo, una linea por N.
# 2. Puntos de entrada/salida (take-profit/stop-loss/trailing-stop/fin de
#    ventana) por N - reusa reconstruir_operaciones()/graficar_entrada_salida()
#    de 38, generalizado con horizonte_tp (agregado ahi mismo para esto).

import importlib

import matplotlib.pyplot as plt
import pandas as pd

graficos38_mod = importlib.import_module("38_graficos_entrada_salida_multidia")

RESULTADOS_DIR = "../datos/resultados"
H_FIJO = 4
VALORES_N = [3, 5, 7]
COLORES_N = {3: "steelblue", 5: "mediumseagreen", 7: "crimson"}


def graficar_retornos_en_el_tiempo(path_salida):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for N in VALORES_N:
        r = pd.read_csv(f"{RESULTADOS_DIR}/grilla_nh_N{N}_h{H_FIJO}_operaciones.csv", parse_dates=["ds"])
        retorno_pct = 100 * (r["capital"] / 100.0 - 1)
        ax.plot(r["ds"], retorno_pct, label=f"N={N}", color=COLORES_N[N], linewidth=1.8)
    ax.axhline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_ylabel("Retorno acumulado (%)")
    ax.set_xlabel("Fecha")
    ax.set_title(f"Grilla Issue #10 - h{H_FIJO} fijo, comparando ventana de holding (N)")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    graficar_retornos_en_el_tiempo(f"{RESULTADOS_DIR}/grilla_nh_h{H_FIJO}_retornos_por_N.png")
    print(f"Grafico de retornos -> grilla_nh_h{H_FIJO}_retornos_por_N.png")

    adaptativo_mod = importlib.import_module("36_entorno_trading_rl_diario_tp_adaptativo")  # reusa cargar_dataset_h3 como referencia de patron
    entorno_diario_mod = importlib.import_module("27_entorno_trading_rl_diario")
    df_h5 = entorno_diario_mod.cargar_dataset(f"{RESULTADOS_DIR}/dataset_entrenamiento_rl_diario_h5.csv")
    fondo = df_h5[df_h5["ds"] >= pd.Timestamp("2025-06-01")][["ds", "y"]]

    for N in VALORES_N:
        ops = graficos38_mod.reconstruir_operaciones(
            df_h5, f"{RESULTADOS_DIR}/grilla_nh_N{N}_h{H_FIJO}_operaciones.csv", dias_holding=N, horizonte_tp=H_FIJO)
        graficos38_mod.graficar_entrada_salida(
            ops, fondo, f"Grilla Issue #10: N={N}, h={H_FIJO} - entrada/salida por operacion",
            f"{RESULTADOS_DIR}/grilla_nh_N{N}_h{H_FIJO}_entrada_salida.png")
        print(f"N={N} h={H_FIJO}: {len(ops)} operaciones graficadas -> grilla_nh_N{N}_h{H_FIJO}_entrada_salida.png")
