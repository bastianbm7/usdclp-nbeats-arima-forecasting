# Graficos de entrada/salida por operacion para los experimentos de holding
# multi-dia (Issue #9, N=2/3/5) y take-profit adaptativo (h1/h2/h3). Reusa
# las posiciones YA jugadas por cada agente (guardadas en los CSV de
# operaciones de 33/37) - NO reentrena nada, solo reconstruye el mecanismo de
# TP/SL/trailing con el registro de fecha de salida agregado en 32/36 (antes
# solo se guardaba COMO cerro cada operacion, no CUANDO dentro del bloque).

import importlib

import matplotlib.pyplot as plt
import pandas as pd

entorno_diario_mod = importlib.import_module("27_entorno_trading_rl_diario")
multidia_mod = importlib.import_module("32_entorno_trading_rl_diario_multidia")
adaptativo_mod = importlib.import_module("36_entorno_trading_rl_diario_tp_adaptativo")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")

RESULTADOS_DIR = "../datos/resultados"
N_WINDOWS_WF = 5
N_TEST_POR_VENTANA = 60

COLORES_RAZON = {"take_profit": "green", "stop_loss": "red", "trailing_stop": "purple", "cierre_periodo": "gray", "plano": "lightgray"}
ETIQUETAS_RAZON = {"take_profit": "Salida: take-profit", "stop_loss": "Salida: stop-loss", "trailing_stop": "Salida: trailing stop", "cierre_periodo": "Salida: fin de ventana"}


def reconstruir_operaciones(df_completo, csv_posiciones, dias_holding=None, adaptativo=False):
    # Reconstruye, ventana por ventana (mismos limites que el walk-forward
    # original), el dataframe de decisiones YA instrumentado con
    # fecha_salida_* - y le pega encima la posicion/razon/pnl que el agente
    # ya jugo (guardados en el CSV del backtest original), emparejando por
    # fecha de decision. No hace falta reentrenar PPO: la posicion elegida ya
    # esta grabada, solo falta la fecha exacta de salida.
    posiciones = pd.read_csv(csv_posiciones, parse_dates=["ds"])
    partes = []
    for df_train, df_test in wf_mod.ventanas_walkforward(df_completo, N_WINDOWS_WF, N_TEST_POR_VENTANA):
        if adaptativo:
            env = adaptativo_mod.USDCLPTradingEnvDiarioTPAdaptativo(df=df_test)
        else:
            env = multidia_mod.USDCLPTradingEnvDiarioMultidia(df=df_test, dias_holding=dias_holding)
        decisiones = env.df.copy()
        decisiones = decisiones.merge(posiciones[["ds", "posicion", "razon_cierre", "pnl"]], on="ds", how="inner")
        partes.append(decisiones)
    return pd.concat(partes, ignore_index=True)


def graficar_entrada_salida(df_operaciones, df_precio_fondo, titulo, path_salida):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df_precio_fondo["ds"], df_precio_fondo["y"], color="black", linewidth=1, label="USD/CLP (diario)", zorder=1)

    razones_presentes = set()
    for _, op in df_operaciones.iterrows():
        if op["posicion"] == 0:
            continue
        direccion = "largo" if op["posicion"] > 0 else "corto"
        marcador = "^" if op["posicion"] > 0 else "v"
        fecha_salida = op[f"fecha_salida_{direccion}"]
        precio_salida = op[f"precio_salida_{direccion}"]
        razon = op[f"razon_cierre_{direccion}"]
        razones_presentes.add(razon)
        color_salida = COLORES_RAZON.get(razon, "black")
        ax.plot([op["ds"], fecha_salida], [op["y"], precio_salida], color="lightsteelblue", linewidth=0.8, zorder=2)
        ax.scatter(op["ds"], op["y"], marker=marcador, color="blue", s=40, zorder=3)
        ax.scatter(fecha_salida, precio_salida, marker="o", color=color_salida, s=22, zorder=3)

    for etiqueta, color in [("Entrada larga (▲) / corta (▼)", "blue")] + [(ETIQUETAS_RAZON[r], COLORES_RAZON[r]) for r in ETIQUETAS_RAZON if r in razones_presentes]:
        ax.scatter([], [], color=color, label=etiqueta)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(titulo)
    ax.set_ylabel("USD/CLP")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # --- Issue #9: holding fijo N=2,3,5 (mismo dataset que 33) ---
    df_diario = entorno_diario_mod.cargar_dataset()
    fondo = df_diario[df_diario["ds"] >= pd.Timestamp("2025-06-01")][["ds", "y"]]

    for n in [2, 3, 5]:
        ops = reconstruir_operaciones(df_diario, f"{RESULTADOS_DIR}/walkforward_diario_multidia_ppo_h{n}_operaciones.csv", dias_holding=n)
        graficar_entrada_salida(ops, fondo, f"PPO holding={n}d: puntos de entrada/salida (Issue #9)",
                                 f"{RESULTADOS_DIR}/entrada_salida_multidia_h{n}.png")
        print(f"holding={n}: {len(ops)} operaciones graficadas -> entrada_salida_multidia_h{n}.png")

    # --- TP adaptativo (mismo dataset con h3 que 37) ---
    df_h3 = adaptativo_mod.cargar_dataset_h3()
    fondo_h3 = df_h3[df_h3["ds"] >= pd.Timestamp("2025-06-01")][["ds", "y"]]

    ops_base = reconstruir_operaciones(df_h3, f"{RESULTADOS_DIR}/walkforward_diario_tp_adaptativo_baseline_operaciones.csv", dias_holding=3)
    graficar_entrada_salida(ops_base, fondo_h3, "PPO baseline (TP=h1 siempre, holding=3d): entrada/salida",
                             f"{RESULTADOS_DIR}/entrada_salida_tp_baseline.png")
    print(f"tp_baseline: {len(ops_base)} operaciones graficadas -> entrada_salida_tp_baseline.png")

    ops_adap = reconstruir_operaciones(df_h3, f"{RESULTADOS_DIR}/walkforward_diario_tp_adaptativo_operaciones.csv", adaptativo=True)
    graficar_entrada_salida(ops_adap, fondo_h3, "PPO TP adaptativo (h1/h2/h3, holding=3d): entrada/salida",
                             f"{RESULTADOS_DIR}/entrada_salida_tp_adaptativo.png")
    print(f"tp_adaptativo: {len(ops_adap)} operaciones graficadas -> entrada_salida_tp_adaptativo.png")
