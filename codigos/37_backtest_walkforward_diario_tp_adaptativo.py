# Compara, sobre el MISMO dataset (dataset_entrenamiento_rl_diario_h3.csv,
# generado por 35 - necesario para tener nhits_h3 y comparar apples-to-apples,
# ya que NHITS no tiene semilla fija y el dataset original de 26/27 no sirve
# como baseline exacto aca): un holding fijo de 3 dias con take-profit SIEMPRE
# en h1 (el mismo comportamiento de 9.17-9.20, solo que sobre datos
# reentrenados) contra la version con take-profit ADAPTATIVO de
# 36_entorno_trading_rl_diario_tp_adaptativo.py (h1/h2/h3 segun consistencia
# de la trayectoria del forecast).
#
# Pregunta que responde: si se le da al mecanismo de salida (no al agente)
# la posibilidad de apuntar mas lejos cuando el propio forecast de NHITS
# sugiere que vale la pena, ¿mejora sobre el TP fijo en h1 de siempre?

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

adaptativo_mod = importlib.import_module("36_entorno_trading_rl_diario_tp_adaptativo")
multidia_mod = importlib.import_module("32_entorno_trading_rl_diario_multidia")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")  # reusa ventanas_walkforward()
kelly_diario_mod = importlib.import_module("22_kelly_diario_cobre")  # reusa posiciones_umbral_cobre()

RESULTADOS_DIR = "../datos/resultados"

CAPITAL_INICIAL = 100.0
RIESGO_MAX_PCT = multidia_mod.RIESGO_MAX_PCT
K_STOP_LOSS = multidia_mod.K_STOP_LOSS
SLIPPAGE_PCT = multidia_mod.SLIPPAGE_PCT

N_WINDOWS_WF = 5
N_TEST_POR_VENTANA = 60  # mismo tamano de ventana de calendario que 9.20
DIAS_HOLDING = adaptativo_mod.DIAS_HOLDING  # 3, fijo
TOTAL_TIMESTEPS_PPO = 100_000
SEED = 42


def posiciones_ppo_baseline(df_train, df_test):
    env_train = multidia_mod.USDCLPTradingEnvDiarioMultidia(df=df_train, dias_holding=DIAS_HOLDING)
    modelo = PPO("MlpPolicy", env_train, seed=SEED, verbose=0)
    modelo.learn(total_timesteps=TOTAL_TIMESTEPS_PPO)

    env_test = multidia_mod.USDCLPTradingEnvDiarioMultidia(df=df_test, dias_holding=DIAS_HOLDING)
    obs, _ = env_test.reset()
    posiciones = []
    terminado = False
    while not terminado:
        accion, _ = modelo.predict(obs, deterministic=True)
        posiciones.append(multidia_mod.POSICION_POR_ACCION[int(accion)])
        obs, _, terminado, _, _ = env_test.step(accion)
    return posiciones, env_test.df


def posiciones_ppo_adaptativo(df_train, df_test):
    env_train = adaptativo_mod.USDCLPTradingEnvDiarioTPAdaptativo(df=df_train)
    modelo = PPO("MlpPolicy", env_train, seed=SEED, verbose=0)
    modelo.learn(total_timesteps=TOTAL_TIMESTEPS_PPO)

    env_test = adaptativo_mod.USDCLPTradingEnvDiarioTPAdaptativo(df=df_test)
    obs, _ = env_test.reset()
    posiciones = []
    terminado = False
    while not terminado:
        accion, _ = modelo.predict(obs, deterministic=True)
        posiciones.append(adaptativo_mod.POSICION_POR_ACCION[int(accion)])
        obs, _, terminado, _, _ = env_test.step(accion)
    return posiciones, env_test.df


def simular_con_gestion_riesgo(df_decisiones, posiciones, nombre, capital_inicial):
    capital = capital_inicial
    posicion_previa = 0.0
    filas = []
    for i, fila in df_decisiones.reset_index(drop=True).iterrows():
        posicion = posiciones[i]
        entrada = fila["y"]
        if posicion == 0:
            notional, pnl, razon, precio_salida = 0.0, 0.0, "plano", entrada
        else:
            direccion = "largo" if posicion > 0 else "corto"
            distancia_riesgo = K_STOP_LOSS * fila["vol_garch"]
            precio_salida, razon = fila[f"precio_salida_{direccion}"], fila[f"razon_cierre_{direccion}"]
            notional = (RIESGO_MAX_PCT * capital) / distancia_riesgo if distancia_riesgo > 0 else 0.0
            retorno_pct = posicion * (precio_salida - entrada) / entrada
            costo_slippage = SLIPPAGE_PCT * notional if posicion != posicion_previa else 0.0
            pnl = notional * retorno_pct - costo_slippage
        capital += pnl
        posicion_previa = posicion
        filas.append({"ds": fila["ds"], "estrategia": nombre, "posicion": posicion, "notional": notional, "razon_cierre": razon, "pnl": pnl, "capital": capital})
    return pd.DataFrame(filas)


def calcular_metricas(resultado, nombre):
    capital_previo = resultado["capital"].shift(1).fillna(CAPITAL_INICIAL)
    r = resultado["pnl"] / capital_previo
    capital = resultado["capital"]
    retorno_total_pct = 100 * (capital.iloc[-1] / CAPITAL_INICIAL - 1)
    periodos_por_anio = 252 / DIAS_HOLDING  # mismos DIAS_HOLDING para baseline y adaptativo - directamente comparable
    sharpe = (r.mean() / r.std()) * np.sqrt(periodos_por_anio) if r.std() > 0 else np.nan
    drawdown = capital / capital.cummax() - 1
    operaciones = (resultado["posicion"] != 0).sum()
    win_rate = 100 * (resultado.loc[resultado["posicion"] != 0, "pnl"] > 0).mean() if operaciones > 0 else np.nan
    return {"estrategia": nombre, "n_decisiones": len(resultado), "capital_final": capital.iloc[-1], "retorno_total_%": retorno_total_pct,
            "sharpe_anualizado": sharpe, "max_drawdown_%": 100 * drawdown.min(), "win_rate_%": win_rate, "operaciones": operaciones}


def graficar(curvas, path_salida):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    colores = {"PPO baseline (TP=h1 siempre)": "steelblue", "PPO TP adaptativo (h1/h2/h3)": "crimson", "Umbral cobre": "darkorange"}
    for nombre, r in curvas.items():
        ax.plot(r["ds"], r["capital"], label=nombre, color=colores.get(nombre), linewidth=1.6)
    ax.axhline(CAPITAL_INICIAL, color="gray", linestyle=":", linewidth=1, label="Capital inicial ($100)")
    ax.set_ylabel("Capital ($)")
    ax.set_title(f"Take-profit adaptativo (h1/h2/h3) vs. TP fijo en h1 - holding={DIAS_HOLDING}d")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df_completo = adaptativo_mod.cargar_dataset_h3()
    print(f"Dataset diario (con h3): {len(df_completo)} dias, {df_completo['ds'].min().date()} a {df_completo['ds'].max().date()}")
    print(f"Walk-forward: {N_WINDOWS_WF} ventanas x {N_TEST_POR_VENTANA} dias de calendario, holding fijo={DIAS_HOLDING}d\n")

    capital_base, capital_adap, capital_cobre = CAPITAL_INICIAL, CAPITAL_INICIAL, CAPITAL_INICIAL
    partes_base, partes_adap, partes_cobre = [], [], []
    horizontes_todos = []

    for w, (df_train, df_test) in enumerate(wf_mod.ventanas_walkforward(df_completo, N_WINDOWS_WF, N_TEST_POR_VENTANA)):
        pos_base, df_dec_base = posiciones_ppo_baseline(df_train, df_test)
        r_base = simular_con_gestion_riesgo(df_dec_base, pos_base, "PPO baseline (TP=h1 siempre)", capital_base)
        capital_base = r_base["capital"].iloc[-1]
        partes_base.append(r_base)

        pos_adap, df_dec_adap = posiciones_ppo_adaptativo(df_train, df_test)
        r_adap = simular_con_gestion_riesgo(df_dec_adap, pos_adap, "PPO TP adaptativo (h1/h2/h3)", capital_adap)
        capital_adap = r_adap["capital"].iloc[-1]
        partes_adap.append(r_adap)
        horizontes_todos.append(df_dec_adap["horizonte_tp_elegido"])

        pos_cobre = list(kelly_diario_mod.posiciones_umbral_cobre(df_dec_base))
        r_cobre = simular_con_gestion_riesgo(df_dec_base, pos_cobre, "Umbral cobre", capital_cobre)
        capital_cobre = r_cobre["capital"].iloc[-1]
        partes_cobre.append(r_cobre)

        print(f"  ventana {w+1}/{N_WINDOWS_WF}: baseline=${capital_base:.2f}, adaptativo=${capital_adap:.2f}, umbral cobre=${capital_cobre:.2f}")

    r_base_concat = pd.concat(partes_base, ignore_index=True)
    r_adap_concat = pd.concat(partes_adap, ignore_index=True)
    r_cobre_concat = pd.concat(partes_cobre, ignore_index=True)
    horizontes_concat = pd.concat(horizontes_todos, ignore_index=True)

    r_base_concat.to_csv(f"{RESULTADOS_DIR}/walkforward_diario_tp_adaptativo_baseline_operaciones.csv", index=False)
    r_adap_concat.to_csv(f"{RESULTADOS_DIR}/walkforward_diario_tp_adaptativo_operaciones.csv", index=False)
    r_cobre_concat.to_csv(f"{RESULTADOS_DIR}/walkforward_diario_tp_adaptativo_cobre_operaciones.csv", index=False)

    tabla = pd.DataFrame([
        calcular_metricas(r_base_concat, "PPO baseline (TP=h1 siempre)"),
        calcular_metricas(r_adap_concat, "PPO TP adaptativo (h1/h2/h3)"),
        calcular_metricas(r_cobre_concat, "Umbral cobre"),
    ]).sort_values("sharpe_anualizado", ascending=False).reset_index(drop=True)
    tabla.to_csv(f"{RESULTADOS_DIR}/walkforward_diario_tp_adaptativo_metricas.csv", index=False)

    distribucion_horizonte = horizontes_concat.value_counts().sort_index()
    distribucion_horizonte.to_csv(f"{RESULTADOS_DIR}/walkforward_diario_tp_adaptativo_distribucion_horizonte.csv")

    graficar({"PPO baseline (TP=h1 siempre)": r_base_concat, "PPO TP adaptativo (h1/h2/h3)": r_adap_concat, "Umbral cobre": r_cobre_concat},
             f"{RESULTADOS_DIR}/walkforward_diario_tp_adaptativo_curva_capital.png")

    print("\n=== Metricas ===")
    print(tabla.to_string(index=False))
    print("\n=== Distribucion de horizonte de TP elegido (solo variante adaptativa) ===")
    print(distribucion_horizonte)
