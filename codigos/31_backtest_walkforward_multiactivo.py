# Issue #6: walk-forward final - entrena una politica PPO MULTI-ACTIVO
# (30_entorno_trading_rl_multiactivo.py) sobre USD/CLP + USD/MXN + USD/BRL +
# USD/COP simultaneamente, evalua deterministicamente sobre el test set de
# CLP en cada ventana (mismo esquema de walk-forward que 14/17/18/20/22: 5
# ventanas x 20 semanas), y compara contra el agente solo-CLP.
#
# El agente solo-CLP NO se reentrena aca - se reusa la fila ya versionada
# "PPO (RL)" de walkforward_gestion_riesgo_metricas.csv (mismo criterio que
# 18_kelly_validacion.py y 20_agente_rl_ronda2.py, que reusan esa misma
# referencia en vez de reentrenar cada vez): con SEED=42 fijo y el mismo
# codigo de 11_entorno_trading_rl.py (sin tocar en este Issue), reentrenar
# reproduciria esos numeros identicos (ya verificado a 6 decimales en el
# Issue #2, ver NOTAS-CLAUDE.md) - reentrenarlo de nuevo seria ~15-20 min de
# computo redundante sin agregar informacion nueva.
#
# Para los pares nuevos (MXN/BRL/COP) en cada ventana se usa TODA su historia
# hasta la MISMA fecha de corte que el train de CLP de esa ventana (sin
# look-ahead) - mismo patron ya usado y revisado en
# 23_dataset_multi_par_diario.py (entrenar_pooled_vs_single).
#
# Presupuesto de entrenamiento IGUAL al agente solo-CLP (100k timesteps por
# ventana) a proposito, no 4x mas: la pregunta que responde este script es
# "¿ayuda diversificar el MISMO presupuesto de computo entre 4 pares en vez
# de gastarlo todo en CLP?", no "¿ayuda si se le da mucho mas computo?" - ver
# nota de limitaciones en la seccion 9.18 del paper. Con muestreo uniforme
# entre 4 pares, el agente multi-activo ve en promedio ~25k pasos "de CLP"
# por ventana, contra 100k del agente solo-CLP - un trade-off inherente al
# diseno, no un descuido.

import importlib

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO

entorno_mod = importlib.import_module("11_entorno_trading_rl")
multi_env_mod = importlib.import_module("30_entorno_trading_rl_multiactivo")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")

RESULTADOS_DIR = "../datos/resultados"
BASES_DIR = "../datos/bases"

N_WINDOWS_WF = wf_mod.N_WINDOWS_WF  # 5
N_TEST_POR_VENTANA = wf_mod.N_TEST_POR_VENTANA  # 20
TOTAL_TIMESTEPS_PPO = wf_mod.TOTAL_TIMESTEPS_PPO  # 100_000, mismo presupuesto que el agente solo-CLP (ver nota arriba)
SEED = wf_mod.SEED  # 42
CAPITAL_INICIAL = wf_mod.CAPITAL_INICIAL
NOMBRE_ESTRATEGIA = "PPO multi-activo (CLP+MXN+BRL+COP)"

PARES_NUEVOS = ["USDMXN", "USDBRL", "USDCOP"]
PARES_CONFIG = {
    "USDCLP": {"dataset": entorno_mod.DATASET_PATH, "diario": entorno_mod.DATOS_DIARIOS_PATH},
    "USDMXN": {"dataset": f"{RESULTADOS_DIR}/dataset_entrenamiento_rl_usdmxn.csv", "diario": f"{BASES_DIR}/usdmxn_long.csv"},
    "USDBRL": {"dataset": f"{RESULTADOS_DIR}/dataset_entrenamiento_rl_usdbrl.csv", "diario": f"{BASES_DIR}/usdbrl_long.csv"},
    "USDCOP": {"dataset": f"{RESULTADOS_DIR}/dataset_entrenamiento_rl_usdcop.csv", "diario": f"{BASES_DIR}/usdcop_long.csv"},
}


def cargar_todos_los_pares():
    return {par: entorno_mod.cargar_dataset(cfg["dataset"]) for par, cfg in PARES_CONFIG.items()}


def entrenar_multiactivo(dfs_train_por_par, diario_paths, seed=SEED):
    env_train = multi_env_mod.MultiFXTradingEnv(dfs_train_por_par, diario_paths, modo_muestreo="aleatorio", seed=seed)
    modelo = PPO("MlpPolicy", env_train, seed=seed, verbose=0)
    modelo.learn(total_timesteps=TOTAL_TIMESTEPS_PPO)
    return modelo


def posiciones_multiactivo_para_clp(modelo, dfs_eval_por_par, diario_paths):
    # par_fijo="USDCLP": el one-hot sigue teniendo dimension = numero de pares
    # de ENTRENAMIENTO (dfs_eval_por_par debe seguir trayendo los 4 pares, no
    # solo CLP) - si no, la politica entrenada con un one-hot de 4 posiciones
    # recibiria un vector de forma distinta al evaluar y fallaria.
    env_eval = multi_env_mod.MultiFXTradingEnv(dfs_eval_por_par, diario_paths, par_fijo="USDCLP")
    obs, _ = env_eval.reset()
    posiciones = []
    terminado = False
    while not terminado:
        accion, _ = modelo.predict(obs, deterministic=True)
        posiciones.append(entorno_mod.POSICION_POR_ACCION[int(accion)])
        obs, _, terminado, _, _ = env_eval.step(accion)
    return posiciones


def graficar_comparacion(resultado_multi, ppo_solo_clp, path_salida):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(resultado_multi["ds"], resultado_multi["capital"], label=NOMBRE_ESTRATEGIA, color="steelblue", linewidth=1.8)
    ax.plot(ppo_solo_clp["ds"], ppo_solo_clp["capital"], label="PPO solo-CLP (referencia, seccion 9.4/9.9)", color="crimson", linewidth=1.8, linestyle="--")
    ax.axhline(CAPITAL_INICIAL, color="gray", linestyle=":", linewidth=1, label="Capital inicial ($100)")
    ax.set_ylabel("Capital ($)")
    ax.set_title("Agente PPO multi-activo (CLP+MXN+BRL+COP) vs. solo-CLP, evaluado sobre el mismo walk-forward de CLP")
    ax.legend(fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    dfs = cargar_todos_los_pares()
    for par, df in dfs.items():
        print(f"{par}: {len(df)} semanas, {df['ds'].min().date()} a {df['ds'].max().date()}")

    diario_clp = pd.read_csv(wf_mod.DATOS_LARGO, parse_dates=["ds"])
    diario_paths = {par: cfg["diario"] for par, cfg in PARES_CONFIG.items()}

    capital_multi = CAPITAL_INICIAL
    partes_multi = []
    for w, (clp_train, clp_test) in enumerate(wf_mod.ventanas_walkforward(dfs["USDCLP"])):
        fecha_corte = clp_test["ds"].iloc[0]
        print(f"\n--- Ventana {w + 1}/{N_WINDOWS_WF}: train CLP={len(clp_train)} semanas, test={clp_test['ds'].min().date()} a {clp_test['ds'].max().date()} ---")

        dfs_train = {"USDCLP": clp_train}
        for par in PARES_NUEVOS:
            dfs_train[par] = dfs[par][dfs[par]["ds"] < fecha_corte].reset_index(drop=True)
            print(f"  {par}: {len(dfs_train[par])} semanas de entrenamiento hasta {fecha_corte.date()} (sin look-ahead)")

        modelo = entrenar_multiactivo(dfs_train, diario_paths, seed=SEED)

        dfs_eval = dict(dfs_train)
        dfs_eval["USDCLP"] = clp_test
        posiciones = posiciones_multiactivo_para_clp(modelo, dfs_eval, diario_paths)

        r = wf_mod.simular_con_gestion_riesgo(clp_test, diario_clp, posiciones, NOMBRE_ESTRATEGIA, capital_inicial=capital_multi)
        capital_multi = r["capital"].iloc[-1]
        partes_multi.append(r)
        print(f"  Capital al cierre de la ventana: ${capital_multi:.2f} ({(r['posicion'] != 0).sum()} operaciones)")

    resultado_multi = pd.concat(partes_multi, ignore_index=True)
    resultado_multi.to_csv(f"{RESULTADOS_DIR}/walkforward_multiactivo_operaciones.csv", index=False)

    metricas_multi = wf_mod.calcular_metricas(resultado_multi, NOMBRE_ESTRATEGIA)
    referencia = pd.read_csv(f"{RESULTADOS_DIR}/walkforward_gestion_riesgo_metricas.csv")
    tabla = pd.concat([pd.DataFrame([metricas_multi]), referencia], ignore_index=True)
    tabla = tabla.sort_values("sharpe_anualizado", ascending=False, na_position="last").reset_index(drop=True)
    tabla.to_csv(f"{RESULTADOS_DIR}/walkforward_multiactivo_metricas.csv", index=False)

    ppo_solo_clp = pd.read_csv(f"{RESULTADOS_DIR}/walkforward_ppo_operaciones.csv", parse_dates=["ds"])
    graficar_comparacion(resultado_multi, ppo_solo_clp, f"{RESULTADOS_DIR}/walkforward_multiactivo_curva_capital.png")

    print(f"\n=== Walk-forward multi-activo completo: {N_WINDOWS_WF * N_TEST_POR_VENTANA} semanas de test (out-of-sample) de USD/CLP ===\n")
    print(tabla.to_string(index=False))
