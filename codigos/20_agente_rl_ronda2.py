# Hoja de ruta del radar-baseline (2026-09-13), Tier 1: prueba 2 cambios
# independientes sobre el agente PPO, mas baratos que buscar variables nuevas
# porque no requieren datos externos (a diferencia de 19_features_nuevas_validacion.py,
# que SI encontro que rate_diff/copper/momentum no superan |r|=0.11 - esto se
# corre igual porque una red puede combinar features no-linealmente, ver nota
# en 11_entorno_trading_rl.py):
#
# 1. incluir_momentum=True: agrega mom_4s/mom_12s (retorno normalizado por vol
#    a 4 y 12 semanas) al estado.
# 2. modo_recompensa="dsr": Differential Sharpe Ratio (Moody & Saffell) en vez
#    de retorno % crudo o exceso_bh (ya probado en Issue #2).
#
# Mismo esquema que 17_agente_rl_mejoras.py: walk-forward de
# 14_backtest_walkforward_gestion_riesgo.py (5 ventanas x 20 semanas), mismo
# capital/riesgo/seed, comparable a la seccion 9.4 del paper y a la tabla de
# mejoras_rl_metricas.csv del Issue #2.

import importlib

import pandas as pd
from stable_baselines3 import PPO

entorno_mod = importlib.import_module("11_entorno_trading_rl")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")

RESULTADOS_DIR = "../datos/resultados"
REFERENCIA_CSV = f"{RESULTADOS_DIR}/mejoras_rl_metricas.csv"

CAPITAL_INICIAL = wf_mod.CAPITAL_INICIAL
TOTAL_TIMESTEPS_PPO = wf_mod.TOTAL_TIMESTEPS_PPO
SEED = wf_mod.SEED

CONFIGURACIONES = {
    "PPO + momentum": dict(slug="momentum", incluir_momentum=True, modo_recompensa="cruda"),
    "PPO + DSR reward": dict(slug="dsr", incluir_momentum=False, modo_recompensa="dsr"),
    "PPO + momentum + DSR": dict(slug="momentum_dsr", incluir_momentum=True, modo_recompensa="dsr"),
}


def posiciones_ppo(df_train, df_test, incluir_momentum, modo_recompensa):
    env_train = entorno_mod.USDCLPTradingEnv(df=df_train, incluir_momentum=incluir_momentum, modo_recompensa=modo_recompensa)
    modelo = PPO("MlpPolicy", env_train, seed=SEED, verbose=0)
    modelo.learn(total_timesteps=TOTAL_TIMESTEPS_PPO)

    env_test = entorno_mod.USDCLPTradingEnv(df=df_test, incluir_momentum=incluir_momentum, modo_recompensa=modo_recompensa)
    obs, _ = env_test.reset()
    posiciones = []
    terminado = False
    while not terminado:
        accion, _ = modelo.predict(obs, deterministic=True)
        obs, _, terminado, _, info = env_test.step(accion)
        posiciones.append(info["posicion"])
    return posiciones


def correr_config(nombre, incluir_momentum, modo_recompensa, df_completo, diario):
    capital = CAPITAL_INICIAL
    partes = []
    for w, (df_train, df_test) in enumerate(wf_mod.ventanas_walkforward(df_completo)):
        print(f"  [{nombre}] ventana {w + 1}/{wf_mod.N_WINDOWS_WF} (train={len(df_train)} semanas)...", flush=True)
        posiciones = posiciones_ppo(df_train, df_test, incluir_momentum, modo_recompensa)
        r = wf_mod.simular_con_gestion_riesgo(df_test, diario, posiciones, nombre, capital_inicial=capital)
        capital = r["capital"].iloc[-1]
        partes.append(r)
    return pd.concat(partes, ignore_index=True)


if __name__ == "__main__":
    diario = pd.read_csv(wf_mod.DATOS_LARGO, parse_dates=["ds"])

    filas_metricas = []
    for nombre, cfg in CONFIGURACIONES.items():
        df_completo = entorno_mod.cargar_dataset(incluir_momentum=cfg["incluir_momentum"])
        print(f"=== {nombre} (incluir_momentum={cfg['incluir_momentum']}, modo_recompensa={cfg['modo_recompensa']}, "
              f"{len(df_completo)} semanas) ===")
        resultado = correr_config(nombre, cfg["incluir_momentum"], cfg["modo_recompensa"], df_completo, diario)
        resultado.to_csv(f"{RESULTADOS_DIR}/ronda2_rl_{cfg['slug']}_operaciones.csv", index=False)
        metricas = wf_mod.calcular_metricas(resultado, nombre)
        filas_metricas.append(metricas)
        print(f"  -> retorno_total={metricas['retorno_total_%']:.1f}%  sharpe={metricas['sharpe_anualizado']}  operaciones={metricas['operaciones']}\n")

    referencia = pd.read_csv(REFERENCIA_CSV)
    referencia_otros = referencia[~referencia["estrategia"].isin(CONFIGURACIONES.keys())]

    tabla = pd.concat([pd.DataFrame(filas_metricas), referencia_otros], ignore_index=True)
    tabla = tabla.sort_values("sharpe_anualizado", ascending=False).reset_index(drop=True)
    tabla.to_csv(f"{RESULTADOS_DIR}/ronda2_rl_metricas.csv", index=False)

    print(f"\n=== Ronda 2 completa, comparado contra la tabla del Issue #2 (mejoras_rl_metricas.csv) ===\n")
    print(tabla.to_string(index=False))
