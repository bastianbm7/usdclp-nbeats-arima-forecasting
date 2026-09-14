# Issue #2: prueba 3 cambios independientes sobre el agente PPO de
# 11_entorno_trading_rl.py, frente al hallazgo de la seccion 9.4-9.5 del paper
# (con la recompensa real, PPO aprende a no operar nunca en las 5 ventanas del
# walk-forward - respuesta racional a que ninguna de las 7 variables del
# estado supera |r|=0.11 de correlacion con el retorno futuro):
#
# 1. mas exploracion en PPO (ent_coef mas alto que el default de sb3)
# 2. accion continua (tamano de posicion, no solo largo/plano/corto)
# 3. recompensa como exceso sobre buy-and-hold en vez de retorno % crudo
#
# Independientes entre si -> se evaluan cada uno por separado y combinados,
# todos contra el MISMO walk-forward de 14_backtest_walkforward_gestion_riesgo.py
# (5 ventanas x 20 semanas, mismo capital/riesgo/seed) para ser comparables con
# la tabla de referencia de la seccion 9.4 del paper - nunca un cambio evaluado
# en aislado.
#
# La fila "PPO base" reusa los parametros default de USDCLPTradingEnv
# (accion_continua=False, modo_recompensa="cruda") y con el mismo SEED
# reproduce exactamente los resultados ya versionados en
# walkforward_gestion_riesgo_metricas.csv (Buy-and-hold y Umbral simple no
# dependen de estos cambios, se leen de ese CSV en vez de recalcularse).

import importlib

import pandas as pd
from stable_baselines3 import PPO

entorno_mod = importlib.import_module("11_entorno_trading_rl")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")

RESULTADOS_DIR = "../datos/resultados"
REFERENCIA_CSV = f"{RESULTADOS_DIR}/walkforward_gestion_riesgo_metricas.csv"

CAPITAL_INICIAL = wf_mod.CAPITAL_INICIAL
TOTAL_TIMESTEPS_PPO = wf_mod.TOTAL_TIMESTEPS_PPO
SEED = wf_mod.SEED
ENT_COEF_ALTO = 0.02

CONFIGURACIONES = {
    "PPO base": dict(slug="base", ent_coef=0.0, accion_continua=False, modo_recompensa="cruda"),
    "PPO + ent_coef alto": dict(slug="entcoef", ent_coef=ENT_COEF_ALTO, accion_continua=False, modo_recompensa="cruda"),
    "PPO + accion continua": dict(slug="continua", ent_coef=0.0, accion_continua=True, modo_recompensa="cruda"),
    "PPO + exceso sobre buy-and-hold": dict(slug="excesobh", ent_coef=0.0, accion_continua=False, modo_recompensa="exceso_bh"),
    "PPO + las 3 combinadas": dict(slug="combinado", ent_coef=ENT_COEF_ALTO, accion_continua=True, modo_recompensa="exceso_bh"),
}


def posiciones_ppo(df_train, df_test, ent_coef, accion_continua, modo_recompensa):
    env_train = entorno_mod.USDCLPTradingEnv(df=df_train, accion_continua=accion_continua, modo_recompensa=modo_recompensa)
    modelo = PPO("MlpPolicy", env_train, seed=SEED, ent_coef=ent_coef, verbose=0)
    modelo.learn(total_timesteps=TOTAL_TIMESTEPS_PPO)

    env_test = entorno_mod.USDCLPTradingEnv(df=df_test, accion_continua=accion_continua, modo_recompensa=modo_recompensa)
    obs, _ = env_test.reset()
    posiciones = []
    terminado = False
    while not terminado:
        accion, _ = modelo.predict(obs, deterministic=True)
        obs, _, terminado, _, info = env_test.step(accion)
        posiciones.append(info["posicion"])
    return posiciones


def correr_config(nombre, ent_coef, accion_continua, modo_recompensa, df_completo, diario):
    capital = CAPITAL_INICIAL
    partes = []
    for w, (df_train, df_test) in enumerate(wf_mod.ventanas_walkforward(df_completo)):
        print(f"  [{nombre}] ventana {w + 1}/{wf_mod.N_WINDOWS_WF} (train={len(df_train)} semanas)...", flush=True)
        posiciones = posiciones_ppo(df_train, df_test, ent_coef, accion_continua, modo_recompensa)
        r = wf_mod.simular_con_gestion_riesgo(df_test, diario, posiciones, nombre, capital_inicial=capital)
        capital = r["capital"].iloc[-1]
        partes.append(r)
    return pd.concat(partes, ignore_index=True)


if __name__ == "__main__":
    diario = pd.read_csv(wf_mod.DATOS_LARGO, parse_dates=["ds"])
    df_completo = entorno_mod.cargar_dataset()
    print(f"Dataset: {len(df_completo)} semanas. Walk-forward: {wf_mod.N_WINDOWS_WF} ventanas x {wf_mod.N_TEST_POR_VENTANA} semanas "
          f"(mismo esquema de 14, comparable a la seccion 9.4 del paper)\n")

    filas_metricas = []
    for nombre, cfg in CONFIGURACIONES.items():
        print(f"=== {nombre} (ent_coef={cfg['ent_coef']}, accion_continua={cfg['accion_continua']}, modo_recompensa={cfg['modo_recompensa']}) ===")
        resultado = correr_config(nombre, cfg["ent_coef"], cfg["accion_continua"], cfg["modo_recompensa"], df_completo, diario)
        resultado.to_csv(f"{RESULTADOS_DIR}/mejoras_rl_{cfg['slug']}_operaciones.csv", index=False)
        metricas = wf_mod.calcular_metricas(resultado, nombre)
        filas_metricas.append(metricas)
        print(f"  -> retorno_total={metricas['retorno_total_%']:.1f}%  sharpe={metricas['sharpe_anualizado']}  operaciones={metricas['operaciones']}\n")

    referencia = pd.read_csv(REFERENCIA_CSV)
    referencia_otros = referencia[referencia["estrategia"] != "PPO (RL)"]

    tabla = pd.concat([pd.DataFrame(filas_metricas), referencia_otros], ignore_index=True)
    tabla = tabla.sort_values("sharpe_anualizado", ascending=False).reset_index(drop=True)
    tabla.to_csv(f"{RESULTADOS_DIR}/mejoras_rl_metricas.csv", index=False)

    print(f"\n=== Walk-forward completo, {wf_mod.N_WINDOWS_WF * wf_mod.N_TEST_POR_VENTANA} semanas de test, comparado contra la tabla de referencia (seccion 9.4) ===\n")
    print(tabla.to_string(index=False))
