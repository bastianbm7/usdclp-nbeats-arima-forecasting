# CORRECCION (2026-09-23) - reentrena los agentes PPO diarios de 9.17 (28),
# 9.20 (33), 9.23 (37), 9.24/9.26 (40/42, subconjunto) y 9.28 (48-50) sobre los
# datasets REALINEADOS (63) y con el entorno de costos corregido (27/32/36 ya
# cobran spread ida+vuelta en cada operacion). Solo entrena y guarda las
# posiciones out-of-sample; las metricas, baselines, IC y sensibilidad a costos
# las calcula 65_resumen_rl_diario_corregido.py.
#
# Cambios de protocolo (correcciones, no configuraciones nuevas):
#   - varias semillas de PPO por configuracion (antes: SEED=42 unica).
#   - spread ida+vuelta del par dentro del entorno de ENTRENAMIENTO (el
#     agente aprende con la economia real, igual que 9.4 hizo con el
#     apalancamiento): CLP 0.15%, AUD 0.02%, CAD 0.02%, NZD 0.03%.
#   - mismo walk-forward (5 ventanas x 60 filas diarias de test), mismo
#     presupuesto (100k timesteps por ventana), misma red por defecto.
#
# Uso:
#   python 64_entrenar_ppo_diario_corregido.py --lote A      (ver LOTES abajo)
#   python 64_entrenar_ppo_diario_corregido.py --exp diario --par clp --seed 42
# Cada corrida escribe datos/resultados/correccion_rl/<exp>_<par>_N<N>_h<h>_seed<seed>.csv
# (si el archivo ya existe se salta: permite relanzar lotes interrumpidos).

import argparse
import importlib
import os
import time

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

ce = importlib.import_module("costos_y_estadistica")
entorno_mod = importlib.import_module("27_entorno_trading_rl_diario")
multidia_mod = importlib.import_module("32_entorno_trading_rl_diario_multidia")
tpadapt_mod = importlib.import_module("36_entorno_trading_rl_diario_tp_adaptativo")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")

RESULTADOS_DIR = "../datos/resultados"
SALIDA_DIR = f"{RESULTADOS_DIR}/correccion_rl"
N_WINDOWS_WF, N_TEST_POR_VENTANA = 5, 60
TOTAL_TIMESTEPS_PPO = 100_000
SEMILLAS = [42, 7, 123]

DATASET = {
    "clp": "dataset_entrenamiento_rl_diario_alineado.csv",
    "clp_h3": "dataset_entrenamiento_rl_diario_h3_alineado.csv",
    "clp_h5": "dataset_entrenamiento_rl_diario_h5_alineado.csv",
    "aud": "dataset_entrenamiento_rl_diario_aud_alineado.csv",
    "cad": "dataset_entrenamiento_rl_diario_cad_alineado.csv",
    "nzd": "dataset_entrenamiento_rl_diario_nzd_alineado.csv",
}
CODIGO_PAR = {"clp": "CLP", "clp_h3": "CLP", "clp_h5": "CLP", "aud": "AUD", "cad": "CAD", "nzd": "NZD"}

LOTES = {
    # A: 9.17 paso 2 + 9.28 (1 dia, TP=h1), 3 semillas
    "A": [("diario", p, 1, 1, s) for p in ["clp", "aud", "cad", "nzd"] for s in SEMILLAS],
    # B: 9.20 holding N dias (N=1 es A/clp), 3 semillas
    "B": [("multidia", "clp", n, 1, s) for n in [2, 3, 5] for s in SEMILLAS],
    # C: 9.23 TP adaptativo vs TP=h1 con holding 3 sobre el dataset h3, 3 semillas
    "C": [("tpadapt", "clp_h3", 3, 0, s) for s in SEMILLAS] + [("multidia", "clp_h3", 3, 1, s) for s in SEMILLAS],
    # D: 9.24/9.26 subconjunto representativo de la grilla N x h (dataset h5), 1 semilla
    "D": [("multidia", "clp_h5", n, h, 42) for n in [3, 5, 7, 14] for h in [1, 2, 4]],
}


def construir_env(exp, df, n, h, spread):
    if exp == "diario":
        return entorno_mod.USDCLPTradingEnvDiario(df=df, costo_ida_vuelta_pct=spread)
    if exp == "multidia":
        return multidia_mod.USDCLPTradingEnvDiarioMultidia(df=df, dias_holding=n, horizonte_tp=h, costo_ida_vuelta_pct=spread)
    if exp == "tpadapt":
        return tpadapt_mod.USDCLPTradingEnvDiarioTPAdaptativo(df=df, costo_ida_vuelta_pct=spread)
    raise ValueError(exp)


def correr(exp, par, n, h, seed):
    os.makedirs(SALIDA_DIR, exist_ok=True)
    salida = f"{SALIDA_DIR}/{exp}_{par}_N{n}_h{h}_seed{seed}.csv"
    if os.path.exists(salida):
        print(f"[saltado, ya existe] {salida}")
        return
    df = entorno_mod.cargar_dataset(f"{RESULTADOS_DIR}/{DATASET[par]}")
    spread = ce.SPREAD_IDA_VUELTA[CODIGO_PAR[par]]
    t0 = time.time()
    partes = []
    for w, (df_train, df_test) in enumerate(wf_mod.ventanas_walkforward(df, N_WINDOWS_WF, N_TEST_POR_VENTANA)):
        env_train = construir_env(exp, df_train, n, h, spread)
        modelo = PPO("MlpPolicy", env_train, seed=seed, verbose=0)
        modelo.learn(total_timesteps=TOTAL_TIMESTEPS_PPO)
        env_test = construir_env(exp, df_test, n, h, spread)
        obs, _ = env_test.reset()
        pos, terminado = [], False
        while not terminado:
            accion, _ = modelo.predict(obs, deterministic=True)
            pos.append(entorno_mod.POSICION_POR_ACCION[int(accion)])
            obs, _, terminado, _, _ = env_test.step(accion)
        dec = env_test.df.copy()
        dec["pos_ppo"] = pos
        dec["ventana"] = w + 1
        # umbral simple y signo de cobre: se estiman SOLO con train (65 los usa)
        fr = ((df_train["nhits_h1"] - df_train["y"]) / df_train["y"]).abs()
        dec["umbral_simple_train"] = fr.median()
        ret_sig_train = (df_train["y_next"] - df_train["y"]) / df_train["y"]
        dec["signo_cobre_train"] = np.sign(df_train["copper_ret_1d"].corr(ret_sig_train))
        partes.append(dec)
        print(f"  {exp} {par} N={n} h={h} seed={seed} ventana {w+1}/{N_WINDOWS_WF}: {time.time()-t0:.0f}s acumulados", flush=True)
    out = pd.concat(partes, ignore_index=True)
    out["exp"], out["par"], out["N"], out["h"], out["seed"], out["spread"] = exp, par, n, h, seed, spread
    out.to_csv(salida, index=False)
    print(f"[ok] {salida} ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lote")
    ap.add_argument("--parte", type=int, default=0, help="indice de parte si el lote se reparte en varios procesos")
    ap.add_argument("--n_partes", type=int, default=1)
    ap.add_argument("--exp")
    ap.add_argument("--par")
    ap.add_argument("--N", type=int, default=1)
    ap.add_argument("--h", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hilos", type=int, default=2)
    ap.add_argument("--smoke", action="store_true", help="2000 timesteps, salida en correccion_rl_smoke/ (solo para validar la logica)")
    a = ap.parse_args()
    torch.set_num_threads(a.hilos)
    if a.smoke:
        TOTAL_TIMESTEPS_PPO = 2000
        SALIDA_DIR = f"{RESULTADOS_DIR}/correccion_rl_smoke"
    if a.lote:
        tareas = [t for lote in a.lote.split(",") for t in LOTES[lote]]
        tareas = tareas[a.parte::a.n_partes]
        print(f"Lote {a.lote} parte {a.parte}/{a.n_partes}: {len(tareas)} corridas")
        for t in tareas:
            correr(*t)
    else:
        correr(a.exp, a.par, a.N, a.h, a.seed)
