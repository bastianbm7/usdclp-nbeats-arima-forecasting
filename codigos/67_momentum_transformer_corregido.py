# CORRECCION (2026-09-23) - rehace 9.34 (Issue #13 Fase 4: port del Momentum
# Transformer). Reemplaza para efectos del paper a 57.
#
# Misma arquitectura, perdida (SharpeLoss), hiperparametros y volatility
# targeting (15%) que 57 - se importan de ahi, no se reimplementan. Cambia:
#   1. Panel alineado (60): el Sharpe 5.29 de 57 venia de aprender la relacion
#      CONTEMPORANEA cobre -> FX (la feature copper_ret_1d de la fila t incluia
#      el settlement de cobre ~17h posterior al precio de entrada).
#   2. Costos: spread ida+vuelta de USD/CLP (0.15%) sobre |posicion| cada dia
#      (cota superior) y cota inferior de rotacion (solo se paga el cambio de
#      posicion - relevante aca porque la posicion es continua y persistente).
#   3. 5 semillas en vez de 1; Sharpe con IC95.
#   4. Se elimina el argumento de "escalar 3x": multiplicar la posicion por una
#      constante no cambia el Sharpe (salvo por el clip a +-1) - solo el retorno
#      total y el riesgo -, el factor 3x se eligio mirando el resultado de test
#      para cruzar el criterio de +83%, y el CSV
#      fase4_momentum_transformer_escalado_apalancamiento.csv no lo produce
#      ningun script del repo. No es evidencia de nada; no se repite.

import argparse
import importlib
import time

import numpy as np
import pandas as pd
import torch

mt = importlib.import_module("57_momentum_transformer_port")
ce = importlib.import_module("costos_y_estadistica")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")

BASES_DIR = "../datos/bases"
RESULTADOS_DIR = "../datos/resultados"
PANEL_ALINEADO = f"{BASES_DIR}/panel_fx_diario_alineado.csv"
SEMILLAS = [42, 7, 123, 2024, 99]
TARGET_VOL = 0.15
SPREAD = ce.SPREAD_IDA_VUELTA["CLP"]


def entrenar_ventana(panel_train, panel_eval, monedas_a_id, n_monedas, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    X_tr, ids_tr, y_tr, _, _ = mt.construir_dataset_ventanas(panel_train, monedas_a_id)
    X_all, ids_all, y_all, f_all, m_all = mt.construir_dataset_ventanas(panel_eval, monedas_a_id)
    X_tr, X_all = mt.normalizar_features(X_tr, X_all)
    modelo = mt.MomentumTransformerLite(n_monedas)
    opt = torch.optim.Adam(modelo.parameters(), lr=mt.LEARNING_RATE)
    Xt, it, yt = torch.tensor(X_tr), torch.tensor(ids_tr, dtype=torch.long), torch.tensor(y_tr)
    modelo.train()
    for _ in range(mt.STEPS_POR_VENTANA):
        idx = torch.randint(0, len(Xt), (min(mt.BATCH_SIZE, len(Xt)),))
        pos, _ = modelo(Xt[idx], it[idx])
        loss = mt.sharpe_loss(pos, yt[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
        opt.step()
    modelo.eval()
    with torch.no_grad():
        p, _ = modelo(torch.tensor(X_all), torch.tensor(ids_all, dtype=torch.long))
    return pd.DataFrame({"ds": f_all, "par": m_all, "posicion_cruda": p.numpy()})


def correr_semilla(panel, seed):
    monedas = sorted(panel["par"].unique())
    a_id = {m: i for i, m in enumerate(monedas)}
    clp = panel[panel["par"] == "CLP=X"].sort_values("ds").reset_index(drop=True)
    partes = []
    for i, (_, test) in enumerate(wf_mod.ventanas_walkforward(clp, mt.N_WINDOWS_WF, mt.N_TEST_POR_VENTANA)):
        ini, fin = test["ds"].iloc[0], test["ds"].iloc[-1]
        sal = entrenar_ventana(panel[panel["ds"] < ini], panel[panel["ds"] <= fin], a_id, len(monedas), seed)
        s = sal[(sal["par"] == "CLP=X") & (sal["ds"] >= ini) & (sal["ds"] <= fin)]
        d = test[["ds", "y", "y_next", "vol_realizada"]].merge(s[["ds", "posicion_cruda"]], on="ds", how="inner")
        vol_anual = d["vol_realizada"].to_numpy() * np.sqrt(252)
        d["pos"] = np.clip(d["posicion_cruda"].to_numpy() * TARGET_VOL / np.maximum(vol_anual, 1e-6), -1, 1)
        d["ventana"], d["seed"] = i + 1, seed
        partes.append(d)
    return pd.concat(partes, ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hilos", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.hilos)
    panel = pd.read_csv(PANEL_ALINEADO, parse_dates=["ds"])
    todas, filas = [], []
    for seed in SEMILLAS:
        t0 = time.time()
        d = correr_semilla(panel, seed)
        todas.append(d)
        pos, ret = d["pos"].to_numpy(), ((d["y_next"] - d["y"]) / d["y"]).to_numpy()
        diag = {"seed": seed, "acierto_direccion_%": 100 * np.mean(np.sign(pos) == np.sign(ret)),
                "corr_posicion_retorno": np.corrcoef(pos, ret)[0, 1], "exposicion_media": np.abs(pos).mean(),
                "pct_dias_saturado": 100 * np.mean(np.abs(pos) >= 0.999)}
        for costo, r in [("bruto", ce.retornos_posicion_fija(pos, ret, 0.0)),
                         (f"ida+vuelta diaria {100*SPREAD:.2f}%", ce.retornos_posicion_fija(pos, ret, SPREAD)),
                         (f"rotacion {100*SPREAD:.2f}%", ce.retornos_posicion_fija(pos, ret, SPREAD, modo="rotacion"))]:
            filas.append(ce.metricas_desde_retornos(r, "Momentum Transformer Lite (corregido)", extra={**diag, "costo": costo}))
        print(f"seed {seed}: {time.time()-t0:.0f}s | bruto Sharpe={filas[-3]['sharpe']:.2f} | rotacion={filas[-1]['sharpe']:.2f}", flush=True)
    pd.concat(todas, ignore_index=True).to_csv(f"{RESULTADOS_DIR}/correccion_momentum_transformer_posiciones.csv", index=False)
    t = pd.DataFrame(filas)
    resumen = t.groupby("costo").agg(sharpe_medio=("sharpe", "mean"), sharpe_min=("sharpe", "min"), sharpe_max=("sharpe", "max"),
                                     retorno_medio_pct=("retorno_total_%", "mean"), acierto_medio=("acierto_direccion_%", "mean"),
                                     corr_media=("corr_posicion_retorno", "mean")).reset_index()
    t.to_csv(f"{RESULTADOS_DIR}/correccion_momentum_transformer_metricas.csv", index=False)
    resumen.to_csv(f"{RESULTADOS_DIR}/correccion_momentum_transformer_resumen.csv", index=False)
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    print(resumen.round(3).to_string(index=False))
