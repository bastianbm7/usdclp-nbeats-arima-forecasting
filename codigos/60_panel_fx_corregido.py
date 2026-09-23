# CORRECCION (2026-09-23) - rehace la parte "panel de 13 pares" de 9.14
# (generalizacion del efecto cobre, backtest por par, pooled vs. solo-CLP) y
# construye el panel ALINEADO que usan 66 (TFT) y 67 (Momentum Transformer).
# Reemplaza para efectos del paper a 23 y 55 (quedan como registro historico).
#
# Que cambia respecto a 23/55:
#   - copper_ret_1d/copper_mom_5d con la regla estricta de alineacion_temporal
#     (settlement < timestamp del precio FX). En el panel la correlacion
#     "predictiva" de 0.02-0.37 que se reporto era en realidad contemporanea.
#   - Precios repetidos (feriados / huecos de pares poco liquidos) eliminados
#     antes de calcular retornos.
#   - Signo de la regla "umbral cobre" estimado solo con train en cada ventana.
#   - Costos ida+vuelta por par (costos_y_estadistica.SPREAD_IDA_VUELTA).
#   - Sharpe con IC95.
# Las features que dependen solo del FX (retorno_1d, macd_rel, rsi_norm,
# vol_realizada) se recalculan con la MISMA receta de 23 (identicas salvo por
# las filas repetidas eliminadas).

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

alin = importlib.import_module("alineacion_temporal")
ce = importlib.import_module("costos_y_estadistica")
features_mod = importlib.import_module("19_features_nuevas_validacion")
generar_mod = importlib.import_module("10_generar_dataset_rl")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")
clp_mod = importlib.import_module("59_senal_cobre_clp_corregida")

BASES_DIR = "../datos/bases"
RESULTADOS_DIR = "../datos/resultados"
PANEL_ORIGINAL = f"{BASES_DIR}/panel_fx_diario_extendido.csv"  # 13 pares de 23 + NOK de 55
PANEL_ALINEADO = f"{BASES_DIR}/panel_fx_diario_alineado.csv"
VENTANA_VOL = 20
N_WINDOWS_WF, N_TEST_POR_VENTANA = 5, 60
FEATURES = clp_mod.FEATURES_CON_COBRE


def construir_panel_alineado(macro):
    original = pd.read_csv(PANEL_ORIGINAL, parse_dates=["ds"])
    partes = []
    for par, g in original.groupby("par", sort=False):
        g = g.sort_values("ds").reset_index(drop=True)
        df = g[["ds", "y"]].copy()
        df = df[~alin.marcar_precios_repetidos(df["y"])].reset_index(drop=True)
        df["retorno_1d"] = np.log(df["y"] / df["y"].shift(1))
        df["macd_rel"] = generar_mod.calcular_macd(df["y"]) / df["y"]
        df["rsi_norm"] = generar_mod.calcular_rsi(df["y"]) / 100
        df["vol_realizada"] = df["retorno_1d"].rolling(VENTANA_VOL).std()
        df = alin.agregar_features_cobre(df, macro)
        df["y_next"] = df["y"].shift(-1)
        df["par"], df["nombre_par"], df["es_commodity"] = par, g["nombre_par"].iloc[0], g["es_commodity"].iloc[0]
        partes.append(df.dropna().reset_index(drop=True))
    panel = pd.concat(partes, ignore_index=True)
    panel["ds_reloj"] = alin.fecha_reloj_fx_yahoo(panel["ds"])  # documentacion: hora real del precio FX
    panel.to_csv(PANEL_ALINEADO, index=False)
    return panel


def preparar_para_wf(g):
    g = g.sort_values("ds").reset_index(drop=True).copy()
    g["ret_siguiente"] = (g["y_next"] - g["y"]) / g["y"]
    return g


def correlaciones(panel_orig, panel_alin):
    filas = []
    for par in panel_alin["par"].unique():
        go = panel_orig[panel_orig["par"] == par].sort_values("ds")
        ga = panel_alin[panel_alin["par"] == par].sort_values("ds")
        eo = alin.escaneo_rezagos_filas_fx(go, "copper_ret_1d")
        ea = alin.escaneo_rezagos_filas_fx(ga, "copper_ret_1d")
        filas.append({"par": ga["nombre_par"].iloc[0], "ticker": par, "es_commodity": bool(ga["es_commodity"].iloc[0]),
                      "original_rezago_+1_(reportado_como_predictivo)": eo[1], "original_rezago_0": eo[0],
                      "corregido_rezago_0_(contemporaneo)": ea[0], "corregido_rezago_+1_(operable)": ea[1],
                      "n": len(ga)})
    return pd.DataFrame(filas).sort_values("corregido_rezago_0_(contemporaneo)").reset_index(drop=True)


def backtest_por_par(panel_alin):
    filas, sens = [], []
    for par in panel_alin["par"].unique():
        g = preparar_para_wf(panel_alin[panel_alin["par"] == par])
        spread = ce.spread_de(par)
        res, signos = clp_mod.correr_walkforward(g)
        for nombre in ["Umbral cobre", "Kelly diario (con cobre)"]:
            d = res[nombre]
            pos, ret = d["pos"].to_numpy(), d["ret_siguiente"].to_numpy()
            for costo, r in [("bruto", ce.retornos_posicion_fija(pos, ret, 0.0)),
                             (f"ida+vuelta {100*spread:.2f}%", ce.retornos_posicion_fija(pos, ret, spread))]:
                filas.append(ce.metricas_desde_retornos(r, nombre, extra={
                    "par": g["nombre_par"].iloc[0], "costo": costo, "spread_%": 100 * spread,
                    "signo_train": ",".join(str(int(s)) for s in signos) if nombre == "Umbral cobre" else "",
                    "dias_con_posicion": int((pos != 0).sum()), "desde": d["ds"].min().date(), "hasta": d["ds"].max().date()}))
    return pd.DataFrame(filas)


def pooled_vs_single(panel_alin):
    clp = preparar_para_wf(panel_alin[panel_alin["par"] == "CLP=X"])
    panel_wf = pd.concat([preparar_para_wf(g) for _, g in panel_alin.groupby("par")], ignore_index=True)
    partes = {"Kelly diario (solo CLP)": [], "Kelly diario (panel pooled, 14 pares)": []}
    for clp_train, clp_test in wf_mod.ventanas_walkforward(clp, N_WINDOWS_WF, N_TEST_POR_VENTANA):
        corte = clp_test["ds"].iloc[0]
        pooled_train = panel_wf[panel_wf["ds"] < corte]
        for nombre, train in [("Kelly diario (solo CLP)", clp_train), ("Kelly diario (panel pooled, 14 pares)", pooled_train)]:
            b = clp_test[["ds", "ret_siguiente"]].copy()
            b["pos"] = clp_mod.pos_kelly(train, clp_test, FEATURES)
            partes[nombre].append(b)
    filas = []
    spread = ce.SPREAD_IDA_VUELTA["CLP"]
    for nombre, pp in partes.items():
        d = pd.concat(pp, ignore_index=True)
        pos, ret = d["pos"].to_numpy(), d["ret_siguiente"].to_numpy()
        for costo, r in [("bruto", ce.retornos_posicion_fija(pos, ret, 0.0)),
                         (f"ida+vuelta {100*spread:.2f}%", ce.retornos_posicion_fija(pos, ret, spread))]:
            filas.append(ce.metricas_desde_retornos(r, nombre, extra={"costo": costo, "pct_dias_saturado": 100 * np.mean(np.abs(pos) >= 0.999)}))
    return pd.DataFrame(filas)


def graficar(tabla, path):
    fig, ax = plt.subplots(figsize=(10, 6.5))
    y = np.arange(len(tabla))
    ax.barh(y + 0.27, tabla["original_rezago_+1_(reportado_como_predictivo)"], height=0.27, color="lightgray", label="Original: 'rezago +1' (reportado como predictivo)")
    ax.barh(y, tabla["corregido_rezago_0_(contemporaneo)"], height=0.27, color="steelblue", label="Corregido: rezago 0 (contemporaneo, no operable)")
    ax.barh(y - 0.27, tabla["corregido_rezago_+1_(operable)"], height=0.27, color="darkorange", label="Corregido: rezago +1 (operable)")
    ax.set_yticks(y)
    ax.set_yticklabels(tabla["par"])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Correlacion de copper_ret_1d con el retorno del par (convencion USD por unidad)")
    ax.set_title("Panel FX: la correlacion 'predictiva' del cobre era contemporanea")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    macro = features_mod.cargar_macro()
    panel_orig = pd.read_csv(PANEL_ORIGINAL, parse_dates=["ds"])
    panel_alin = construir_panel_alineado(macro)
    print(f"Panel alineado: {len(panel_alin)} filas, {panel_alin['par'].nunique()} pares -> {PANEL_ALINEADO}")

    tabla_corr = correlaciones(panel_orig, panel_alin)
    tabla_corr.to_csv(f"{RESULTADOS_DIR}/correccion_panel_fx_correlacion_cobre_por_par.csv", index=False)
    graficar(tabla_corr, f"{RESULTADOS_DIR}/correccion_panel_fx_correlacion_cobre_por_par.png")
    print("\n=== Correlacion cobre -> par: original vs corregido ===")
    print(tabla_corr.round(3).to_string(index=False))

    bt = backtest_por_par(panel_alin)
    bt.to_csv(f"{RESULTADOS_DIR}/correccion_panel_fx_backtest_por_par.csv", index=False)
    print("\n=== Backtest por par (walk-forward 5x60, signo solo con train) ===")
    print(bt[["par", "estrategia", "costo", "retorno_total_%", "sharpe", "ic95_bajo", "ic95_alto", "signo_train"]].round(3).to_string(index=False))

    pv = pooled_vs_single(panel_alin)
    pv.to_csv(f"{RESULTADOS_DIR}/correccion_panel_fx_pooled_vs_single.csv", index=False)
    print("\n=== Pooled vs solo-CLP (corregido) ===")
    print(pv[["estrategia", "costo", "retorno_total_%", "sharpe", "ic95_bajo", "ic95_alto", "pct_dias_saturado"]].round(3).to_string(index=False))
