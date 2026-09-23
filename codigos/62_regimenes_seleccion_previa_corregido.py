# CORRECCION (2026-09-23) - rehace 9.32 (Issue #13 Fase 2: 3 regimenes
# historicos). Reemplaza para efectos del paper a 54.
#
# Problema de 54 (pseudo fuera de muestra): los 3 regimenes (2014-2016, 2020,
# 2022) estan DENTRO de la muestra 2010-2026 con la que Fase 1 eligio los pares
# y el signo (54:42-46,118 lee fase1_supervivientes_fdr.csv). Un par elegido
# porque correlaciona en toda la historia va a tender a correlacionar en
# cualquier sub-tramo de esa historia - no es validacion temporal.
#
# Correccion: para cada regimen, el MISMO procedimiento de seleccion de Fase 1
# (escaneo de 90 pruebas + FDR BH 5% + |r|>=0.11, ahora en el rezago operable)
# se aplica SOLO con datos ANTERIORES al inicio del regimen; el signo tambien
# sale de esos datos. Despues se evalua dentro del regimen, con costos. Para
# que el resultado no dependa de que el filtro deje pasar algo, se reporta
# ademas el universo completo: los 15 pares en el rezago operable +1, con el
# signo pre-regimen, dentro de cada regimen (no es una busqueda: es la tabla
# entera, sin elegir).

import importlib

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ce = importlib.import_module("costos_y_estadistica")
scr = importlib.import_module("61_screening_spa_universo_corregido")

RESULTADOS_DIR = "../datos/resultados"
REGIMENES = {
    "2014-2016 (crash petroleo)": ("2014-01-01", "2016-12-31"),
    "2020 (shock COVID)": ("2020-01-01", "2020-12-31"),
    "2022 (guerra/inflacion)": ("2022-01-01", "2022-12-31"),
}


def screening_previo(pares, fecha_inicio):
    filas = []
    for (m, c), df in pares.items():
        pre = df[df["ds"] < fecha_inicio]
        for lag in scr.LAGS:
            fut = pre["retorno_1d"].shift(-lag)
            v = fut.notna()
            r, p = stats.pearsonr(pre.loc[v, "commodity_ret_1d"], fut[v])
            filas.append({"moneda": m, "commodity": c, "rezago": lag, "correlacion_pre": r, "p_valor": p, "n_pre": int(v.sum())})
    t = pd.DataFrame(filas)
    rech, pfdr, _, _ = multipletests(t["p_valor"], alpha=scr.ALPHA_FDR, method="fdr_bh")
    t["p_fdr"], t["sobrevive_fdr"] = pfdr, rech
    t["seleccionado"] = t["sobrevive_fdr"] & (t["correlacion_pre"].abs() >= scr.UMBRAL_CORR) & (t["rezago"] >= 1)
    return t


def evaluar_en_regimen(df, ini, fin, lag, signo, spread):
    ventana = df[(df["ds"] >= ini) & (df["ds"] <= fin)].reset_index(drop=True)
    pos = signo * np.sign(ventana["commodity_ret_1d"].to_numpy())
    ret = ventana["ret_simple"].shift(-lag).to_numpy()
    ok = ~np.isnan(ret)
    rb = ce.retornos_posicion_fija(pos[ok], ret[ok], 0.0)
    rn = ce.retornos_posicion_fija(pos[ok], ret[ok], spread)
    return rb, rn, ventana["commodity_ret_1d"].corr(ventana["retorno_1d"].shift(-lag))


if __name__ == "__main__":
    panel = pd.read_csv(scr.PANEL_ALINEADO, parse_dates=["ds"])
    pares = {(m, c): scr.construir_par(scr.cargar_moneda(m, panel), scr.cargar_commodity(c), c)
             for m in scr.MONEDAS for c in scr.COMMODITIES}

    filas_sel, filas_univ = [], []
    for regimen, (ini, fin) in REGIMENES.items():
        t = screening_previo(pares, pd.Timestamp(ini))
        sel = t[t["seleccionado"]]
        print(f"\n=== {regimen}: seleccion con datos < {ini} (n_pre ~{int(t['n_pre'].median())}) ===")
        print(f"  pruebas que sobreviven FDR: {int(t['sobrevive_fdr'].sum())} (por rezago: {t[t['sobrevive_fdr']].groupby('rezago').size().to_dict()})")
        print(f"  seleccionados (FDR + |r|>=0.11 + rezago operable): {len(sel)}")
        for _, f in sel.iterrows():
            spread = ce.SPREAD_IDA_VUELTA[f["moneda"]]
            rb, rn, c_in = evaluar_en_regimen(pares[(f["moneda"], f["commodity"])], ini, fin, f["rezago"], np.sign(f["correlacion_pre"]), spread)
            filas_sel.append({"regimen": regimen, "moneda": f["moneda"], "commodity": f["commodity"], "rezago": f["rezago"],
                              "corr_pre": f["correlacion_pre"], "corr_en_regimen": c_in,
                              "sharpe_bruto": ce.sharpe(rb), "sharpe_neto": ce.sharpe(rn), "n": len(rb)})
        # universo completo: 15 pares, rezago +1 operable, signo pre-regimen
        for (m, c), df in pares.items():
            fila_pre = t[(t["moneda"] == m) & (t["commodity"] == c) & (t["rezago"] == 1)].iloc[0]
            signo = np.sign(fila_pre["correlacion_pre"])
            rb, rn, c_in = evaluar_en_regimen(df, ini, fin, 1, signo, ce.SPREAD_IDA_VUELTA[m])
            lo, hi = ce.ic_sharpe_bootstrap(rb)
            filas_univ.append({"regimen": regimen, "moneda": m, "commodity": scr.COMMODITIES[c], "corr_pre_rezago1": fila_pre["correlacion_pre"],
                               "signo_pre": signo, "corr_en_regimen_rezago1": c_in, "sharpe_bruto": ce.sharpe(rb),
                               "ic95_bruto": f"[{lo:.2f}, {hi:.2f}]", "sharpe_neto": ce.sharpe(rn), "n": len(rb)})

    tsel = pd.DataFrame(filas_sel)
    tuniv = pd.DataFrame(filas_univ)
    tsel.to_csv(f"{RESULTADOS_DIR}/correccion_fase2_seleccionados_pre_regimen.csv", index=False)
    tuniv.to_csv(f"{RESULTADOS_DIR}/correccion_fase2_universo_por_regimen.csv", index=False)
    print("\n=== Seleccionados con datos previos, evaluados en el regimen ===")
    print(tsel.round(3).to_string(index=False) if len(tsel) else "(ninguno)")
    print("\n=== Universo completo (15 pares, rezago +1, signo pre-regimen) ===")
    print(tuniv.round(3).to_string(index=False))
    resumen = tuniv.groupby("regimen").agg(sharpe_bruto_medio=("sharpe_bruto", "mean"), sharpe_neto_medio=("sharpe_neto", "mean"),
                                           pares_sharpe_bruto_pos=("sharpe_bruto", lambda s: int((s > 0).sum())),
                                           pares_sharpe_neto_pos=("sharpe_neto", lambda s: int((s > 0).sum())))
    resumen.to_csv(f"{RESULTADOS_DIR}/correccion_fase2_resumen.csv")
    print("\n=== Resumen por regimen ===")
    print(resumen.round(3).to_string())
