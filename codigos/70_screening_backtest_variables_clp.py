# Issue #14: screening + backtest simple de las variables candidatas nuevas
# (69) sobre USD/CLP a frecuencia semanal y mensual, con el protocolo de
# protocolo_evaluacion.py:
#   - Solo datos ANTERIORES al hold-out (2025-01-01) para todo lo que se elige.
#   - Cada variable se une con el ultimo dato conocido ESTRICTAMENTE antes del
#     precio FX de la fila (alineacion_temporal.valor_conocido), usando la hora
#     real de publicacion guardada por 69.
#   - Criterio de supervivencia (fijado antes de mirar resultados):
#       q-valor FDR (Benjamini-Hochberg, sobre TODO el universo: las 18
#       nuevas + las 6 re-verificadas en 68) < 0.05, |r| >= max(0.11, 2/sqrt(n)),
#       y mismo signo en las dos mitades de la muestra previa al hold-out.
#   - Backtest de TODAS las variables (no solo sobrevivientes) para responder
#     "que rentabilidad dan", con la correccion por pruebas multiples al lado:
#     walk-forward expansivo con reestimacion anual (signo de la correlacion y
#     mediana de la variable, solo con train), posicion +-1 en USD/CLP, carry
#     incluido (largo USD paga el diferencial Chile - EE.UU.), costo por
#     rotacion (medio spread por unidad de cambio de posicion, 0.15% ida+vuelta)
#     y cota superior de costo ida+vuelta en cada periodo.
#   - El hold-out solo se abre para una variable que sobreviva el screening Y
#     tenga Sharpe neto con IC95 por encima de 0 antes del hold-out.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import alineacion_temporal as alin
import costos_y_estadistica as ce
import protocolo_evaluacion as prot

RESULTADOS_DIR = "../datos/resultados"
DATOS_LARGO = "../datos/bases/usdclp_long.csv"
EXTERNAS = "../datos/bases/issue14_variables_externas.csv"
REVERIFICADAS = f"{RESULTADOS_DIR}/issue14_features_semanales_mensuales_corregido.csv"
SCRIPT = "70_screening_backtest_variables_clp.py"
UMBRAL_CORR = 0.11
ALPHA_FDR = 0.05
INICIO_TEST_BACKTEST = pd.Timestamp("2015-01-01")  # primer anio de test; train expansivo desde 2010
SPREAD = ce.SPREAD_IDA_VUELTA["CLP"]
FRECUENCIAS = [("W", "semanal", 4, 52), ("ME", "mensual", 3, 12)]


def serie_clp():
    d = pd.read_csv(DATOS_LARGO, parse_dates=["ds"])[["ds", "y"]]
    return d[~alin.marcar_precios_repetidos(d["y"])].reset_index(drop=True)


def periodos(diario, freq):
    g = diario.set_index("ds")["y"]
    s = pd.DataFrame({"y": g.resample(freq).last(),
                      "ds_fila": g.resample(freq).apply(lambda x: x.index.max())}).dropna().reset_index()
    s = s.iloc[:-1]
    s["ret_fut"] = s["y"].shift(-1) / s["y"] - 1
    s["ts_fx"] = alin.ts_fx_yahoo(s["ds_fila"])
    return s


def unir_externas(s, ext):
    for serie, d in ext.groupby("serie"):
        vals, _ = alin.valor_conocido(s["ts_fx"], pd.DatetimeIndex(pd.to_datetime(d["ts_conocido_utc"], utc=True)), d["valor"])
        s[serie] = vals
    return s


def construir_features(s, k):
    f = pd.DataFrame(index=s.index)
    f["vix_log"] = np.log(s["VIXCLS"])
    f["vix_cambio_1"] = f["vix_log"].diff()
    for serie, nombre in [("^GSPC", "spx"), ("EEM", "eem"), ("ECH", "ech"), ("CL=F", "wti")]:
        f[f"{nombre}_ret_1"] = np.log(s[serie] / s[serie].shift(1))
    f[f"dxy_mom_{k}"] = s["DTWEXBGS"] / s["DTWEXBGS"].shift(k) - 1
    f["ust2y_cambio_1"] = s["DGS2"].diff()
    f["carry_cl_us_3m"] = s["IR3TIB01CLM156N"] - s["DTB3"]
    return f


def correlacion(x, y):
    v = x.notna() & y.notna()
    if v.sum() < 10:
        return np.nan, np.nan, int(v.sum())
    r, p = stats.pearsonr(x[v], y[v])
    return r, p, int(v.sum())


def screening(s, f, nombre_freq):
    pre = (s["ds"] < prot.HOLDOUT_INICIO).to_numpy()
    idx_pre = np.flatnonzero(pre)
    mitad = idx_pre[len(idx_pre) // 2]
    filas = []
    for col in f.columns:
        r, p, n = correlacion(f.loc[pre, col], s.loc[pre, "ret_fut"])
        r1, _, _ = correlacion(f.loc[idx_pre[idx_pre < mitad], col], s.loc[idx_pre[idx_pre < mitad], "ret_fut"])
        r2, _, _ = correlacion(f.loc[idx_pre[idx_pre >= mitad], col], s.loc[idx_pre[idx_pre >= mitad], "ret_fut"])
        filas.append({"frecuencia": nombre_freq, "variable": col, "origen": "nueva (69)", "correlacion": r, "p_valor": p,
                      "n": n, "corr_mitad_1": r1, "corr_mitad_2": r2})
    return filas


def backtest(s, x, periodos_anio, carry_periodo):
    """Walk-forward expansivo, reestimacion al inicio de cada anio de test."""
    pos = np.full(len(s), np.nan)
    pre = s["ds"] < prot.HOLDOUT_INICIO
    anios = sorted(s.loc[(s["ds"] >= INICIO_TEST_BACKTEST) & pre, "ds"].dt.year.unique())
    for anio in anios:
        inicio = pd.Timestamp(f"{anio}-01-01")
        # train: filas cuyo retorno siguiente ya se realizo antes del inicio del anio de test
        train = (s["ds"] < inicio) & s["ret_fut"].notna() & x.notna()
        train &= s["ds"].shift(-1) < inicio
        if train.sum() < 30:
            continue
        r, _, _ = correlacion(x[train], s.loc[train, "ret_fut"])
        signo = np.sign(r) if np.isfinite(r) and r != 0 else 0.0
        mediana = x[train].median()
        test = (s["ds"].dt.year == anio) & pre
        pos[test.to_numpy()] = signo * np.sign(x[test] - mediana).fillna(0).to_numpy()
    valido = ~np.isnan(pos) & s["ret_fut"].notna().to_numpy()
    p = pos[valido]
    ret = s["ret_fut"].to_numpy()[valido]
    carry = carry_periodo.to_numpy()[valido]
    # largo USD/CLP (+1) paga el diferencial Chile - EE.UU.; corto lo cobra
    bruto = p * (ret - carry)
    neto_rot = ce.retornos_posicion_fija(p, ret - carry, SPREAD, modo="rotacion")
    neto_iv = ce.retornos_posicion_fija(p, ret - carry, SPREAD, modo="ida_vuelta_diaria")
    return bruto, neto_rot, neto_iv, p, s["ds"].to_numpy()[valido]


def main():
    diario = serie_clp()
    ext = pd.read_csv(EXTERNAS, parse_dates=["ds"])
    screen, bts, series_bt = [], [], {}
    for freq, nombre, k, ppa in FRECUENCIAS:
        s = unir_externas(periodos(diario, freq), ext)
        f = construir_features(s, k)
        screen += screening(s, f, nombre)
        # carry por periodo (fraccion): diferencial anual conocido / periodos por anio
        carry = ((s["IR3TIB01CLM156N"] - s["DTB3"]) / 100 / ppa).fillna(0)
        base = {"largo USD/CLP permanente": pd.Series(1.0, index=s.index),
                "corto USD/CLP permanente (largo CLP + carry)": pd.Series(-1.0, index=s.index)}
        for col in list(f.columns) + list(base):
            if col in base:
                pre = (s["ds"] >= INICIO_TEST_BACKTEST) & (s["ds"] < prot.HOLDOUT_INICIO) & s["ret_fut"].notna()
                p = base[col][pre].to_numpy()
                r = s.loc[pre, "ret_fut"].to_numpy() - carry[pre].to_numpy()
                bruto, rot, iv = p * r, ce.retornos_posicion_fija(p, r, SPREAD, "rotacion"), ce.retornos_posicion_fija(p, r, SPREAD, "ida_vuelta_diaria")
                pos = p
            else:
                bruto, rot, iv, pos, fechas = backtest(s, f[col], ppa, carry)
            fila = {"frecuencia": nombre, "estrategia": col, "n_periodos": len(bruto),
                    "rotacion_media": float(np.mean(np.abs(np.diff(np.r_[0, pos])))) if len(pos) else np.nan}
            for etiqueta, r in [("bruto", bruto), ("neto_rotacion", rot), ("neto_ida_vuelta", iv)]:
                lo, hi = ce.ic_sharpe_bootstrap(r, ppa, bloque=4 if ppa == 12 else 8)
                fila[f"sharpe_{etiqueta}"] = ce.sharpe(r, ppa)
                fila[f"ic95_bajo_{etiqueta}"] = lo
                fila[f"ic95_alto_{etiqueta}"] = hi
            fila["retorno_anual_neto_rotacion_%"] = 100 * (np.prod(1 + rot) ** (ppa / max(len(rot), 1)) - 1)
            bts.append(fila)
            if col not in base:
                series_bt[(nombre, col)] = rot

    # --- screening: universo = 18 nuevas + 6 re-verificadas (68), FDR conjunto ---
    tabla = pd.DataFrame(screen)
    rev = pd.read_csv(REVERIFICADAS)
    rev = rev[rev["alineacion"] == "corr"]
    tabla = pd.concat([tabla, pd.DataFrame({
        "frecuencia": rev["frecuencia"], "variable": rev["variable"], "origen": "re-verificada (68)",
        "correlacion": rev["correlacion_pre_holdout"], "p_valor": rev["p_valor_pre_holdout"], "n": rev["n_pre_holdout"]})],
        ignore_index=True)
    _, q, _, _ = multipletests(tabla["p_valor"], alpha=ALPHA_FDR, method="fdr_bh")
    tabla["q_valor_fdr"] = q
    tabla["umbral_r"] = np.maximum(UMBRAL_CORR, 2 / np.sqrt(tabla["n"]))
    tabla["estable_mitades"] = np.sign(tabla["corr_mitad_1"]) == np.sign(tabla["corr_mitad_2"])
    tabla["sobrevive"] = (tabla["q_valor_fdr"] < ALPHA_FDR) & (tabla["correlacion"].abs() >= tabla["umbral_r"]) & tabla["estable_mitades"]
    tabla.to_csv(f"{RESULTADOS_DIR}/issue14_screening_variables_clp.csv", index=False)

    bt = pd.DataFrame(bts)
    n_pruebas_bt = int((~bt["estrategia"].str.contains("permanente")).sum())
    for nombre, _, _, ppa in [(f[1], None, None, f[3]) for f in FRECUENCIAS]:
        m = bt["frecuencia"] == nombre
        bt.loc[m, "sharpe_max_nulo_esperado"] = ce.sharpe_maximo_esperado_nulo(n_pruebas_bt, int(bt.loc[m, "n_periodos"].max()), ppa)
    bt.to_csv(f"{RESULTADOS_DIR}/issue14_backtest_variables_clp.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Screening (solo datos < 2025-01-01) ===")
    print(tabla[["frecuencia", "variable", "origen", "correlacion", "p_valor", "q_valor_fdr", "n", "corr_mitad_1", "corr_mitad_2", "sobrevive"]]
          .sort_values(["frecuencia", "p_valor"]).round(3).to_string(index=False))
    print(f"\nSobreviven: {int(tabla['sobrevive'].sum())} de {len(tabla)}")
    print(f"\n=== Backtest walk-forward {INICIO_TEST_BACKTEST.year}-2024 (antes del hold-out), carry incluido, spread {SPREAD:.2%} ===")
    cols = ["frecuencia", "estrategia", "n_periodos", "rotacion_media", "sharpe_bruto", "sharpe_neto_rotacion",
            "ic95_bajo_neto_rotacion", "ic95_alto_neto_rotacion", "sharpe_neto_ida_vuelta", "retorno_anual_neto_rotacion_%", "sharpe_max_nulo_esperado"]
    print(bt[cols].sort_values(["frecuencia", "sharpe_neto_rotacion"], ascending=[True, False]).round(2).to_string(index=False))

    # --- registro de pruebas ---
    filas_reg = [{"prueba": f"corr {r.variable}", "frecuencia": r.frecuencia, "metrica": "correlacion_pre_holdout",
                  "valor": r.correlacion, "n": r.n, "usa_holdout": False, "nota": f"q_fdr={r.q_valor_fdr:.3f}; sobrevive={r.sobrevive}"}
                 for r in tabla[tabla["origen"] == "nueva (69)"].itertuples()]
    filas_reg += [{"prueba": f"backtest {r.estrategia}", "frecuencia": r.frecuencia, "metrica": "sharpe_neto_rotacion_pre_holdout",
                   "valor": r.sharpe_neto_rotacion, "n": r.n_periodos, "usa_holdout": False, "nota": "walk-forward anual, carry incluido"}
                  for r in bt.itertuples()]
    prot.registrar_pruebas(filas_reg, issue="#14", script=SCRIPT)

    # --- hold-out: solo si alguna variable cumple AMBAS condiciones ---
    candidatas = bt.merge(tabla[tabla["sobrevive"]][["frecuencia", "variable"]], left_on=["frecuencia", "estrategia"], right_on=["frecuencia", "variable"])
    candidatas = candidatas[candidatas["ic95_bajo_neto_rotacion"] > 0]
    print(f"\nCandidatas para abrir el hold-out (sobreviven screening y IC95 neto > 0): {len(candidatas)}")
    if len(candidatas) == 0:
        print("Hold-out NO abierto: ninguna variable cumple el criterio fijado de antemano.")

    # --- grafico ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for ax, (nombre, n_ref) in zip(axes, [("semanal", None), ("mensual", None)]):
        t = tabla[tabla["frecuencia"] == nombre].sort_values("correlacion")
        colores = ["#4c78a8" if o == "nueva (69)" else "#b0b0b0" for o in t["origen"]]
        ax.barh(t["variable"], t["correlacion"], color=colores)
        lim = t["umbral_r"].max()
        ax.axvspan(-lim, lim, color="#f2f2f2", zorder=0)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(f"USD/CLP {nombre}: correlacion con el retorno siguiente\n(antes de 2025, zona gris = |r| < {lim:.3f})")
        ax.set_xlabel("correlacion de Pearson")
    fig.suptitle(f"Issue #14: {int(tabla['sobrevive'].sum())} de {len(tabla)} variables (azul = nuevas, gris = re-verificadas) "
                 f"sobreviven FDR + umbral + estabilidad")
    fig.tight_layout()
    fig.savefig(f"{RESULTADOS_DIR}/issue14_screening_variables_clp.png", dpi=120)


if __name__ == "__main__":
    main()
