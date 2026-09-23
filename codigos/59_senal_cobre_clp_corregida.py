# CORRECCION (2026-09-23) - rehace 9.13 / 9.14 (parte USD/CLP) / 9.17 paso 1
# con la alineacion temporal corregida (alineacion_temporal.py), costos
# ida+vuelta realistas (costos_y_estadistica.py) y sin elegir nada con la
# muestra completa. Reemplaza, para efectos del paper, a 21 (parte diaria), 22
# y 25 - esos scripts quedan como registro historico de los numeros invalidos.
#
# Que cambia respecto a 22/25 (cada punto es un bug o supuesto irreal que se
# corrige, no una nueva configuracion a probar):
#   1. Timestamp: copper_ret_1d de la fila FX D = retorno del cobre hasta el
#      ultimo settlement ANTERIOR al precio FX de D (antes: el del mismo dia D,
#      cuyo settlement ocurre ~17h DESPUES de ese precio FX).
#   2. Ejecucion: se entra al precio de la fila D (~20:00 NY de D-1, ya con la
#      senal conocida) y se sale en la fila siguiente - operable.
#   3. Signo de la regla "umbral cobre": se estima en cada ventana SOLO con la
#      historia de entrenamiento (signo de la correlacion senal vs. retorno
#      siguiente); antes estaba fijo en el codigo (-1), elegido mirando la
#      correlacion de toda la muestra (22:73-77, heredado por 25).
#   4. Costos: spread ida+vuelta sobre la exposicion de CADA dia con posicion
#      (0.15% para USD/CLP, sensibilidad 0-0.30%); cota inferior "rotacion"
#      (solo se paga al cambiar de posicion) reportada al lado.
#   5. Precios repetidos (feriados/huecos de CLP=X) se eliminan antes de
#      construir retornos: no hubo cotizacion nueva a la que operar.
#   6. Incertidumbre: Sharpe con IC95 por bootstrap de bloques.
#
# Se reproduce ademas la version ORIGINAL (con el artefacto) para la tabla de
# errata, con exactamente el mismo codigo de simulacion.

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

alin = importlib.import_module("alineacion_temporal")
ce = importlib.import_module("costos_y_estadistica")
features_mod = importlib.import_module("19_features_nuevas_validacion")
generar_mod = importlib.import_module("10_generar_dataset_rl")
wf_mod = importlib.import_module("14_backtest_walkforward_gestion_riesgo")

RESULTADOS_DIR = "../datos/resultados"
DATOS_LARGO = "../datos/bases/usdclp_long.csv"

VENTANA_VOL = 20
N_WINDOWS_WF = 5
N_TEST_POR_VENTANA = 60
LIMITE_APALANCAMIENTO = 1.0
SPREAD_CLP = ce.SPREAD_IDA_VUELTA["CLP"]
SLIPPAGE_ORIGINAL = 0.0005
FEATURES_CON_COBRE = ["retorno_1d", "macd_rel", "rsi_norm", "copper_ret_1d", "copper_mom_5d"]
FEATURES_SIN_COBRE = ["retorno_1d", "macd_rel", "rsi_norm"]

CORTES_HISTORICOS = {  # mismos periodos que 25
    "2025-2026 (reciente)": None,
    "2020-2021": "2021-12-31",
    "2017-2018": "2018-12-31",
    "2015-2016": "2016-12-31",
}


def preparar(macro, corregido=True):
    df = pd.read_csv(DATOS_LARGO, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)[["ds", "y"]]
    if corregido:
        df = df[~alin.marcar_precios_repetidos(df["y"])].reset_index(drop=True)
    df["retorno_1d"] = np.log(df["y"] / df["y"].shift(1))
    df["macd_rel"] = generar_mod.calcular_macd(df["y"]) / df["y"]
    df["rsi_norm"] = generar_mod.calcular_rsi(df["y"]) / 100
    df["vol_realizada"] = df["retorno_1d"].rolling(VENTANA_VOL).std()
    df = alin.agregar_features_cobre(df, macro) if corregido else alin.agregar_features_cobre_original(df, macro)
    df["y_next"] = df["y"].shift(-1)
    df["ret_siguiente"] = (df["y_next"] - df["y"]) / df["y"]
    return df.dropna().reset_index(drop=True)


def ajustar_ols(df_train, features):
    X = np.column_stack([np.ones(len(df_train)), df_train[features].to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(X, df_train["ret_siguiente"].to_numpy(dtype=float), rcond=None)
    return coef


def pos_kelly(df_train, df_test, features):
    coef = ajustar_ols(df_train, features)
    X = np.column_stack([np.ones(len(df_test)), df_test[features].to_numpy(dtype=float)])
    mu = X @ coef
    s2 = df_test["vol_realizada"].to_numpy(dtype=float) ** 2
    f = np.divide(mu, s2, out=np.zeros_like(mu), where=s2 > 0)
    return np.clip(f, -LIMITE_APALANCAMIENTO, LIMITE_APALANCAMIENTO)


def pos_umbral_cobre_train(df_train, df_test):
    signo = np.sign(df_train["copper_ret_1d"].corr(df_train["ret_siguiente"]))
    return signo * np.sign(df_test["copper_ret_1d"].to_numpy()), signo


def pos_umbral_cobre_original(df_test):
    return np.where(df_test["copper_ret_1d"] > 0, -1.0, np.where(df_test["copper_ret_1d"] < 0, 1.0, 0.0))


def retornos_original(pos, ret):
    # replica EXACTA del costo de 22.simular(): 0.05% de |pos|*capital solo cuando la posicion cambia
    pos = np.asarray(pos, dtype=float)
    prev = np.r_[0.0, pos[:-1]]
    return pos * ret - np.where(pos != prev, SLIPPAGE_ORIGINAL * np.abs(pos), 0.0)


def correr_walkforward(df, n_windows=N_WINDOWS_WF, n_test=N_TEST_POR_VENTANA, original=False):
    """Devuelve dict estrategia -> DataFrame(ds, pos, ret) sobre el test concatenado."""
    salida = {k: [] for k in ["Umbral cobre", "Kelly diario (con cobre)", "Kelly diario (sin cobre)", "Buy-and-hold"]}
    signos = []
    for df_train, df_test in wf_mod.ventanas_walkforward(df, n_windows, n_test):
        if original:
            p_cobre, s = pos_umbral_cobre_original(df_test), -1.0
        else:
            p_cobre, s = pos_umbral_cobre_train(df_train, df_test)
        signos.append(s)
        base = df_test[["ds", "ret_siguiente"]].reset_index(drop=True)
        for nombre, p in [("Umbral cobre", p_cobre), ("Kelly diario (con cobre)", pos_kelly(df_train, df_test, FEATURES_CON_COBRE)),
                          ("Kelly diario (sin cobre)", pos_kelly(df_train, df_test, FEATURES_SIN_COBRE)),
                          ("Buy-and-hold", np.ones(len(df_test)))]:
            b = base.copy()
            b["pos"] = p
            salida[nombre].append(b)
    return {k: pd.concat(v, ignore_index=True) for k, v in salida.items()}, signos


def metricas_estrategias(res, etiqueta, spread=SPREAD_CLP, original=False):
    filas = []
    for nombre, d in res.items():
        pos, ret = d["pos"].to_numpy(), d["ret_siguiente"].to_numpy()
        bruto = ce.retornos_posicion_fija(pos, ret, 0.0)
        extra = {"variante": etiqueta, "dias_con_posicion": int((pos != 0).sum()),
                 "exposicion_media": float(np.abs(pos).mean()),
                 "breakeven_spread_ida_vuelta_%": 100 * ce.breakeven_spread(bruto, np.abs(pos)) if nombre != "Buy-and-hold" else np.nan,
                 "desde": d["ds"].min().date(), "hasta": d["ds"].max().date()}
        if original:
            filas.append(ce.metricas_desde_retornos(retornos_original(pos, ret), nombre, extra={**extra, "costo": "original (0.05% solo al cambiar)"}))
            continue
        filas.append(ce.metricas_desde_retornos(bruto, nombre, extra={**extra, "costo": "bruto (spread 0)"}))
        if nombre == "Buy-and-hold":
            continue  # una sola entrada: costo despreciable
        filas.append(ce.metricas_desde_retornos(ce.retornos_posicion_fija(pos, ret, spread), nombre,
                                                extra={**extra, "costo": f"ida+vuelta diaria {100*spread:.2f}%"}))
        filas.append(ce.metricas_desde_retornos(ce.retornos_posicion_fija(pos, ret, spread, modo="rotacion"), nombre,
                                                extra={**extra, "costo": f"cota inferior rotacion {100*spread:.2f}%"}))
    return filas


def sensibilidad(res, etiqueta):
    filas = []
    for nombre in ["Umbral cobre", "Kelly diario (con cobre)"]:
        d = res[nombre]
        for s in ce.GRILLA_SPREADS:
            r = ce.retornos_posicion_fija(d["pos"].to_numpy(), d["ret_siguiente"].to_numpy(), s)
            filas.append({"variante": etiqueta, "estrategia": nombre, "spread_ida_vuelta_%": 100 * s,
                          "sharpe": ce.sharpe(r), "retorno_total_%": 100 * (np.prod(1 + r) - 1)})
    return filas


def graficar(res_orig, res_corr, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, res, titulo, original in [(axes[0], res_orig, "ORIGINAL (artefacto de timestamp, costo 0.05% solo al cambiar)", True),
                                      (axes[1], res_corr, f"CORREGIDO (senal previa a la entrada, spread ida+vuelta {100*SPREAD_CLP:.2f}%)", False)]:
        for nombre, color in [("Umbral cobre", "darkorange"), ("Kelly diario (con cobre)", "crimson"), ("Buy-and-hold", "black")]:
            d = res[nombre]
            if original:
                r = retornos_original(d["pos"].to_numpy(), d["ret_siguiente"].to_numpy())
            else:
                r = ce.retornos_posicion_fija(d["pos"].to_numpy(), d["ret_siguiente"].to_numpy(), 0.0 if nombre == "Buy-and-hold" else SPREAD_CLP)
            ax.plot(d["ds"], 100 * np.cumprod(1 + r), label=nombre, color=color)
        ax.axhline(100, color="gray", linestyle=":")
        ax.set_title(titulo, fontsize=9)
        ax.set_ylabel("Capital ($, arranca en 100)")
        ax.legend(fontsize=8)
    fig.suptitle("USD/CLP + cobre, walk-forward 5x60 dias (mismo tramo de test que 22)")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def tabla_features_diarias(macro, df_orig, df_corr):
    """Tabla de 9.13 (21, parte diaria) + chequeo por mitades de 9.22, en el
    marco original y en el corregido. 'rate_diff' (tasas mensuales con
    forward-fill) se incluye por completitud."""
    filas = []
    for etiqueta, df in [("original", df_orig), ("corregido", df_corr)]:
        d = df.copy()
        d = d.merge(macro[["ds", "tasa_chile", "tasa_us"]], on="ds", how="left")
        d["rate_diff"] = d["tasa_chile"] - d["tasa_us"]
        d["copper_mom_20d"] = (d["copper"] - d["copper"].shift(20)) / d["copper"].shift(20)
        for feat in ["copper_ret_1d", "copper_mom_5d", "copper_mom_20d", "rate_diff"]:
            fila = {"variante": etiqueta, "feature": feat,
                    "corr_con_retorno_siguiente_(operado)": d[feat].corr(d["ret_siguiente"]),
                    "corr_con_retorno_que_termina_en_la_fila": d[feat].corr(d["retorno_1d"])}
            for a, b in [("2010", "2018"), ("2018", "2027")]:
                s = d[(d["ds"] >= a) & (d["ds"] < b)]
                fila[f"corr_operado_{a}-{int(b)-1 if b != '2027' else 2026}"] = s[feat].corr(s["ret_siguiente"])
            filas.append(fila)
    return pd.DataFrame(filas)


if __name__ == "__main__":
    macro = features_mod.cargar_macro()
    df_orig = preparar(macro, corregido=False)
    df_corr = preparar(macro, corregido=True)
    tf = tabla_features_diarias(macro, df_orig, df_corr)
    tf.to_csv(f"{RESULTADOS_DIR}/correccion_clp_features_diarias_913.csv", index=False)
    print("=== 9.13 / 9.22: features diarias de cobre/tasas (original vs corregido) ===")
    print(tf.round(3).to_string(index=False))
    print(f"Original: {len(df_orig)} filas | Corregido (sin precios repetidos): {len(df_corr)} filas")

    # --- escaneo de rezagos en el propio dataset (muestra completa y tramo de test) ---
    filas_rez = []
    for etiqueta, df in [("original", df_orig), ("corregido", df_corr)]:
        for tramo, sub in [("2010-2026", df), ("tramo test (ultimas 300 filas)", df.iloc[-300:])]:
            esc = alin.escaneo_rezagos_filas_fx(sub, "copper_ret_1d")
            filas_rez.append({"variante": etiqueta, "tramo": tramo, "n": len(sub), **{f"rezago_{k}": v for k, v in esc.items()}})
    tabla_rez = pd.DataFrame(filas_rez)
    tabla_rez.to_csv(f"{RESULTADOS_DIR}/correccion_clp_cobre_rezagos.csv", index=False)
    print("\n=== Escaneo de rezagos (rezago k = corr(copper_ret_1d fila t, retorno_1d fila t+k)) ===")
    print(tabla_rez.round(3).to_string(index=False))

    # --- walk-forward reciente (mismo esquema que 22) ---
    res_orig, _ = correr_walkforward(df_orig, original=True)
    res_corr, signos = correr_walkforward(df_corr)
    print(f"\nSigno de la regla umbral cobre estimado en cada ventana (solo train): {signos}")
    filas = metricas_estrategias(res_orig, "original (reproduce 22)", original=True) + metricas_estrategias(res_corr, "corregido")
    tabla = pd.DataFrame(filas)
    tabla.to_csv(f"{RESULTADOS_DIR}/correccion_clp_cobre_walkforward_metricas.csv", index=False)
    cols = ["variante", "estrategia", "costo", "retorno_total_%", "sharpe", "ic95_bajo", "ic95_alto", "max_drawdown_%",
            "dias_con_posicion", "breakeven_spread_ida_vuelta_%", "desde", "hasta"]
    print("\n=== Walk-forward 5x60 (reciente) ===")
    print(tabla[cols].round(3).to_string(index=False))

    sens = pd.DataFrame(sensibilidad(res_corr, "corregido"))
    sens.to_csv(f"{RESULTADOS_DIR}/correccion_clp_cobre_sensibilidad_spread.csv", index=False)
    print("\n=== Sensibilidad al spread (corregido) ===")
    print(sens.round(3).to_string(index=False))

    for nombre, d in res_corr.items():
        d.to_csv(f"{RESULTADOS_DIR}/correccion_clp_cobre_{nombre.lower().replace(' ', '_').replace('(', '').replace(')', '')}_posiciones.csv", index=False)
    graficar(res_orig, res_corr, f"{RESULTADOS_DIR}/correccion_clp_cobre_curva_capital.png")

    # --- validacion historica (25), signo solo con train de cada ventana ---
    filas_hist = []
    for periodo, corte in CORTES_HISTORICOS.items():
        for etiqueta, df, original in [("original (reproduce 25)", df_orig, True), ("corregido", df_corr, False)]:
            sub = df if corte is None else df[df["ds"] <= pd.Timestamp(corte)].reset_index(drop=True)
            res, _ = correr_walkforward(sub, original=original)
            for f in metricas_estrategias(res, etiqueta, original=original):
                f["periodo"] = periodo
                filas_hist.append(f)
    hist = pd.DataFrame(filas_hist)
    hist.to_csv(f"{RESULTADOS_DIR}/correccion_clp_cobre_validacion_historica.csv", index=False)
    print("\n=== Validacion historica (Umbral cobre y Kelly con cobre) ===")
    m = hist["estrategia"].isin(["Umbral cobre", "Kelly diario (con cobre)"]) & ~hist["costo"].str.startswith("cota")
    print(hist.loc[m, ["periodo", "variante", "estrategia", "costo", "retorno_total_%", "sharpe", "ic95_bajo", "ic95_alto", "desde", "hasta"]].round(3).to_string(index=False))
