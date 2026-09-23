# INVALIDADO / SUPERADO (2026-09-23) - los regimenes (pares y signo elegidos con una muestra que incluye los regimenes): ver script 62.
# Motivo: artefacto de timestamp (la barra diaria Yahoo FX con fecha D es el precio
# de ~20:00 NY de D-1, el cierre de un futuro de commodity con fecha D es su
# settlement de ~13:00-14:30 ET de D; el merge_asof(direction='backward') por
# fecha usaba informacion posterior al precio de entrada) + costos cobrados una
# sola vez al cambiar de posicion en vez de ida+vuelta en cada operacion. Ver
# alineacion_temporal.py, costos_y_estadistica.py y la errata en 9.35 del paper.
# Se conserva sin cambios de logica como registro de los numeros originales; no
# reproduce exactamente sus CSV si se vuelve a correr despues de la correccion de
# costos en 27/32/36.
#
# Issue #13, Fase 2: para los 10 pares moneda-commodity que sobrevivieron
# Fase 1 (FDR + bootstrap SPA, rezago +1 - ver 52/53), valida que la relacion
# no sea un artefacto de todo-el-historico mirando MULTIPLES regimenes
# historicos distintos por separado, no solo el tramo reciente. Mismo
# principio que 25_validacion_historica_cobre_diario.py (Issue #5 paso 1:
# valido que el Sharpe de cobre-CLP se sostuviera en 2015-2016/2017-2018/2020-2021
# antes de comprometerse a reconstruir el pipeline diario completo).
#
# 3 regimenes deliberadamente distintos para commodities (crash de petroleo
# 2014-2016, shock COVID 2020 con el WTI negativo de por medio, spike de
# guerra/inflacion 2022):
#   - 2014-2016: crash prolongado de petroleo y fin del superciclo de commodities
#   - 2020: shock de demanda COVID, con el propio WTI llegando a precio negativo
#   - 2022: invasion a Ucrania, spike de energia/alimentos, inflacion alta
#
# Para cada par y cada regimen, DOS chequeos (no solo el backtest):
#   1. Escaneo de rezagos -2..+3 restringido a ESE regimen (n mucho mas chico
#      que las 4300+ obs de Fase 1) - ¿el patron "efecto concentrado en +1"
#      se repite dentro del regimen, o es un promedio de toda la historia
#      escondiendo regimenes donde no aplica?
#   2. Backtest de la MISMA regla sign(r_fase1) x sign(commodity_ret_1d) ya
#      validada con SPA en Fase 1 - sin refitear nada dentro del regimen (la
#      direccion ya esta fijada por Fase 1, esto es un chequeo de
#      generalizacion fuera de muestra temporal, no una nueva calibracion).
#
# Se reporta Sharpe POR VENTANA, no solo agregado - un par puede tener buen
# Sharpe promediado en toda la historia y aun asi fallar en un regimen
# especifico, y eso es exactamente lo que este chequeo esta diseñado para
# revelar.

import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

screening_mod = importlib.import_module("52_screening_correlacion_commodities_fx")

RESULTADOS_DIR = "../datos/resultados"

REGIMENES = {
    "2014-2016 (crash petroleo)": ("2014-01-01", "2016-12-31"),
    "2020 (shock COVID)": ("2020-01-01", "2020-12-31"),
    "2022 (guerra/inflacion)": ("2022-01-01", "2022-12-31"),
}


def escaneo_de_rezagos_ventana(df_ventana, moneda, commodity_nombre, regimen):
    filas = []
    for lag in screening_mod.LAGS:
        retorno_futuro = df_ventana["retorno_1d"].shift(-lag)
        valido = retorno_futuro.notna() & df_ventana["commodity_ret_1d"].notna()
        n = int(valido.sum())
        if n < 30:  # muy pocas obs para una correlacion confiable dentro del regimen
            filas.append({"moneda": moneda, "commodity": commodity_nombre, "regimen": regimen,
                          "rezago": lag, "correlacion": np.nan, "p_valor": np.nan, "n_obs": n})
            continue
        x, y = df_ventana.loc[valido, "commodity_ret_1d"], retorno_futuro[valido]
        r, p = stats.pearsonr(x, y)
        filas.append({"moneda": moneda, "commodity": commodity_nombre, "regimen": regimen,
                      "rezago": lag, "correlacion": r, "p_valor": p, "n_obs": n})
    return filas


def backtest_regla_fija(df_ventana, moneda, commodity_nombre, regimen, signo_r):
    senal = np.sign(df_ventana["commodity_ret_1d"]) * np.sign(signo_r)
    retorno_futuro = df_ventana["retorno_1d"].shift(-1)
    valido = retorno_futuro.notna()
    retorno_estrategia = (senal * retorno_futuro)[valido].to_numpy()
    n = len(retorno_estrategia)
    if n < 30:
        return {"moneda": moneda, "commodity": commodity_nombre, "regimen": regimen, "n_obs": n,
                "retorno_total_%": np.nan, "sharpe_anualizado": np.nan, "win_rate_%": np.nan}
    retorno_total_pct = 100 * (np.prod(1 + retorno_estrategia) - 1)
    sharpe = (retorno_estrategia.mean() / retorno_estrategia.std()) * np.sqrt(252) if retorno_estrategia.std() > 0 else np.nan
    win_rate = 100 * (retorno_estrategia > 0).mean()
    return {"moneda": moneda, "commodity": commodity_nombre, "regimen": regimen, "n_obs": n,
            "retorno_total_%": retorno_total_pct, "sharpe_anualizado": sharpe, "win_rate_%": win_rate}


def graficar_sharpe_por_regimen(tabla_bt, path_salida):
    tabla_bt = tabla_bt.copy()
    tabla_bt["par"] = tabla_bt["moneda"] + " x " + tabla_bt["commodity"]
    pivot = tabla_bt.pivot_table(index="par", columns="regimen", values="sharpe_anualizado")
    pivot = pivot[list(REGIMENES.keys())]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(pivot.index))
    ancho = 0.25
    colores = ["#4C72B0", "#DD8452", "#55A868"]
    for i, regimen in enumerate(pivot.columns):
        ax.bar(x + (i - 1) * ancho, pivot[regimen], width=ancho, label=regimen, color=colores[i])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=45, ha="right")
    ax.set_ylabel("Sharpe anualizado (regla fija de Fase 1, sin refitear)")
    ax.set_title("Fase 2 (Issue #13): Sharpe por regimen historico, 10 pares de Fase 1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_salida, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    supervivientes = pd.read_csv(f"{RESULTADOS_DIR}/fase1_supervivientes_fdr.csv")
    supervivientes_lag1 = supervivientes[supervivientes["rezago"] == 1].reset_index(drop=True)
    print(f"Validando {len(supervivientes_lag1)} pares en {len(REGIMENES)} regimenes historicos distintos:")
    for nombre, (ini, fin) in REGIMENES.items():
        print(f"  - {nombre}: {ini} a {fin}")

    monedas_data = {m: screening_mod.cargar_moneda(m) for m in screening_mod.MONEDAS}
    codigo_por_nombre = {v: k for k, v in screening_mod.COMMODITIES.items()}
    commodities_data = {c: screening_mod.cargar_commodity(c) for c in screening_mod.COMMODITIES}

    filas_rezagos, filas_backtest = [], []
    for _, fila in supervivientes_lag1.iterrows():
        moneda, commodity_nombre, r_signo = fila["moneda"], fila["commodity"], fila["correlacion"]
        codigo_commodity = codigo_por_nombre[commodity_nombre]
        par_completo = screening_mod.construir_par(monedas_data[moneda], commodities_data[codigo_commodity])

        for nombre_regimen, (ini, fin) in REGIMENES.items():
            ventana = par_completo[(par_completo["ds"] >= ini) & (par_completo["ds"] <= fin)].reset_index(drop=True)
            filas_rezagos.extend(escaneo_de_rezagos_ventana(ventana, moneda, commodity_nombre, nombre_regimen))
            filas_backtest.append(backtest_regla_fija(ventana, moneda, commodity_nombre, nombre_regimen, r_signo))

    tabla_rezagos = pd.DataFrame(filas_rezagos)
    tabla_rezagos.to_csv(f"{RESULTADOS_DIR}/fase2_escaneo_rezagos_por_regimen.csv", index=False)

    tabla_backtest = pd.DataFrame(filas_backtest)
    tabla_backtest.to_csv(f"{RESULTADOS_DIR}/fase2_backtest_por_regimen.csv", index=False)

    print("\n=== Escaneo de rezagos, solo rezago +1, por par y regimen (¿se sostiene el pico en +1?) ===")
    tabla_rezagos_lag1 = tabla_rezagos[tabla_rezagos["rezago"] == 1]
    print(tabla_rezagos_lag1.to_string(index=False))

    print("\n=== Backtest de la regla fija de Fase 1 (sin refitear), por par y regimen ===")
    print(tabla_backtest.to_string(index=False))

    graficar_sharpe_por_regimen(tabla_backtest, f"{RESULTADOS_DIR}/fase2_sharpe_por_regimen.png")

    # Resumen: en cuantos de los 3 regimenes cada par tiene Sharpe > 0
    resumen = tabla_backtest.groupby(["moneda", "commodity"]).agg(
        regimenes_sharpe_positivo=("sharpe_anualizado", lambda s: int((s > 0).sum())),
        sharpe_promedio=("sharpe_anualizado", "mean"),
        sharpe_minimo=("sharpe_anualizado", "min"),
    ).reset_index().sort_values("regimenes_sharpe_positivo", ascending=False)
    resumen.to_csv(f"{RESULTADOS_DIR}/fase2_resumen_consistencia.csv", index=False)
    print(f"\n=== Resumen: en cuantos de los {len(REGIMENES)} regimenes cada par tuvo Sharpe > 0 ===")
    print(resumen.to_string(index=False))
