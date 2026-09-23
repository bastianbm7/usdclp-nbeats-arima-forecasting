# Issue #14 (paso previo): re-verifica con la alineacion corregida las variables
# CROSS-ASSET que se habian probado a frecuencia semanal/mensual (19 y la parte
# mensual de 21) y que la errata de 9.35 no rehizo.
#
# Por que hacia falta: el artefacto de timestamp (9.35.1) no afecta a las
# variables derivadas del propio precio de USD/CLP (MACD, RSI, SMA/EMA,
# Bollinger, CCI, ADX, momentum, GARCH, NHITS, min-max): variable y objetivo
# estan en el mismo reloj, y correrlos juntos no cambia la correlacion. Pero en
# 19/21 el cobre se unia por etiqueta de periodo (domingo / fin de mes) contra
# el cache con forward-fill calendario: la fila semanal de CLP (ultima barra
# Yahoo de la semana = ~20:00 NY del jueves) recibia el settlement del VIERNES,
# ~17h posterior - la misma contaminacion que en la linea diaria, diluida en
# una ventana de 5 dias. Las tasas (OECD mensual) se usaban sin el rezago de
# publicacion de 2-4 meses.
#
# Correccion: cobre = ultimo settlement ESTRICTAMENTE anterior al precio FX real
# de la ultima fila del periodo (alineacion_temporal.valor_conocido); tasas con
# 3 meses de rezago de publicacion; precios repetidos de CLP=X eliminados.
# Muestra completa 2010-2026 (es una re-verificacion de pruebas ya hechas, no
# una seleccion: por eso no se recorta el hold-out), se reporta ademas la
# muestra previa al hold-out.

import numpy as np
import pandas as pd
from scipy import stats

import alineacion_temporal as alin
import protocolo_evaluacion as prot

RESULTADOS_DIR = "../datos/resultados"
DATOS_LARGO = "../datos/bases/usdclp_long.csv"
MACRO = "../datos/bases/macro_tasas_cobre.csv"
REZAGO_PUBLICACION_TASAS_MESES = 3
UMBRAL_CORR = 0.11  # umbral historico del proyecto (9.5)


def serie_clp():
    d = pd.read_csv(DATOS_LARGO, parse_dates=["ds"])[["ds", "y"]]
    return d[~alin.marcar_precios_repetidos(d["y"])].reset_index(drop=True)


def agregar_periodo(diario, macro, freq):
    """Ultimo precio de cada periodo + fecha real de esa fila Yahoo, retorno del
    periodo siguiente y cobre/tasas en version original y corregida."""
    g = diario.set_index("ds")["y"]
    s = pd.DataFrame({"y": g.resample(freq).last(),
                      "ds_fila": g.resample(freq).apply(lambda x: x.index.max())}).dropna().reset_index()
    s = s.iloc[:-1]  # periodo en curso incompleto (mismo criterio que 10/24)
    s["ret_fut"] = s["y"].shift(-1) / s["y"] - 1

    orig = pd.merge_asof(s[["ds"]], macro[["ds", "copper", "tasa_chile", "tasa_us"]], on="ds", direction="backward")
    s["cobre_orig"] = orig["copper"].to_numpy()
    s["rd_orig"] = (orig["tasa_chile"] - orig["tasa_us"]).to_numpy()

    cu = alin.serie_commodity_dias_habiles(macro)
    corr = alin.agregar_commodity_conocido(s[["ds_fila"]].rename(columns={"ds_fila": "ds"}), cu, "HG_F", "cobre")
    s["cobre_corr"] = corr["cobre"].to_numpy()
    m2 = macro[["ds", "tasa_chile", "tasa_us"]].copy()
    m2["ds"] = m2["ds"] + pd.DateOffset(months=REZAGO_PUBLICACION_TASAS_MESES)
    tasas = pd.merge_asof(s[["ds"]], m2, on="ds", direction="backward")
    s["rd_corr"] = (tasas["tasa_chile"] - tasas["tasa_us"]).to_numpy()
    return s


def features(s, k_mom, version):
    c = s[f"cobre_{version}"]
    return {"cobre_ret_1": np.log(c / c.shift(1)),
            f"cobre_mom_{k_mom}": c / c.shift(k_mom) - 1,
            "rate_diff": s[f"rd_{version}"]}


def correlacion(x, y):
    v = x.notna() & y.notna()
    r, p = stats.pearsonr(x[v], y[v])
    return r, p, int(v.sum())


def main():
    diario = serie_clp()
    macro = pd.read_csv(MACRO, parse_dates=["ds"])
    filas = []
    for freq, nombre, k in [("W", "semanal", 4), ("ME", "mensual", 3)]:
        s = agregar_periodo(diario, macro, freq)
        previo = s["ds"] < prot.HOLDOUT_INICIO
        for version in ["orig", "corr"]:
            for feat, x in features(s, k, version).items():
                r, p, n = correlacion(x, s["ret_fut"])
                r_pre, p_pre, n_pre = correlacion(x[previo], s.loc[previo, "ret_fut"])
                filas.append({"frecuencia": nombre, "variable": feat, "alineacion": version,
                              "correlacion": r, "p_valor": p, "n": n,
                              "correlacion_pre_holdout": r_pre, "p_valor_pre_holdout": p_pre, "n_pre_holdout": n_pre,
                              "umbral_2_sqrt_n": 2 / np.sqrt(n)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(f"{RESULTADOS_DIR}/issue14_features_semanales_mensuales_corregido.csv", index=False)

    pv = tabla.pivot_table(index=["frecuencia", "variable"], columns="alineacion", values="correlacion")
    pv["n"] = tabla[tabla["alineacion"] == "corr"].set_index(["frecuencia", "variable"])["n"]
    pv["2/sqrt(n)"] = 2 / np.sqrt(pv["n"])
    print("\nCorrelacion con el retorno del periodo siguiente de USD/CLP (orig = como en 19/21, corr = alineacion estricta)")
    print(pv.round(3).to_string())

    corr = tabla[tabla["alineacion"] == "corr"]
    prot.registrar_pruebas(
        [{"prueba": f"{r.variable} (reverificacion)", "frecuencia": r.frecuencia, "metrica": "correlacion",
          "valor": r.correlacion, "n": r.n, "usa_holdout": True,
          "nota": f"re-verificacion con alineacion corregida; original {tabla[(tabla.frecuencia == r.frecuencia) & (tabla.variable == r.variable) & (tabla.alineacion == 'orig')].correlacion.iloc[0]:.3f}"}
         for r in corr.itertuples()],
        issue="#14", script="68_features_semanales_mensuales_corregido.py")
    supera = corr[(corr["correlacion"].abs() >= UMBRAL_CORR) & (corr["p_valor"] < 0.05)]
    print(f"\nSuperan |r|>={UMBRAL_CORR} y p<0.05 con la alineacion corregida: {len(supera)}")


if __name__ == "__main__":
    main()
