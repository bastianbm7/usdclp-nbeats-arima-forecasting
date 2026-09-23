# CORRECCION (2026-09-23) - realinea (NO regenera) los datasets walk-forward
# diarios de 26 (CLP, h2), 35 (CLP, h3), 39 (CLP, h5), 45/46/47 (AUD/CAD/NZD).
#
# Por que realinear y no regenerar (justificacion, ver tambien
# alineacion_temporal.py): cada fila D de esos datasets contiene
#   - y: precio FX de la barra Yahoo D (~20:00 NY de D-1)
#   - nhits_h*: forecast NHITS calculado con la serie FX hasta y(D) (cutoff <= D)
#   - vol_garch, macd, rsi, precio_min/max_ventana: solo la serie FX hasta y(D)
#   - copper_ret_1d, copper_mom_5d: cobre con merge fecha <= D  <-- MAL ALINEADO
# Todo salvo las columnas de cobre usa exclusivamente informacion de la propia
# serie FX disponible al momento del precio y(D): su timestamp real cambia de
# nombre (D -> tarde de D-1) pero no su contenido ni su causalidad. Regenerar
# NHITS (~50 min por dataset, y sin semilla fija en 26/35/39/45-47) solo
# agregaria ruido de reentrenamiento (9.23 ya documento que dos corridas de
# NHITS cambian resultados aguas abajo), sin corregir nada. Lo unico que se
# recalcula es:
#   1. copper_ret_1d/copper_mom_5d con la regla estricta (settlement < precio FX)
#   2. y_next desde la serie FX cruda (proximo precio real), no desde la fila
#      siguiente del dataset: 26 lo calculaba con dataset["y"].shift(-1) y en 3
#      filas el dataset salta un dia de la serie cruda (auditoria align26.py)
#   3. se eliminan las filas cuyo precio es un precio repetido de Yahoo
#      (feriado / hueco de liquidez), y y_next apunta al proximo precio NUEVO
#   4. se agrega ds_reloj (fecha real del precio FX) solo como documentacion

import importlib

import numpy as np
import pandas as pd

alin = importlib.import_module("alineacion_temporal")
features_mod = importlib.import_module("19_features_nuevas_validacion")

RESULTADOS_DIR = "../datos/resultados"
BASES_DIR = "../datos/bases"

DATASETS = {  # nombre -> (csv original, fuente de la serie FX cruda)
    "clp": ("dataset_entrenamiento_rl_diario.csv", "CLP"),
    "clp_h3": ("dataset_entrenamiento_rl_diario_h3.csv", "CLP"),
    "clp_h5": ("dataset_entrenamiento_rl_diario_h5.csv", "CLP"),
    "aud": ("dataset_entrenamiento_rl_diario_aud.csv", "AUDUSD=X"),
    "cad": ("dataset_entrenamiento_rl_diario_cad.csv", "CAD=X"),
    "nzd": ("dataset_entrenamiento_rl_diario_nzd.csv", "NZDUSD=X"),
}


def serie_fx_cruda(fuente):
    if fuente == "CLP":
        s = pd.read_csv(f"{BASES_DIR}/usdclp_long.csv", parse_dates=["ds"])[["ds", "y"]]
    else:  # 45-47 tomaban la serie del panel de 23 (ya invertida a USD por unidad)
        p = pd.read_csv(f"{BASES_DIR}/panel_fx_diario.csv", parse_dates=["ds"])
        s = p[p["par"] == fuente][["ds", "y"]]
    return s.sort_values("ds").reset_index(drop=True)


def realinear(df, fuente, macro):
    cruda = serie_fx_cruda(fuente)
    fresca = cruda[~alin.marcar_precios_repetidos(cruda["y"])].reset_index(drop=True)
    fresca = alin.agregar_features_cobre(fresca, macro)
    fresca["y_next_crudo"] = fresca["y"].shift(-1)
    fresca["ds_salida"] = fresca["ds"].shift(-1)

    out = df.drop(columns=["copper_ret_1d", "copper_mom_5d", "y_next"]).merge(
        fresca[["ds", "y", "copper_ret_1d", "copper_mom_5d", "y_next_crudo", "ds_salida"]], on="ds", how="inner", suffixes=("", "_crudo"))
    assert np.allclose(out["y"], out["y_crudo"]), "el precio del dataset no coincide con la serie cruda"
    out = out.drop(columns=["y_crudo"]).rename(columns={"y_next_crudo": "y_next"})
    out["ds_reloj"] = alin.fecha_reloj_fx_yahoo(out["ds"])
    out = out.dropna(subset=["y_next", "copper_ret_1d", "copper_mom_5d"]).reset_index(drop=True)
    return out, len(df) - len(out)


if __name__ == "__main__":
    macro = features_mod.cargar_macro()
    filas = []
    for nombre, (csv, fuente) in DATASETS.items():
        orig = pd.read_csv(f"{RESULTADOS_DIR}/{csv}", parse_dates=["ds"])
        nuevo, eliminadas = realinear(orig, fuente, macro)
        salida = csv.replace(".csv", "_alineado.csv")
        nuevo.to_csv(f"{RESULTADOS_DIR}/{salida}", index=False)

        def chequeo(d):
            ret_hoy = np.log(d["y"] / d["y"].shift(1))
            ret_sig = (d["y_next"] - d["y"]) / d["y"]
            return d["copper_ret_1d"].corr(ret_hoy), d["copper_ret_1d"].corr(ret_sig)
        c0_o, c1_o = chequeo(orig)
        c0_n, c1_n = chequeo(nuevo)
        nh = [c for c in orig.columns if c.startswith("nhits_")]
        comun = orig.merge(nuevo[["ds"] + nh], on="ds", suffixes=("", "_n"))
        filas.append({"dataset": nombre, "filas_original": len(orig), "filas_alineado": len(nuevo), "filas_eliminadas": eliminadas,
                      "nhits_intactos": bool(all(np.allclose(comun[c], comun[c + "_n"]) for c in nh)),
                      "orig_corr_cobre_vs_retorno_que_termina_en_la_fila": c0_o, "orig_corr_cobre_vs_retorno_siguiente_(el_operado)": c1_o,
                      "alin_corr_cobre_vs_retorno_que_termina_en_la_fila_(contemp)": c0_n, "alin_corr_cobre_vs_retorno_siguiente_(operable)": c1_n,
                      "archivo": salida})
    t = pd.DataFrame(filas)
    t.to_csv(f"{RESULTADOS_DIR}/correccion_realineado_datasets_chequeo.csv", index=False)
    print(t.round(3).to_string(index=False))
