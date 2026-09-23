# Issue #14: prueba de CONFIRMACION de la unica pista de 70 - el nivel del VIX.
#
# En 70, vix_log fue la variable mas consistente sobre USD/CLP (correlacion
# negativa con el retorno siguiente en semanal -0.076 y mensual -0.163, mismo
# signo en las dos mitades; Sharpe neto semanal 0.66 [0.11, 1.20]), pero NO
# sobrevive la correccion por pruebas multiples (q = 0.39) y su Sharpe apenas
# supera el maximo esperado por azar entre las pruebas hechas (0.59). Con el
# criterio fijado en 70, el hold-out no se abre.
#
# Lectura economica de la pista: despues de semanas de estres (VIX alto), las
# monedas emergentes se recuperan (prima por riesgo que se cobra por mantenerlas
# en el panico). Si es real, deberia aparecer en OTRAS monedas emergentes que
# no se usaron para formular la hipotesis. Hipotesis fijada ANTES de correr
# (sin reestimar signo ni parametros por moneda):
#   - Regla: semana t, si log(VIX) conocido > mediana expansiva de log(VIX) del
#     train (hasta el inicio del anio), largo la moneda local contra el USD; si
#     no, corto. Misma regla y mismo walk-forward que 70 (2015-2024).
#   - Grupo principal (EM, la prediccion es Sharpe > 0): BRL, MXN, COP, ZAR, PEN.
#   - Grupo secundario (G10 sensibles al riesgo): AUD, NZD, CAD, NOK.
#   - Control (refugios, la prediccion es lo contrario o nada): JPY, CHF.
# Se mide correlacion (vix_log vs retorno siguiente de la moneda local) y
# Sharpe bruto / neto de spread por rotacion. Limitacion: no incluye el carry
# de cada moneda (la regla esta ~mitad del tiempo larga y ~mitad corta, asi que
# en promedio se compensa, pero no es exacto). Tampoco es una prueba
# independiente pura: las monedas EM estan correlacionadas entre si.
# No se usa ningun dato >= 2025-01-01.

import numpy as np
import pandas as pd
from scipy import stats

import alineacion_temporal as alin
import costos_y_estadistica as ce
import protocolo_evaluacion as prot

RESULTADOS_DIR = "../datos/resultados"
PANEL = "../datos/bases/panel_fx_diario_alineado.csv"
EXTERNAS = "../datos/bases/issue14_variables_externas.csv"
SCRIPT = "71_confirmacion_vix_otras_monedas.py"
INICIO_TEST = pd.Timestamp("2015-01-01")
GRUPOS = {
    "EM (principal)": ["BRL=X", "MXN=X", "COP=X", "ZAR=X", "PEN=X"],
    "G10 riesgo (secundario)": ["AUDUSD=X", "NZDUSD=X", "CAD=X", "USDNOK=X"],
    "refugio (control)": ["JPY=X", "CHF=X"],
    "referencia (formulo la hipotesis)": ["CLP=X"],
}


def semanal_moneda(panel, par, vix):
    d = panel[panel["par"] == par][["ds", "y"]].sort_values("ds")
    # el panel viene en USD por unidad de moneda local (alineado y sin repetidos, 60):
    # un retorno positivo = la moneda local se aprecia
    g = d.set_index("ds")["y"]
    s = pd.DataFrame({"y": g.resample("W").last(), "ds_fila": g.resample("W").apply(lambda x: x.index.max())}).dropna().reset_index()
    s = s.iloc[:-1]
    s["ret_local_fut"] = s["y"].shift(-1) / s["y"] - 1
    vals, _ = alin.valor_conocido(alin.ts_fx_yahoo(s["ds_fila"]),
                                  pd.DatetimeIndex(pd.to_datetime(vix["ts_conocido_utc"], utc=True)), vix["valor"])
    s["vix_log"] = np.log(vals)
    return s[s["ds"] < prot.HOLDOUT_INICIO].reset_index(drop=True)


def regla_vix(s):
    pos = np.full(len(s), np.nan)
    for anio in sorted(s.loc[s["ds"] >= INICIO_TEST, "ds"].dt.year.unique()):
        inicio = pd.Timestamp(f"{anio}-01-01")
        mediana = s.loc[s["ds"] < inicio, "vix_log"].median()
        test = (s["ds"].dt.year == anio).to_numpy()
        pos[test] = np.where(s.loc[test, "vix_log"] > mediana, 1.0, -1.0)
    ok = ~np.isnan(pos) & s["ret_local_fut"].notna().to_numpy() & s["vix_log"].notna().to_numpy()
    return pos[ok], s["ret_local_fut"].to_numpy()[ok]


def main():
    panel = pd.read_csv(PANEL, parse_dates=["ds"])
    ext = pd.read_csv(EXTERNAS, parse_dates=["ds"])
    vix = ext[ext["serie"] == "VIXCLS"]
    filas, retornos_grupo = [], {}
    for grupo, pares in GRUPOS.items():
        for par in pares:
            s = semanal_moneda(panel, par, vix)
            v = s["vix_log"].notna() & s["ret_local_fut"].notna()
            r, p = stats.pearsonr(s.loc[v, "vix_log"], s.loc[v, "ret_local_fut"])
            pos, ret = regla_vix(s)
            spread = ce.spread_de(par)
            bruto = pos * ret
            neto = ce.retornos_posicion_fija(pos, ret, spread, "rotacion")
            lo, hi = ce.ic_sharpe_bootstrap(neto, 52, bloque=8)
            filas.append({"grupo": grupo, "par": par, "corr_vix_ret_local_sig": r, "p_valor": p, "n_corr": int(v.sum()),
                          "sharpe_bruto": ce.sharpe(bruto, 52), "sharpe_neto_rotacion": ce.sharpe(neto, 52),
                          "ic95_bajo": lo, "ic95_alto": hi, "spread_ida_vuelta": spread, "n_semanas": len(neto)})
            retornos_grupo.setdefault(grupo, []).append(pd.Series(neto))
    tabla = pd.DataFrame(filas)

    # cartera equiponderada por grupo (misma longitud: semanas 2015-2024)
    for grupo, lista in retornos_grupo.items():
        n = min(len(x) for x in lista)
        cartera = np.mean([x.to_numpy()[-n:] for x in lista], axis=0)
        lo, hi = ce.ic_sharpe_bootstrap(cartera, 52, bloque=8)
        tabla = pd.concat([tabla, pd.DataFrame([{"grupo": grupo, "par": "CARTERA equiponderada",
                                                 "sharpe_neto_rotacion": ce.sharpe(cartera, 52), "ic95_bajo": lo, "ic95_alto": hi,
                                                 "n_semanas": n}])], ignore_index=True)
    tabla.to_csv(f"{RESULTADOS_DIR}/issue14_confirmacion_vix_otras_monedas.csv", index=False)
    pd.set_option("display.width", 200)
    print("Regla fija: VIX sobre su mediana de train -> largo moneda local vs USD (semanal, 2015-2024, sin hold-out)")
    print(tabla.round(3).to_string(index=False))

    prot.registrar_pruebas(
        [{"prueba": f"confirmacion vix_log {r.par}", "frecuencia": "semanal", "metrica": "sharpe_neto_rotacion_pre_holdout",
          "valor": r.sharpe_neto_rotacion, "n": r.n_semanas, "usa_holdout": False, "nota": r.grupo}
         for r in tabla.itertuples() if r.par != "CARTERA equiponderada" and r.par != "CLP=X"],
        issue="#14", script=SCRIPT)


if __name__ == "__main__":
    main()
