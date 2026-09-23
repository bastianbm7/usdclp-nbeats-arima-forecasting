# Issue #15, tercera extension pedida por Bastian:
#   A. Momentum de 3 meses vs. 6 meses. Las versiones solas ya estaban en la
#      seleccion de 81 (no se re-prueban, se leen de su CSV); se agrega UNA
#      prueba nueva: la combinacion carry + momentum 3m, que es como se usaria.
#   B. Carry: ¿el stop funciona distinto en la pata LARGA (las 2 monedas de
#      tasa mas alta, las que sufren en los crashes) que en la CORTA (las 2 de
#      tasa mas baja, tipicamente refugios como JPY/CHF)? Stop por moneda y
#      trailing por moneda (reglas de 84) aplicados SOLO a una pata.
#   C. Momentum 6m: las monedas se dividen en cada rebalanceo en dos mitades
#      por su volatilidad ex-ante (mas volatiles / menos volatiles); stop y
#      trailing aplicados SOLO a una mitad.
#   Ademas, sin stops, cuanto aporta cada grupo al retorno (contexto: no son
#   pruebas nuevas, es una descomposicion de la misma estrategia).
#   Grilla fijada antes de correr: k en {1.0, 1.5, 2.0}; B y C = 2 reglas x 2
#   grupos x 3 k = 12 variantes cada una. Total nuevo: 25 pruebas, anotadas en
#   el registro. Solo datos < 2025-01-01; el hold-out no se reabre.

import importlib
import json

import numpy as np
import pandas as pd

cf = importlib.import_module("cartera_fx")
proto = importlib.import_module("protocolo_evaluacion")
ce = importlib.import_module("costos_y_estadistica")
m83 = importlib.import_module("83_cartera_fx_stop_take_profit")
m84 = importlib.import_module("84_cartera_fx_stops_por_posicion_trailing")

RESULTADOS_DIR = "../datos/resultados"
SCRIPT = "85_cartera_fx_stops_por_grupo.py"
KS = [1.0, 1.5, 2.0]
METODOS = ["stop_moneda", "trailing_moneda"]


def pata_larga(D, t, w):
    return [m for m in w.index if w[m] > 0]


def pata_corta(D, t, w):
    return [m for m in w.index if w[m] < 0]


def _mitades_vol(D, t, w):
    activos = [m for m in w.index if w[m] != 0]
    v = cf.vol_activos(D, t).reindex(activos).dropna().sort_values()
    corte = len(v) // 2
    return list(v.index[corte:]), list(v.index[:corte])


def vol_alta(D, t, w):
    return _mitades_vol(D, t, w)[0]


def vol_baja(D, t, w):
    return _mitades_vol(D, t, w)[1]


def aporte_por_grupo(D, cfg, grupos):
    """Sin stops: retorno bruto aportado por cada grupo (fraccion del capital)."""
    bt, W = cf.backtest(D, cfg)
    W = W.fillna(0.0)
    filas = []
    for t, d in enumerate(D["ds"][:-1]):
        if d not in W.index:
            continue
        w = W.loc[d]
        x = D["X"].iloc[t].reindex(w.index).fillna(0.0)
        fila = {"ds": D["ds"][t + 1]}
        for nombre, f in grupos.items():
            g = f(D, t, w)
            fila[nombre] = float((w[g] * x[g]).sum()) if g else 0.0
        filas.append(fila)
    a = cf.recortar(pd.DataFrame(filas), "pre")
    return {n: {"aporte_anual_%": 100 * a[n].mean() * 12, "sharpe_bruto_del_aporte": ce.sharpe(a[n], 12)} for n in grupos}


def fila_metricas(bt, nombre, riesgo, extra):
    m = cf.metricas(bt, nombre, riesgo)
    m.update({"cierres_por_anio": 12 * bt["n_cierres"].mean(),
              "%_exposicion_fuera": 100 * bt["frac_expuesta_fuera"].mean()})
    m.update(extra)
    return m


def main():
    D = cf.cargar()
    P = m83.grilla_diaria(D)
    TRI = m84.indice_retorno_total(D, P)
    elegida = json.load(open(f"{RESULTADOS_DIR}/issue15_config_elegida.json"))["config"]
    riesgo = cf.cargar_riesgo()
    c81 = pd.read_csv(f"{RESULTADOS_DIR}/issue15_pre_holdout_configuraciones.csv").set_index("estrategia")
    cols = ["sharpe_neto", "ic95_bajo_neto", "ic95_alto_neto", "ret_anual_%_neto", "max_dd_%_neto", "peor_mes_%",
            "ret_%_2008_ago_dic", "ret_%_2020_feb_mar"]
    pd.set_option("display.width", 250)
    nuevas = []

    # --- A. momentum 3m vs 6m ---
    tabla_a = c81.loc[["tsmom_k3_vt5", "tsmom_k6_vt5", "combo[carry_n2_vt5+tsmom_k6_vt5]_vt5"], cols + ["turnover_anual"]].copy()
    tabla_a["origen"] = "81 (ya probada)"
    cfg_combo3 = {"tipo": "combo", "carry": elegida["carry"], "tsmom": {"tipo": "tsmom", "lookback": 3, "vol_obj": 0.05}, "vol_obj": 0.05}
    m = cf.metricas(cf.recortar(cf.backtest(D, cfg_combo3)[0], "pre"), "combo[carry_n2_vt5+tsmom_k3_vt5]_vt5", riesgo)
    fila = pd.DataFrame([m]).set_index("estrategia")[cols + ["turnover_anual"]]
    fila["origen"] = "85 (nueva)"
    tabla_a = pd.concat([tabla_a, fila])
    nuevas.append(m)
    print("=== A. Momentum 3 meses vs 6 meses (2000-2024, neto) ===")
    print(tabla_a.round(2).to_string())

    # --- B y C ---
    bloques = {
        "B. Carry: stop solo en una pata": (elegida["carry"], {"pata larga (tasas altas)": pata_larga, "pata corta (tasas bajas)": pata_corta}),
        "C. Momentum 6m: stop solo en una mitad por volatilidad": (elegida["tsmom"], {"mitad mas volatil": vol_alta, "mitad menos volatil": vol_baja}),
    }
    todas = [tabla_a.reset_index().assign(bloque="A")]
    for titulo, (cfg, grupos) in bloques.items():
        aporte = aporte_por_grupo(D, cfg, grupos)
        print(f"\n=== {titulo} ===")
        print("Aporte de cada grupo SIN stops (bruto, antes de costos):")
        for g, v in aporte.items():
            print(f"   {g:28s} {v['aporte_anual_%']:+.2f}% al anio   Sharpe del aporte {v['sharpe_bruto_del_aporte']:.2f}")
        filas = [dict(cf.metricas(cf.recortar(cf.backtest(D, cfg)[0], "pre"), "sin stop", riesgo),
                      grupo="-", metodo="sin stop", k=np.nan, cierres_por_anio=0.0)]
        for g, f in grupos.items():
            for metodo in METODOS:
                for k in KS:
                    bt = cf.recortar(m84.backtest(D, P, TRI, cfg, metodo, k, filtro=f), "pre")
                    mt = fila_metricas(bt, f"{titulo[:1]} | {g} | {metodo} {k}", riesgo, {"grupo": g, "metodo": metodo, "k": k})
                    filas.append(mt)
                    nuevas.append(mt)
        t = pd.DataFrame(filas)
        print(t[["grupo", "metodo", "k"] + cols + ["cierres_por_anio"]].round(2).to_string(index=False))
        todas.append(t.assign(bloque=titulo[:1]))

    pd.concat(todas, ignore_index=True).to_csv(f"{RESULTADOS_DIR}/issue15_stops_por_grupo_pre_holdout.csv", index=False)
    proto.registrar_pruebas(
        [{"prueba": r["estrategia"], "frecuencia": "mensual (stop diario)", "metrica": "sharpe_neto_pre_holdout",
          "valor": r["sharpe_neto"], "n": r["meses"], "usa_holdout": False, "nota": "extension stops por grupo / momentum 3m"}
         for r in nuevas], issue="#15", script=SCRIPT)
    n_total = proto.n_pruebas_registradas("#15")
    print(f"\nPruebas registradas en el Issue #15: {n_total}. Sharpe maximo esperado por azar: "
          f"{ce.sharpe_maximo_esperado_nulo(n_total, 300, periodos=12):.2f}")


if __name__ == "__main__":
    main()
