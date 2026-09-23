# Issue #15, segunda extension pedida por Bastian (solo stop-loss, sin
# take-profit): ¿el monitoreo diario sirve para cerrar antes las POSICIONES que
# caen mucho, y un trailing stop asegura ganancias sin esperar a que se den
# vuelta?
#
# 83 probo un stop sobre la cartera COMPLETA dentro del mes (ninguna variante
# mejoro el Sharpe). Aca, tres reglas distintas, fijadas antes de correr:
#   A. "stop_moneda": cada moneda se cierra sola si su retorno desde que se
#      abrio la posicion (spot + carry, en la direccion de la posicion) cae a
#      <= -k * sigma_i. El resto de la cartera sigue abierta.
#   B. "trailing_moneda": igual, pero medido desde el MAXIMO que alcanzo esa
#      posicion desde que se abrio: cierra si retrocede k * sigma_i desde su
#      mejor punto. Si la posicion sigue abierta varios meses (el carry suele
#      mantener las mismas monedas), el maximo se arrastra entre meses: asegura
#      ganancias acumuladas, no solo las del mes.
#   C. "trailing_cartera": sobre la cartera completa, desde el maximo del
#      retorno acumulado del mes (el maximo parte en 0 al rebalancear).
#   - sigma_i = vol anual ex-ante de la moneda al abrir la posicion / sqrt(12)
#     (su movimiento tipico de un mes; ej. BRL ~4%, EUR ~2.5%); en C,
#     sigma = vol objetivo / sqrt(12) = 1.44%.
#   - k en {1.0, 1.5, 2.0}: 9 variantes por estrategia, 27 en total.
#   - Monitoreo cada dia habil H.10 (mediodia NY) con el ultimo precio conocido;
#     la salida se ejecuta el dia habil SIGUIENTE al gatillo; spread ida+vuelta
#     sobre lo que se cierra. Lo cerrado queda en caja hasta el proximo
#     rebalanceo, donde vuelve a entrar si la senal lo pide (con seguimiento
#     nuevo).
#   - Solo datos < 2025-01-01; el hold-out no se reabre.

import importlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cf = importlib.import_module("cartera_fx")
proto = importlib.import_module("protocolo_evaluacion")
ce = importlib.import_module("costos_y_estadistica")
m83 = importlib.import_module("83_cartera_fx_stop_take_profit")

RESULTADOS_DIR = "../datos/resultados"
SCRIPT = "84_cartera_fx_stops_por_posicion_trailing.py"
METODOS = ["stop_moneda", "trailing_moneda", "trailing_cartera"]
KS = [1.0, 1.5, 2.0]


def indice_retorno_total(D, P):
    """Indice diario de retorno total por moneda (spot + carry a prorrata),
    encadenado mes a mes sobre la grilla H.10 - lo que 've' el monitoreo."""
    ds = D["ds"]
    TRI = pd.DataFrame(np.nan, index=P.index, columns=P.columns)
    nivel = pd.Series(1.0, index=P.columns)
    for t in range(len(ds) - 1):
        tramo = P[(P.index >= ds[t]) & (P.index <= ds[t + 1])]
        if tramo.empty:
            continue
        base = tramo.iloc[0]
        fact = D["fact"].iloc[t].reindex(P.columns).fillna(1.0)
        frac = np.array([(d - ds[t]).days / max((ds[t + 1] - ds[t]).days, 1) for d in tramo.index])
        rel = (tramo / base).ffill()
        tr = rel * np.power(fact.to_numpy()[None, :], frac[:, None])
        TRI.loc[tramo.index] = (tr * nivel).to_numpy()
        ultimo = TRI.loc[tramo.index[-1]]
        nivel = ultimo.where(ultimo.notna(), nivel)
    return TRI


def backtest(D, P, TRI, cfg, metodo, k, filtro=None):
    """filtro(D, t, w) -> conjunto de monedas a las que se les aplica el stop
    (None = todas). Lo usa 85 para probar el stop solo en un grupo."""
    ds, mon = D["ds"], D["monedas"]
    s = pd.Series(cf.SPREADS).reindex(mon)
    sigma_cartera = cfg["vol_obj"] / np.sqrt(12)
    w_prev_deriva = pd.Series(0.0, index=mon)
    seg = {}  # moneda -> {"signo", "entrada", "pico", "sigma"} (posiciones con seguimiento abierto)
    filas = []
    for t in range(len(ds) - 1):
        w, v_ex_ante, sat = cf.pesos_config(D, t, cfg)
        w = (w if w is not None else pd.Series(dtype=float)).reindex(mon).fillna(0.0)
        x = D["X"].iloc[t].reindex(mon).fillna(0.0)
        xs = D["X_spot"].iloc[t].reindex(mon).fillna(0.0)
        base = D["PE"].iloc[t].reindex(mon)
        fact = D["fact"].iloc[t].reindex(mon).fillna(1.0)
        dias_mes = (ds[t + 1] - ds[t]).days
        dw = (w - w_prev_deriva).abs()
        costo_rot = float((s * dw).sum())

        # seguimiento por moneda: se mantiene si la posicion sigue en la misma direccion
        if metodo != "trailing_cartera":
            vol_i = cf.vol_activos(D, t).reindex(mon)
            elegibles_stop = set(mon) if filtro is None else set(filtro(D, t, w))
            for m in mon:
                sg = np.sign(w[m])
                if sg == 0 or m not in elegibles_stop:
                    seg.pop(m, None)
                elif m not in seg or seg[m]["signo"] != sg:
                    if ds[t] in TRI.index and np.isfinite(TRI.at[ds[t], m]) and np.isfinite(vol_i[m]):
                        seg[m] = {"signo": sg, "entrada": TRI.at[ds[t], m], "pico": 0.0, "sigma": vol_i[m] / np.sqrt(12)}

        camino = P[(P.index > ds[t]) & (P.index < ds[t + 1])]
        salida = {}  # moneda -> relativo spot a la salida y fraccion del mes
        pico_cartera = 0.0
        for j in range(len(camino) - 1):  # el ultimo dia no puede ejecutar al dia siguiente dentro del mes
            d = camino.index[j]
            gatillo = []
            if metodo == "trailing_cartera":
                rel = (camino.iloc[j] / base).where(w != 0, 1.0)
                R = float((w * (rel * fact ** ((d - ds[t]).days / dias_mes) - 1)).fillna(0.0).sum())
                pico_cartera = max(pico_cartera, R)
                if pico_cartera - R >= k * sigma_cartera:
                    gatillo = [m for m in mon if w[m] != 0]
            else:
                for m, st in list(seg.items()):
                    if m in salida or not np.isfinite(TRI.at[d, m]):
                        continue
                    r_pos = st["signo"] * (TRI.at[d, m] / st["entrada"] - 1)
                    st["pico"] = max(st["pico"], r_pos)
                    umbral = -k * st["sigma"] if metodo == "stop_moneda" else st["pico"] - k * st["sigma"]
                    if r_pos <= umbral:
                        gatillo.append(m)
            if gatillo:
                d2 = camino.index[j + 1]
                rel2 = (camino.iloc[j + 1] / base)
                for m in gatillo:
                    if m in salida:
                        continue
                    r_rel = rel2[m] if np.isfinite(rel2[m]) else (camino.iloc[j][m] / base[m])
                    salida[m] = (r_rel, (d2 - ds[t]).days / dias_mes, (ds[t + 1] - d2).days)
                    seg.pop(m, None)
                if metodo == "trailing_cartera":
                    break

        r_bruto, r_spot, costo_salida, dias_fuera_pond = 0.0, 0.0, 0.0, 0.0
        w_deriva = pd.Series(0.0, index=mon)
        for m in mon:
            if w[m] == 0:
                continue
            if m in salida:
                rel, frac, dias_fuera = salida[m]
                r_i, rs_i = rel * fact[m] ** frac - 1, rel - 1
                costo_salida += s[m] * abs(w[m] * rel)
                dias_fuera_pond += abs(w[m]) * dias_fuera
            else:
                r_i, rs_i = x[m], xs[m]
                w_deriva[m] = w[m] * (1 + xs[m])
            r_bruto += w[m] * r_i
            r_spot += w[m] * rs_i
        w_prev_deriva = w_deriva / (1 + r_bruto) if (1 + r_bruto) > 0 else w_deriva * 0
        filas.append({"ds": ds[t + 1], "ds_decision": ds[t], "r_bruto": r_bruto,
                      "r_neto": r_bruto - costo_rot - costo_salida,
                      "r_neto_estricto": r_bruto - costo_rot - costo_salida - float((s * w.abs()).sum()),
                      "r_spot": r_spot, "r_carry": r_bruto - r_spot,
                      "turnover": float(dw.sum()), "bruto_exposicion": float(w.abs().sum()),
                      "neto_exposicion": float(w.sum()), "vol_ex_ante": v_ex_ante, "saturado": bool(sat),
                      "n_cierres": len(salida),
                      "frac_expuesta_fuera": dias_fuera_pond / (float(w.abs().sum()) * dias_mes) if w.abs().sum() > 0 else 0.0})
    return pd.DataFrame(filas)


def main():
    D = cf.cargar()
    P = m83.grilla_diaria(D)
    TRI = indice_retorno_total(D, P)
    elegida = json.load(open(f"{RESULTADOS_DIR}/issue15_config_elegida.json"))["config"]
    estrategias = {"Carry": elegida["carry"], "Momentum 6m": elegida["tsmom"], "Combinacion 50/50": elegida}
    riesgo = cf.cargar_riesgo()
    base83 = pd.read_csv(f"{RESULTADOS_DIR}/issue15_stop_take_profit_pre_holdout.csv")
    filas, curvas = [], {}
    for nombre, cfg in estrategias.items():
        b = base83[(base83["estrategia_base"] == nombre) & base83["k_sl"].isna() & base83["k_tp"].isna()].iloc[0].to_dict()
        b.update({"metodo": "sin stop", "k": np.nan, "cierres_por_anio": 0.0, "%_exposicion_fuera": 0.0})
        filas.append(b)
        for metodo in METODOS:
            for k in KS:
                bt = cf.recortar(backtest(D, P, TRI, cfg, metodo, k), "pre")
                mt = cf.metricas(bt, f"{nombre} | {metodo} {k}", riesgo)
                mt.update({"estrategia_base": nombre, "metodo": metodo, "k": k,
                           "cierres_por_anio": 12 * bt["n_cierres"].mean(),
                           "%_exposicion_fuera": 100 * bt["frac_expuesta_fuera"].mean()})
                filas.append(mt)
                curvas[(nombre, metodo, k)] = bt.set_index("ds")["r_neto"]
    tabla = pd.DataFrame(filas)
    tabla.to_csv(f"{RESULTADOS_DIR}/issue15_stops_por_posicion_trailing_pre_holdout.csv", index=False)

    nuevas = tabla[tabla["metodo"] != "sin stop"]
    proto.registrar_pruebas(
        [{"prueba": r.estrategia, "frecuencia": "mensual (stop diario)", "metrica": "sharpe_neto_pre_holdout",
          "valor": r.sharpe_neto, "n": r.meses, "usa_holdout": False, "nota": "extension stops por posicion / trailing"}
         for r in nuevas.itertuples()], issue="#15", script=SCRIPT)
    n_total = proto.n_pruebas_registradas("#15")
    umbral = ce.sharpe_maximo_esperado_nulo(n_total, int(nuevas["meses"].max()), periodos=12)
    pd.set_option("display.width", 250)
    cols = ["estrategia_base", "metodo", "k", "sharpe_neto", "ic95_bajo_neto", "ic95_alto_neto", "ret_anual_%_neto",
            "max_dd_%_neto", "peor_mes_%", "ret_%_2008_ago_dic", "ret_%_2020_feb_mar", "cierres_por_anio", "%_exposicion_fuera"]
    print(f"2000-01 a 2024-12 (antes del hold-out). Sharpe maximo esperado por azar con {n_total} pruebas del #15: {umbral:.2f}\n")
    print(tabla[cols].round(2).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), facecolor="#fcfcfb")
    colores = {"sin stop": "#2a78d6", "stop_moneda": "#eb6834", "trailing_moneda": "#1baf7a", "trailing_cartera": "#898781"}
    for ax, nombre in zip(axes, estrategias):
        base = base83[(base83["estrategia_base"] == nombre)]
        bt0 = cf.recortar(cf.backtest(D, estrategias[nombre])[0], "pre").set_index("ds")["r_neto"]
        ax.plot(bt0.index, 100 * np.cumprod(1 + bt0), color=colores["sin stop"], lw=1.9, label="sin stop")
        for metodo in METODOS:
            sub = tabla[(tabla["estrategia_base"] == nombre) & (tabla["metodo"] == metodo)].sort_values("sharpe_neto")
            k = sub.iloc[-1]["k"]
            r = curvas[(nombre, metodo, k)]
            ax.plot(r.index, 100 * np.cumprod(1 + r), color=colores[metodo], lw=1.2, label=f"{metodo.replace('_', ' ')} {k}σ (el mejor k)")
        ax.set_facecolor("#fcfcfb")
        ax.set_title(f"{nombre}: $100 en 2000, neto de costos", loc="left", fontsize=11)
        ax.grid(axis="y", color="#e1e0d9"); ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for a, b in [("2008-08-01", "2008-12-31"), ("2020-02-01", "2020-03-31")]:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#f0efe9", zorder=0)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(0.01, 0.01, "k en multiplos de sigma mensual (por moneda: su vol ex-ante/sqrt(12); cartera: 1.44%). "
             "Para cada metodo se muestra el mejor k de 3 - favorece al stop. Sombreado: 2008-08..12 y 2020-02..03.",
             fontsize=8, color="#898781")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{RESULTADOS_DIR}/issue15_stops_por_posicion_trailing.png", dpi=120, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
