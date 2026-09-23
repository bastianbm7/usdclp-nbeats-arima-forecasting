# Issue #15, extension pedida por Bastian: stop-loss y take-profit DENTRO del
# mes para carry, momentum de 6 meses y la combinacion elegida en 81.
#
# Pregunta: ¿cortar la posicion cuando el mes va mal evita las caidas grandes
# del carry (2008, 2020) sin destruir el retorno del resto del tiempo?
#
# Como se implementa (sin mirar el futuro):
#   - La cartera se arma a fin de mes igual que en 81 (mismos pesos, misma
#     vol objetivo). Durante el mes se marca a mercado TODOS LOS DIAS habiles
#     de H.10 (mediodia NY) con el ultimo precio conocido de cada moneda
#     (CLP: ultima barra Yahoo anterior), mas el carry devengado a prorrata.
#   - Stop-loss: si el retorno acumulado del mes cae a <= -k_sl * sigma_mes,
#     se cierra TODO al dia habil SIGUIENTE (no al precio del gatillo: la
#     senal se conoce con el precio del dia, la orden se ejecuta despues; si
#     el mercado sigue cayendo, se sale mas abajo - "gap"). Se queda en caja
#     hasta el proximo rebalanceo mensual.
#   - Take-profit: igual, si el retorno acumulado del mes llega a >= +k_tp * sigma_mes.
#   - sigma_mes = vol objetivo / sqrt(12) (5% anual -> 1.44% mensual): el
#     umbral queda en unidades del riesgo que la propia cartera se propone.
#   - Costo: spread ida+vuelta sobre lo que se cierra (y la reentrada del mes
#     siguiente se cobra sola via turnover, como en 81).
#   - Grilla fijada antes de correr: k_sl en {sin, 1.0, 1.5, 2.0}, k_tp en
#     {sin, 1.5, 2.5}; 11 variantes nuevas por estrategia (33 en total), todas
#     anotadas en el registro de pruebas. Solo datos < 2025-01-01: el hold-out
#     de 82 ya se abrio para la configuracion sin stops y NO se reabre para
#     variantes elegidas mirando este resultado.

import importlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

cf = importlib.import_module("cartera_fx")
alin = importlib.import_module("alineacion_temporal")
proto = importlib.import_module("protocolo_evaluacion")
ce = importlib.import_module("costos_y_estadistica")

RESULTADOS_DIR = "../datos/resultados"
SCRIPT = "83_cartera_fx_stop_take_profit.py"
K_SL = [None, 1.0, 1.5, 2.0]
K_TP = [None, 1.5, 2.5]


def grilla_diaria(D):
    """Precio conocido de cada moneda en cada dia habil H.10 (mediodia NY)."""
    d = pd.read_csv(cf.DIARIO, parse_dates=["ds"])
    h10 = d[d["fuente"].str.startswith("FRED")]
    fechas = pd.DatetimeIndex(sorted(h10["ds"].unique()))
    T = alin.ts_fred_h10(fechas) + pd.Timedelta(minutes=1)
    P = pd.DataFrame(index=fechas, columns=D["monedas"], dtype=float)
    for m in D["monedas"]:
        dm = d[d["moneda"] == m].sort_values("ts_utc")
        vals, ts_usado = alin.valor_conocido(T, pd.DatetimeIndex(pd.to_datetime(dm["ts_utc"], utc=True)), dm["precio"])
        vals = pd.Series(vals, index=fechas)
        viejo = (T - pd.DatetimeIndex(pd.to_datetime(ts_usado, utc=True))) > pd.Timedelta(days=5)
        vals[np.asarray(viejo)] = np.nan
        P[m] = vals
    return P


def backtest_con_stops(D, P, cfg, k_sl, k_tp):
    ds, mon = D["ds"], D["monedas"]
    s = pd.Series(cf.SPREADS).reindex(mon).to_numpy()
    sigma_mes = cfg["vol_obj"] / np.sqrt(12)
    w_prev_deriva = pd.Series(0.0, index=mon)
    filas = []
    for t in range(len(ds) - 1):
        w, v_ex_ante, sat = cf.pesos_config(D, t, cfg)
        w = (w if w is not None else pd.Series(dtype=float)).reindex(mon).fillna(0.0)
        x = D["X"].iloc[t].reindex(mon).fillna(0.0)
        xs = D["X_spot"].iloc[t].reindex(mon).fillna(0.0)
        dw = (w - w_prev_deriva).abs()
        costo_rot = float((s * dw.to_numpy()).sum())
        r_bruto, r_spot = float((w * x).sum()), float((w * xs).sum())
        w_fin = w * (1 + xs)
        evento, dias_fuera = "", 0
        if (k_sl is not None or k_tp is not None) and (w != 0).any():
            base = D["PE"].iloc[t].reindex(mon)
            fact_mes = D["fact"].iloc[t].reindex(mon).fillna(1.0)
            dias_mes = (ds[t + 1] - ds[t]).days
            camino = P[(P.index > ds[t]) & (P.index < ds[t + 1])]
            activos = w != 0
            for j in range(len(camino)):
                rel = (camino.iloc[j] / base).where(activos, 1.0)
                frac = (camino.index[j] - ds[t]).days / dias_mes
                R = float((w * (rel * fact_mes ** frac - 1)).fillna(0.0).sum())
                gatillo = ("stop" if (k_sl is not None and R <= -k_sl * sigma_mes) else
                           "take" if (k_tp is not None and R >= k_tp * sigma_mes) else "")
                if not gatillo:
                    continue
                if j + 1 < len(camino):  # salida el dia habil siguiente
                    rel2 = (camino.iloc[j + 1] / base).where(activos, 1.0)
                    frac2 = (camino.index[j + 1] - ds[t]).days / dias_mes
                    rel2 = rel2.fillna(rel)  # si falta el precio de salida, el ultimo conocido
                    r_bruto = float((w * (rel2 * fact_mes ** frac2 - 1)).sum())
                    r_spot = float((w * (rel2 - 1)).sum())
                    w_fin = w * rel2
                    dias_fuera = (ds[t + 1] - camino.index[j + 1]).days
                    evento = gatillo
                    break
                break  # el gatillo cae el ultimo dia: se sale igual en el rebalanceo
        costo_salida = float((s * w_fin.abs().to_numpy()).sum()) if evento else 0.0
        costo_estr = costo_rot + float((s * w.abs().to_numpy()).sum())
        filas.append({"ds": ds[t + 1], "ds_decision": ds[t], "r_bruto": r_bruto,
                      "r_neto": r_bruto - costo_rot - costo_salida,
                      "r_neto_estricto": r_bruto - costo_estr - costo_salida,
                      "r_spot": r_spot, "r_carry": r_bruto - r_spot,
                      "turnover": float(dw.sum()) + (float(w_fin.abs().sum()) if evento else 0.0),
                      "bruto_exposicion": float(w.abs().sum()), "neto_exposicion": float(w.sum()),
                      "vol_ex_ante": v_ex_ante, "saturado": bool(sat), "evento": evento,
                      "dias_fuera": dias_fuera, "dias_mes": (ds[t + 1] - ds[t]).days})
        if evento:
            w_prev_deriva = pd.Series(0.0, index=mon)
        else:
            w_prev_deriva = w * (1 + xs) / (1 + r_bruto) if (1 + r_bruto) > 0 else w * 0
    return pd.DataFrame(filas)


def main():
    D = cf.cargar()
    P = grilla_diaria(D)
    elegida = json.load(open(f"{RESULTADOS_DIR}/issue15_config_elegida.json"))["config"]
    estrategias = {"Carry": elegida["carry"], "Momentum 6m": elegida["tsmom"], "Combinacion 50/50": elegida}
    riesgo = cf.cargar_riesgo()
    filas, curvas = [], {}
    for nombre, cfg in estrategias.items():
        for k_sl in K_SL:
            for k_tp in K_TP:
                bt = cf.recortar(backtest_con_stops(D, P, cfg, k_sl, k_tp), "pre")
                etiqueta = f"SL {'-' if k_sl is None else k_sl} / TP {'-' if k_tp is None else k_tp}"
                m = cf.metricas(bt, f"{nombre} | {etiqueta}", riesgo)
                m.update({"estrategia_base": nombre, "k_sl": k_sl, "k_tp": k_tp,
                          "stops_por_anio": 12 * (bt["evento"] == "stop").mean(),
                          "takes_por_anio": 12 * (bt["evento"] == "take").mean(),
                          "%_tiempo_fuera": 100 * bt["dias_fuera"].sum() / bt["dias_mes"].sum()})
                filas.append(m)
                curvas[(nombre, etiqueta)] = bt.set_index("ds")["r_neto"]
    tabla = pd.DataFrame(filas)
    tabla.to_csv(f"{RESULTADOS_DIR}/issue15_stop_take_profit_pre_holdout.csv", index=False)

    n_total = proto.n_pruebas_registradas("#15") + int(((tabla["k_sl"].notna()) | (tabla["k_tp"].notna())).sum())
    umbral = ce.sharpe_maximo_esperado_nulo(n_total, int(tabla["meses"].max()), periodos=12)
    pd.set_option("display.width", 250)
    cols = ["estrategia", "sharpe_neto", "ic95_bajo_neto", "ic95_alto_neto", "ret_anual_%_neto", "max_dd_%_neto",
            "peor_mes_%", "ret_%_2008_ago_dic", "ret_%_2020_feb_mar", "stops_por_anio", "takes_por_anio", "%_tiempo_fuera"]
    print(f"2000-01 a 2024-12 (antes del hold-out). Sharpe maximo esperado por azar con {n_total} pruebas del #15: {umbral:.2f}\n")
    print(tabla[cols].round(2).to_string(index=False))

    prot_filas = [{"prueba": r.estrategia, "frecuencia": "mensual (stop diario)", "metrica": "sharpe_neto_pre_holdout",
                   "valor": r.sharpe_neto, "n": r.meses, "usa_holdout": False, "nota": "extension stop-loss/take-profit"}
                  for r in tabla.itertuples() if not (pd.isna(r.k_sl) and pd.isna(r.k_tp))]
    proto.registrar_pruebas(prot_filas, issue="#15", script=SCRIPT)

    # grafico: capital acumulado de carry y momentum, sin stops vs. el mejor stop de cada uno
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), sharey=False, facecolor="#fcfcfb")
    for ax, nombre in zip(axes, ["Carry", "Momentum 6m"]):
        sub = tabla[tabla["estrategia_base"] == nombre]
        mejor = sub[sub["k_sl"].notna() & sub["k_tp"].isna()].sort_values("sharpe_neto").iloc[-1]
        for etiqueta, col, lw in [("SL - / TP -", "#2a78d6", 1.8),
                                  (f"SL {mejor.k_sl} / TP -", "#eb6834", 1.4),
                                  ("SL 1.0 / TP 1.5", "#898781", 1.0)]:
            r = curvas[(nombre, etiqueta)]
            ax.plot(r.index, 100 * np.cumprod(1 + r), color=col, lw=lw, label={"SL - / TP -": "sin stop ni take-profit"}.get(etiqueta, etiqueta))
        ax.set_facecolor("#fcfcfb")
        ax.set_title(f"{nombre}: $100 invertidos en 2000 (neto de costos)", loc="left", fontsize=11)
        ax.grid(axis="y", color="#e1e0d9"); ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for a, b in [("2008-08-01", "2008-12-31"), ("2020-02-01", "2020-03-31")]:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#f0efe9", zorder=0)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.text(0.01, 0.01, "SL/TP en multiplos de la vol mensual objetivo (5%/sqrt(12) = 1.44%). Sombreado: 2008-08..12 y 2020-02..03.", fontsize=8, color="#898781")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{RESULTADOS_DIR}/issue15_stop_take_profit.png", dpi=120, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
