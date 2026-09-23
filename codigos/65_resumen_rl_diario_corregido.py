# CORRECCION (2026-09-23) - metricas de todo lo entrenado por 64 (9.17, 9.20,
# 9.23, 9.24/9.26, 9.28) + los mismos baselines de siempre, recalculados sobre
# las MISMAS decisiones out-of-sample:
#   - PPO: una fila por semilla + resumen (media, min, max del Sharpe).
#   - Umbral cobre: signo(copper_ret_1d) x signo estimado SOLO con el train de
#     cada ventana (antes: signo fijo elegido con toda la muestra).
#   - Umbral simple (forecast NHITS): umbral = mediana de |forecast_rel| del
#     TRAIN de cada ventana (antes: mediana de TODO el dataset, incluido el
#     test - 28:176-177).
#   - Buy-and-hold (sin apalancar; se reporta igual porque el Sharpe no depende
#     del apalancamiento, el retorno total si - leer con eso en mente).
# Economia identica a los entornos corregidos: notional = 3% del capital /
# (1 x vol_garch), salida precomputada por el entorno (TP/SL/trailing/cierre)
# y spread ida+vuelta del par sobre el notional en CADA operacion. Como el
# apalancamiento L = 0.03/vol_garch no depende del capital, el retorno de cada
# decision es r = L*(signo*retorno_operacion - spread): eso permite recalcular
# la sensibilidad a cualquier spread sin volver a simular.

import glob
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ce = importlib.import_module("costos_y_estadistica")
entorno_mod = importlib.import_module("27_entorno_trading_rl_diario")

RESULTADOS_DIR = "../datos/resultados"
ENTRADA_DIR = f"{RESULTADOS_DIR}/correccion_rl"
RIESGO, K = entorno_mod.RIESGO_MAX_PCT, entorno_mod.K_STOP_LOSS


def retornos_riesgo(dec, pos, spread):
    pos = np.asarray(pos, dtype=float)
    y = dec["y"].to_numpy(dtype=float)
    salida = np.where(pos > 0, dec["precio_salida_largo"], np.where(pos < 0, dec["precio_salida_corto"], y)).astype(float)
    L = RIESGO / (K * dec["vol_garch"].to_numpy(dtype=float))
    L = np.where(pos != 0, L, 0.0)
    bruto = L * np.sign(pos) * (salida - y) / y
    return bruto - spread * L, bruto, L


def posiciones_baseline(dec, nombre):
    if nombre == "Umbral cobre":
        return dec["signo_cobre_train"].to_numpy() * np.sign(dec["copper_ret_1d"].to_numpy())
    if nombre == "Umbral simple (forecast)":
        fr = ((dec["nhits_h1"] - dec["y"]) / dec["y"]).to_numpy()
        u = dec["umbral_simple_train"].to_numpy()
        return np.where(fr > u, 1.0, np.where(fr < -u, -1.0, 0.0))
    raise ValueError(nombre)


def fila_metricas(dec, pos, nombre, spread, periodos, extra):
    r_neto, r_bruto, L = retornos_riesgo(dec, pos, spread)
    m = ce.metricas_desde_retornos(r_neto, nombre, periodos=periodos)
    lo_b, hi_b = ce.ic_sharpe_bootstrap(r_bruto, periodos)
    opera = np.asarray(pos) != 0
    m.update({"sharpe_bruto": ce.sharpe(r_bruto, periodos), "ic95_bruto": f"[{lo_b:.2f}, {hi_b:.2f}]",
              "retorno_bruto_%": 100 * (np.prod(1 + r_bruto) - 1), "operaciones": int(opera.sum()),
              "apalancamiento_mediano": float(np.median(L[opera])) if opera.any() else np.nan,
              "breakeven_spread_%": 100 * ce.breakeven_spread(r_bruto[opera], L[opera]) if opera.any() else np.nan,
              "spread_%": 100 * spread, "n_decisiones": len(dec)})
    m.update(extra)
    return m


def resumir_config(archivos):
    decs = {int(pd.read_csv(f, nrows=1)["seed"].iloc[0]): pd.read_csv(f, parse_dates=["ds"]) for f in archivos}
    seed0 = sorted(decs)[0]
    dec0 = decs[seed0]
    exp, par, N, h, spread = dec0["exp"].iloc[0], dec0["par"].iloc[0], int(dec0["N"].iloc[0]), int(dec0["h"].iloc[0]), float(dec0["spread"].iloc[0])
    periodos = 252 / N
    base = {"exp": exp, "par": par, "N": N, "h": h}
    filas, sens = [], []
    sharpes_ppo = []
    for s, dec in sorted(decs.items()):
        pos_c = posiciones_baseline(dec, "Umbral cobre")
        opera = dec["pos_ppo"].to_numpy() != 0
        coinc = 100 * np.mean(np.sign(dec["pos_ppo"].to_numpy()[opera]) == np.sign(pos_c[opera])) if opera.any() else np.nan
        f = fila_metricas(dec, dec["pos_ppo"], f"PPO seed={s}", spread, periodos, {**base, "coincidencia_con_cobre_%": coinc})
        filas.append(f)
        sharpes_ppo.append(f["sharpe"])
    filas.append({**base, "estrategia": "PPO (resumen semillas)", "sharpe": np.mean(sharpes_ppo),
                  "sharpe_min_semillas": np.min(sharpes_ppo), "sharpe_max_semillas": np.max(sharpes_ppo),
                  "n_semillas": len(sharpes_ppo), "spread_%": 100 * spread,
                  "sharpe_bruto": np.mean([x["sharpe_bruto"] for x in filas if x["estrategia"].startswith("PPO seed")]),
                  "retorno_total_%": np.mean([x["retorno_total_%"] for x in filas if x["estrategia"].startswith("PPO seed")])})
    for nombre in ["Umbral cobre", "Umbral simple (forecast)"]:
        filas.append(fila_metricas(dec0, posiciones_baseline(dec0, nombre), nombre, spread, periodos, base))
    # CONTROL DEL SIMULADOR (agregado al ver Sharpe brutos positivos sin correlacion operable,
    # auditoria stops.py): el TP/SL con precios de cierre solo dispara si el CIERRE cruza el
    # nivel; un stop que se toco intradia y el precio volvio no se ejecuta (optimista), un TP
    # tocado intradia que se devolvio no se cobra (pesimista). Con direccion al azar, un
    # simulador insesgado deberia dar Sharpe bruto ~0. Se reporta la distribucion.
    rng = np.random.default_rng(0)
    sh_azar = []
    for _ in range(300):
        p = rng.choice([-1.0, 1.0], size=len(dec0))
        _, rb, _ = retornos_riesgo(dec0, p, 0.0)
        sh_azar.append(ce.sharpe(rb, periodos))
    filas.append({**base, "estrategia": "Control: direccion al azar (300 sorteos, bruto)", "sharpe_bruto": np.mean(sh_azar),
                  "sharpe_min_semillas": np.percentile(sh_azar, 5), "sharpe_max_semillas": np.percentile(sh_azar, 95),
                  "spread_%": 0.0, "n_decisiones": len(dec0)})
    if exp == "diario":
        # misma regla del cobre y mismo sizing, pero saliendo SIEMPRE al cierre siguiente (sin TP/SL)
        d_cc = dec0.copy()
        d_cc["precio_salida_largo"] = d_cc["y_next"]
        d_cc["precio_salida_corto"] = d_cc["y_next"]
        filas.append(fila_metricas(d_cc, posiciones_baseline(dec0, "Umbral cobre"), "Umbral cobre, salida al cierre (sin TP/SL)", spread, periodos, base))
        sh_cc = []
        for _ in range(300):
            p = rng.choice([-1.0, 1.0], size=len(dec0))
            _, rb, _ = retornos_riesgo(d_cc, p, 0.0)
            sh_cc.append(ce.sharpe(rb, periodos))
        filas.append({**base, "estrategia": "Control: azar, salida al cierre (300 sorteos, bruto)", "sharpe_bruto": np.mean(sh_cc),
                      "sharpe_min_semillas": np.percentile(sh_cc, 5), "sharpe_max_semillas": np.percentile(sh_cc, 95),
                      "spread_%": 0.0, "n_decisiones": len(dec0)})
    if exp == "diario":
        bh = ((dec0["y_next"] - dec0["y"]) / dec0["y"]).to_numpy()
        filas.append({**base, **ce.metricas_desde_retornos(bh, "Buy-and-hold (sin apalancar)", periodos=periodos), "spread_%": 0.0, "n_decisiones": len(dec0)})
    # sensibilidad al spread (posiciones fijas)
    for nombre, pos in [("PPO seed=%d" % seed0, decs[seed0]["pos_ppo"].to_numpy()), ("Umbral cobre", posiciones_baseline(dec0, "Umbral cobre"))]:
        for sp in ce.GRILLA_SPREADS:
            r, _, _ = retornos_riesgo(decs[seed0] if nombre.startswith("PPO") else dec0, pos, sp)
            sens.append({**base, "estrategia": nombre, "spread_ida_vuelta_%": 100 * sp, "sharpe": ce.sharpe(r, periodos),
                         "retorno_total_%": 100 * (np.prod(1 + r) - 1)})
    return filas, sens, decs


def desglose_tpadapt(decs, spread):
    filas = []
    for s, dec in decs.items():
        r, _, _ = retornos_riesgo(dec, dec["pos_ppo"], spread)
        d = dec.assign(r=r, opera=dec["pos_ppo"] != 0)
        d["razon"] = np.where(d["pos_ppo"] > 0, d["razon_cierre_largo"], np.where(d["pos_ppo"] < 0, d["razon_cierre_corto"], "plano"))
        for hz, g in d[d["opera"]].groupby("horizonte_tp_elegido"):
            filas.append({"seed": s, "horizonte_elegido": hz, "n": len(g), "r_medio_%": 100 * g["r"].mean(),
                          "r_total_%": 100 * g["r"].sum(), "pct_stop_loss": 100 * (g["razon"] == "stop_loss").mean()})
    return pd.DataFrame(filas)


def graficar_clp(decs, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    dec0 = decs[sorted(decs)[0]]
    sp = float(dec0["spread"].iloc[0])
    for s, dec in sorted(decs.items()):
        r, _, _ = retornos_riesgo(dec, dec["pos_ppo"], sp)
        ax.plot(dec["ds"], 100 * np.cumprod(1 + r), color="crimson", alpha=0.5, label=f"PPO seed={s}")
    r, _, _ = retornos_riesgo(dec0, posiciones_baseline(dec0, "Umbral cobre"), sp)
    ax.plot(dec0["ds"], 100 * np.cumprod(1 + r), color="darkorange", label="Umbral cobre (signo de train)")
    r, rb, _ = retornos_riesgo(dec0, posiciones_baseline(dec0, "Umbral cobre"), 0.0)
    ax.plot(dec0["ds"], 100 * np.cumprod(1 + rb), color="darkorange", linestyle="--", label="Umbral cobre, bruto (spread 0)")
    bh = ((dec0["y_next"] - dec0["y"]) / dec0["y"]).to_numpy()
    ax.plot(dec0["ds"], 100 * np.cumprod(1 + bh), color="black", label="Buy-and-hold")
    ax.axhline(100, color="gray", linestyle=":")
    ax.set_title(f"USD/CLP diario CORREGIDO (senal previa a la entrada, spread ida+vuelta {100*sp:.2f}%, riesgo 3%/op)")
    ax.set_ylabel("Capital ($)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:  # valida la logica sobre las corridas cortas de 64 --smoke (no escribe tablas finales)
        ENTRADA_DIR = f"{RESULTADOS_DIR}/correccion_rl_smoke"
        RESULTADOS_DIR = f"{RESULTADOS_DIR}/correccion_rl_smoke"
    archivos = sorted(glob.glob(f"{ENTRADA_DIR}/*.csv"))
    archivos = [a for a in archivos if not a.replace("\\", "/").split("/")[-1].startswith("correccion_")]
    grupos = {}
    for f in archivos:
        clave = f.split("\\")[-1].split("/")[-1].rsplit("_seed", 1)[0]
        grupos.setdefault(clave, []).append(f)
    print(f"{len(archivos)} corridas en {len(grupos)} configuraciones")

    todas, todas_sens = [], []
    for clave, fs in sorted(grupos.items()):
        filas, sens, decs = resumir_config(fs)
        todas += filas
        todas_sens += sens
        if clave == "diario_clp_N1_h1":
            graficar_clp(decs, f"{RESULTADOS_DIR}/correccion_rl_diario_clp_curva_capital.png")
        if clave.startswith("tpadapt"):
            d = desglose_tpadapt(decs, float(decs[sorted(decs)[0]]["spread"].iloc[0]))
            d.to_csv(f"{RESULTADOS_DIR}/correccion_rl_tpadapt_desglose_horizonte.csv", index=False)
            print("\n=== TP adaptativo: desglose por horizonte elegido ===")
            print(d.round(3).to_string(index=False))

    t = pd.DataFrame(todas)
    t.to_csv(f"{RESULTADOS_DIR}/correccion_rl_metricas_todas.csv", index=False)
    pd.DataFrame(todas_sens).to_csv(f"{RESULTADOS_DIR}/correccion_rl_sensibilidad_spread.csv", index=False)
    cols = ["exp", "par", "N", "h", "estrategia", "retorno_total_%", "sharpe", "ic95_bajo", "ic95_alto", "sharpe_min_semillas", "sharpe_max_semillas",
            "sharpe_bruto", "ic95_bruto", "max_drawdown_%", "operaciones", "apalancamiento_mediano", "breakeven_spread_%", "coincidencia_con_cobre_%"]
    cols = [c for c in cols if c in t.columns]
    pd.set_option("display.width", 250)
    for exp in t["exp"].unique():
        print(f"\n=== {exp} ===")
        print(t[t["exp"] == exp][cols].round(3).to_string(index=False))
