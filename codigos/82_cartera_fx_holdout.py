# Issue #15 - paso 3: APERTURA UNICA del hold-out (retornos >= 2025-01-01)
# para la configuracion elegida por 81 (issue15_config_elegida.json).
#
# Correr desde codigos/ (despues de 81):  python 82_cartera_fx_holdout.py
#
# Reglas (protocolo_evaluacion.py):
#   - La configuracion se lee del JSON que escribio 81 (commit f023f1e, antes
#     de abrir el hold-out); aca no se elige nada.
#   - Se evalua UNA vez. Si el registro ya tiene la apertura de este script,
#     el script se niega a correr (salvo --forzar, que deja constancia en la
#     nota del registro).
#   - Los componentes (carry y TSMOM de la combinacion) y los benchmarks
#     (equiponderada larga, no hacer nada) se muestran como DESCOMPOSICION de
#     la misma apertura, no son candidatas: no hay nada que elegir entre ellos.

import argparse
import importlib
import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
cfx = importlib.import_module("cartera_fx")
ce = importlib.import_module("costos_y_estadistica")
proto = importlib.import_module("protocolo_evaluacion")

RES = "../datos/resultados"
SCRIPT = "82_cartera_fx_holdout.py"


def ya_abierto():
    if not os.path.exists(proto.RUTA_REGISTRO):
        return False
    reg = pd.read_csv(proto.RUTA_REGISTRO)
    return bool(((reg["script"] == SCRIPT) & (reg["usa_holdout"].astype(str) == "True")).any())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forzar", action="store_true")
    args = ap.parse_args()
    if ya_abierto() and not args.forzar:
        print("El hold-out de la configuracion elegida ya se abrio (ver registro_pruebas.csv). No se vuelve a correr.")
        return

    with open(f"{RES}/issue15_config_elegida.json") as f:
        elegida = json.load(f)
    cfg = elegida["config"]
    D = cfx.cargar()
    riesgo = cfx.cargar_riesgo()

    estrategias = {"elegida: " + elegida["estrategia"]: cfg,
                   "componente carry: " + cfx.config_nombre(cfg["carry"]): cfg["carry"],
                   "componente TSMOM: " + cfx.config_nombre(cfg["tsmom"]): cfg["tsmom"],
                   "benchmark equiponderada larga": {"tipo": "ew", "vol_obj": None}}
    filas, series = [], {}
    for nombre, c in estrategias.items():
        bt, _ = cfx.backtest(D, c)
        ho = cfx.recortar(bt, "holdout")
        pre = cfx.recortar(bt, "pre")
        m = cfx.metricas(ho, nombre, riesgo)
        m["sharpe_neto_pre_holdout"] = ce.sharpe(pre["r_neto"], 12)
        filas.append(m)
        series[nombre] = ho.set_index("ds")
    filas.append({"estrategia": "no hacer nada (caja USD; retorno en exceso = 0)", "meses": filas[0]["meses"],
                  "sharpe_bruto": 0.0, "sharpe_neto": 0.0, "ret_anual_%_neto": 0.0, "max_dd_%_neto": 0.0})
    tab = pd.DataFrame(filas)
    se = np.sqrt(12 / max(filas[0]["meses"], 1))
    tab["se_aprox_sharpe"] = se
    tab.to_csv(f"{RES}/issue15_holdout_resultado.csv", index=False)
    pd.concat({k: v[["r_bruto", "r_neto", "r_neto_estricto", "turnover", "bruto_exposicion"]]
               for k, v in series.items()}, axis=1).to_csv(f"{RES}/issue15_retornos_mensuales_holdout.csv")

    pd.set_option("display.width", 250)
    cols = ["estrategia", "desde", "hasta", "meses", "sharpe_bruto", "sharpe_neto", "ic95_bajo_neto", "ic95_alto_neto",
            "sharpe_neto_estricto", "ret_anual_%_neto", "vol_anual_%", "max_dd_%_neto", "turnover_anual",
            "sharpe_neto_pre_holdout"]
    print(tab[cols].round(3).T.to_string())
    print(f"Error estandar aproximado de un Sharpe anual con {filas[0]['meses']} meses: {se:.2f}")

    r0 = tab.iloc[0]
    proto.registrar_pruebas([{
        "prueba": elegida["estrategia"], "frecuencia": "mensual", "metrica": "sharpe_neto_holdout",
        "valor": round(r0["sharpe_neto"], 4), "n": int(r0["meses"]), "usa_holdout": True,
        "nota": (f"APERTURA UNICA del hold-out {r0['desde']}..{r0['hasta']}; bruto {r0['sharpe_bruto']:.2f}; "
                 f"IC95 neto [{r0['ic95_bajo_neto']:.2f},{r0['ic95_alto_neto']:.2f}]; "
                 f"config elegida por 81 entre {elegida['n_pruebas_pre_holdout']} pruebas"
                 + ("; RE-CORRIDA con --forzar" if args.forzar else ""))}], issue="#15", script=SCRIPT)

    # grafico
    colores = ["#1baf7a", "#2a78d6", "#eb6834", "#eda100"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for (k, v), col in zip(series.items(), colores):
        cap = np.r_[1.0, (1 + v["r_neto"]).cumprod().to_numpy()]
        x = [v.index[0] - pd.offsets.MonthEnd(1)] + list(v.index)
        ax.plot(x, cap, color=col, lw=2.2 if k.startswith("elegida") else 1.5, label=k)
    ax.axhline(1, color="#9a9994", lw=0.8)
    ax.grid(color="#e4e3df", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel("Capital (neto de costos)")
    ax.set_title(f"Issue #15 — hold-out (apertura única), {r0['desde']} a {r0['hasta']}", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    fig.tight_layout()
    fig.savefig(f"{RES}/issue15_holdout.png", dpi=140)


if __name__ == "__main__":
    main()
