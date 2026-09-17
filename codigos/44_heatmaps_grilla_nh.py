# Heatmaps de la grilla completa N x h (Issue #10: N=3,5,7 + Issue #11:
# N=10,12,14,20) - eje Y = h (horizonte de take-profit), eje X = N (dias de
# holding), un panel por metrica (drawdown, % stop-loss, % trailing-stop,
# % take-profit).

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTADOS_DIR = "../datos/resultados"
VALORES_N = [3, 5, 7, 10, 12, 14, 20]
VALORES_H = [1, 2, 3, 4, 5]

METRICAS = [
    ("retorno_total_%", "Retorno total (%)", "RdYlGn"),       # mas = mejor -> verde
    ("capital_final", "Saldo final ($, capital inicial=$100)", "RdYlGn"),  # mas = mejor -> verde
    ("max_drawdown_%", "Max drawdown (%)", "RdYlGn"),        # menos negativo = mejor -> verde
    ("%_stop_loss", "% cierres por stop-loss", "RdYlGn_r"),   # menos = mejor -> verde
    ("%_trailing_stop", "% cierres por trailing-stop", "RdYlGn"),  # mas = mejor (objetivo de diseno) -> verde
    ("%_take_profit", "% cierres por take-profit", "RdYlGn"),      # mas = mejor -> verde
]

if __name__ == "__main__":
    tabla = pd.concat([
        pd.read_csv(f"{RESULTADOS_DIR}/grilla_nh_metricas.csv"),
        pd.read_csv(f"{RESULTADOS_DIR}/grilla_nh_largo_metricas.csv"),
    ], ignore_index=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    for ax, (col, titulo, cmap) in zip(axes.flat, METRICAS):
        matriz = np.full((len(VALORES_H), len(VALORES_N)), np.nan)
        for i, h in enumerate(VALORES_H):
            for j, N in enumerate(VALORES_N):
                fila = tabla[(tabla["h"] == h) & (tabla["N"] == N)]
                if len(fila) == 1:
                    matriz[i, j] = fila[col].iloc[0]

        im = ax.imshow(matriz, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(VALORES_N)))
        ax.set_xticklabels(VALORES_N)
        ax.set_yticks(range(len(VALORES_H)))
        ax.set_yticklabels([f"h{h}" for h in VALORES_H])
        ax.set_xlabel("N (días de holding)")
        ax.set_ylabel("Horizonte de take-profit")
        ax.set_title(titulo)
        for i in range(len(VALORES_H)):
            for j in range(len(VALORES_N)):
                valor = matriz[i, j]
                if not np.isnan(valor):
                    ax.text(j, i, f"{valor:.1f}", ha="center", va="center", fontsize=9,
                             color="black" if 0.25 < (valor - np.nanmin(matriz)) / (np.nanmax(matriz) - np.nanmin(matriz) + 1e-9) < 0.75 else "white")
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Grilla N × h completa (Issue #10 + #11): drawdown y razón de cierre", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{RESULTADOS_DIR}/grilla_nh_heatmaps.png", dpi=150)
    plt.close(fig)
    print(f"Heatmaps -> grilla_nh_heatmaps.png")
