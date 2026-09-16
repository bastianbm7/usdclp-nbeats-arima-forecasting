# Issue #11: resumen visual de la grilla completa (N=3,5,7 del Issue #10 +
# N=10,12,14,20 de esta extension) - promedio sobre h1-h5 por cada N, para
# ver si el patron de N=7 (menos stop-loss, menos drawdown) se sostiene al
# alargar mas la ventana de holding. Spoiler visible en el grafico: NO es
# monotono - hay una zona buena en N=5-7 y otra en N=14, con un bache en
# N=10-12 y un colapso en N=20 (Sharpe cerca de 0, solo 15 decisiones en
# todo el walk-forward - la combinacion mas ruidosa de toda la grilla).

import pandas as pd
import matplotlib.pyplot as plt

RESULTADOS_DIR = "../datos/resultados"

if __name__ == "__main__":
    tabla = pd.concat([
        pd.read_csv(f"{RESULTADOS_DIR}/grilla_nh_metricas.csv"),
        pd.read_csv(f"{RESULTADOS_DIR}/grilla_nh_largo_metricas.csv"),
    ], ignore_index=True)

    resumen = tabla.groupby("N").agg(
        sharpe_promedio=("sharpe_anualizado", "mean"),
        drawdown_promedio=("max_drawdown_%", "mean"),
        stoploss_promedio=("%_stop_loss", "mean"),
        n_decisiones=("n_decisiones", "first"),
    ).reset_index().sort_values("N")
    resumen.to_csv(f"{RESULTADOS_DIR}/grilla_nh_resumen_por_N.csv", index=False)
    print(resumen.to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col, titulo, color in [
        (axes[0], "sharpe_promedio", "Sharpe promedio (h1-h5)", "steelblue"),
        (axes[1], "drawdown_promedio", "Max drawdown promedio (%)", "crimson"),
        (axes[2], "stoploss_promedio", "% cierres por stop-loss, promedio", "darkorange"),
    ]:
        ax.plot(resumen["N"], resumen[col], marker="o", color=color, linewidth=2)
        ax.set_xlabel("N (dias de holding)")
        ax.set_title(titulo)
        ax.grid(alpha=0.3)
    fig.suptitle("Grilla N x h: promedio sobre h1-h5 por N - Issue #10 (N=3,5,7) + Issue #11 (N=10,12,14,20)")
    fig.tight_layout()
    fig.savefig(f"{RESULTADOS_DIR}/grilla_nh_resumen_por_N.png", dpi=150)
    plt.close(fig)
    print(f"\nGrafico -> grilla_nh_resumen_por_N.png")
