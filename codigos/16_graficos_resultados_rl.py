# Graficos finales para documentar la Fase 1 (Issue #1): lee los CSV ya
# generados por 09/14/15, no vuelve a correr nada caro (GARCH/N-HiTS/PPO).

import matplotlib.pyplot as plt
import pandas as pd

RESULTADOS_DIR = "../datos/resultados"

COLOR_ESTRATEGIA = {"Buy-and-hold": "black", "PPO (RL)": "steelblue", "Umbral simple (Opcion A)": "darkorange"}


def graficar_comparacion_volatilidad():
    tabla = pd.read_csv(f"{RESULTADOS_DIR}/comparacion_volatilidad_metricas.csv")
    tabla = tabla.sort_values("RMSE")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colores = ["seagreen" if m == tabla["modelo"].iloc[0] else "gray" for m in tabla["modelo"]]
    ax.bar(tabla["modelo"], tabla["RMSE"], color=colores)
    for i, v in enumerate(tabla["RMSE"]):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("RMSE (volatilidad realizada semanal)")
    ax.set_title("Comparacion de modelos de volatilidad (walk-forward, 100 ventanas)")
    fig.tight_layout()
    fig.savefig(f"{RESULTADOS_DIR}/rmse_comparacion_volatilidad.png", dpi=150)
    plt.close(fig)


def graficar_curva_capital():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for nombre, archivo in [("Buy-and-hold", "buy-and-hold"), ("PPO (RL)", "ppo"), ("Umbral simple (Opcion A)", "umbral")]:
        r = pd.read_csv(f"{RESULTADOS_DIR}/walkforward_{archivo}_operaciones.csv", parse_dates=["ds"])
        ax.plot(r["ds"], r["capital"], label=nombre, color=COLOR_ESTRATEGIA[nombre], linewidth=1.8)
    ax.axhline(100, color="gray", linestyle=":", linewidth=1, label="Capital inicial ($100)")
    ax.set_ylabel("Capital ($)")
    ax.set_title("Evolucion del capital - walk-forward, 100 semanas out-of-sample (2024-09 a 2026-08)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{RESULTADOS_DIR}/walkforward_curva_capital.png", dpi=150)
    plt.close(fig)


def graficar_metricas_por_estrategia():
    tabla = pd.read_csv(f"{RESULTADOS_DIR}/walkforward_gestion_riesgo_metricas.csv")
    metricas = [("retorno_total_%", "Retorno total (%)"), ("sharpe_anualizado", "Sharpe anualizado"),
                ("max_drawdown_%", "Max drawdown (%)"), ("win_rate_%", "Win rate (%)")]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (columna, titulo) in zip(axes.flat, metricas):
        colores = [COLOR_ESTRATEGIA[e] for e in tabla["estrategia"]]
        valores = tabla[columna].fillna(0)
        ax.bar(tabla["estrategia"], valores, color=colores)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(titulo)
        ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Metricas por estrategia - walk-forward con gestion de riesgo (100 semanas)")
    fig.tight_layout()
    fig.savefig(f"{RESULTADOS_DIR}/walkforward_metricas_por_estrategia.png", dpi=150)
    plt.close(fig)


COLOR_MEJORA = {
    "Buy-and-hold": "black", "Umbral simple (Opcion A)": "darkorange", "PPO base": "steelblue",
    "PPO + ent_coef alto": "mediumpurple", "PPO + accion continua": "crimson",
    "PPO + exceso sobre buy-and-hold": "goldenrod", "PPO + las 3 combinadas": "seagreen",
}
ARCHIVO_OPERACIONES_MEJORA = {
    "Buy-and-hold": "walkforward_buy-and-hold_operaciones", "Umbral simple (Opcion A)": "walkforward_umbral_operaciones",
    "PPO base": "mejoras_rl_base_operaciones", "PPO + ent_coef alto": "mejoras_rl_entcoef_operaciones",
    "PPO + accion continua": "mejoras_rl_continua_operaciones", "PPO + exceso sobre buy-and-hold": "mejoras_rl_excesobh_operaciones",
    "PPO + las 3 combinadas": "mejoras_rl_combinado_operaciones",
}


def graficar_metricas_mejoras():
    # Issue #2: mismas 4 metricas que graficar_metricas_por_estrategia, pero
    # con las 5 configuraciones nuevas + los 2 baselines que no cambian
    # (Buy-and-hold, Umbral simple) para comparar todo en un solo grafico.
    tabla = pd.read_csv(f"{RESULTADOS_DIR}/mejoras_rl_metricas.csv")
    metricas = [("retorno_total_%", "Retorno total (%)"), ("sharpe_anualizado", "Sharpe anualizado"),
                ("max_drawdown_%", "Max drawdown (%)"), ("win_rate_%", "Win rate (%)")]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (columna, titulo) in zip(axes.flat, metricas):
        colores = [COLOR_MEJORA[e] for e in tabla["estrategia"]]
        valores = tabla[columna].fillna(0)
        ax.bar(tabla["estrategia"], valores, color=colores)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(titulo)
        ax.tick_params(axis="x", rotation=30)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
    fig.suptitle("Issue #2: mejoras al agente PPO vs. baselines - walk-forward (100 semanas)")
    fig.tight_layout()
    fig.savefig(f"{RESULTADOS_DIR}/mejoras_rl_metricas_por_estrategia.png", dpi=150)
    plt.close(fig)


def graficar_curva_capital_mejoras():
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for nombre, archivo in ARCHIVO_OPERACIONES_MEJORA.items():
        r = pd.read_csv(f"{RESULTADOS_DIR}/{archivo}.csv", parse_dates=["ds"])
        ax.plot(r["ds"], r["capital"], label=nombre, color=COLOR_MEJORA[nombre], linewidth=1.6)
    ax.axhline(100, color="gray", linestyle=":", linewidth=1, label="Capital inicial ($100)")
    ax.set_ylabel("Capital ($)")
    ax.set_title("Issue #2: evolucion del capital por configuracion - walk-forward (100 semanas)")
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{RESULTADOS_DIR}/mejoras_rl_curva_capital.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    graficar_comparacion_volatilidad()
    graficar_curva_capital()
    graficar_metricas_por_estrategia()
    graficar_metricas_mejoras()
    graficar_curva_capital_mejoras()
    print("Graficos guardados en datos/resultados/:")
    print("- rmse_comparacion_volatilidad.png")
    print("- walkforward_curva_capital.png")
    print("- walkforward_metricas_por_estrategia.png")
    print("- mejoras_rl_metricas_por_estrategia.png")
    print("- mejoras_rl_curva_capital.png")
