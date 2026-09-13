# "Redes que van aprendiendo a medida que aparecen nuevos datos" = refit=True en
# cross_validation, con use_init_models=False (default) para que cada reentreno
# parta de los pesos anteriores (warm start), no de cero.
#
# statsforecast ya reentrena por ventana por default (refit=True es el default ahi,
# por eso AutoARIMA podia elegir un orden distinto en cada ventana en los analisis
# anteriores) - el cambio real es habilitar lo mismo en neuralforecast, que por
# default (refit=False) entrena una sola vez y nunca mas actualiza con datos nuevos.
#
# Corre 3 escalas (diaria/semanal/mensual) x 3 horizontes (1/2/3 pasos) = 9 combos.
# max_steps se baja mucho respecto a 03/06 (de 1500 a MAX_STEPS_REFIT) porque con
# refit=True cada ventana ya parte entrenada (warm start) - repetir 1500 pasos en
# cada una de las N_WINDOWS ventanas sobreentrenaria y tardaria mucho mas.

import matplotlib.pyplot as plt
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MQLoss
from neuralforecast.models import NBEATS, NHITS
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, Naive
from utilsforecast.losses import mae, mape, rmse

DATOS_LARGO = "../datos/bases/usdclp_long.csv"
RESULTADOS_DIR = "../datos/resultados"

N_WINDOWS = 10
NIVELES = [80, 90]
SCALER_TYPE = "standard"

# Presupuesto por reentreno (no total - warm start entre ventanas). Mensual necesita
# mas: con 100 pasos, NBEATS en h=2 daba RMSE=20 (todavia gana al naive=30, pero lejos
# de su potencial); con 300 pasos baja a RMSE=3.25. Verificado que NO es un problema
# generico de "poco entrenamiento" en diaria/semanal (ahi 100 ya da resultados solidos
# y consistentes) - es especifico de que mensual tiene la serie mas corta (200 obs.).
MAX_STEPS_REFIT = {"diaria": 100, "semanal": 100, "mensual": 300}

ESCALAS = {
    "diaria": {"freq": "B", "input_size": 30},
    "semanal": {"freq": "W", "input_size": 12},
    "mensual": {"freq": "ME", "input_size": 12},
}
HORIZONTES = [1, 2, 3]

COLUMNA_PUNTUAL = {
    "Naive": "Naive",
    "AutoARIMA": "AutoARIMA",
    "NBEATS": "NBEATS-median",
    "NHITS": "NHITS-median",
}


def resamplear(df_diario, freq):
    s = df_diario.set_index("ds")["y"].resample(freq).last()
    s = s.iloc[:-1]  # descarta el ultimo periodo, incompleto
    df = s.reset_index()
    df.insert(0, "unique_id", "USDCLP")
    return df.dropna().reset_index(drop=True)


def correr_combo(df, freq, horizonte, input_size, max_steps_refit):
    sf = StatsForecast(models=[AutoARIMA(season_length=1), Naive()], freq=freq, n_jobs=-1)
    cv_baseline = sf.cross_validation(
        df=df, h=horizonte, n_windows=N_WINDOWS, step_size=horizonte, level=NIVELES, refit=True,
    )

    # NBEATS con los stacks interpretables por default (trend+seasonality) no admite
    # h=1 (la base armonica/polinomica colapsa con un solo paso) - restriccion real
    # de la libreria, no un bug. Con h=1 se usa la variante generica (3 stacks
    # "identity"); NHITS no tiene esta restriccion, se deja igual en todos los casos.
    stack_types_nbeats = ["identity", "identity", "identity"] if horizonte == 1 else ["identity", "trend", "seasonality"]
    trainer_kwargs = dict(enable_progress_bar=False, enable_model_summary=False, logger=False)
    modelos_nn = [
        NBEATS(
            h=horizonte, input_size=input_size, loss=MQLoss(), max_steps=max_steps_refit,
            scaler_type=SCALER_TYPE, stack_types=stack_types_nbeats, **trainer_kwargs,
        ),
        NHITS(h=horizonte, input_size=input_size, loss=MQLoss(), max_steps=max_steps_refit, scaler_type=SCALER_TYPE, **trainer_kwargs),
    ]
    nf = NeuralForecast(models=modelos_nn, freq=freq)
    cv_nn = nf.cross_validation(
        df=df, n_windows=N_WINDOWS, step_size=horizonte, level=NIVELES, refit=True, use_init_models=False,
    )

    claves = ["unique_id", "ds", "cutoff", "y"]
    return cv_baseline.merge(cv_nn, on=claves, how="inner")


def calcular_metricas(cv_df, escala, horizonte):
    columnas = list(COLUMNA_PUNTUAL.values())
    tablas = {"RMSE": rmse(cv_df, columnas), "MAE": mae(cv_df, columnas), "MAPE": mape(cv_df, columnas)}
    filas = []
    for nombre_modelo, columna in COLUMNA_PUNTUAL.items():
        fila = {"escala": escala, "horizonte": horizonte, "modelo": nombre_modelo}
        for metrica, tabla in tablas.items():
            fila[metrica] = tabla[columna].iloc[0]
        filas.append(fila)
    return pd.DataFrame(filas)


def graficar_heatmap(metricas):
    metricas["combo"] = metricas["escala"] + " h=" + metricas["horizonte"].astype(str)
    pivot = metricas.pivot(index="modelo", columns="combo", values="RMSE")
    pivot = pivot.reindex(columns=[f"{e} h={h}" for e in ESCALAS for h in HORIZONTES])

    pivot_norm = pivot.div(pivot.max(axis=0), axis=1)  # normaliza por columna: 1=peor de esa columna

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(pivot_norm.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("RMSE por modelo x escala x horizonte (rojo = peor de esa columna, verde = mejor) - online learning (refit=True)")
    fig.tight_layout()
    fig.savefig(f"{RESULTADOS_DIR}/online_learning_heatmap.png", dpi=150)
    plt.close(fig)


def graficar_ejemplo(cv_df, escala, horizonte):
    ganador_col = "NBEATS-median"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(cv_df["ds"], cv_df["y"], label="real", color="black", linewidth=2, marker="o")
    for modelo, columna in COLUMNA_PUNTUAL.items():
        ax.plot(cv_df["ds"], cv_df[columna], label=modelo, linestyle="--", marker="x", alpha=0.8)
    ax.set_title(f"USD/CLP {escala}, h={horizonte}, online learning (refit=True, warm start)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(f"{RESULTADOS_DIR}/online_learning_ejemplo_{escala}_h{horizonte}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    diario = pd.read_csv(DATOS_LARGO, parse_dates=["ds"])
    todas_las_metricas = []

    for escala, cfg in ESCALAS.items():
        serie = resamplear(diario, cfg["freq"])
        print(f"\n=== {escala}: {len(serie)} obs, {serie['ds'].min().date()} a {serie['ds'].max().date()} ===")
        for h in HORIZONTES:
            print(f"--- horizonte={h} ---")
            cv_df = correr_combo(serie, cfg["freq"], h, cfg["input_size"], MAX_STEPS_REFIT[escala])
            cv_df.to_csv(f"{RESULTADOS_DIR}/online_learning_cv_{escala}_h{h}.csv", index=False)
            metricas = calcular_metricas(cv_df, escala, h)
            todas_las_metricas.append(metricas)
            print(metricas.to_string(index=False))
            if escala == "diaria" and h == 1:
                graficar_ejemplo(cv_df, escala, h)

    metricas_finales = pd.concat(todas_las_metricas, ignore_index=True)
    metricas_finales.to_csv(f"{RESULTADOS_DIR}/online_learning_metricas.csv", index=False)
    graficar_heatmap(metricas_finales)
    print("\n=== listo ===")
    print(metricas_finales.to_string(index=False))
