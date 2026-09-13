# Entorno Gym liviano para la estrategia de trading (Issue #1, Fase 1) - propio en
# vez de adoptar FinRL completo, porque FinRL esta pensado para carteras
# multi-activo y aca es un solo par (USD/CLP). Consume el dataset semanal de
# 10_generar_dataset_rl.py (precio + forecast N-HiTS + volatilidad GARCH +
# indicadores tecnicos).
#
# Todas las features del estado son relativas/acotadas (retorno, forecast como
# desviacion % del precio actual, RSI/100, posicion en rango min-max) en vez de
# precio nivel - el USD/CLP paso de ~500 a ~950 en 16 anios, un precio nivel
# crudo no es estacionario y el agente no generalizaria entre epocas.

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

DATASET_PATH = "../datos/resultados/dataset_entrenamiento_rl.csv"

ACCION_CORTO, ACCION_PLANO, ACCION_LARGO = 0, 1, 2
POSICION_POR_ACCION = {ACCION_CORTO: -1.0, ACCION_PLANO: 0.0, ACCION_LARGO: 1.0}

SLIPPAGE_PCT = 0.0005  # spread simulado por cambio de posicion; comision = 0 en esta v1 (decision explicita, ver Tarea de Notion)


def cargar_dataset(path=DATASET_PATH):
    df = pd.read_csv(path, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)
    df["retorno_1s"] = np.log(df["y"] / df["y"].shift(1))
    df["nhits_h1_rel"] = (df["nhits_h1"] - df["y"]) / df["y"]
    df["nhits_h2_rel"] = (df["nhits_h2"] - df["y"]) / df["y"]
    df["macd_rel"] = df["macd"] / df["y"]
    df["rsi_norm"] = df["rsi"] / 100
    rango = (df["precio_max_ventana"] - df["precio_min_ventana"]).replace(0, np.nan)
    df["posicion_en_rango"] = (df["y"] - df["precio_min_ventana"]) / rango
    return df.dropna().reset_index(drop=True)


FEATURES_ESTADO = ["retorno_1s", "nhits_h1_rel", "nhits_h2_rel", "vol_garch", "macd_rel", "rsi_norm", "posicion_en_rango"]


class USDCLPTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset_path=DATASET_PATH, slippage_pct=SLIPPAGE_PCT):
        super().__init__()
        self.df = cargar_dataset(dataset_path)
        self.slippage_pct = slippage_pct
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(FEATURES_ESTADO),), dtype=np.float32)
        self._paso = 0
        self._posicion_previa = 0.0
        self.valor_portafolio = 1.0

    def _obs(self):
        return self.df.loc[self._paso, FEATURES_ESTADO].to_numpy(dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._paso = 0
        self._posicion_previa = 0.0
        self.valor_portafolio = 1.0
        return self._obs(), {}

    def step(self, action):
        posicion = POSICION_POR_ACCION[int(action)]
        fila = self.df.loc[self._paso]

        retorno_semana = (fila["y_next"] - fila["y"]) / fila["y"]
        costo_slippage = self.slippage_pct * abs(posicion - self._posicion_previa)
        retorno_neto = posicion * retorno_semana - costo_slippage

        self.valor_portafolio *= (1 + retorno_neto)
        self._posicion_previa = posicion
        self._paso += 1

        terminated = self._paso >= len(self.df) - 1
        obs = self._obs() if not terminated else np.zeros(len(FEATURES_ESTADO), dtype=np.float32)
        info = {"retorno_semana": retorno_semana, "posicion": posicion, "valor_portafolio": self.valor_portafolio}
        return obs, float(retorno_neto), terminated, False, info


if __name__ == "__main__":
    env = USDCLPTradingEnv()
    print(f"Dataset cargado: {len(env.df)} semanas, {env.df['ds'].min().date()} a {env.df['ds'].max().date()}")
    print(f"Estado: {FEATURES_ESTADO}")

    obs, _ = env.reset()
    terminado = False
    while not terminado:
        accion = env.action_space.sample()
        obs, reward, terminado, _, info = env.step(accion)

    print(f"\nEpisodio de prueba (accion aleatoria) - valor final del portafolio: {env.valor_portafolio:.4f} (arranca en 1.0)")
