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
#
# La recompensa de entrenamiento es la MISMA economia real que usa el backtest
# (14_backtest_walkforward_gestion_riesgo.py): apalancamiento via risk sizing
# (arriesgar RIESGO_MAX_PCT del capital vigente) + take-profit/stop-loss
# chequeados dia a dia contra precios reales - no una version simplificada.
# Se probo primero con una recompensa simplificada (retorno % sin apalancar,
# ver historial de commits) y el agente convergio a una politica degenerada de
# "siempre largo" que, evaluada con la gestion de riesgo real, perdia -52% -
# la hipotesis es que el agente nunca "sintio" el costo real de esa decision
# durante el entrenamiento. Este cambio lo hace entrenar sobre lo mismo que se
# usa para evaluarlo.

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

DATASET_PATH = "../datos/resultados/dataset_entrenamiento_rl.csv"
DATOS_DIARIOS_PATH = "../datos/bases/usdclp_long.csv"

ACCION_CORTO, ACCION_PLANO, ACCION_LARGO = 0, 1, 2
POSICION_POR_ACCION = {ACCION_CORTO: -1.0, ACCION_PLANO: 0.0, ACCION_LARGO: 1.0}

SLIPPAGE_PCT = 0.0005  # spread simulado por cambio de posicion; comision = 0 en esta v1 (decision explicita, ver Tarea de Notion)
CAPITAL_INICIAL = 100.0
RIESGO_MAX_PCT = 0.03
K_STOP_LOSS = 1.0


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


def precios_diarios_entre(diario, fecha_inicio, fecha_fin):
    mask = (diario["ds"] > fecha_inicio) & (diario["ds"] <= fecha_fin)
    return diario.loc[mask, "y"].tolist()


def ejecutar_operacion(take_profit, stop_loss_inicial, k_stop_loss, vol, posicion, precios_diarios, precio_cierre_semana):
    # Trailing stop: a medida que el precio se mueve a favor, el stop se
    # "sube" (largo) o "baja" (corto) para seguir al extremo mas favorable
    # visto hasta ese dia, manteniendo siempre la misma distancia de riesgo
    # (k_stop_loss * vol) - nunca retrocede. El take-profit fijo sigue
    # disponible (si se llega de un salto, se toma), pero ya no hace falta
    # esperarlo para asegurar ganancias parciales si el precio se da vuelta
    # antes de llegar. Aproximacion sobre precios de cierre diario (no
    # intra-dia) - estandar para backtesting sin datos tick a tick.
    stop_actual = stop_loss_inicial
    extremo_favorable = None
    for precio_dia in precios_diarios:
        if posicion > 0:
            extremo_favorable = precio_dia if extremo_favorable is None else max(extremo_favorable, precio_dia)
            stop_actual = max(stop_actual, extremo_favorable * (1 - k_stop_loss * vol))
            if precio_dia >= take_profit:
                return take_profit, "take_profit"
            if precio_dia <= stop_actual:
                razon = "trailing_stop" if stop_actual > stop_loss_inicial else "stop_loss"
                return stop_actual, razon
        else:
            extremo_favorable = precio_dia if extremo_favorable is None else min(extremo_favorable, precio_dia)
            stop_actual = min(stop_actual, extremo_favorable * (1 + k_stop_loss * vol))
            if precio_dia <= take_profit:
                return take_profit, "take_profit"
            if precio_dia >= stop_actual:
                razon = "trailing_stop" if stop_actual < stop_loss_inicial else "stop_loss"
                return stop_actual, razon
    return precio_cierre_semana, "cierre_semana"


def precomputar_salidas_tp_sl(df, diario, k_stop_loss=K_STOP_LOSS):
    # Precalcula, para CADA semana y CADA direccion posible (largo/corto), que
    # habria pasado si se hubiera tomado esa posicion - independiente del
    # tamano de la posicion (que depende del capital vigente, eso se resuelve
    # recien en el step()). Evita recorrer precios diarios en cada step().
    filas = {"largo": [], "corto": []}
    for _, fila in df.iterrows():
        entrada, vol, take_profit = fila["y"], fila["vol_garch"], fila["nhits_h1"]
        fecha_fin_semana = fila["ds"] + pd.Timedelta(days=7)
        precios_semana = precios_diarios_entre(diario, fila["ds"], fecha_fin_semana)

        sl_largo = entrada * (1 - k_stop_loss * vol)
        precio_largo, razon_largo = ejecutar_operacion(take_profit, sl_largo, k_stop_loss, vol, 1, precios_semana, fila["y_next"])
        filas["largo"].append({"stop_loss": sl_largo, "precio_salida": precio_largo, "razon_cierre": razon_largo})

        sl_corto = entrada * (1 + k_stop_loss * vol)
        precio_corto, razon_corto = ejecutar_operacion(take_profit, sl_corto, k_stop_loss, vol, -1, precios_semana, fila["y_next"])
        filas["corto"].append({"stop_loss": sl_corto, "precio_salida": precio_corto, "razon_cierre": razon_corto})

    df = df.copy()
    for direccion in ["largo", "corto"]:
        for campo in ["stop_loss", "precio_salida", "razon_cierre"]:
            df[f"{campo}_{direccion}"] = [f[campo] for f in filas[direccion]]
    return df


FEATURES_ESTADO = ["retorno_1s", "nhits_h1_rel", "nhits_h2_rel", "vol_garch", "macd_rel", "rsi_norm", "posicion_en_rango"]


class USDCLPTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset_path=DATASET_PATH, diario_path=DATOS_DIARIOS_PATH, slippage_pct=SLIPPAGE_PCT,
                 capital_inicial=CAPITAL_INICIAL, riesgo_max_pct=RIESGO_MAX_PCT, k_stop_loss=K_STOP_LOSS, df=None):
        super().__init__()
        # df explicito = pasar un slice ya cargado (train/test split) sin releer
        # ni reprocesar el CSV en cada split - ver 12_entrenar_agente_rl.py /
        # 14_backtest_walkforward_gestion_riesgo.py.
        base = df.reset_index(drop=True) if df is not None else cargar_dataset(dataset_path)
        diario = pd.read_csv(diario_path, parse_dates=["ds"])
        self.df = precomputar_salidas_tp_sl(base, diario, k_stop_loss)

        self.slippage_pct = slippage_pct
        self.capital_inicial = capital_inicial
        self.riesgo_max_pct = riesgo_max_pct
        self.k_stop_loss = k_stop_loss

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(FEATURES_ESTADO),), dtype=np.float32)
        self._paso = 0
        self._posicion_previa = 0.0
        self.capital = capital_inicial
        self.valor_portafolio = 1.0  # = capital / capital_inicial, se mantiene por compatibilidad con 12/13

    def _obs(self):
        return self.df.loc[self._paso, FEATURES_ESTADO].to_numpy(dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._paso = 0
        self._posicion_previa = 0.0
        self.capital = self.capital_inicial
        self.valor_portafolio = 1.0
        return self._obs(), {}

    def step(self, action):
        posicion = POSICION_POR_ACCION[int(action)]
        fila = self.df.loc[self._paso]
        capital_previo = self.capital

        if posicion == 0:
            notional, pnl, razon = 0.0, 0.0, "plano"
        else:
            direccion = "largo" if posicion > 0 else "corto"
            distancia_riesgo = self.k_stop_loss * fila["vol_garch"]
            notional = (self.riesgo_max_pct * capital_previo) / distancia_riesgo if distancia_riesgo > 0 else 0.0
            precio_salida, razon = fila[f"precio_salida_{direccion}"], fila[f"razon_cierre_{direccion}"]
            retorno_pct = posicion * (precio_salida - fila["y"]) / fila["y"]
            costo_slippage = self.slippage_pct * notional if posicion != self._posicion_previa else 0.0
            pnl = notional * retorno_pct - costo_slippage

        self.capital += pnl
        self.valor_portafolio = self.capital / self.capital_inicial
        self._posicion_previa = posicion
        self._paso += 1

        # >= len(self.df), no len(self.df)-1: cada fila ya trae su propio y_next
        # (ver 10_generar_dataset_rl.py), asi que la ULTIMA fila tambien es
        # operable - terminar un paso antes descartaba el 5% de una ventana de
        # test de 20 semanas (bug encontrado evaluando walk-forward multi-ventana).
        terminated = self._paso >= len(self.df)
        obs = self._obs() if not terminated else np.zeros(len(FEATURES_ESTADO), dtype=np.float32)
        reward = pnl / capital_previo  # % de retorno sobre el capital vigente - mas estable para entrenar que dolares crudos, pero ya refleja apalancamiento y TP/SL reales
        info = {"posicion": posicion, "pnl": pnl, "notional": notional, "razon_cierre": razon, "capital": self.capital}
        return obs, float(reward), terminated, False, info


if __name__ == "__main__":
    env = USDCLPTradingEnv()
    print(f"Dataset cargado: {len(env.df)} semanas, {env.df['ds'].min().date()} a {env.df['ds'].max().date()}")
    print(f"Estado: {FEATURES_ESTADO}")

    obs, _ = env.reset()
    terminado = False
    while not terminado:
        accion = env.action_space.sample()
        obs, reward, terminado, _, info = env.step(accion)

    print(f"\nEpisodio de prueba (accion aleatoria) - capital final: ${env.capital:.2f} (arranca en ${env.capital_inicial:.0f})")
