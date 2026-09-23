# Issue #5, paso 2 (reconstruccion RL diaria) - entorno Gym a frecuencia
# DIARIA, hermano de 11_entorno_trading_rl.py (semanal) pero archivo NUEVO en
# vez de modificar el semanal in-place: el Issue #6 (multi-activo FX) se
# trabaja EN PARALELO en otra rama y tambien toca 11_entorno_trading_rl.py -
# duplicar en vez de compartir minimiza el conflicto de merge entre ambos PRs
# (indicacion explicita del Issue #5). Consume el dataset de
# 26_generar_dataset_rl_diario.py (forecast NHITS walk-forward + GARCH diario +
# MACD/RSI/minmax + copper_ret_1d/copper_mom_5d).
#
# DECISION DE DISENO: el trailing stop / take-profit intra-periodo del agente
# semanal (ejecutar_operacion() en 11_entorno_trading_rl.py) chequea precios
# DIARIOS dentro de cada semana para "subir" el stop a medida que el precio se
# mueve a favor - eso requiere al menos 2 puntos de precio dentro del periodo
# de una posicion para que "trailing" signifique algo (el stop se mueve entre
# el punto 1 y el punto 2). A frecuencia diaria, el periodo de una posicion es
# UN dia (se abre al cierre de hoy, se resuelve contra el cierre de manana) -
# no hay un segundo punto intradia para trailear. Un trailing stop "de verdad"
# necesitaria mantener la posicion abierta varios dias, pero eso contradice
# que el agente tome una decision NUEVA cada dia (la premisa central de llevar
# el agente a frecuencia diaria).
#
# Por eso el mecanismo se simplifica, de forma explicita y documentada, a un
# CHEQUEO DE UN SOLO PASO por operacion (no un trailing real): la posicion de
# hoy se resuelve contra el UNICO precio disponible (el cierre de manana) -
# si ese precio ya cruzo el take-profit (el forecast NHITS a 1 dia) o el
# stop-loss (distancia = K_STOP_LOSS x vol_garch diaria), la salida se acota
# ahi; si no, la salida es el cierre de manana tal cual. Matematicamente esto
# es exactamente ejecutar_operacion() del entorno semanal evaluado con una
# lista de precios de un solo elemento (donde "trailing" colapsa a un check
# simple, sin nada que trailear) - se reimplementa directo en vez de importar
# esa funcion para no depender de su logica de ventana calendario semanal
# (fecha_fin_semana = ds + 7 dias), que no aplica aca: el dataset diario YA
# trae "y_next" precalculado (el precio del proximo dia habil, sea cual sea
# la brecha de calendario - fin de semana, feriado), no hace falta volver a
# consultar una tabla de precios diarios por rango de fechas.
#
# El resto de la economia (RIESGO_MAX_PCT, notional via risk sizing, slippage,
# capital de $100) es IDENTICO al agente semanal - en particular, el sizing
# acotado por riesgo (no el f*=mu/sigma^2 sin acotar de Kelly) es justamente
# la correccion que 9.14/9.15 del paper identifico como necesaria antes de
# construir cualquier cosa sobre el hallazgo del cobre diario (el apalancamiento
# de Kelly sin acotar saturaba al limite el 97% de los dias) - se construye
# esta version YA con esa correccion incorporada desde el diseño, no como un
# parche posterior.

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

DATASET_PATH = "../datos/resultados/dataset_entrenamiento_rl_diario.csv"

ACCION_CORTO, ACCION_PLANO, ACCION_LARGO = 0, 1, 2
POSICION_POR_ACCION = {ACCION_CORTO: -1.0, ACCION_PLANO: 0.0, ACCION_LARGO: 1.0}

SLIPPAGE_PCT = 0.0005  # mismo valor que el agente semanal (11_entorno_trading_rl.py) y que 22_kelly_diario_cobre.py
# CORRECCION (2026-09-23): BUG DE COSTOS. step() cobraba SLIPPAGE_PCT una sola
# vez, solo al entrar y solo si la posicion cambiaba respecto al dia anterior
# (linea "costo_slippage = ... if posicion != self._posicion_previa"). Pero en
# este entorno CADA decision es una operacion completa: se abre al precio de
# la fila y se cierra en y_next (o al tocar TP/SL) - una ida+vuelta por dia,
# siempre. Con el notional del risk sizing (3.8-5.9x el capital) ese costo
# omitido cambiaba el signo del resultado. Ahora se cobra el spread
# ida+vuelta completo sobre el notional en CADA operacion. El valor por
# defecto es el supuesto para USD/CLP (costos_y_estadistica.SPREAD_IDA_VUELTA);
# los scripts corregidos (64+) pasan el spread de cada par. SLIPPAGE_PCT se
# conserva solo como registro del valor original (ya no se usa en step()).
COSTO_IDA_VUELTA_PCT = 0.0015
CAPITAL_INICIAL = 100.0
RIESGO_MAX_PCT = 0.03  # identico al agente semanal - ver nota de diseno arriba (corrige el bug de apalancamiento sin acotar de 9.14/9.15)
K_STOP_LOSS = 1.0

FEATURES_ESTADO = [
    "retorno_1d", "nhits_h1_rel", "nhits_h2_rel", "vol_garch", "macd_rel", "rsi_norm",
    "posicion_en_rango", "copper_ret_1d", "copper_mom_5d",
]


def cargar_dataset(path=DATASET_PATH):
    df = pd.read_csv(path, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)
    df["retorno_1d"] = np.log(df["y"] / df["y"].shift(1))
    df["nhits_h1_rel"] = (df["nhits_h1"] - df["y"]) / df["y"]
    df["nhits_h2_rel"] = (df["nhits_h2"] - df["y"]) / df["y"]
    df["macd_rel"] = df["macd"] / df["y"]
    df["rsi_norm"] = df["rsi"] / 100
    rango = (df["precio_max_ventana"] - df["precio_min_ventana"]).replace(0, np.nan)
    df["posicion_en_rango"] = (df["y"] - df["precio_min_ventana"]) / rango
    return df.dropna().reset_index(drop=True)


def calcular_salida_dia(entrada, precio_manana, vol, take_profit, direccion, k_stop_loss=K_STOP_LOSS):
    # Ver nota de diseno arriba: equivalente a ejecutar_operacion() del
    # entorno semanal con una lista de precios de un solo elemento - sin
    # trailing real posible (hace falta >=2 puntos para que el stop "se mueva"),
    # asi que es un chequeo de TP/SL de un solo paso contra el unico precio
    # disponible (el cierre de manana).
    #
    # BUG encontrado y corregido (Issue #5, al revisar por que "Umbral cobre"
    # daba -99.75% en 28_backtest_walkforward_diario.py - un numero tan malo
    # que ameritaba desconfiar antes de reportarlo, mismo criterio que "BUG #3"
    # de NOTAS-CLAUDE.md): take_profit=nhits_h1 es un precio ABSOLUTO, no
    # necesariamente del lado ganador de la posicion. El agente semanal (11) y
    # la estrategia "Umbral simple" (28, posiciones_umbral_simple) SIEMPRE
    # eligen la direccion en base al signo de (nhits_h1 - entrada), asi que
    # take_profit termina del lado correcto por construccion. Pero cualquier
    # estrategia cuya direccion venga de OTRA señal (ej. "Umbral cobre", que
    # usa copper_ret_1d - nada que ver con el forecast NHITS) puede terminar
    # con un take_profit del lado PERDEDOR (ej. corto con take_profit arriba
    # de la entrada) - la version anterior de este chequeo igual "tomaba
    # ganancia" ahi, cerrando la posicion con PERDIDA pero etiquetada
    # take_profit (confirmado en los datos: 170/300 cierres "take_profit" con
    # pnl PROMEDIO NEGATIVO de -0.57, sumando -96.2 de los -99.75 puntos
    # totales perdidos). La correccion: el take-profit solo es un gatillo
    # valido si esta genuinamente del lado ganador (take_profit > entrada
    # para largo, < entrada para corto); si no, se ignora y solo quedan
    # stop-loss/cierre de dia como salida - el pnl de cada trade se sigue
    # calculando con el precio de salida real, nunca se inventa nada.
    if direccion == "largo":
        stop = entrada * (1 - k_stop_loss * vol)
        tp_valido = take_profit > entrada
        if tp_valido and precio_manana >= take_profit:
            return take_profit, "take_profit"
        if precio_manana <= stop:
            return stop, "stop_loss"
        return precio_manana, "cierre_dia"
    else:
        stop = entrada * (1 + k_stop_loss * vol)
        tp_valido = take_profit < entrada
        if tp_valido and precio_manana <= take_profit:
            return take_profit, "take_profit"
        if precio_manana >= stop:
            return stop, "stop_loss"
        return precio_manana, "cierre_dia"


def precomputar_salidas_tp_sl(df, k_stop_loss=K_STOP_LOSS):
    # Precalcula, para CADA dia y CADA direccion posible (largo/corto), la
    # salida - independiente del tamano de la posicion (que depende del
    # capital vigente, eso se resuelve en step()). Mismo patron que
    # precomputar_salidas_tp_sl() del entorno semanal.
    filas = {"largo": [], "corto": []}
    for _, fila in df.iterrows():
        entrada, vol, take_profit, precio_manana = fila["y"], fila["vol_garch"], fila["nhits_h1"], fila["y_next"]

        sl_largo = entrada * (1 - k_stop_loss * vol)
        precio_largo, razon_largo = calcular_salida_dia(entrada, precio_manana, vol, take_profit, "largo", k_stop_loss)
        filas["largo"].append({"stop_loss": sl_largo, "precio_salida": precio_largo, "razon_cierre": razon_largo})

        sl_corto = entrada * (1 + k_stop_loss * vol)
        precio_corto, razon_corto = calcular_salida_dia(entrada, precio_manana, vol, take_profit, "corto", k_stop_loss)
        filas["corto"].append({"stop_loss": sl_corto, "precio_salida": precio_corto, "razon_cierre": razon_corto})

    df = df.copy()
    for direccion in ["largo", "corto"]:
        for campo in ["stop_loss", "precio_salida", "razon_cierre"]:
            df[f"{campo}_{direccion}"] = [f[campo] for f in filas[direccion]]
    return df


class USDCLPTradingEnvDiario(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset_path=DATASET_PATH, slippage_pct=SLIPPAGE_PCT, capital_inicial=CAPITAL_INICIAL,
                 riesgo_max_pct=RIESGO_MAX_PCT, k_stop_loss=K_STOP_LOSS, df=None, accion_continua=False,
                 costo_ida_vuelta_pct=COSTO_IDA_VUELTA_PCT):
        super().__init__()
        self.costo_ida_vuelta_pct = costo_ida_vuelta_pct  # CORRECCION (2026-09-23), ver nota junto a COSTO_IDA_VUELTA_PCT
        # df explicito = pasar un slice ya cargado (train/test split) sin releer
        # ni reprocesar el CSV - mismo patron que 11_entorno_trading_rl.py,
        # usado por el walk-forward de 28_backtest_walkforward_diario.py.
        base = df.reset_index(drop=True) if df is not None else cargar_dataset(dataset_path)
        self.df = precomputar_salidas_tp_sl(base, k_stop_loss)

        self.slippage_pct = slippage_pct
        self.capital_inicial = capital_inicial
        self.riesgo_max_pct = riesgo_max_pct
        self.k_stop_loss = k_stop_loss
        self.accion_continua = accion_continua
        self.features_estado = FEATURES_ESTADO

        if accion_continua:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.features_estado),), dtype=np.float32)

        self._paso = 0
        self._posicion_previa = 0.0
        self.capital = capital_inicial
        self.valor_portafolio = 1.0

    def _obs(self):
        return self.df.loc[self._paso, self.features_estado].to_numpy(dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._paso = 0
        self._posicion_previa = 0.0
        self.capital = self.capital_inicial
        self.valor_portafolio = 1.0
        return self._obs(), {}

    def step(self, action):
        if self.accion_continua:
            posicion = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        else:
            posicion = POSICION_POR_ACCION[int(action)]
        fila = self.df.loc[self._paso]
        capital_previo = self.capital

        if posicion == 0:
            notional, pnl, razon = 0.0, 0.0, "plano"
        else:
            direccion = "largo" if posicion > 0 else "corto"
            signo = 1.0 if direccion == "largo" else -1.0
            distancia_riesgo = self.k_stop_loss * fila["vol_garch"]
            notional = abs(posicion) * (self.riesgo_max_pct * capital_previo) / distancia_riesgo if distancia_riesgo > 0 else 0.0
            precio_salida, razon = fila[f"precio_salida_{direccion}"], fila[f"razon_cierre_{direccion}"]
            retorno_pct = signo * (precio_salida - fila["y"]) / fila["y"]
            # CORRECCION (2026-09-23): antes "self.slippage_pct * notional if posicion != self._posicion_previa else 0.0"
            costo_slippage = self.costo_ida_vuelta_pct * notional
            pnl = notional * retorno_pct - costo_slippage

        self.capital += pnl
        self.valor_portafolio = self.capital / self.capital_inicial
        self._posicion_previa = posicion
        self._paso += 1

        terminated = self._paso >= len(self.df)  # >= len(df), no len(df)-1 - misma correccion que el entorno semanal (11), cada fila ya trae su y_next
        obs = self._obs() if not terminated else np.zeros(len(FEATURES_ESTADO), dtype=np.float32)
        reward = pnl / capital_previo
        info = {"posicion": posicion, "pnl": pnl, "notional": notional, "razon_cierre": razon, "capital": self.capital}
        return obs, float(reward), terminated, False, info


if __name__ == "__main__":
    env = USDCLPTradingEnvDiario()
    print(f"Dataset cargado: {len(env.df)} dias, {env.df['ds'].min().date()} a {env.df['ds'].max().date()}")
    print(f"Estado: {FEATURES_ESTADO}")

    obs, _ = env.reset()
    terminado = False
    while not terminado:
        accion = env.action_space.sample()
        obs, reward, terminado, _, info = env.step(accion)

    print(f"\nEpisodio de prueba (accion aleatoria) - capital final: ${env.capital:.2f} (arranca en ${env.capital_inicial:.0f})")
