# Issue #9: explorar holding de mas de 1 dia en el agente RL diario, sin
# perder la granularidad diaria del estado (ver comentario de diseno al
# inicio de 27_entorno_trading_rl_diario.py, que explica por que el holding
# de 1 dia fue una decision deliberada, no un descuido: el trailing stop
# real necesita >=2 precios DENTRO de una misma posicion para que el stop se
# pueda ir moviendo a favor).
#
# ENFOQUE (Propuesta A del Issue, la mas barata de las 3 propuestas):
# horizonte de holding FIJO de N dias (dias_holding), adaptando el
# mecanismo de trailing stop del agente SEMANAL (ejecutar_operacion() en
# 11_entorno_trading_rl.py, que generaliza sobre CUALQUIER lista de precios
# dentro de un periodo) a precios diarios - aca se le pasan los N dias de
# decision SIGUIENTES del propio dataset diario en vez de los dias de una
# semana calendario. No se reusa esa funcion tal cual: le falta la
# correccion de take-profit invalido que 27_entorno_trading_rl_diario.py
# encontro y corrigio para el caso diario (ver ejecutar_operacion_multidia()
# mas abajo, que combina ambas piezas - el bug reaparecio en el primer
# intento de este archivo, antes del smoke-test, precisamente por reusar
# ejecutar_operacion() sin portar esa correccion).
#
# El punto de decision SIGUE siendo diario (no se pierde el estado fresco de
# copper_ret_1d en la fecha de entrada) - lo que cambia es que, una vez
# tomada la decision, la posicion queda abierta dias_holding dias antes de
# la siguiente decision: el dataset se recorre en bloques NO solapados de
# dias_holding filas (fila 0, dias_holding, 2*dias_holding, ...), mismo
# criterio que un agente semanal decidiendo 1 vez por semana en vez de
# todos los dias.
#
# Los "N dias siguientes" de cada bloque se toman como las PROPIAS filas
# consecutivas del dataset diario (columna "y", ya construido por 26 con un
# precio real por dia de decision) - no se vuelve a consultar el archivo
# crudo de precios (usdclp_long.csv): evita cualquier riesgo de desalinear
# fechas entre dos fuentes distintas, y garantiza que dias_holding=1
# reproduce EXACTAMENTE el entorno diario original (verificado en el
# smoke-test al final del archivo, no solo asumido).

import importlib

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd

entorno_diario_mod = importlib.import_module("27_entorno_trading_rl_diario")

DATASET_PATH = entorno_diario_mod.DATASET_PATH

ACCION_CORTO, ACCION_PLANO, ACCION_LARGO = entorno_diario_mod.ACCION_CORTO, entorno_diario_mod.ACCION_PLANO, entorno_diario_mod.ACCION_LARGO
POSICION_POR_ACCION = entorno_diario_mod.POSICION_POR_ACCION
SLIPPAGE_PCT = entorno_diario_mod.SLIPPAGE_PCT
CAPITAL_INICIAL = entorno_diario_mod.CAPITAL_INICIAL
RIESGO_MAX_PCT = entorno_diario_mod.RIESGO_MAX_PCT
K_STOP_LOSS = entorno_diario_mod.K_STOP_LOSS
FEATURES_ESTADO = entorno_diario_mod.FEATURES_ESTADO


def construir_decisiones_multidia(df, dias_holding):
    # Bloques NO solapados de dias_holding filas sobre el dataset diario ya
    # ordenado cronologicamente (cargar_dataset() de 27 ya ordena por ds). El
    # ultimo bloque incompleto (menos de dias_holding filas futuras
    # disponibles) se descarta - mismo criterio que el dropna() del resto
    # del proyecto, no se rellena con datos parciales.
    #
    # "precios futuros" = df["y"] extendido con UN punto extra al final
    # (df["y_next"] de la ultima fila) - verificado que y_next[i] == y[i+1]
    # para toda fila salvo la ultima (26_generar_dataset_rl_diario.py calcula
    # y_next ANTES del dropna() final, asi que la ultima fila del CSV ya
    # trae un precio futuro real que no quedo como "y" de ninguna fila propia
    # del dataset). Sin este extendido, dias_holding=1 perderia la ultima
    # fila operable respecto al entorno original (27) - el smoke-test de
    # abajo lo verifica explicitamente, no se asume.
    precios_futuros = df["y"].tolist() + [df["y_next"].iloc[-1]]
    # Fechas paralelas a precios_futuros, para poder reportar la fecha REAL de
    # salida de cada operacion (no solo el precio) - usadas por los graficos
    # de entrada/salida. La fecha del punto extra al final (analogo al precio
    # de y_next) se aproxima a "un dia habil despues de la ultima fila" - no
    # hay una columna de fecha real para ese punto en el dataset, y solo
    # afecta al ultimo bloque de la serie (caso de borde, no el grueso de los
    # datos).
    fechas_futuras = df["ds"].tolist() + [df["ds"].iloc[-1] + pd.Timedelta(days=1)]

    filas_idx, precios_periodo, fechas_periodo = [], [], []
    for i in range(0, len(df) - dias_holding + 1, dias_holding):
        filas_idx.append(i)
        precios_periodo.append(precios_futuros[i + 1: i + 1 + dias_holding])
        fechas_periodo.append(fechas_futuras[i + 1: i + 1 + dias_holding])
    decisiones = df.iloc[filas_idx].reset_index(drop=True)
    decisiones["_precios_periodo"] = precios_periodo
    decisiones["_fechas_periodo"] = fechas_periodo
    return decisiones


def ejecutar_operacion_multidia(entrada, take_profit, stop_loss_inicial, k_stop_loss, vol, direccion, precios_periodo, precio_cierre_periodo):
    # Combina DOS piezas que hasta ahora vivian en entornos separados:
    # - El trailing stop real de ejecutar_operacion() en 11_entorno_trading_rl.py
    #   (agente semanal): a medida que pasan los dias DENTRO de la misma
    #   posicion, el stop se mueve a favor siguiendo el extremo mas favorable
    #   visto hasta ese dia, sin retroceder nunca.
    # - La correccion de take-profit invalido de calcular_salida_dia() en
    #   27_entorno_trading_rl_diario.py (BUG documentado ahi en detalle):
    #   take_profit=nhits_h1 es un precio ABSOLUTO que puede caer del lado
    #   PERDEDOR de una posicion cuya direccion no vino del propio signo de
    #   ese forecast. El agente semanal nunca necesito esta correccion
    #   porque siempre elige direccion segun el signo de (nhits_h1-entrada),
    #   asi que el TP termina del lado correcto por construccion - pero aca
    #   (igual que en 27) el TP/SL se precomputa para largo Y corto sin
    #   condicionar en el forecast, asi que el mismo bug podria reaparecer
    #   si no se porta la correccion.
    #
    # Verificado (smoke-test al final del archivo): con precios_periodo de
    # UN solo elemento, esto colapsa exactamente al resultado de
    # calcular_salida_dia() de 27 (mismo precio de salida; el trailing
    # jamas se activa con un solo punto, no hay margen para que el stop se
    # mueva dentro del propio dia de resolucion).
    #
    # dia_salida (1-indexado, 1..len(precios_periodo)): en que dia DENTRO del
    # bloque se resolvio la operacion - antes se descartaba (solo se sabia
    # COMO cerro, no CUANDO); se agrega para poder reportar duracion real de
    # holding y la fecha exacta de salida en los graficos de entrada/salida.
    stop_actual = stop_loss_inicial
    extremo_favorable = None
    if direccion == "largo":
        tp_valido = take_profit > entrada
        for dia_idx, precio_dia in enumerate(precios_periodo, start=1):
            extremo_favorable = precio_dia if extremo_favorable is None else max(extremo_favorable, precio_dia)
            stop_actual = max(stop_actual, extremo_favorable * (1 - k_stop_loss * vol))
            if tp_valido and precio_dia >= take_profit:
                return take_profit, "take_profit", dia_idx
            if precio_dia <= stop_actual:
                razon = "trailing_stop" if stop_actual > stop_loss_inicial else "stop_loss"
                return stop_actual, razon, dia_idx
    else:
        tp_valido = take_profit < entrada
        for dia_idx, precio_dia in enumerate(precios_periodo, start=1):
            extremo_favorable = precio_dia if extremo_favorable is None else min(extremo_favorable, precio_dia)
            stop_actual = min(stop_actual, extremo_favorable * (1 + k_stop_loss * vol))
            if tp_valido and precio_dia <= take_profit:
                return take_profit, "take_profit", dia_idx
            if precio_dia >= stop_actual:
                razon = "trailing_stop" if stop_actual < stop_loss_inicial else "stop_loss"
                return stop_actual, razon, dia_idx
    return precio_cierre_periodo, "cierre_periodo", len(precios_periodo)


def precomputar_salidas_tp_sl_multidia(decisiones, k_stop_loss=K_STOP_LOSS, horizonte_tp=1):
    # horizonte_tp (Issue #10): que columna nhits_h{N} usar como objetivo de
    # take-profit - antes hardcodeado a nhits_h1 (h1 siempre). Requiere que
    # el df tenga esa columna (nhits_h1/h2 vienen en el dataset original de
    # 26; h3 en el de 35; h4/h5 en el de 39_generar_dataset_rl_diario_h5.py).
    columna_tp = f"nhits_h{horizonte_tp}"
    filas = {"largo": [], "corto": []}
    for _, fila in decisiones.iterrows():
        entrada, vol, take_profit = fila["y"], fila["vol_garch"], fila[columna_tp]
        precios_periodo = fila["_precios_periodo"]
        fechas_periodo = fila["_fechas_periodo"]
        precio_cierre_periodo = precios_periodo[-1]

        sl_largo = entrada * (1 - k_stop_loss * vol)
        precio_largo, razon_largo, dia_largo = ejecutar_operacion_multidia(
            entrada, take_profit, sl_largo, k_stop_loss, vol, "largo", precios_periodo, precio_cierre_periodo)
        filas["largo"].append({"stop_loss": sl_largo, "precio_salida": precio_largo, "razon_cierre": razon_largo,
                                "dias_hasta_salida": dia_largo, "fecha_salida": fechas_periodo[dia_largo - 1]})

        sl_corto = entrada * (1 + k_stop_loss * vol)
        precio_corto, razon_corto, dia_corto = ejecutar_operacion_multidia(
            entrada, take_profit, sl_corto, k_stop_loss, vol, "corto", precios_periodo, precio_cierre_periodo)
        filas["corto"].append({"stop_loss": sl_corto, "precio_salida": precio_corto, "razon_cierre": razon_corto,
                                "dias_hasta_salida": dia_corto, "fecha_salida": fechas_periodo[dia_corto - 1]})

    decisiones = decisiones.drop(columns=["_precios_periodo", "_fechas_periodo"]).copy()
    for direccion in ["largo", "corto"]:
        for campo in ["stop_loss", "precio_salida", "razon_cierre", "dias_hasta_salida", "fecha_salida"]:
            decisiones[f"{campo}_{direccion}"] = [f[campo] for f in filas[direccion]]
    return decisiones


class USDCLPTradingEnvDiarioMultidia(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset_path=DATASET_PATH, dias_holding=1, slippage_pct=SLIPPAGE_PCT,
                 capital_inicial=CAPITAL_INICIAL, riesgo_max_pct=RIESGO_MAX_PCT, k_stop_loss=K_STOP_LOSS,
                 df=None, accion_continua=False, horizonte_tp=1):
        super().__init__()
        # df explicito = pasar un slice ya cargado (train/test split) sin
        # releer el CSV - mismo patron que 27_entorno_trading_rl_diario.py.
        # OJO: el slice de train/test tiene que hacerse ANTES de submuestrear
        # (sobre el dataset diario completo), no despues - submuestrear un
        # slice ya recortado desalinearia los bloques de dias_holding entre
        # ventanas del walk-forward. Ver 33_backtest_walkforward_diario_multidia.py.
        #
        # horizonte_tp (Issue #10, default=1 preserva el comportamiento
        # original de 9.17-9.20): que forecast nhits_h{N} usar como objetivo
        # de take-profit, independiente de dias_holding - se pueden combinar
        # libremente (ej. horizonte_tp=5 con dias_holding=3 es un objetivo
        # ambicioso en una ventana corta, valido de correr aunque rara vez
        # se alcance).
        base = df.reset_index(drop=True) if df is not None else entorno_diario_mod.cargar_dataset(dataset_path)
        decisiones = construir_decisiones_multidia(base, dias_holding)
        self.df = precomputar_salidas_tp_sl_multidia(decisiones, k_stop_loss, horizonte_tp)

        self.dias_holding = dias_holding
        self.horizonte_tp = horizonte_tp
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
        # Identico a USDCLPTradingEnvDiario.step() (27) - la unica diferencia
        # esta en como se precomputaron precio_salida_*/razon_cierre_* arriba
        # (contra dias_holding precios consecutivos, con trailing stop real
        # via ejecutar_operacion(), en vez de contra un unico y_next).
        if self.accion_continua:
            posicion = float(np.clip(np.asarray(action, dtype=np.float32).reshape(-1)[0], -1.0, 1.0))
        else:
            posicion = POSICION_POR_ACCION[int(action)]
        fila = self.df.loc[self._paso]
        capital_previo = self.capital

        if posicion == 0:
            notional, pnl, razon, dias_hasta_salida, fecha_salida = 0.0, 0.0, "plano", 0, fila["ds"]
        else:
            direccion = "largo" if posicion > 0 else "corto"
            signo = 1.0 if direccion == "largo" else -1.0
            distancia_riesgo = self.k_stop_loss * fila["vol_garch"]
            notional = abs(posicion) * (self.riesgo_max_pct * capital_previo) / distancia_riesgo if distancia_riesgo > 0 else 0.0
            precio_salida, razon = fila[f"precio_salida_{direccion}"], fila[f"razon_cierre_{direccion}"]
            dias_hasta_salida, fecha_salida = fila[f"dias_hasta_salida_{direccion}"], fila[f"fecha_salida_{direccion}"]
            retorno_pct = signo * (precio_salida - fila["y"]) / fila["y"]
            costo_slippage = self.slippage_pct * notional if posicion != self._posicion_previa else 0.0
            pnl = notional * retorno_pct - costo_slippage

        self.capital += pnl
        self.valor_portafolio = self.capital / self.capital_inicial
        self._posicion_previa = posicion
        self._paso += 1

        terminated = self._paso >= len(self.df)
        obs = self._obs() if not terminated else np.zeros(len(FEATURES_ESTADO), dtype=np.float32)
        reward = pnl / capital_previo
        info = {"posicion": posicion, "pnl": pnl, "notional": notional, "razon_cierre": razon, "capital": self.capital,
                "dias_hasta_salida": dias_hasta_salida, "fecha_salida": fecha_salida}
        return obs, float(reward), terminated, False, info


if __name__ == "__main__":
    # Smoke test: dias_holding=1 tiene que ser matematicamente equivalente al
    # entorno diario original (27) - ejecutar_operacion() con una lista de UN
    # solo precio colapsa al mismo chequeo de un paso que calcular_salida_dia()
    # de 27 (ver nota de diseno al inicio de ese archivo). Se compara columna
    # a columna en vez de asumirlo.
    base = entorno_diario_mod.cargar_dataset()
    env_original = entorno_diario_mod.USDCLPTradingEnvDiario(df=base.copy())
    env_multidia_1 = USDCLPTradingEnvDiarioMultidia(dias_holding=1, df=base.copy())

    assert len(env_original.df) == len(env_multidia_1.df), \
        f"Distinta cantidad de filas: original={len(env_original.df)}, multidia(N=1)={len(env_multidia_1.df)}"
    # "cierre_dia" (27) vs "cierre_periodo" (30) es un cambio de nombre
    # deliberado (el concepto generaliza a un periodo de N dias, no solo 1) -
    # se normalizan ambos a "cierre" antes de comparar, el resto de las
    # categorias (take_profit/stop_loss/trailing_stop) se comparan tal cual.
    normalizar = lambda s: "cierre" if s.startswith("cierre") else s
    for direccion in ["largo", "corto"]:
        diff_precio = (env_original.df[f"precio_salida_{direccion}"].to_numpy(dtype=float)
                        - env_multidia_1.df[f"precio_salida_{direccion}"].to_numpy(dtype=float))
        assert np.abs(diff_precio).max() < 1e-9, f"precio_salida_{direccion} difiere (max abs diff={np.abs(diff_precio).max()})"
        razones_original = [normalizar(s) for s in env_original.df[f"razon_cierre_{direccion}"]]
        razones_multidia = [normalizar(s) for s in env_multidia_1.df[f"razon_cierre_{direccion}"]]
        assert razones_original == razones_multidia, f"razon_cierre_{direccion} difiere (categoria)"
    print(f"Smoke test OK: dias_holding=1 reproduce exactamente el entorno diario original ({len(env_multidia_1.df)} dias)")

    for n in [1, 2, 3, 5]:
        env = USDCLPTradingEnvDiarioMultidia(dias_holding=n)
        print(f"\ndias_holding={n}: {len(env.df)} decisiones ({env.df['ds'].min().date()} a {env.df['ds'].max().date()})")
        obs, _ = env.reset()
        terminado = False
        while not terminado:
            accion = env.action_space.sample()
            obs, reward, terminado, _, info = env.step(accion)
        print(f"  Episodio de prueba (accion aleatoria) - capital final: ${env.capital:.2f}")
