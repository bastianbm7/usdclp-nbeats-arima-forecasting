"""Fuente de datos de respaldo (gratis) vía yfinance, SOLO para MODO=simulado sin token.

ADVERTENCIA: los timestamps diarios de Yahoo son ambiguos (cada ticker cierra su barra
a una hora distinta: HG=F sigue la sesión CME, AUDUSD=X una convención propia). Esa
ambigüedad es justamente lo que generó el artefacto de la señal "cobre día t -> AUD
día t+1" en la investigación. Por eso todo lo que sale de acá se etiqueta como
`fallback_yahoo` y no debe usarse para sacar conclusiones sobre la señal: sirve para
probar la infraestructura mientras no exista la cuenta demo de OANDA.
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

from modelos import Cotizacion, Vela

log = logging.getLogger(__name__)

NY = ZoneInfo("America/New_York")
FUENTE = "fallback_yahoo"
TICKERS = {"AUD_USD": "AUDUSD=X", "XCU_USD": "HG=F", "EUR_USD": "EURUSD=X",
           "USD_CLP": "CLP=X", "XAU_USD": "GC=F"}
PIP = {"AUD_USD": 0.0001, "EUR_USD": 0.0001, "XCU_USD": 0.001}

_advertido = False


def advertir_una_vez() -> None:
    global _advertido
    if not _advertido:
        log.warning("Usando FALLBACK de Yahoo Finance: timestamps diarios AMBIGUOS y spread "
                    "sintético. Trades etiquetados '%s'. No sirven para evaluar la señal; "
                    "crear la cuenta demo OANDA para el forward test real.", FUENTE)
        _advertido = True


def fecha_ultimo_cierre_ny(ahora_utc: datetime) -> datetime.date:
    """Fecha (NY) del último cierre de las 17:00 Nueva York ocurrido antes de `ahora`."""
    ahora_ny = ahora_utc.astimezone(NY)
    if ahora_ny.time() >= time(17, 0):
        return ahora_ny.date()
    return ahora_ny.date() - timedelta(days=1)


def mercado_fx_abierto(ahora_utc: datetime) -> bool:
    """FX spot: cerrado desde viernes 17:00 NY hasta domingo 17:00 NY."""
    ny = ahora_utc.astimezone(NY)
    wd = ny.weekday()  # lunes=0 ... domingo=6
    if wd == 5:
        return False
    if wd == 4 and ny.time() >= time(17, 0):
        return False
    if wd == 6 and ny.time() < time(17, 0):
        return False
    return True


class FuenteYahoo:
    nombre = FUENTE

    def __init__(self, spread_pips: float = 1.4, reloj=None, descargador=None):
        self.spread_pips = spread_pips
        self._reloj = reloj or (lambda: datetime.now(timezone.utc))
        self._descargar = descargador or self._descargar_yf

    @staticmethod
    def _descargar_yf(ticker: str, period: str, interval: str):
        import yfinance as yf  # import perezoso: dependencia opcional del núcleo
        return yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)

    def _ticker(self, instrumento: str) -> str:
        if instrumento not in TICKERS:
            raise ValueError(f"Instrumento sin ticker Yahoo mapeado: {instrumento}")
        return TICKERS[instrumento]

    def velas_diarias(self, instrumento: str, n: int = 10) -> list[Vela]:
        advertir_una_vez()
        df = self._descargar(self._ticker(instrumento), "1mo", "1d")
        if df is None or len(df) == 0:
            raise RuntimeError(f"Yahoo no devolvió datos para {instrumento}")
        limite = fecha_ultimo_cierre_ny(self._reloj())
        velas: list[Vela] = []
        for idx, fila in df.iterrows():
            d = idx.date() if hasattr(idx, "date") else idx
            # Convención (aproximada): la barra con fecha D = día que cierra D 17:00 NY.
            cierre = datetime.combine(d, time(17, 0), tzinfo=NY).astimezone(timezone.utc)
            apertura = cierre - timedelta(days=1)
            if any(v != v for v in (fila["Open"], fila["Close"])):  # NaN
                continue
            velas.append(Vela(
                instrumento=instrumento, tiempo=apertura, cierre=cierre,
                completa=d <= limite,
                o=float(fila["Open"]), h=float(fila["High"]), l=float(fila["Low"]),
                c=float(fila["Close"]), volumen=int(fila.get("Volume", 0) or 0), fuente=FUENTE,
            ))
        return velas[-n:]

    def precio(self, instrumento: str) -> Cotizacion:
        advertir_una_vez()
        df = self._descargar(self._ticker(instrumento), "5d", "1m")
        if df is None or len(df) == 0:
            df = self._descargar(self._ticker(instrumento), "5d", "1d")
        if df is None or len(df) == 0:
            raise RuntimeError(f"Yahoo no devolvió precio para {instrumento}")
        mid = float(df["Close"].dropna().iloc[-1])
        medio = self.spread_pips * PIP.get(instrumento, 0.0001) / 2.0
        ahora = self._reloj()
        return Cotizacion(instrumento, bid=mid - medio, ask=mid + medio, tiempo=ahora,
                          tradeable=mercado_fx_abierto(ahora), fuente=FUENTE)
