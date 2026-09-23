"""Estructuras de datos compartidas (velas, cotizaciones, fills, señales)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Vela:
    """Vela diaria. `tiempo` = apertura (UTC), `cierre` = apertura + 1 día (UTC)."""
    instrumento: str
    tiempo: datetime
    cierre: datetime
    completa: bool
    o: float
    h: float
    l: float
    c: float
    bid_c: float | None = None
    ask_c: float | None = None
    volumen: int = 0
    fuente: str = "oanda"


@dataclass(frozen=True)
class Cotizacion:
    instrumento: str
    bid: float
    ask: float
    tiempo: datetime
    tradeable: bool = True
    fuente: str = "oanda"

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid


@dataclass
class Fill:
    """Resultado de una orden a mercado (apertura o cierre)."""
    instrumento: str
    unidades: int            # con signo: + compra, - venta
    precio: float            # precio efectivo de ejecución
    bid: float
    ask: float
    tiempo: datetime
    costo_spread: float      # medio spread pagado en esta ejecución (moneda de la cuenta)
    pnl: float = 0.0         # P&L realizado (solo cierres)
    financiamiento: float = 0.0
    broker_id: str = ""
    fuente: str = "oanda"


@dataclass
class Senal:
    """Salida de una estrategia para una vela ya cerrada."""
    estrategia: str
    instrumento: str
    direccion: int                    # +1 largo, -1 corto, 0 plano
    vela_tiempo: datetime | None      # apertura de la vela que generó la señal (UTC)
    vela_cierre: datetime | None      # cierre de esa vela (UTC)
    retorno_senal: float | None = None
    motivo: str = ""
    fuente: str = "oanda"
    detalle: dict = field(default_factory=dict)
