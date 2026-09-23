"""Interfaz común de broker: OANDA práctica y simulado exponen los mismos métodos."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from modelos import Cotizacion, Fill, Vela


@dataclass
class Posicion:
    instrumento: str
    unidades: int          # con signo
    precio_promedio: float

    @property
    def direccion(self) -> int:
        return (self.unidades > 0) - (self.unidades < 0)


class Broker(ABC):
    nombre: str = "base"

    @abstractmethod
    def velas_diarias(self, instrumento: str, n: int = 10) -> list[Vela]:
        """Velas diarias (alineadas a 17:00 NY cuando la fuente lo permite)."""

    @abstractmethod
    def precio(self, instrumento: str) -> Cotizacion:
        """Bid/ask actual y si el instrumento se puede operar ahora."""

    @abstractmethod
    def equity(self, cotizacion: Cotizacion | None = None) -> float:
        """Patrimonio actual (balance + P&L no realizado)."""

    @abstractmethod
    def balance(self) -> float:
        """Balance realizado."""

    @abstractmethod
    def posicion_abierta(self, instrumento: str) -> Posicion | None:
        ...

    @abstractmethod
    def abrir_mercado(self, instrumento: str, unidades: int, etiqueta: str = "") -> Fill:
        """Orden a mercado: compra al ASK (unidades > 0) o vende al BID (unidades < 0)."""

    @abstractmethod
    def cerrar_posicion(self, instrumento: str, etiqueta: str = "") -> Fill | None:
        """Cierra toda la posición del instrumento (largo al BID, corto al ASK)."""

    def moneda_cuenta(self) -> str:
        return "USD"
