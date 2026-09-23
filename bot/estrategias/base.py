"""Interfaz mínima de estrategia. Agregar una nueva = una clase + registrarla en __init__.py."""
from __future__ import annotations

from abc import ABC, abstractmethod

from modelos import Senal, Vela


class Estrategia(ABC):
    nombre: str = "base"
    version: str = "0"

    #: instrumento que se opera
    instrumento: str
    #: instrumentos cuyas velas diarias necesita `calcular_senal`
    instrumentos_datos: list[str]
    #: cuántas velas diarias pedir por instrumento
    n_velas: int = 10

    @abstractmethod
    def calcular_senal(self, velas: dict[str, list[Vela]]) -> Senal:
        """Recibe velas diarias por instrumento y devuelve la posición objetivo (-1/0/+1)
        para las próximas ~24h. Debe usar SOLO velas completas (sin mirar el futuro)."""
