"""Regla v1 (congelada): signo del retorno diario del cobre -> posición en AUD/USD.

Contexto honesto: en la investigación, la señal "retorno del cobre en t -> moneda
commodity en t+1" resultó ser un artefacto de timestamps de Yahoo. Esta es la versión
CORREGIDA, con ambas velas cerrando exactamente a la misma hora (17:00 Nueva York,
`dailyAlignment=17` en OANDA), para hacer un forward test en vivo. Se espera edge ~0
después de costos; el objetivo es medirlo, no prometer ganancias.

Regla:
- Se toma la última vela diaria COMPLETA de XCU_USD y de AUD_USD (misma hora de cierre).
- retorno_cobre = close_t / close_{t-1} - 1 (mid, cierre a cierre).
- Señal: +1 (largo AUD_USD) si retorno > 0, -1 (corto) si < 0, 0 (plano) si = 0 o si
  faltan datos / las velas no están alineadas (p.ej. feriado en el cobre).
"""
from __future__ import annotations

from datetime import timedelta

from estrategias.base import Estrategia
from modelos import Senal, Vela


class CobreAud(Estrategia):
    nombre = "cobre_aud"
    version = "v1"
    n_velas = 6

    def __init__(self, instrumento: str = "AUD_USD", instrumento_senal: str = "XCU_USD",
                 tolerancia_alineacion: timedelta = timedelta(minutes=1)):
        self.instrumento = instrumento
        self.instrumento_senal = instrumento_senal
        self.instrumentos_datos = [instrumento_senal, instrumento]
        self.tolerancia = tolerancia_alineacion

    def _plano(self, motivo: str, vela: Vela | None = None, fuente: str = "oanda",
               detalle: dict | None = None) -> Senal:
        return Senal(self.nombre, self.instrumento, 0,
                     vela.tiempo if vela else None, vela.cierre if vela else None,
                     None, motivo, fuente, detalle or {})

    def calcular_senal(self, velas: dict[str, list[Vela]]) -> Senal:
        cobre = sorted([v for v in velas.get(self.instrumento_senal, []) if v.completa],
                       key=lambda v: v.tiempo)
        aud = sorted([v for v in velas.get(self.instrumento, []) if v.completa],
                     key=lambda v: v.tiempo)
        fuente = (aud or cobre)[-1].fuente if (aud or cobre) else "oanda"

        if not aud:
            return self._plano(f"sin velas completas de {self.instrumento}", fuente=fuente)
        ultima_aud = aud[-1]
        if len(cobre) < 2:
            return self._plano(f"menos de 2 velas completas de {self.instrumento_senal}",
                               ultima_aud, fuente)

        ultima_cu, previa_cu = cobre[-1], cobre[-2]
        detalle = {
            "vela_cobre_tiempo": ultima_cu.tiempo.isoformat(),
            "vela_cobre_previa_tiempo": previa_cu.tiempo.isoformat(),
            "vela_aud_tiempo": ultima_aud.tiempo.isoformat(),
            "cobre_close": ultima_cu.c,
            "cobre_close_previo": previa_cu.c,
            "aud_close": ultima_aud.c,
            "version": self.version,
        }
        # La vela del cobre debe cerrar a la misma hora que la última del AUD.
        if abs(ultima_cu.cierre - ultima_aud.cierre) > self.tolerancia:
            return self._plano("velas no alineadas (posible feriado o dato faltante)",
                               ultima_aud, fuente, detalle)
        if previa_cu.c <= 0:
            return self._plano("precio previo del cobre inválido", ultima_aud, fuente, detalle)

        ret = ultima_cu.c / previa_cu.c - 1.0
        direccion = (ret > 0) - (ret < 0)
        motivo = {1: "cobre sube -> largo AUD", -1: "cobre baja -> corto AUD",
                  0: "cobre sin cambio -> plano"}[direccion]
        return Senal(self.nombre, self.instrumento, direccion, ultima_aud.tiempo,
                     ultima_aud.cierre, ret, motivo, fuente, detalle)
