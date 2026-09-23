"""Configuración del bot de paper trading.

Lee variables desde `bot/.env` (ver `.env.example`) o desde el entorno del sistema,
y aplica los chequeos de seguridad DUROS antes de que cualquier otra parte del bot
toque la red:

- Solo se permite el host de práctica de OANDA (`api-fxpractice.oanda.com`).
- Si se configura el host real (`api-fxtrade.oanda.com`) o cualquier otro, el bot
  se niega a correr (`ErrorSeguridad`).
- El token nunca se imprime: `repr(Config)` lo enmascara y `registro.py` agrega un
  filtro que lo borra de cualquier línea de log.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parent

HOST_PRACTICA = "api-fxpractice.oanda.com"
HOST_REAL = "api-fxtrade.oanda.com"
MODOS_VALIDOS = ("simulado", "oanda_practice")


class ErrorSeguridad(RuntimeError):
    """Configuración peligrosa (p.ej. apuntar a la cuenta real). El bot no debe correr."""


class ErrorConfig(ValueError):
    """Configuración incompleta o inválida."""


def validar_host(host: str) -> str:
    """Devuelve el host normalizado si es el de práctica; si no, lanza ErrorSeguridad.

    Es una lista blanca de UN solo host: cualquier cosa distinta (incluido el host
    real `api-fxtrade`) se rechaza, así un typo nunca termina operando dinero real.
    """
    h = (host or "").strip().lower()
    for prefijo in ("https://", "http://"):
        if h.startswith(prefijo):
            h = h[len(prefijo):]
    h = h.split("/")[0].split(":")[0]
    if "fxtrade" in h or h == HOST_REAL:
        raise ErrorSeguridad(
            f"Host de CUENTA REAL detectado ({h}). Este bot solo opera en DEMO "
            f"({HOST_PRACTICA}). Abortando."
        )
    if h != HOST_PRACTICA:
        raise ErrorSeguridad(
            f"Host OANDA no permitido: '{h}'. Único host permitido: {HOST_PRACTICA}."
        )
    return h


def _float(nombre: str, defecto: float) -> float:
    valor = os.getenv(nombre)
    if valor is None or valor.strip() == "":
        return defecto
    try:
        return float(valor)
    except ValueError as e:
        raise ErrorConfig(f"{nombre} debe ser numérico (valor actual: {valor!r})") from e


def _bool(nombre: str, defecto: bool) -> bool:
    valor = os.getenv(nombre)
    if valor is None or valor.strip() == "":
        return defecto
    return valor.strip().lower() in ("1", "true", "si", "sí", "yes", "y")


def _ruta(valor: str) -> Path:
    p = Path(valor)
    return p if p.is_absolute() else (BOT_DIR / p)


@dataclass
class Config:
    modo: str = "simulado"
    oanda_token: str = field(default="", repr=False)
    oanda_account_id: str = ""
    oanda_host: str = HOST_PRACTICA
    capital_inicial: float = 10_000.0
    fraccion_notional: float = 1.0       # notional = fracción x equity
    apalancamiento_max: float = 2.0      # tope duro: notional <= apalancamiento_max x equity
    estrategia: str = "cobre_aud"
    instrumento: str = "AUD_USD"
    instrumento_senal: str = "XCU_USD"
    minutos_espera_cierre: int = 15      # no operar antes de cierre de vela + N min (rollover)
    reabrir_si_misma_senal: bool = False  # True = cerrar y reabrir cada día aunque la señal no cambie
    fuente_datos_simulado: str = "auto"  # auto | oanda | yahoo
    spread_fallback_pips: float = 1.4    # spread sintético para el fallback de Yahoo
    db_path: Path = field(default_factory=lambda: BOT_DIR / "data" / "bot.db")
    log_dir: Path = field(default_factory=lambda: BOT_DIR / "logs")

    def __repr__(self) -> str:  # nunca mostrar el token
        partes = []
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name == "oanda_token":
                v = "***" if v else "(vacío)"
            partes.append(f"{f.name}={v!r}")
        return "Config(" + ", ".join(partes) + ")"

    __str__ = __repr__

    @property
    def tiene_token(self) -> bool:
        return bool(self.oanda_token and self.oanda_token.strip())

    def validar(self) -> "Config":
        """Chequeos de seguridad y consistencia. Lanza ErrorSeguridad / ErrorConfig."""
        # El host se valida SIEMPRE (también en simulado, que puede usar datos OANDA).
        self.oanda_host = validar_host(self.oanda_host)
        if self.modo not in MODOS_VALIDOS:
            raise ErrorConfig(f"MODO inválido: {self.modo!r}. Opciones: {MODOS_VALIDOS}")
        if self.modo == "oanda_practice":
            if not self.tiene_token or not self.oanda_account_id.strip():
                raise ErrorConfig(
                    "MODO=oanda_practice requiere OANDA_TOKEN y OANDA_ACCOUNT_ID en bot/.env"
                )
        if self.fuente_datos_simulado not in ("auto", "oanda", "yahoo"):
            raise ErrorConfig("FUENTE_DATOS_SIMULADO debe ser auto | oanda | yahoo")
        if self.fuente_datos_simulado == "oanda" and not self.tiene_token:
            raise ErrorConfig("FUENTE_DATOS_SIMULADO=oanda requiere OANDA_TOKEN")
        if self.capital_inicial <= 0:
            raise ErrorConfig("CAPITAL_INICIAL debe ser > 0")
        if self.fraccion_notional <= 0:
            raise ErrorConfig("FRACCION_NOTIONAL debe ser > 0")
        if self.apalancamiento_max <= 0 or self.apalancamiento_max > 20:
            raise ErrorConfig("APALANCAMIENTO_MAX debe estar en (0, 20]")
        if self.minutos_espera_cierre < 0:
            raise ErrorConfig("MINUTOS_ESPERA_CIERRE debe ser >= 0")
        return self

    @property
    def fuente_datos(self) -> str:
        """De dónde salen velas y precios: 'oanda' o 'yahoo'."""
        if self.modo == "oanda_practice":
            return "oanda"
        if self.fuente_datos_simulado == "auto":
            return "oanda" if self.tiene_token else "yahoo"
        return self.fuente_datos_simulado


def cargar_config(env_file: Path | str | None = None, validar: bool = True) -> Config:
    """Carga la config desde `bot/.env` (si existe) + variables de entorno."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv es opcional en tiempo de ejecución
        load_dotenv = None
    ruta_env = Path(env_file) if env_file else BOT_DIR / ".env"
    if load_dotenv is not None and ruta_env.exists():
        load_dotenv(ruta_env, override=False)

    cfg = Config(
        modo=os.getenv("MODO", "simulado").strip().lower(),
        oanda_token=os.getenv("OANDA_TOKEN", "").strip(),
        oanda_account_id=os.getenv("OANDA_ACCOUNT_ID", "").strip(),
        oanda_host=os.getenv("OANDA_HOST", HOST_PRACTICA).strip(),
        capital_inicial=_float("CAPITAL_INICIAL", 10_000.0),
        fraccion_notional=_float("FRACCION_NOTIONAL", 1.0),
        apalancamiento_max=_float("APALANCAMIENTO_MAX", 2.0),
        estrategia=os.getenv("ESTRATEGIA", "cobre_aud").strip(),
        instrumento=os.getenv("INSTRUMENTO", "AUD_USD").strip().upper(),
        instrumento_senal=os.getenv("INSTRUMENTO_SENAL", "XCU_USD").strip().upper(),
        minutos_espera_cierre=int(_float("MINUTOS_ESPERA_CIERRE", 15)),
        reabrir_si_misma_senal=_bool("REABRIR_SI_MISMA_SENAL", False),
        fuente_datos_simulado=os.getenv("FUENTE_DATOS_SIMULADO", "auto").strip().lower(),
        spread_fallback_pips=_float("SPREAD_FALLBACK_PIPS", 1.4),
        db_path=_ruta(os.getenv("DB_PATH", "data/bot.db")),
        log_dir=_ruta(os.getenv("LOG_DIR", "logs")),
    )
    return cfg.validar() if validar else cfg
