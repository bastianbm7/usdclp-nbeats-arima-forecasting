"""Logging del bot: archivo mensual en bot/logs/ + consola, con el token SIEMPRE redactado."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


class FiltroSecretos(logging.Filter):
    """Reemplaza cualquier secreto (token OANDA) por '***' en los mensajes de log."""

    def __init__(self, secretos: list[str]):
        super().__init__()
        self.secretos = [s for s in secretos if s and len(s) >= 4]

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.secretos:
            return True
        msg = record.getMessage()
        limpio = msg
        for s in self.secretos:
            limpio = limpio.replace(s, "***")
        if limpio != msg:
            record.msg = limpio
            record.args = ()
        return True


def configurar_logging(log_dir: Path, secretos: list[str] | None = None,
                       nivel: int = logging.INFO) -> Path:
    """Configura el logger raíz. Devuelve la ruta del archivo de log del mes."""
    log_dir.mkdir(parents=True, exist_ok=True)
    archivo = log_dir / f"bot_{datetime.now():%Y-%m}.log"
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    filtro = FiltroSecretos(secretos or [])

    raiz = logging.getLogger()
    raiz.setLevel(nivel)
    for h in list(raiz.handlers):
        raiz.removeHandler(h)

    fh = logging.FileHandler(archivo, encoding="utf-8")
    ch = logging.StreamHandler(sys.stdout)
    for h in (fh, ch):
        h.setFormatter(fmt)
        h.addFilter(filtro)
        raiz.addHandler(h)
    # urllib3 en DEBUG podría loguear headers; lo dejamos en WARNING.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    return archivo
