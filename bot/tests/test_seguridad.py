import logging

import pytest

from broker.oanda import OandaClient
from config import Config, ErrorConfig, ErrorSeguridad, cargar_config, validar_host
from registro import FiltroSecretos

TOKEN = "abcd1234secreto-5678efgh"


@pytest.mark.parametrize("host", [
    "api-fxtrade.oanda.com",
    "https://api-fxtrade.oanda.com",
    "https://api-fxtrade.oanda.com/v3",
    "API-FXTRADE.OANDA.COM",
    "stream-fxtrade.oanda.com",
])
def test_host_real_rechazado(host):
    with pytest.raises(ErrorSeguridad):
        validar_host(host)


@pytest.mark.parametrize("host", ["", "example.com", "api-fxpractice.oanda.com.evil.com"])
def test_host_desconocido_rechazado(host):
    with pytest.raises(ErrorSeguridad):
        validar_host(host)


def test_host_practica_ok():
    assert validar_host("https://api-fxpractice.oanda.com/") == "api-fxpractice.oanda.com"


def test_config_con_host_real_no_valida():
    with pytest.raises(ErrorSeguridad):
        Config(oanda_host="api-fxtrade.oanda.com").validar()


def test_cargar_config_host_real_desde_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("MODO=oanda_practice\nOANDA_TOKEN=x\nOANDA_ACCOUNT_ID=1\n"
                   "OANDA_HOST=api-fxtrade.oanda.com\n", encoding="utf-8")
    with pytest.raises(ErrorSeguridad):
        cargar_config(env)


def test_cliente_oanda_rechaza_host_real():
    with pytest.raises(ErrorSeguridad):
        OandaClient(TOKEN, "101-001-1", host="api-fxtrade.oanda.com")


def test_oanda_practice_requiere_credenciales():
    with pytest.raises(ErrorConfig):
        Config(modo="oanda_practice").validar()


def test_modo_invalido():
    with pytest.raises(ErrorConfig):
        Config(modo="real").validar()


def test_token_no_aparece_en_repr():
    c = Config(oanda_token=TOKEN, oanda_account_id="101-001-1")
    assert TOKEN not in repr(c) and TOKEN not in str(c)
    assert TOKEN not in repr(OandaClient(TOKEN, "101-001-1"))


def test_filtro_redacta_token_en_logs():
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "Header: Bearer %s", (TOKEN,), None)
    FiltroSecretos([TOKEN]).filter(rec)
    assert TOKEN not in rec.getMessage()
    assert "***" in rec.getMessage()
