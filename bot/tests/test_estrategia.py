from datetime import timedelta

from conftest import hacer_velas
from estrategias import crear_estrategia
from estrategias.cobre_aud import CobreAud


def _senal(cu, aud, **kw):
    return CobreAud().calcular_senal({"XCU_USD": cu, "AUD_USD": aud, **kw})


def test_cobre_sube_largo():
    s = _senal(hacer_velas("XCU_USD", [4.00, 4.10], "2026-09-22"),
               hacer_velas("AUD_USD", [0.65, 0.66], "2026-09-22"))
    assert s.direccion == 1
    assert abs(s.retorno_senal - 0.025) < 1e-12
    assert s.vela_cierre.isoformat().startswith("2026-09-22T21:00")  # 17:00 NY (EDT)


def test_cobre_baja_corto():
    s = _senal(hacer_velas("XCU_USD", [4.10, 4.00], "2026-09-22"),
               hacer_velas("AUD_USD", [0.65, 0.66], "2026-09-22"))
    assert s.direccion == -1


def test_cobre_sin_cambio_plano():
    s = _senal(hacer_velas("XCU_USD", [4.10, 4.10], "2026-09-22"),
               hacer_velas("AUD_USD", [0.65, 0.66], "2026-09-22"))
    assert s.direccion == 0


def test_vela_incompleta_se_ignora():
    # La última vela del cobre (incompleta) bajaría; la señal debe usar solo las completas.
    cu = hacer_velas("XCU_USD", [4.00, 4.10, 3.00], "2026-09-23", ultima_completa=False)
    aud = hacer_velas("AUD_USD", [0.65, 0.66, 0.67], "2026-09-23", ultima_completa=False)
    s = _senal(cu, aud)
    assert s.direccion == 1
    assert s.vela_cierre.date().isoformat() == "2026-09-22"


def test_datos_faltantes_plano():
    s = _senal([], hacer_velas("AUD_USD", [0.65, 0.66], "2026-09-22"))
    assert s.direccion == 0 and "velas" in s.motivo
    s2 = _senal(hacer_velas("XCU_USD", [4.0, 4.1], "2026-09-22"), [])
    assert s2.direccion == 0 and s2.vela_tiempo is None


def test_velas_desalineadas_plano():
    # Feriado en el cobre: su última vela es de un día antes que la del AUD.
    s = _senal(hacer_velas("XCU_USD", [4.0, 4.1], "2026-09-21"),
               hacer_velas("AUD_USD", [0.65, 0.66], "2026-09-22"))
    assert s.direccion == 0
    assert "alineadas" in s.motivo


def test_registro_estrategias(cfg):
    e = crear_estrategia(cfg)
    assert e.nombre == "cobre_aud" and e.instrumento == "AUD_USD"
    assert e.instrumentos_datos == ["XCU_USD", "AUD_USD"]
    assert CobreAud().tolerancia == timedelta(minutes=1)
