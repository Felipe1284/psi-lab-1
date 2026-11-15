
import numpy as np
import pytest

from psi_lab.lab1.muestreo import (
    muestrear,
    frecuencia_alias,
)


def test_numero_muestras():
    """Verifica cantidad y tiempos de muestreo."""
    fun = lambda t: np.sin(2 * np.pi * 3 * t)

    t, x = muestrear(fun, fs=10, duracion=2)

    assert len(t) == 20
    assert len(x) == 20
    assert t[0] == pytest.approx(0)
    assert t[-1] == pytest.approx(1.9)


@pytest.mark.parametrize(
    "f, esperado",
    [
        (3, 3),
        (7, 3),
        (13, 3),
        (23, 3),
    ],
)
def test_frecuencia_alias(f, esperado):
    """Verifica las frecuencias aparentes."""
    resultado = frecuencia_alias(f, fs=10)

    assert resultado == pytest.approx(esperado)


def test_muestras_identicas():
    """Dos cosenos diferentes producen las mismas muestras."""
    fun1 = lambda t: np.cos(2 * np.pi * 3 * t)
    fun2 = lambda t: np.cos(2 * np.pi * 7 * t)

    t1, x1 = muestrear(fun1, 10, 1)
    t2, x2 = muestrear(fun2, 10, 1)

    assert np.allclose(t1, t2)
    assert np.allclose(x1, x2, atol=1e-12)


def test_frecuencia_muestreo_invalida():
    """La frecuencia de muestreo debe ser positiva."""
    with pytest.raises(ValueError):
        muestrear(lambda t: t, fs=0, duracion=2)

    with pytest.raises(ValueError):
        frecuencia_alias(7, fs=0)


def test_duracion_invalida():
    """La duración debe ser positiva."""
    with pytest.raises(ValueError):
        muestrear(lambda t: t, fs=10, duracion=-1)
