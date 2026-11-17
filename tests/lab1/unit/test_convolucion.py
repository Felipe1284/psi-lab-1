import numpy as np
import pytest

from psi_lab.lab1.convolucion import convolucion_fft, convolucion_directa

FUNCTIONS = [convolucion_directa, convolucion_fft]
LARGOS = [(1, 1), (5, 3), (3, 5), (16, 16), (100, 37), (1000, 101)]

@pytest.mark.parametrize("convolucion", FUNCTIONS)
@pytest.mark.parametrize("x, h", LARGOS)
def test_coincide_npconvolve(convolucion, x, h):
    """Comprobar que la convolución hecha coincide con np.convolve."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(x)
    h = rng.standard_normal(h)

    y1 = convolucion(x, h) # Función propia
    y2 = np.convolve(x, h) # Función numpy

    assert np.allclose(y1, y2)


@pytest.mark.parametrize("convolucion", FUNCTIONS)
def test_ejemplo_simple(convolucion):
    # [1, 2, 3] * [-1, 0, 1] = [-1, -2, -2, 2, 3]
    assert np.allclose(convolucion([1, 2, 3], [-1, 0, 1]), [-1, -2, -2, 2, 3])


@pytest.mark.parametrize("convolucion", FUNCTIONS)
def test_largo_de_salida(convolucion):
    """La salida de la convolución lineal completa tiene largo len(x) + len(h) - 1."""
    x = np.ones(10)
    h = np.ones(4)
    y = convolucion(x, h)
    assert len(y) == len(x) + len(h) - 1


@pytest.mark.parametrize("convolucion", FUNCTIONS)
def test_convolucion_vacia(convolucion):
    """Si alguna de las señales es vacía, la salida es vacía."""
    assert len(convolucion([], [1, 2, 3])) == 0
    assert len(convolucion([1, 2, 3], [])) == 0
    assert len(convolucion([], [])) == 0


@pytest.mark.parametrize("convolucion", FUNCTIONS)
def test_conmutativa(convolucion):
    """La convolución es conmutativa: x * h = h * x."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(10)
    h = rng.standard_normal(4)

    y1 = convolucion(x, h)
    y2 = convolucion(h, x)

    assert np.allclose(y1, y2)


def test_directa_fft_equivalentes():
    """Comprobar que la convolución directa y la FFT dan el mismo resultado."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal(500)
    h = rng.standard_normal(64)

    y1 = convolucion_directa(x, h)
    y2 = convolucion_fft(x, h)

    assert np.allclose(y1, y2)

def test_salida_real_si_entradas_reales():
    """Si las entradas son reales, la salida también es real."""
    x = np.ones(10)
    h = np.ones(4)
    
    y = convolucion_fft(x, h)
    assert np.isrealobj(y)

def test_salida_compleja_si_entradas_complejas():
    """Si alguna de las entradas es compleja, la salida también es compleja."""
    x = np.ones(10) + 1j * np.ones(10)
    h = np.ones(4)
    
    y = convolucion_fft(x, h)
    assert not np.isrealobj(y)

def test_sin_relleno_es_circular():
    """Si no se rellena con ceros, la convolución es circular."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal(20)
    h = rng.standard_normal(8)
    N = len(x)

    referencia = np.real(np.fft.ifft(np.fft.fft(x) * np.fft.fft(h, N)))
    y = convolucion_fft(x, h, rellenar=False)
    assert np.allclose(y, referencia)