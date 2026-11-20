import numpy as np

from psi_lab.lab1.dft import dft, idft


def test_dft_coincide_con_numpy():
    """dft coincide con np.fft.fft"""
    rng = np.random.default_rng(42)
    x = rng.normal(size=64)
    np.testing.assert_allclose(dft(x), np.fft.fft(x), rtol=1e-10, atol=1e-10)


def test_idft_recupera_senal():
    """idft(dft(x)) recupera x"""
    rng = np.random.default_rng(42)
    x = rng.normal(size=64)
    np.testing.assert_allclose(idft(dft(x)), x, rtol=1e-10, atol=1e-10)


def test_simetria_conjugada_senal_real():
    """para x real, X[N - k] = X[k];"""
    rng = np.random.default_rng(42)
    x = rng.normal(size=64)
    X = dft(x)
    # Comparamos desde k=1 hasta N-1 contra el arreglo invertido
    np.testing.assert_allclose(X[1:], np.conj(X[1:][::-1]), rtol=1e-10, atol=1e-10)


def test_teorema_parseval():
    """se cumple Parseval"""
    rng = np.random.default_rng(42)
    x = rng.normal(size=64)
    X = dft(x)
    energia_tiempo = np.sum(np.abs(x) ** 2)
    energia_frecuencia = np.sum(np.abs(X) ** 2) / 64
    assert np.isclose(energia_tiempo, energia_frecuencia, rtol=1e-10)