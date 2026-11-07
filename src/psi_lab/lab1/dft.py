import numpy as np

def dft(x):
    """DFT por definicion, O(N^2): X[k] = sum_n x[n] exp(-2j pi k n / N)."""

    # convertir x en un array de numpy si no lo es y obtiene su longitud
    x = np.asarray(x)
    N = len(x)

    # vectores de indices n (tiempo) y k (frecuencia)
    n = np.arange(N)
    k = n.reshape((N, 1))
    
    # matriz exponencial de Fourier para cada par (k, n)
    e = np.exp(-2j * np.pi * k * n / N)
    
    # multiplicacion de matriz exponencial por vector de señales (esto hace la sumatoria)
    return np.dot(e, x)

def idft(X):
    """DFT inversa por definicion."""

def espectro(x, fs, ventana="rect", nfft=None):
    """Espectro de amplitud de un lado. ventana en {"rect", "hann", "hamming"};
    nfft > len(x) rellena con ceros. Retorna (freqs, amp)."""