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

    # convertir X en un array de numpy si no lo es y obtiene su longitud
    X = np.asarray(X)
    N = len(X)

    # vectores de indices n (tiempo) y k (frecuencia)
    n = np.arange(N)
    k = n.reshape((N, 1))

    # matriz exponencial de la DFT inversa para cada par (k, n) (exponente positivo)
    e = np.exp(2j * np.pi * k * n / N)

    # Se divide por N para normalizar la transformada inversa
    return np.dot(e, X) / N

def espectro(x, fs, ventana="rect", nfft=None):
    """Espectro de amplitud de un lado. ventana en {"rect", "hann", "hamming"};
    nfft > len(x) rellena con ceros. Retorna (freqs, amp)."""

    x = np.asarray(x, dtype=float)
    N = len(x)

    # ventana
    if ventana == "hann":
        w = np.hanning(N)
    elif ventana == "hamming":
        w = np.hamming(N)
    else:
        w = np.ones(N)

    x_w = x * w

    # nfft
    if nfft is not None and nfft > N:
        x_final = np.zeros(nfft)
        x_final[:N] = x_w
        N_fft = nfft
    else:
        x_final = x_w
        N_fft = N

    # dft
    X = dft(x_final)

    # frecuencia
    freqs = np.arange(N_fft) * fs / N_fft

    # espectro de amplitud de un lado
    N_mitad = N_fft // 2
    freqs_final = freqs[:N_mitad]

    # magnitud de la DFT normalizada por N_fft y multiplicada por 2 para obtener el espectro de un lado
    amp_final = np.abs(X[:N_mitad]) / N_fft

    # duplicar amplitud ignorando la frecuencia 0
    amp_final[1:] = amp_final[1:] * 2

    return freqs_final, amp_final