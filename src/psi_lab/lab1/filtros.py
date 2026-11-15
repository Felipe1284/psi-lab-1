import numpy as np
from .convolucion import convolucion_directa, convolucion_fft

# --------------------------------------------------------------------------- #

def _ventana(nombre, N):
    if nombre == "hamming":
        return np.hamming(N)
    if nombre == "hann":
        return np.hanning(N)
    if nombre == "rect":
        return np.ones(N)
    raise ValueError(f"Ventana desconocida: {nombre}. Debe ser 'hamming', 'hann' o 'rect'.")

def _delta(n_coef):
    # Impulso centrado: 1 en la posición (n_coef - 1) / 2, 0 en el resto
    d = np.zeros(n_coef)

    # Colocamos el 1 en la posición central del array
    d[(n_coef - 1) // 2] = 1

    # Devolvemos el array delta.
    return d 

# --------------------------------------------------------------------------- #

def fir_pasabajos(fc, fs, n_coef, ventana="hamming"):
    """Sinc truncada y enventanada, normalizada a ganancia 1 en DC."""

    # Validar n_coef impar
    if n_coef % 2 == 0: raise ValueError("El número de coeficientes debe ser impar.")

    M = (n_coef - 1) // 2
    n = np.arange(n_coef)

    # Como sinc es infinita la truncamos a n_coef y la centramos en M
    h = 2 * fc / fs * np.sinc(2 * fc / fs * (n - M))

    # Aplicamos la ventana seleccionada
    h *= _ventana(ventana, n_coef)

    # Normalizamos a ganancia 1 en DC
    return h / np.sum(h)

def fir_pasaaltos(fc, fs, n_coef, ventana="hamming"):
    """Invercion espectral: invertir los coeficientes del pasabajos y sumar 1 en el centro."""

    if n_coef % 2 == 0: raise ValueError("El número de coeficientes debe ser impar.")
    
    # Hpa(f) = 1 - Hpb(f)
    return _delta(n_coef) - fir_pasabajos(fc, fs, n_coef, ventana)

def fir_pasabanda(f1, f2, fs, n_coef, ventana="hamming"):
    if n_coef % 2 == 0: raise ValueError("El número de coeficientes debe ser impar.")
    return fir_pasabajos(f2, fs, n_coef, ventana) - fir_pasabajos(f1, fs, n_coef, ventana)

def fir_rechazabanda(f1, f2, fs, n_coef, ventana="hamming"):
    if n_coef % 2 == 0: raise ValueError("El número de coeficientes debe ser impar.")
    return _delta(n_coef) - fir_pasabanda(f1, f2, fs, n_coef, ventana)

def respuesta_frecuencia(h, fs, n=4096):
    """Retorna (freqs, H) con H complejo."""

    h = np.asarray(h)
    N = len(h)

    # vector de indices k de los coeficientes (fila)
    k = np.arange(N)

    # vector de frecuencias
    freqs = np.linspace(0, fs / 2, n)

    # frecuencias como columnas
    f = freqs.reshape((n, 1))

    # Matriz exponencial para cada (f, k)
    e = np.exp(-2j * np.pi * f * k / fs)
    
    # Retorna la multiplicación de matriz x vector
    return freqs, np.dot(e, h)

def filtrar(x, h):
    """Aplica h a x (largo igual a x), compensando el retardo (n_coef - 1) / 2."""

    if h % 2 == 0: 
        raise ValueError("El número de coeficientes debe ser impar.")

    x = np.asarray(x)
    h = np.asarray(h)

    # Calculamos el retardo del filtro
    retardo = (len(h) - 1) // 2

    # Aplicamos la convolución lineal completa
    if len(h) >= 500:
        y = convolucion_fft(x, h)
    else:
        y = convolucion_directa(x, h)

    # Compensamos el retardo y recortamos la señal de salida
    return y[retardo:retardo + len(x)]