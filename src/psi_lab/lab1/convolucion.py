import numpy as np
from .dft import dft, idft

# Se prohibe usar la función np.fft, por lo que crearemos una propia
# Tampoco se usara la función dft, por que toma un tiempo O(N^2) y no es eficiente para señales grandes
# En su lugar, se implementará la FFT, que toma un tiempo O(N log N)
# la FFT calcula la misma DFT en NlogN

def _fft(x, inverso=False):

    x = np.asarray(x, dtype=complex)
    N = len(x)
    
    # Si N no es potencia de 2
    if 2 ** int(np.log2(N)) != N: 
        raise ValueError("La longitud de la señal debe ser una potencia de 2.")

    # Reordenar los elementos de x en orden bit-reverso
    
    bits = N.bit_length() - 1
    indices = np.arange(N)
    
    # Rellenamos el array al inicio con ceros
    reverso = np.zeros(N, dtype=int)

    # Reordenamos los indices en orden bit-reverso
    for i in range(bits):
        
        # El operador >> desplaza los bits a la derecha, y el operador & obtiene el bit menos significativo
        # Luego, se desplaza el bit menos significativo a la posición correspondiente en el orden bit-reverso
        reverso |= ((indices >> i) & 1) << (bits - 1 - i)

    x = x[reverso].copy()

    # ----------------------------------------- Implementamos las mariposas de la FFT ----------------------------------------- #
    signo = 1 if inverso else -1
    tam = 2

    while tam <= N:

        # Calculamos la división entera
        mitad = tam // 2

        # Calculamos el factor de giro
        w = np.exp(signo * 2j * np.pi * np.arange(mitad) / tam)

        # Cada fila es un bloque de tamaño 'tam'
        # No copia, X y x comparten la misma memoria, por lo que los cambios en X afectan a x
        X = x.reshape(-1, tam)

        # Partimos el array en dos mitades, y multiplicamos la segunda mitad por el factor de giro
        par = X[:, :mitad].copy()
        impar = X[:, mitad:] * w

        # Actualizamos los valores de x con las mariposas
        X[:, :mitad] = par + impar
        X[:, mitad:] = par - impar

        tam *= 2

    # Si es la FFT inversa, dividimos por N
    if inverso:
        x /= N
    
    return x

def _ifft(x):
    return _fft(x, inverso=True)

# ------------------------------------------------------------------------------------------------- #

def convolucion_directa(x, h):
    """Convolucion lineal completa (largo len(x) + len(h) - 1), por definicion.
    
    Como se calcula:
    y[n] = ∑k x[k] * h[n − k]           (Clase convolución) """

    x = np.asarray(x)
    h = np.asarray(h)
    Nx = len(x)
    Nh = len(h)

    if Nx == 0 or Nh == 0:
        return np.array([])

    # Determina el tipo de dato de salida, para que sea compatible con los tipos de entrada
    tipo = np.result_type(x, h, float)

    
    # y es el Array de salida, con el largo de la convolución lineal completa
    y = np.zeros(Nx + Nh - 1, dtype=tipo)

    
    # Si el filtro es mas largo que la señal, se invierten para que el bucle sea mas corto
    if Nx < Nh:
        x, h = h, x
        Nx, Nh = Nh, Nx

    
    # Por cada coeficiente del filtro, se multiplica por toda la señal y se suma al resultado
    # k:k+Nx empieza de k y termina en k+Nx, por lo que se suman los valores de x multiplicados por h[k] en la posición correspondiente de y
    for k in range(Nh):
        y[k:k + Nx] += h[k] * x

    
    # Retorna el resultado de la convolución lineal completa
    return y


def convolucion_fft(x, h, rellenar=True):
    """Misma convolucion lineal, via el teorema de la convolucion.
    Usamos la FFT para calcular la convolución lineal completa, por el teorema de la convolución. 
    
    Como se calcula:
    y[n] = IFFT( FFT(x) * FFT(h) )          (Clase convolución)"""

    x = np.asarray(x)
    h = np.asarray(h)
    Nx = len(x)
    Nh = len(h)

    # Si alguna de las señales es vacía, retorna un array vacío
    if Nx == 0 or Nh == 0: 
        return np.array([])

    # Determina el tipo de dato de salida, para que sea compatible con los tipos de entrada
    # FFT trabaja con números complejos, por lo que el tipo de salida será complejo si alguna de las señales es compleja
    es_real = np.isrealobj(x) and np.isrealobj(h)

    if rellenar:
        # Largo de la convolución lineal completa 
        largo = Nx + Nh - 1
        
        # Menor potencia de 2 que es mayor o igual a largo
        L = 2 ** int(np.ceil(np.log2(largo)))
        
        xp = np.zeros(L, dtype=tipo)
        hp = np.zeros(L, dtype=tipo)
        xp[:Nx] = x
        hp[:Nh] = h
        
        # Usamos FFT
        # Usamos [:largo], ya que solo se necitan los primeros 'largo' datos
        y = _ifft( _fft(xp) * _fft(hp) )[:largo]

    else:
        # Convolución circular
        N = max(Nx, Nh)
        xp = np.zeros(N, dtype=tipo)
        hp = np.zeros(N, dtype=tipo)
        xp[:Nx] = x
        hp[:Nh] = h
        y = idft(dft(xp) * dft(hp))

    # Si las entradas eran reales, se retorna solo la parte real
    return y.real if es_real else y