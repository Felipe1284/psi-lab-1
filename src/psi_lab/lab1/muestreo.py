
import numpy as np


def muestrear(fun, fs, duracion):
    """
    Muestrea una señal continua.

    Parámetros:
        fun: función que representa la señal continua.
        fs: frecuencia de muestreo en Hz.
        duracion: duración de la señal en segundos.

    Retorna:
        t: instantes de muestreo.
        x: valores de la señal muestreada.
    """
    if fs <= 0 or duracion <= 0:
        raise ValueError("fs y duracion deben ser positivos")

    N = round(fs * duracion)

    t = np.arange(N) / fs

    x = np.asarray(fun(t))

    return t, x



def frecuencia_alias(f, fs):
    """
    Calcula la frecuencia aparente de una señal
    después de muestrearla.

    Parámetros:
        f: frecuencia original en Hz.
        fs: frecuencia de muestreo en Hz.

    Retorna:
        Frecuencia aparente entre 0 y fs/2.
    """
    if fs <= 0:
        raise ValueError("fs debe ser positiva")

    return abs((f + fs / 2) % fs - fs / 2)
