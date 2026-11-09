
import numpy as np
import matplotlib.pyplot as plt

from psi_lab.lab1.muestreo import muestrear, frecuencia_alias


# Parámetros del experimento
fs = 10
duracion = 2
frecuencias = [3, 7, 13, 23]

# Tiempo continuo para representar las señales originales
t_cont = np.linspace(0, duracion, 2000, endpoint=False)

# Crear cuatro gráficos, uno por frecuencia
fig, axes = plt.subplots(4, 1, figsize=(10, 12))

for ax, f in zip(axes, frecuencias):

    # Definir la señal senoidal
    fun = lambda t, f=f: np.sin(2 * np.pi * f * t)

    # Obtener las muestras
    t, x = muestrear(fun, fs, duracion)

    # Calcular la frecuencia aparente
    f_alias = frecuencia_alias(f, fs)

    # Dibujar señal original
    ax.plot(t_cont, fun(t_cont), label="Señal continua")

    # Dibujar las muestras
    ax.stem(
        t, x,
        linefmt="r-",
        markerfmt="ro",
        basefmt="k-",
        label="Muestras"
    )

    ax.set_title(
        f"Frecuencia original: {f} Hz | "
        f"Frecuencia aparente: {f_alias} Hz"
    )

    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.grid(True)
    ax.legend()

plt.tight_layout(h_pad=3.5)

from pathlib import Path

carpeta = Path("resultados")
carpeta.mkdir(exist_ok=True)

plt.savefig(
    carpeta / "aliasing_senoidales.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()