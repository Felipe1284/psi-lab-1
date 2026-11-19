import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from psi_lab.lab1.convolucion import convolucion_fft, convolucion_directa

# ---------------------------------------------------------
# 2. Muestren qué ocurre si en convolucion_fft no se rellena con ceros: grafiquen el resultado
# (convolución circular) junto al correcto y expliquen la diferencia.

# Señales de ejemplo
# x: pulso cerca del final, para que la cola de la convolución no quepa
# h: exponencial decreciente
x = np.zeros(20)
x[12:] = 1
h = 0.7 ** np.arange(8)

# Convolución con y sin relleno de ceros
lineal = convolucion_fft(x, h)                     # largo 20 + 8 - 1 = 27
circular = convolucion_fft(x, h, rellenar=False)   # largo max(20, 8) = 20
N = len(circular)

# Comprobar la explicación: la circular es la lineal con la cola
doblada = lineal[:N].copy()
doblada[:len(lineal) - N] += lineal[N:]
print("¿Circular = lineal doblada?", np.allclose(circular, doblada))

# Gráfico
plt.figure(figsize=(10, 5))

plt.stem(np.arange(len(lineal)), lineal, linefmt="gray", markerfmt="o", basefmt="k-", label="Lineal, con relleno (correcta)")

plt.scatter(np.arange(N), circular, color="red", marker="x", s=80, label="Circular, sin relleno", zorder=5)

plt.axvline(N - 0.5, color="orange", linestyle="--", label=f"Fin de la circular (N = {N})")

plt.xlabel("n (muestras)")
plt.ylabel("Amplitud")
plt.title("Convolución lineal vs. circular")
plt.legend()
plt.grid(True)
plt.tight_layout()

carpeta = Path("resultados/lab1/convolucion")
carpeta.mkdir(parents=True, exist_ok=True)
plt.savefig(
    carpeta / "convolucion_circular.png", 
    dpi=300, 
    bbox_inches="tight")

plt.close()

# ---------------------------------------------------------
# 3. Comparen tiempos de ambas implementaciones en función del largo del filtro. ¿Desde qué
# largo conviene la FFT?

def medir(funcion, x, h, repeticiones=5):
    """Mejor tiempo (en segundos) de varias repeticiones."""
    tiempos = []
    for _ in range(repeticiones):
        inicio = time.perf_counter()
        funcion(x, h)
        tiempos.append(time.perf_counter() - inicio)
    return min(tiempos)


# Señal fija y filtros de distintos largos
rng = np.random.default_rng(0)
x = rng.standard_normal(10_000)
largos = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

t_directa = []
t_fft = []
for nh in largos:
    h = rng.standard_normal(nh)
    t_directa.append(medir(convolucion_directa, x, h))
    t_fft.append(medir(convolucion_fft, x, h))

# Primer largo donde la FFT es más rápida
umbral = next((nh for nh, td, tf in zip(largos, t_directa, t_fft) if tf < td), None)

# Guardar tabla de datos
with open(carpeta / "tiempos_convolucion.csv", "w") as f:
    f.write("Tarea 3:\n")
    f.write("3. Comparen tiempos de ambas implementaciones en función del largo del filtro. ¿Desde qué largo conviene la FFT?\n\n")
    f.write("largo, directa_[ms], fft_[ms]\n")
    for nh, td, tf in zip(largos, t_directa, t_fft):
        f.write(f"{nh}, {td*1e3:.3f}, {tf*1e3:.3f}\n")
    f.write(f"La fft conviene desde un largo de {umbral}\n")


# Gráfico
plt.figure(figsize=(10, 5))
plt.loglog(largos, np.array(t_directa) * 1e3, "o-", label="Directa")
plt.loglog(largos, np.array(t_fft) * 1e3, "s-", label="FFT")
if umbral is not None:
    plt.axvline(umbral, color="gray", linestyle="--", label=f"Cruce ≈ {umbral}")

plt.xlabel("Largo del filtro (coeficientes)")
plt.ylabel("Tiempo (ms)")
plt.title(f"Tiempo de convolución vs. largo del filtro (señal de {len(x)} muestras)")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()

plt.savefig(carpeta / "tiempos_convolucion.png", dpi=300, bbox_inches="tight")
plt.close()