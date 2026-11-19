import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from psi_lab.lab1.dft import dft

potencias = range(4, 13)
Ns = [2**p for p in potencias]
tiempos_dft = []
tiempos_fft = []
repeticiones = 5

rng = np.random.default_rng(42)

for N in Ns:
    x = rng.normal(size=N)
    
    t0 = time.perf_counter()
    for i in range(repeticiones):
        dft(x)
    tiempos_dft.append((time.perf_counter() - t0) / repeticiones)
    
    t0 = time.perf_counter()
    for i in range(repeticiones):
        np.fft.fft(x)
    tiempos_fft.append((time.perf_counter() - t0) / repeticiones)

pendiente_dft, _ = np.polyfit(np.log10(Ns), np.log10(tiempos_dft), 1)
pendiente_fft, _ = np.polyfit(np.log10(Ns), np.log10(tiempos_fft), 1)

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(Ns, tiempos_dft, "o-", label=f"dft propia (pendiente ≈ {pendiente_dft:.2f})")
ax.loglog(Ns, tiempos_fft, "s-", label=f"np.fft.fft (pendiente ≈ {pendiente_fft:.2f})")

ax.set_title("Complejidad Computacional: DFT O(N²) vs FFT O(N log N)")
ax.set_xlabel("Tamaño de la señal N (muestras)")
ax.set_ylabel("Tiempo de ejecución promedio (s)")
ax.grid(True, which="both", linestyle="--", alpha=0.6)
ax.legend()

carpeta = Path("resultados/lab1/convolucion")
carpeta.mkdir(parents=True, exist_ok=True)
plt.savefig(
    carpeta / "tiempos_ejecucion.png", 
    dpi=300, 
    bbox_inches="tight")

plt.close()