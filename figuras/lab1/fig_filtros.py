import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from psi_lab.lab1.filtros import fir_pasabajos, fir_pasaaltos, fir_pasabanda, fir_rechazabanda, respuesta_frecuencia, filtrar

# ----------------------------------------------------------
# 1. Obtengan pasaaltos, pasabanda y rechazabanda a partir del pasabajos (inversión espectral,
# suma/resta) y explíquenlo.


# Parámetros comunes
fs = 1000
n_coef = 101
f1 = 100
f2 = 200    # cortes (Hz)

# Los cuatro filtros, todos construidos a partir del pasabajos
pb_f1 = fir_pasabajos(f1, fs, n_coef)
pb_f2 = fir_pasabajos(f2, fs, n_coef)

pa = fir_pasaaltos(f1, fs, n_coef)            # delta - PB(f1)
pbanda = fir_pasabanda(f1, f2, fs, n_coef)    # PB(f2) - PB(f1)
rb = fir_rechazabanda(f1, f2, fs, n_coef)     # delta - pasabanda

# Respuestas en frecuencia (magnitud)
freqs, H_pb_f1 = respuesta_frecuencia(pb_f1, fs)
_, H_pb_f2 = respuesta_frecuencia(pb_f2, fs)
_, H_pa = respuesta_frecuencia(pa, fs)
_, H_pbanda = respuesta_frecuencia(pbanda, fs)
_, H_rb = respuesta_frecuencia(rb, fs)

# Comprobar las relaciones de la explicación: sumadas dan un retardo puro,
# H = e^{-j 2 pi f M / fs}, cuya magnitud es 1 en todas las frecuencias
print("¿|H_PB + H_PA| = 1?", np.allclose(np.abs(H_pb_f1 + H_pa), 1))
print("¿|H_pasabanda + H_rechazabanda| = 1?", np.allclose(np.abs(H_pbanda + H_rb), 1))
print("¿H_pasabanda = H_PB(200) - H_PB(100)?", np.allclose(H_pbanda, H_pb_f2 - H_pb_f1))

# Gráfico: 2 x 2, uno por filtro
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)

# Pasabajos
ax = axes[0, 0]
ax.plot(freqs, np.abs(H_pb_f1), label="Pasabajos (100 Hz)")
ax.set_title("Pasabajos: sinc enventanada")

# Pasaaltos = delta - pasabajos
ax = axes[0, 1]
ax.plot(freqs, np.abs(H_pa), color="C1", label="Pasaaltos = δ − PB(100)")
ax.plot(freqs, np.abs(H_pb_f1), "k--", alpha=0.4, label="PB(100)")
ax.set_title("Pasaaltos: inversión espectral")

# Pasabanda = PB(f2) - PB(f1)
ax = axes[1, 0]
ax.plot(freqs, np.abs(H_pbanda), color="C2", label="Pasabanda = PB(200) − PB(100)")
ax.plot(freqs, np.abs(H_pb_f2), "k--", alpha=0.4, label="PB(200)")
ax.plot(freqs, np.abs(H_pb_f1), "k:", alpha=0.4, label="PB(100)")
ax.set_title("Pasabanda: resta de dos pasabajos")

# Rechazabanda = delta - pasabanda
ax = axes[1, 1]
ax.plot(freqs, np.abs(H_rb), color="C3", label="Rechazabanda = δ − Pasabanda")
ax.plot(freqs, np.abs(H_pbanda), "k--", alpha=0.4, label="Pasabanda")
ax.set_title("Rechazabanda: inversión del pasabanda")

for ax in axes.flat:
    ax.axvline(f1, color="gray", linewidth=0.8)
    ax.axvline(f2, color="gray", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=8)
for ax in axes[1]:
    ax.set_xlabel("Frecuencia (Hz)")
for ax in axes[:, 0]:
    ax.set_ylabel("|H(f)|")

fig.suptitle(f"Filtros obtenidos a partir del pasabajos (fs = {fs} Hz, {n_coef} coef., Hamming)")
plt.tight_layout()

carpeta = Path("resultados/lab1/filtros")
carpeta.mkdir(parents=True, exist_ok=True)
plt.savefig(carpeta / "filtros_desde_pasabajos.png", dpi=300, bbox_inches="tight")
plt.close()

# ----------------------------------------------------------
# 2. Para un pasabajos con fc = 100 Hz, fs = 1000 Hz, grafiquen |H(f)| en dB variando
# n_coef ∈ {21, 51, 101, 201} y la ventana (rectangular, Hamming, Hann). Midan ancho de
# transición y atenuación en la banda de rechazo; resuman en una tabla.

# Parámetros del experimento
fc = 100
fs = 1000
lista_n_coef = [21, 51, 101, 201]
ventanas = ["rect", "hamming", "hann"]
nombres = {"rect": "Rectangular", "hamming": "Hamming", "hann": "Hann"}


def medir(freqs, H_db, fc):
    """Mide ancho de transición y atenuación de un pasabajos.

    - Inicio de la transición: última frecuencia con |H| >= -3 dB.
    - Inicio de la banda de rechazo: primer mínimo (cero) de |H| después de fc.
    - Ancho de transición: diferencia entre ambas frecuencias.
    - Atenuación: el máximo de |H| en dB dentro de la banda de rechazo
      (el lóbulo lateral más alto).
    """
    f_3db = freqs[H_db >= -3][-1]

    # primer mínimo local después de fc
    i = np.argmax(freqs > fc)
    while i < len(H_db) - 1 and H_db[i + 1] < H_db[i]:
        i += 1
    f_rechazo = freqs[i]

    atenuacion = np.max(H_db[i:])
    return f_rechazo - f_3db, atenuacion


# Gráfico: una imagen por ventana, una curva por n_coef
filas = []

for ventana in ventanas:
    plt.figure(figsize=(10, 5))

    for n_coef in lista_n_coef:
        h = fir_pasabajos(fc, fs, n_coef, ventana)
        freqs, H = respuesta_frecuencia(h, fs)

        # magnitud en dB (se suma un número muy pequeño para evitar log(0))
        H_db = 20 * np.log10(np.abs(H) + 1e-12)

        ancho, atenuacion = medir(freqs, H_db, fc)
        filas.append((ventana, n_coef, ancho, atenuacion))

        plt.plot(freqs, H_db, label=f"{n_coef} coef.")

    plt.axvline(fc, color="gray", linestyle="--", linewidth=0.8)
    plt.axhline(-40, color="red", linestyle=":", linewidth=0.8, label="-40 dB")
    plt.title(f"Pasabajos fc = {fc} Hz, fs = {fs} Hz, ventana {nombres[ventana]}")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel(" |H(f)| (dB)")
    plt.ylim(-120, 5)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(carpeta / f"pasabajos_{ventana}.png", dpi=300, bbox_inches="tight")


# Tabla: se imprime y se guarda en CSV
print(f"{'Ventana':<12}{'n_coef':>8}{'Transición (Hz)':>18}{'Atenuación (dB)':>18}")
for ventana, n_coef, ancho, atenuacion in filas:
    print(f"{nombres[ventana]:<12}{n_coef:>8}{ancho:>18.1f}{atenuacion:>18.1f}")

with open(carpeta / "pasabajos_ventanas.csv", "w") as archivo:
    archivo.write("ventana, n_coef, transicion_hz, atenuacion_db\n")
    for ventana, n_coef, ancho, atenuacion in filas:
        archivo.write(f"{ventana}, {n_coef}, {ancho:.1f}, {atenuacion:.1f}\n")


# ----------------------------------------------------------
# 3. Filtren un chirp de 0 a 500 Hz con su pasabajos y muestren en el tiempo dónde actúa el
# filtro.

# Parámetros
fs = 1000          # frecuencia de muestreo (Hz)
duracion = 2       # segundos
f0, f1 = 0, 500    # el chirp va de 0 a 500 Hz
fc = 100           # corte del pasabajos (Hz)
n_coef = 101

# Chirp lineal: la frecuencia instantánea crece de f0 a f1 durante la duración
# f(t) = f0 + (f1 - f0) * t / duracion
t = np.arange(0, duracion, 1 / fs)
k = (f1 - f0) / duracion                       # Hz por segundo
x = np.sin(2 * np.pi * (f0 * t + k * t**2 / 2))

# Filtrar con el pasabajos
h = fir_pasabajos(fc, fs, n_coef, "hamming")
y = filtrar(x, h)

# Instante en que el chirp pasa por la frecuencia de corte
t_corte = (fc - f0) / k
print(f"El chirp pasa por {fc} Hz en t = {t_corte:.2f} s")

# Gráfico
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(t, x, color="lightgray", linewidth=0.6, label="Chirp original")
ax.plot(t, y, color="C0", linewidth=0.8, label=f"Filtrado (pasabajos {fc} Hz)")
ax.axvline(t_corte, color="red", linestyle="--", label=f"f = {fc} Hz (t = {t_corte:.2f} s)")

ax.set_xlabel("Tiempo (s)")
ax.set_ylabel("Amplitud")
ax.set_title(f"Chirp de {f0} a {f1} Hz filtrado con un pasabajos de {fc} Hz ({n_coef} coef., Hamming)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)

# Eje superior: frecuencia instantánea del chirp en cada instante
eje_f = ax.secondary_xaxis("top", functions=(lambda t: f0 + k * t, lambda f: (f - f0) / k))
eje_f.set_xlabel("Frecuencia instantánea del chirp (Hz)")

plt.tight_layout()

plt.savefig(carpeta / "chirp_filtrado.png", dpi=300, bbox_inches="tight")
plt.close()