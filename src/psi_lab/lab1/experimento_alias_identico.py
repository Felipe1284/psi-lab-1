
import numpy as np
import matplotlib.pyplot as plt

from psi_lab.lab1.muestreo import muestrear

fs = 10
duracion = 1

f1 = 3
f2 = 7

fun1 = lambda t: np.cos(2 * np.pi * f1 * t)
fun2 = lambda t: np.cos(2 * np.pi * f2 * t)

t1, x1 = muestrear(fun1, fs, duracion)
t2, x2 = muestrear(fun2, fs, duracion)

# Comprobar si las muestras son iguales
print("¿Muestras iguales?", np.allclose(x1, x2))
print("Diferencia máxima:", np.max(np.abs(x1 - x2)))

# Señales continuas
t_cont = np.linspace(0, duracion, 1000)

plt.figure(figsize=(10, 5))

plt.plot(t_cont, fun1(t_cont), label="Coseno 3 Hz")
plt.plot(t_cont, fun2(t_cont), label="Coseno 7 Hz")

plt.scatter(t1, x1, color="red", s=60,
            label="Muestras de ambas señales", zorder=5)

plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.title("Aliasing: dos señales diferentes, mismas muestras")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
