import csv
from pathlib import Path

import numpy as np
import wfdb
from scipy import ndimage, signal

from psi_lab.lab1.convolucion import convolucion_directa
from psi_lab.lab1.filtros import filtrar, fir_pasaaltos, fir_pasabanda, fir_rechazabanda

## CONSTANTES GLOBALES

# Frecuencia de muestreo
FS = 360.0

# Registros limpios y difíciles
REGISTROS_LIMPIOS = ["100", "101", "103"]
REGISTROS_DIFICILES = ["105", "108", "203"]

# Otras constantes
NIVELES_SNR_DB = [12, 6, 0]
SIMBOLOS_LATIDO = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']


def cargar_datos(id_registro):
    """
    Carga una señal de ECG y sus anotaciones de latidos desde la base de datos PhysioNet.

    :param id_registro: Identificador único del registro en la base de datos.
    :return: Una tupla que contiene la señal de ECG y los índices de los latidos válidos (por ejemplo, id = "100").
    """

    registro = wfdb.rdrecord(id_registro, pn_dir="mitdb")
    anotaciones = wfdb.rdann(id_registro, "atr", pn_dir="mitdb")
    ecg_limpio = registro.p_signal[:, 0]

    picos_reales = []
    for muestra, simbolo in zip(anotaciones.sample, anotaciones.symbol):
        if simbolo in SIMBOLOS_LATIDO:
            picos_reales.append(muestra)

    return ecg_limpio, picos_reales


def calcular_snr(senal_pura, senal_contaminada):
    """
    Hace el cálculo de la relación entre la señal y ruido.

    :param senal_pura: La señal ideal (pura).
    :param senal_contaminada: La señal contaminada (el ruido).
    :return: La relación entre la señal y el ruido como número flotante (float).
    """

    senal_pura = np.asarray(senal_pura, dtype=float)
    ruido_residual = np.asarray(senal_contaminada, dtype=float) - senal_pura
    potencia_senal = np.mean(senal_pura**2)
    potencia_ruido = np.mean(ruido_residual**2)

    return 10.0 * np.log10(potencia_senal / potencia_ruido)


def ajustar_longitud(ruido, longitud):
    """
    Ajusta la longitud de un vector de ruido mediante el truncamiento
    o la repetición para garantizar que coincida con la cantidad de muestras
    de la señal de ECG.

    :param ruido: Vector de ruido de entrada.
    :param longitud: Cantidad de muestras deseadas.
    :return: Vector de ruido con la longitud exacta.
    """

    ruido = np.asarray(ruido, dtype=float)
    if ruido.size == 0:
        raise ValueError("La señal de ruido no puede estar vacía.")
    if ruido.size >= longitud:
        return ruido[:longitud]

    repeticiones = int(np.ceil(longitud / ruido.size))
    return np.tile(ruido, repeticiones)[:longitud]


def generar_interferencia_red(longitud):
    """
    Genera los tonos de 50 Hz y 150 Hz usados en la contaminación controlada.
    :param longitud: La longitud de la onda.
    :return:  Un número flotante (float) que correspondería a la interferencia de red.
    """

    tiempo = np.arange(longitud) / FS
    return 0.5 * np.sin(2.0 * np.pi * 50.0 * tiempo) + 0.25 * np.sin(
        2.0 * np.pi * 150.0 * tiempo
    )


def contaminar_ecg(ecg_limpio, bw, ma, snr_objetivo_db):
    """
    Contamina una señal de ECG limpia añadiendo ruido para alcanzar
    un SNR objetivo en decibelios.

    :param ecg_limpio: Señal de ECG original sin ruido.
    :param bw: Señal de ruido bw.
    :param ma: Señal de ruido ma.
    :param snr_objetivo_db: Relación señal-ruido deseada expresada en decibelios (dB).
    :return: Señal de ECG contaminada con el nivel de ruido controlado.
    """

    ecg_limpio = np.asarray(ecg_limpio, dtype=float)
    n = len(ecg_limpio)
    bw_ajustado = ajustar_longitud(bw, n)
    ma_ajustado = ajustar_longitud(ma, n)

    interferencia_red = generar_interferencia_red(n)
    ruido_total = bw_ajustado + ma_ajustado + interferencia_red

    potencia_ecg = np.mean(ecg_limpio**2)
    potencia_ruido = np.mean(ruido_total**2)
    factor_escala = np.sqrt(potencia_ecg
                            / (potencia_ruido * 10.0 ** (snr_objetivo_db / 10.0)))
    return ecg_limpio + factor_escala * ruido_total


def analizar_espectro(senal):
    """
    Calcula la densidad espectral de potencia mediante el método de Welch.

    :param senal: Un array que corresponde a los datos de la señal.
    :return: Los valores de frecuencia, y la densidad espectral de potencia calculados.
    """

    senal = np.asarray(senal, dtype=float)
    if senal.size == 0:
        raise ValueError("La señal no puede estar vacía.")

    frecuencias, densidad_potencia = signal.welch(
        senal,
        fs=FS,
        window="hann",
        nperseg=min(1024, len(senal)),
    )

    return frecuencias, densidad_potencia


def convolucion_mismo_tamano(senal, kernel):
    """
    Calcula la convolución lineal entre una señal y un kernel, recortando el
    resultado centrado para conservar exactamente la longitud de la señal original.

    :param senal: Señal unidimensional de entrada a la que se aplicará la convolución (array).
    :param kernel: Respuesta al impulso o coeficientes del filtro FIR a aplicar (array).
    :return: Señal resultante de la convolución recortada a la longitud de senal (np.ndarray).
    """

    senal = np.asarray(senal)
    kernel = np.asarray(kernel)
    completa = convolucion_directa(senal, kernel)
    inicio = (len(kernel) - 1) // 2
    return completa[inicio : inicio + len(senal)]


def aplicar_notches(senal):
    """
    Aplica dos filtros FIR rechazabanda (notch) en cascada para eliminar la
    interferencia de la red eléctrica (50 Hz) y su tercer armónico (150 Hz).

    :param senal: Señal de ECG unidimensional de entrada (array).
    :return: Señal filtrada sin las componentes de frecuencia de 50 Hz y 150 Hz (np.ndarray).
    """

    h_notch_50 = fir_rechazabanda(
        48.5, 51.5, fs=FS, n_coef=361
    )
    h_notch_150 = fir_rechazabanda(
        148.5, 151.5, fs=FS, n_coef=361
    )
    senal = filtrar(senal, h_notch_50)
    return filtrar(senal, h_notch_150)


def limpiar_ecg_metodo_fir(ecg_ruidoso):
    """
    Limpia una señal de ECG contaminada eliminando la deriva de línea base mediante un
    filtro FIR pasaaltos (corte en 0.5 Hz) y suprimiendo la interferencia de red eléctrica.

    :param ecg_ruidoso: Señal de ECG contaminada con ruido (array).
    :return: Señal de ECG filtrada y libre de deriva e interferencia de red (np.ndarray).
    """

    h_pasaaltos = fir_pasaaltos(
        fc=0.5, fs=FS, n_coef=2881, ventana="hamming"
    )
    ecg_sin_deriva = filtrar(ecg_ruidoso, h_pasaaltos)
    return aplicar_notches(ecg_sin_deriva)


def limpiar_ecg_metodo_alternativo(ecg_ruidoso):
    """
    Limpia una señal de ECG estimando y sustrayendo la deriva de línea base
    (ventana de 0.6 s), seguido de filtros notch.

    :param ecg_ruidoso: Señal de ECG contaminada con ruido (array).
    :return: Señal de ECG limpia tras sustraer la línea base estimada y filtrar la red (np.ndarray).
    """

    ventana_mediana = round(0.6 * FS)
    if ventana_mediana % 2 == 0:
        ventana_mediana += 1

    linea_base_estimada = ndimage.median_filter(
        ecg_ruidoso, size=ventana_mediana
    )
    ecg_sin_deriva = np.asarray(ecg_ruidoso) - linea_base_estimada
    return aplicar_notches(ecg_sin_deriva)


def detectar_picos_r(ecg_filtrado):
    """
    Detecta los picos R en una señal de ECG mediante una adaptación del algoritmo de
    Pan-Tompkins (pasabanda 5–15 Hz, derivada, cuadrado, promedio móvil de ∼150 ms,
    umbral adaptativo con período refractario de 200 ms).

    :param ecg_filtrado: Señal de ECG previamente limpia y filtrada (array).
    :return: Lista de índices enteros correspondientes a las muestras de los picos R detectados.
    """

    ecg_filtrado = np.asarray(ecg_filtrado, dtype=float)
    h_pasabanda = fir_pasabanda(
        f1=5.0, f2=15.0, fs=FS, n_coef=181
    )
    senal_bp = filtrar(ecg_filtrado, h_pasabanda)

    h_derivada = np.array([-1.0, -2.0, 0.0, 2.0, 1.0]) * (FS / 8.0)
    senal_deriv = convolucion_mismo_tamano(senal_bp, h_derivada)
    senal_cuadrada = senal_deriv**2

    m_ventana = round(0.150 * FS)
    h_promedio = np.ones(m_ventana) / m_ventana
    senal_integrada = convolucion_mismo_tamano(senal_cuadrada, h_promedio)

    refractario_muestras = round(0.200 * FS)
    muestras_iniciales = min(len(senal_integrada), int(2 * FS))
    if muestras_iniciales == 0:
        return []

    segmento_inicial = senal_integrada[:muestras_iniciales]
    spki = np.max(segmento_inicial) * 0.5
    npki = np.mean(segmento_inicial) * 0.5
    umbral_i1 = npki + 0.25 * (spki - npki)
    ultimo_pico = -refractario_muestras
    candidatos, _ = signal.find_peaks(senal_integrada)
    picos_detectados = []

    for idx in candidatos:
        valor = senal_integrada[idx]
        if idx - ultimo_pico < refractario_muestras:
            continue

        if valor > umbral_i1:
            spki = 0.125 * valor + 0.875 * spki
            ultimo_pico = idx

            inicio = max(0, idx - 27)
            fin = min(len(ecg_filtrado), idx + 27)
            pico_r_exacto = inicio + np.argmax(
                np.abs(ecg_filtrado[inicio:fin])
            )
            picos_detectados.append(int(pico_r_exacto))
        else:
            npki = 0.125 * valor + 0.875 * npki

        umbral_i1 = npki + 0.25 * (spki - npki)

    return picos_detectados


def evaluar_deteccion(picos_reales, picos_detectados):
    """
    Evalúa el desempeño del detector comparando los picos R detectados contra las
    anotaciones reales usando una ventana de tolerancia de 150 ms.

    :param picos_reales: Índices de muestra de los latidos reales de referencia (array).
    :param picos_detectados: Índices de muestra de los latidos estimados por el algoritmo (array).
    :return: Tupla (sensibilidad, vpp, tp, fp, fn) con el porcentaje de sensibilidad,
            valor predictivo positivo (%), verdaderos positivos, falsos positivos y falsos negativos.
    """

    tolerancia_muestras = round(0.150 * FS)
    emparejados_reales = np.zeros(len(picos_reales), dtype=bool)
    tp = 0
    fp = 0

    for deteccion in picos_detectados:
        mejor_idx = -1
        mejor_distancia = np.inf
        for idx, real in enumerate(picos_reales):
            if emparejados_reales[idx]:
                continue
            distancia = abs(deteccion - real)
            if distancia <= tolerancia_muestras and distancia < mejor_distancia:
                mejor_distancia = distancia
                mejor_idx = idx

        if mejor_idx >= 0:
            tp += 1
            emparejados_reales[mejor_idx] = True
        else:
            fp += 1

    fn = int(np.count_nonzero(~emparejados_reales))
    sensibilidad = 100.0 * tp / (tp + fn) if tp + fn else 0.0
    vpp = 100.0 * tp / (tp + fp) if tp + fp else 0.0
    return sensibilidad, vpp, tp, fp, fn


def calcular_fc_instantanea(picos_detectados):
    """
    Calcula la frecuencia cardíaca instantánea (en latidos por minuto) a partir de los
    intervalos R-R entre latidos consecutivos detectados.

    :param picos_detectados: Secuencia ordenada de índices de muestra de los picos R (array).
    :return: Tupla (tiempos_fc, fc_bpm) con el vector de tiempos en segundos asociado a cada
            intervalo y el vector de frecuencia cardíaca instantánea.
    """

    picos_detectados = np.asarray(picos_detectados, dtype=float)
    intervalos_rr_seg = np.diff(picos_detectados) / FS
    fc_bpm = 60.0 / intervalos_rr_seg
    tiempos_fc = picos_detectados[1:] / FS
    return tiempos_fc, fc_bpm


def cargar_ruidos():
    """
    Carga los registros de ruido de deriva de línea base (bw) y muscular (ma).

    :return: Tupla con las señales bw y ma, ambas como vectores unidimensionales.
    """

    ruido_bw = wfdb.rdrecord("bw", pn_dir="nstdb").p_signal[:, 0]
    ruido_ma = wfdb.rdrecord("ma", pn_dir="nstdb").p_signal[:, 0]
    return ruido_bw, ruido_ma


def evaluar_caso_ecg(registros=None, niveles_snr_db=None, ruido_bw=None, ruido_ma=None):
    """
    Evalúa ambos métodos de limpieza para cada registro y nivel de SNR.
    Si no se entrega ``registros``, se cargan los registros limpios y difíciles
    definidos en este módulo. Si no se entregan los ruidos, se descargan bw y ma.
    Para pruebas sin red se pueden pasar ``registros`` como un diccionario
    ``id: (senal_limpia, picos_reales)`` y proporcionar señales sintéticas
    ``ruido_bw`` y ``ruido_ma``.

    :param registros: Los registros del SNR.
    :param niveles_snr_db: Los niveles pertenecientes al SNR.
    :param ruido_bw: Parametro de ruido.
    :param ruido_ma: Parametro de ruido.
    :return: Los resultados obtenidos tras aplicar una contaminacion controlada,
    Analisis espectral, limpieza, detección de picos, y evaluación correspondiente.
    """

    if registros is None:
        ids_registros = REGISTROS_LIMPIOS + REGISTROS_DIFICILES
        registros = {
            id_registro: cargar_datos(id_registro)
            for id_registro in ids_registros
        }

    if (ruido_bw is None) != (ruido_ma is None):
        raise ValueError("Se deben proporcionar ambos ruidos, bw y ma.")
    if ruido_bw is None:
        ruido_bw, ruido_ma = cargar_ruidos()

    niveles = tuple(
        NIVELES_SNR_DB if niveles_snr_db is None else niveles_snr_db
    )
    metodos = (
        ("fir", limpiar_ecg_metodo_fir),
        ("mediana", limpiar_ecg_metodo_alternativo),
    )
    resultados = []

    for id_registro, (ecg_limpio, picos_reales) in registros.items():
        ecg_limpio = np.asarray(ecg_limpio, dtype=float)
        for snr_objetivo_db in niveles:
            ecg_ruidoso = contaminar_ecg(
                ecg_limpio, ruido_bw, ruido_ma, snr_objetivo_db
            )
            snr_antes_db = calcular_snr(ecg_limpio, ecg_ruidoso)

            for nombre_metodo, limpiar in metodos:
                ecg_filtrado = limpiar(ecg_ruidoso)
                snr_despues_db = calcular_snr(ecg_limpio, ecg_filtrado)
                picos_detectados = detectar_picos_r(ecg_filtrado)
                sensibilidad, vpp, tp, fp, fn = evaluar_deteccion(
                    picos_reales, picos_detectados
                )
                resultados.append(
                    {
                        "registro": id_registro,
                        "snr_objetivo_db": float(snr_objetivo_db),
                        "metodo": nombre_metodo,
                        "snr_antes_db": snr_antes_db,
                        "snr_despues_db": snr_despues_db,
                        "sensibilidad": sensibilidad,
                        "vpp": vpp,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                    }
                )

    return resultados


def generar_figura_espectros(ruta_salida=None, id_registro="100"):
    """
    Genera una figura comparativa de los espectros del caso ECG.

    :param ruta_salida: La ruta de salida de la figura.
    :param id_registro:  El ID del registro.
    :return: La ruta de salida donde se encuentra la figura comparativa de los espectros.
    """

    import matplotlib.pyplot as plt

    ecg_limpio, _ = cargar_datos(id_registro)
    ruido_bw, ruido_ma = cargar_ruidos()
    ruido_bw = ajustar_longitud(ruido_bw, len(ecg_limpio))
    ruido_ma = ajustar_longitud(ruido_ma, len(ecg_limpio))
    interferencia_red = generar_interferencia_red(len(ecg_limpio))
    mezcla_0_db = contaminar_ecg(ecg_limpio, ruido_bw, ruido_ma, 0.0)

    senales = (
        ("ECG limpio", ecg_limpio),
        ("Ruido bw", ruido_bw),
        ("Ruido ma (banda ancha)", ruido_ma),
        ("Interferencia 50/150 Hz", interferencia_red),
        ("Mezcla, SNR 0 dB", mezcla_0_db),
    )

    figura, eje = plt.subplots(figsize=(12, 6))
    epsilon = np.finfo(float).tiny
    for etiqueta, senal in senales:
        frecuencias, potencia = analizar_espectro(senal)
        potencia_db = 10.0 * np.log10(np.maximum(potencia, epsilon))
        eje.plot(frecuencias, potencia_db, label=etiqueta, linewidth=1)

    eje.axvspan(0, 1, color="tab:blue", alpha=0.1, label="Deriva: <1 Hz")
    eje.axvspan(5, 15, color="tab:green", alpha=0.1, label="Banda QRS: 5-15 Hz")
    eje.axvline(50, color="tab:red", linestyle="--", linewidth=1)
    eje.axvline(150, color="tab:red", linestyle=":", linewidth=1)
    eje.text(51, 0.97, "50 Hz", transform=eje.get_xaxis_transform(), va="top")
    eje.text(151, 0.97, "150 Hz", transform=eje.get_xaxis_transform(), va="top")
    eje.set_xlim(0, FS / 2)
    eje.set_xlabel("Frecuencia (Hz)")
    eje.set_ylabel("Densidad espectral de potencia (dB/Hz)")
    eje.set_title(f"Espectros del ECG y ruidos (registro {id_registro})")
    eje.grid(True, which="both", alpha=0.25)
    eje.legend(loc="best", fontsize="small")
    figura.tight_layout()

    if ruta_salida is None:
        ruta_salida = Path.cwd() / "resultados" / "ecg" / "espectros_ecg.png"
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ruta_salida, dpi=160)
    plt.close(figura)
    return ruta_salida


def guardar_tabla_metricas(resultados, ruta_salida):
    """
    Guarda el resumen por registro, SNR y método como CSV.
    :param resultados: Los resultados obtenidos anteriormente tras aplicar la evauación
    del caso ECG.
    :param ruta_salida: La ruta donde se van a guardar los resultados.
    :return: La tabla ya registrada en la ruta de salida.
    """

    columnas = (
        "registro",
        "snr_objetivo_db",
        "metodo",
        "snr_antes_db",
        "snr_despues_db",
        "sensibilidad",
        "vpp",
        "tp",
        "fp",
        "fn",
    )
    ruta_salida = Path(ruta_salida)
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)
    with ruta_salida.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(resultados)
    return ruta_salida


def main():
    ruta_figura = generar_figura_espectros()
    resultados = evaluar_caso_ecg()
    ruta_tabla = guardar_tabla_metricas(
        resultados, Path.cwd() / "resultados" / "ecg" / "evaluacion_ecg.csv"
    )
    print(f"Figura espectral generada: {ruta_figura}")
    print(f"Tabla de evaluación generada: {ruta_tabla}")


if __name__ == "__main__":
    main()
