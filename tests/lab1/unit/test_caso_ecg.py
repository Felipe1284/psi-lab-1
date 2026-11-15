from types import SimpleNamespace

import numpy as np
import pytest
from scipy.ndimage import median_filter
from wfdb import rdann, rdrecord

from psi_lab.lab1 import caso_ecg as ecg
from psi_lab.lab1.filtros import filtrar, fir_pasaaltos, fir_rechazabanda


def test_cargar_datos_filtra_anotaciones_que_no_son_latidos(monkeypatch):
    registro = SimpleNamespace(p_signal=np.array([[0.1], [0.2], [0.3]]))
    anotaciones = SimpleNamespace(
        sample=np.array([10, 20, 30, 40]),
        symbol=["N", "+", "V", "x"],
    )
    monkeypatch.setattr(ecg.wfdb, rdrecord.__name__, lambda *args, **kwargs: registro)
    monkeypatch.setattr(ecg.wfdb, rdann.__name__, lambda *args, **kwargs: anotaciones)

    senal, picos = ecg.cargar_datos("100")

    np.testing.assert_array_equal(senal, [0.1, 0.2, 0.3])
    assert picos == [10, 30]


def test_cargar_ruidos_lee_bw_y_ma_sin_red(monkeypatch):
    llamadas = []
    registros = {
        "bw": SimpleNamespace(p_signal=np.array([[1.0, 10.0], [2.0, 20.0]])),
        "ma": SimpleNamespace(p_signal=np.array([[3.0, 30.0], [4.0, 40.0]])),
    }

    def rdrecord(id_registro, pn_dir):
        llamadas.append((id_registro, pn_dir))
        return registros[id_registro]

    monkeypatch.setattr(ecg.wfdb, rdrecord.__name__, rdrecord)

    ruido_bw, ruido_ma = ecg.cargar_ruidos()

    np.testing.assert_array_equal(ruido_bw, [1.0, 2.0])
    np.testing.assert_array_equal(ruido_ma, [3.0, 4.0])
    assert llamadas == [("bw", "nstdb"), ("ma", "nstdb")]


def test_calcular_snr_usa_potencia_de_senal_y_ruido_residual():
    senal = np.array([1.0, -1.0, 1.0, -1.0])
    contaminada = senal + np.array([0.1, -0.1, 0.1, -0.1])

    assert ecg.calcular_snr(senal, contaminada) == pytest.approx(20.0)


@pytest.mark.parametrize(
    ("ruido", "longitud", "esperado"),
    [
        ([1, 2, 3], 2, [1, 2]),
        ([1, 2], 5, [1, 2, 1, 2, 1]),
        ([1, 2, 3], 3, [1, 2, 3]),
    ],
)


def test_ajustar_longitud_trunca_o_repite(ruido, longitud, esperado):
    np.testing.assert_array_equal(ecg.ajustar_longitud(ruido, longitud), esperado)


def test_ajustar_longitud_rechaza_ruido_vacio():
    with pytest.raises(ValueError, match="no puede estar vacía"):
        ecg.ajustar_longitud([], 4)


def test_generar_interferencia_red_contiene_50_y_150_hz():
    interferencia = ecg.generar_interferencia_red(int(10 * ecg.FS))
    amplitudes = np.abs(np.fft.rfft(interferencia)) / interferencia.size

    assert amplitudes[int(50 * interferencia.size / ecg.FS)] == pytest.approx(
        0.25
    )
    assert amplitudes[int(150 * interferencia.size / ecg.FS)] == pytest.approx(
        0.125
    )


def test_contaminar_ecg_alcanza_snr_objetivo():
    n = 720
    muestras = np.arange(n)
    senal = np.sin(2 * np.pi * muestras / 90)
    bw = np.sin(2 * np.pi * muestras / 180)
    ma = np.cos(2 * np.pi * muestras / 13)

    contaminada = ecg.contaminar_ecg(senal, bw, ma, 6.0)

    assert contaminada.shape == senal.shape
    assert ecg.calcular_snr(senal, contaminada) == pytest.approx(6.0)


def test_analizar_espectro_devuelve_pico_de_frecuencia_sintetico():
    t = np.arange(720) / ecg.FS
    senal = np.sin(2 * np.pi * 10 * t)

    frecuencias, potencia = ecg.analizar_espectro(senal)

    assert frecuencias.shape == potencia.shape
    assert frecuencias[np.argmax(potencia)] == pytest.approx(10, abs=0.5)


def test_convolucion_mismo_tamano_recorta_centrada():
    resultado = ecg.convolucion_mismo_tamano([1, 2, 3], [1, 1, 1])

    np.testing.assert_array_equal(resultado, [3, 6, 5])
    assert resultado.size == 3


def test_aplicar_notches_filtra_50_y_150_hz(monkeypatch):
    filtros = []

    def crear_notch(f1, f2, fs, n_coef):
        filtros.append((f1, f2, fs, n_coef))
        return np.array([f1])

    monkeypatch.setattr(ecg, fir_rechazabanda.__name__, crear_notch)
    monkeypatch.setattr(ecg, filtrar.__name__, lambda x, h: x + h[0])

    resultado = ecg.aplicar_notches(np.array([1.0, 2.0]))

    assert filtros == [
        (48.5, 51.5, ecg.FS, 361),
        (148.5, 151.5, ecg.FS, 361),
    ]
    np.testing.assert_array_equal(resultado, [198.0, 199.0])


def test_limpiar_ecg_metodo_fir_mejora_snr_en_senal_sintetica():
    n_muestras = int(36 * ecg.FS)
    muestras = np.arange(n_muestras)
    tiempo = muestras / ecg.FS
    senal_limpia = np.zeros(n_muestras)
    for pico in np.arange(1, 36) * int(ecg.FS):
        senal_limpia += np.exp(-0.5 * ((muestras - pico) / 4.0) ** 2)

    senal_ruidosa = (
        senal_limpia
        + 0.3 * np.sin(2 * np.pi * 0.25 * tiempo)
        + 0.1 * np.sin(2 * np.pi * 50 * tiempo)
        + 0.05 * np.sin(2 * np.pi * 150 * tiempo)
    )
    snr_antes = ecg.calcular_snr(senal_limpia, senal_ruidosa)

    resultado = ecg.limpiar_ecg_metodo_fir(senal_ruidosa)

    assert resultado.shape == senal_limpia.shape
    assert ecg.calcular_snr(senal_limpia, resultado) > snr_antes


def test_limpiar_ecg_metodo_alternativo_sustrae_mediana_y_aplica_notches(
    monkeypatch,
):
    entrada = np.arange(8, dtype=float)
    linea_base = np.full(8, 2.0)
    filtrada = entrada - linea_base
    sizes = []
    notches = []
    monkeypatch.setattr(
        ecg.ndimage,
        median_filter.__name__,
        lambda x, size: sizes.append(size) or linea_base,
    )
    monkeypatch.setattr(
        ecg, "aplicar_notches", lambda x: notches.append(x) or x
    )

    resultado = ecg.limpiar_ecg_metodo_alternativo(entrada)

    assert sizes == [217]
    np.testing.assert_array_equal(notches[0], filtrada)
    np.testing.assert_array_equal(resultado, filtrada)


def test_detectar_picos_r_detecta_latidos_sinteticos():
    muestras = np.arange(int(8 * ecg.FS))
    picos_reales = np.arange(1, 8) * int(ecg.FS)
    senal = np.zeros_like(muestras, dtype=float)
    for pico in picos_reales:
        senal += np.exp(-0.5 * ((muestras - pico) / 4.0) ** 2)

    detectados = ecg.detectar_picos_r(senal)

    assert len(detectados) == len(picos_reales)
    assert max(abs(d - r) for d, r in zip(detectados, picos_reales)) <= 1


def test_evaluar_deteccion_cuenta_tp_fp_y_fn_con_tolerancia():
    sensibilidad, vpp, tp, fp, fn = ecg.evaluar_deteccion(
        [100, 200, 300], [110, 110, 500]
    )

    assert (tp, fp, fn) == (1, 2, 2)
    assert sensibilidad == pytest.approx(100 / 3)
    assert vpp == pytest.approx(100 / 3)


def test_calcular_fc_instantanea_devuelve_tiempos_y_bpm():
    tiempos, frecuencia = ecg.calcular_fc_instantanea([0, 360, 1080])

    np.testing.assert_allclose(tiempos, [1.0, 3.0])
    np.testing.assert_allclose(frecuencia, [60.0, 30.0])


def test_evaluar_caso_ecg_reporta_snr_y_metricas_por_metodo(monkeypatch):
    senal = np.array([1.0, -1.0, 0.5, -0.5])
    ruido_bw = np.array([0.2, -0.2, 0.1, -0.1])
    ruido_ma = np.array([0.1, 0.1, -0.1, -0.1])
    monkeypatch.setattr(
        ecg,
        ecg.limpiar_ecg_metodo_fir.__name__,
        lambda x: 0.9 * x,
    )
    monkeypatch.setattr(
        ecg,
        ecg.limpiar_ecg_metodo_alternativo.__name__,
        lambda x: 0.8 * x,
    )
    monkeypatch.setattr(
        ecg, ecg.detectar_picos_r.__name__, lambda x: []
    )

    resultados = ecg.evaluar_caso_ecg(
        registros={"105": (senal, [])},
        niveles_snr_db=[6],
        ruido_bw=ruido_bw,
        ruido_ma=ruido_ma,
    )

    assert [resultado["metodo"] for resultado in resultados] == [
        "fir",
        "mediana",
    ]
    for resultado in resultados:
        assert resultado["registro"] == "105"
        assert resultado["snr_objetivo_db"] == 6.0
        assert resultado["snr_antes_db"] == pytest.approx(6.0)
        assert np.isfinite(resultado["snr_despues_db"])
        assert (resultado["tp"], resultado["fp"], resultado["fn"]) == (0, 0, 0)