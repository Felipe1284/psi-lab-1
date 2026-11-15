import numpy as np
import pytest

from psi_lab.lab1 import caso_ecg as ecg


def crear_ecg_sintetico(n_muestras, picos, amplitudes=None):
    muestras = np.arange(n_muestras)
    senal = np.zeros(n_muestras)
    if amplitudes is None:
        amplitudes = np.ones(len(picos))
    for pico, amplitud in zip(picos, amplitudes):
        senal += amplitud * np.exp(-0.5 * ((muestras - pico) / 4.0) ** 2)
    return senal


def test_flujo_ecg_sintetico_compara_metodos_por_snr_y_registro():
    rng = np.random.default_rng(42)
    n_muestras = int(36 * ecg.FS)
    tiempo = np.arange(n_muestras) / ecg.FS

    picos_regulares = np.arange(1, 36) * int(ecg.FS)
    picos_irregulares_18s = np.array(
        [
            360, 720, 936, 1440, 1620, 1980, 2268, 2772, 2952,
            3312, 3672, 3888, 4392, 4572, 4932, 5292, 5508, 6012,
        ]
    )
    picos_irregulares = np.concatenate(
        (picos_irregulares_18s, picos_irregulares_18s + int(18 * ecg.FS))
    )
    amplitudes_irregulares = np.ones(len(picos_irregulares))
    amplitudes_irregulares[8] = -0.8
    amplitudes_irregulares[26] = -0.8
    registros = {
        "100": (
            crear_ecg_sintetico(n_muestras, picos_regulares),
            picos_regulares,
        ),
        "105": (
            crear_ecg_sintetico(
                n_muestras, picos_irregulares, amplitudes_irregulares
            ),
            picos_irregulares,
        ),
    }

    ruido_bw = np.sin(2 * np.pi * 0.25 * tiempo)
    ruido_ma = rng.normal(size=n_muestras)
    resultados = ecg.evaluar_caso_ecg(
        registros=registros,
        niveles_snr_db=ecg.NIVELES_SNR_DB,
        ruido_bw=ruido_bw,
        ruido_ma=ruido_ma,
    )

    assert len(resultados) == len(registros) * len(ecg.NIVELES_SNR_DB) * 2
    assert {resultado["registro"] for resultado in resultados} == {"100", "105"}
    assert {
        resultado["snr_objetivo_db"] for resultado in resultados
    } == set(ecg.NIVELES_SNR_DB)
    assert {resultado["metodo"] for resultado in resultados} == {
        "fir",
        "mediana",
    }
    for resultado in resultados:
        assert resultado["snr_antes_db"] == pytest.approx(
            resultado["snr_objetivo_db"], abs=1e-10
        )
        assert np.isfinite(resultado["snr_despues_db"])
        assert 0.0 <= resultado["sensibilidad"] <= 100.0
        assert 0.0 <= resultado["vpp"] <= 100.0
        assert resultado["tp"] + resultado["fn"] == len(
            registros[resultado["registro"]][1]
        )
        if resultado["registro"] == "100":
            assert resultado["snr_despues_db"] > resultado["snr_antes_db"]
            assert resultado["sensibilidad"] >= 90.0
            assert resultado["vpp"] >= 90.0


def test_flujo_ecg_sintetico_calcula_fc_instantanea():
    rng = np.random.default_rng(42)
    n_muestras = int(18 * ecg.FS)
    muestras = np.arange(n_muestras)
    tiempo = muestras / ecg.FS
    picos_reales = np.arange(1, 17) * int(ecg.FS)
    senal_limpia = crear_ecg_sintetico(n_muestras, picos_reales)
    ruido_bw = np.sin(2 * np.pi * 0.25 * tiempo)
    ruido_ma = rng.normal(size=n_muestras)
    senal_ruidosa = ecg.contaminar_ecg(
        senal_limpia, ruido_bw, ruido_ma, 6.0
    )
    senal_limpiada = ecg.limpiar_ecg_metodo_alternativo(senal_ruidosa)
    picos_detectados = ecg.detectar_picos_r(senal_limpiada)

    tiempos_fc, fc_bpm = ecg.calcular_fc_instantanea(picos_detectados)

    assert len(tiempos_fc) == len(picos_reales) - 1
    np.testing.assert_allclose(fc_bpm, 60.0, atol=0.6)