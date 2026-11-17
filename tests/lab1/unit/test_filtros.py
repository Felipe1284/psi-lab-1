import numpy as np
import pytest

from psi_lab.lab1.filtros import fir_pasabajos, fir_pasaaltos, fir_pasabanda, fir_rechazabanda, respuesta_frecuencia, filtrar

FS = 1000
N_COEF = 101
VENTANA = ["hamming", "hann", "rect"]

def magnitud_db(h, f_min, f_max):
    """|H(f)| en dB entre f_min y f_max."""
    freqs, H = respuesta_frecuencia(h, FS)
    mask = (freqs >= f_min) & (freqs <= f_max)
    return 20 * np.log10(np.abs(H[mask]) + 1e-12)  # Evitar log(0)


def magnitud(h, f_min, f_max):
    """|H(f)| entre f_min y f_max."""
    freqs, H = respuesta_frecuencia(h, FS)
    mask = (freqs >= f_min) & (freqs <= f_max)
    return np.abs(H[mask])

#----------------------------------------------------------
# fir_pasabajos

def test_pasabajos_largo():
    """El largo del filtro es n_coef."""
    assert len(fir_pasabajos(100, FS, N_COEF)) == N_COEF

@pytest.mark.parametrize("ventana", VENTANA)
def test_pasabajos_ganancia_dc(ventana):
    """La ganancia en DC es la suma de los coeficientes."""
    h = fir_pasabajos(100, FS, N_COEF, ventana)
    assert np.isclose(np.sum(h), 1)


@pytest.mark.parametrize("ventana", VENTANA)
def test_pasabajos_simetria(ventana):
    """El filtro es simétrico."""
    h = fir_pasabajos(100, FS, N_COEF, ventana)
    assert np.allclose(h, h[::-1])


def test_pasabajos_ncoef_par():
    """Si n_coef es par, se lanza ValueError."""
    with pytest.raises(ValueError):
        fir_pasabajos(100, FS, 100)


def test_pasabajos_ventana_desconocida():
    """Si la ventana es desconocida, se lanza ValueError."""
    with pytest.raises(ValueError):
        fir_pasabajos(100, FS, N_COEF, "desconocida")


def test_pasabajos_ganancia_1_en_banda():
    """La ganancia en la banda de paso es aproximadamente 1."""
    h = fir_pasabajos(100, FS, N_COEF, "hamming")
    H = magnitud(h, 0, 80)
    assert np.allclose(H, 1, atol=0.01)  # Tolerancia de 0.01

def test_pasabajos_menos_de_40db_en_banda():
    """La ganancia en la banda de rechazo es menor a -40 dB."""
    h = fir_pasabajos(100, FS, N_COEF, "hamming")
    H_db = magnitud_db(h, 130, 500)
    assert np.all(H_db < -40)


#----------------------------------------------------------
# fir_pasaaltos

def test_pasaaltos_simetria():
    """El filtro pasaaltos es simétrico."""
    h = fir_pasaaltos(100, FS, N_COEF)
    assert np.allclose(h, h[::-1])

def test_pasaaltos_atenua_frecuencia_baja():
    """La ganancia en la banda de rechazo es menor a -40 dB."""
    h = fir_pasaaltos(100, FS, N_COEF, "hamming")
    H_db = magnitud_db(h, 0, 70)
    assert np.all(H_db < -40)

def test_pasaaltos_ganancia_1_en_altas_frecuencias():
    """La ganancia en la banda de paso es aproximadamente 1."""
    h = fir_pasaaltos(100, FS, N_COEF, "hamming")
    H = magnitud(h, 120, 500)
    assert np.allclose(H, 1, atol=0.01)  # Tolerancia de 0.01

def test_pasaaltos_ncoef_par():
    """Si n_coef es par, se lanza ValueError."""
    with pytest.raises(ValueError):
        fir_pasaaltos(100, FS, 100)

def test_pasaaltos_inversion_espectral():
    """El filtro pasaaltos es la inversión espectral del pasabajos."""
    h_pb = fir_pasabajos(100, FS, N_COEF)
    h_pa = fir_pasaaltos(100, FS, N_COEF)
    h_delta = np.zeros(N_COEF)
    h_delta[(N_COEF - 1) // 2] = 1
    assert np.allclose(h_pa, h_delta - h_pb)


#----------------------------------------------------------
# fir_pasabanda

def test_pasabanda_simetria():
    """El filtro pasabanda es simétrico."""
    h = fir_pasabanda(100, 200, FS, N_COEF)
    assert np.allclose(h, h[::-1])


def test_pasabanda_ganancia_1_en_banda():
    """La ganancia en la banda de paso es aproximadamente 1."""
    h = fir_pasabanda(100, 200, FS, N_COEF, "hamming")
    H = magnitud(h, 120, 180)
    assert np.allclose(H, 1, atol=0.01)  # Tolerancia de 0.01


def test_pasabanda_atenua_fuera_de_banda():
    """La ganancia fuera de la banda de paso es menor a -40 dB."""
    h = fir_pasabanda(100, 200, FS, N_COEF, "hamming")
    H_db = magnitud_db(h, 0, 70)
    assert np.all(H_db < -40)
    H_db = magnitud_db(h, 230, 500)
    assert np.all(H_db < -40)


def test_pasabanda_es_resta_de_pasabajos():
    """El filtro pasabanda es la resta de dos pasabajos."""
    h_pb1 = fir_pasabajos(100, FS, N_COEF)
    h_pb2 = fir_pasabajos(200, FS, N_COEF)
    h_pbanda = fir_pasabanda(100, 200, FS, N_COEF)
    assert np.allclose(h_pbanda, h_pb2 - h_pb1)


#----------------------------------------------------------
# fir_rechazabanda

def test_rechazabanda_simetria():
    """El filtro rechazabanda es simétrico."""
    h = fir_rechazabanda(100, 200, FS, N_COEF)
    assert np.allclose(h, h[::-1])

def test_rechazabanda_atenua_dentro_banda():
    """La ganancia dentro de la banda de rechazo es menor a -40 dB."""
    h = fir_rechazabanda(100, 200, FS, N_COEF, "hamming")
    H_db = magnitud_db(h, 120, 170)
    assert np.all(H_db < -40)

def test_rechazabanda_ganancia_1_fuera_banda():
    """La ganancia fuera de la banda de rechazo es aproximadamente 1."""
    h = fir_rechazabanda(100, 200, FS, N_COEF, "hamming")
    H = magnitud(h, 0, 80)
    assert np.allclose(H, 1, atol=0.01)  # Tolerancia de 0.01
    H = magnitud(h, 220, 500)
    assert np.allclose(H, 1, atol=0.01)  # Tolerancia de 0.01


#----------------------------------------------------------
# respuesta_frecuencia

def test_rf_forma_y_rango():
    """La respuesta en frecuencia tiene la forma correcta y está en el rango [0, fs/2]."""
    h = np.ones(5)
    freqs, H = respuesta_frecuencia(h, FS, n=256)
    assert len(freqs) == 256
    assert len(H) == 256
    assert np.isclose(freqs[0], 0)
    assert np.isclose(freqs[-1], FS / 2)
    assert np.iscomplexobj(H)  # H debe ser complejo


def test_rf_coincide_con_np_fft():
    """La respuesta en frecuencia coincide con np.fft.fft."""
    rng = np.random.default_rng(5)
    h = rng.standard_normal(31)
    _, H = respuesta_frecuencia(h, FS, n=513)
    H_np = np.fft.rfft(h, 2 * (n - 1))
    assert np.allclose(H, H_np)

def test_rf_de_un_impulso_es_1():
    _, H = respuesta_frecuencia([1], FS)
    assert np.allclose(H, 1)


#----------------------------------------------------------
# filtrar

def test_filtrar_mantiene_largo():
    """La salida de filtrar tiene el mismo largo que la señal de entrada."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(1000)
    h = fir_pasabajos(100, FS, N_COEF)
    y = filtrar(x, h)
    assert len(y) == len(x)

@pytest.mark.parametrize("n_coef", [21, 1025])   # 1025 >= 500
def test_filtrar_no_desplaza_un_pulso(n_coef):
    x = np.zeros(3000)
    x[1500] = 1
    h = fir_pasabajos(100, FS, n_coef)
    y = filtrar(x, h)
    assert np.argmax(y) == 1500

def test_filtrar_con_impulso_unitario():
    """Filtrar con un impulso unitario devuelve la misma señal."""
    x =np.random.default_rng(7).standard_normal(50)
    y = filtrar(x, [1])  # Filtrar con un impulso unitario
    assert np.allclose(x, y)  # La salida debe ser igual a la entrada

def test_filtrar_senoidal():
    """Una senoidal de 20Hz pasa por un pasabajos de 100Hz sin atenuación."""
    t = np.arange(2000) / FS
    x = np.sin(2 * np.pi * 20 * t)
    h = fir_pasabajos(100, FS, N_COEF)
    y = filtrar(x, h)
    assert np.allclose(y[N_COEF:-N_COEF], x[N_COEF:-N_COEF], atol=0.01)  # Tolerancia de 0.01

def test_filtrar_ncoef_par():
    """Si n_coef es par, se lanza ValueError."""
    with pytest.raises(ValueError):
        filtrar([1, 2, 3], [1, 2])  # n_coef = 2 (par)