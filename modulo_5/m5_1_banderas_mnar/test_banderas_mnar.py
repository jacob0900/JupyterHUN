"""Pruebas de la subfase 5.1 con datos sintéticos (no usa datos reales).

    python -m pytest modulo_5/m5_1_banderas_mnar -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modulo_5.m5_1_banderas_mnar import (  # noqa: E402
    ConfigMNAR,
    banderas_estructurales,
    banderas_toma_muestra,
    clasificar_columnas,
    construir_subfase_5_1,
    labs_mas_frecuentes,
    preparar_measurement,
)

T = pd.Timestamp


def _master() -> pd.DataFrame:
    """4 snapshots con las columnas del contrato M1; índice no estándar y orden desordenado."""
    return pd.DataFrame(
        {
            "snapshot_id": [4, 2, 1, 3],
            "episodio_id": [200, 100, 100, 300],
            "person_id": [2, 1, 1, 3],
            "servicio": ["urgencias_observacion", "hospitalizacion", "hospitalizacion", "hospitalizacion"],
            "horas_desde_ingreso": np.array([0, 2, 0, 0], dtype=np.int32),
            "snapshot_time": [T("2024-01-05 08:00"), T("2024-01-02 12:00"), T("2024-01-02 10:00"), T("2024-06-01 09:00")],
            "inicio_episodio": [T("2024-01-05 08:00"), T("2024-01-02 10:00"), T("2024-01-02 10:00"), T("2024-06-01 09:00")],
            "es_entrenamiento": [True, True, True, False],
            "en_riesgo_truncamiento": [False] * 4,
            "en_periodo_estable": [True] * 4,
            "umbral_los_h": [8.9, 141.6, 141.6, 141.6],
            "edad": [40, 70, 70, 25],
            "sexo": ["F", "M", "M", "F"],
            "origen_urgencias": [True, False, False, True],
            "mes": [1, 1, 1, 6],
            "fin_de_semana": [False] * 4,
            "n_mediciones_24h": [0, 3, np.nan, 1],
            "n_dx_historia": [5, 0, 0, 2],
            "horas_desde_ultima_alta_hosp": [np.nan, 100.0, 100.0, np.nan],
            "y_target_los_largo": [0, 1, 1, 0],
            "y_los_h": [1.0, 200.0, 200.0, 50.0],
        },
        index=[103, 7, 55, 9],
    )


def _measurement() -> pd.DataFrame:
    filas = [
        # persona 1, examen 100 a las 10:00 = τ del snapshot de las 10 -> NO cuenta (regla estricta t < τ)
        (1, 10, 100, T("2024-01-02 10:00"), None),
        # persona 1, examen 200 a las 11:00 -> futuro para τ = 10:00, pasado para τ = 12:00
        (1, 10, 200, T("2024-01-02 11:00"), None),
        # persona 1, examen 300: 30 h antes del snapshot de las 12:00 -> fuera de la ventana de 24 h
        (1, 9, 300, T("2024-01-01 06:00"), None),
        # persona 2, examen 100: solo fecha, mismo día del snapshot -> disponible al final del día -> NO cuenta
        (2, 20, 100, None, T("2024-01-05")),
        # persona 2, examen 200: hora 00:00:00 del día anterior -> fin de ese día -> cuenta (8 h antes)
        (2, 20, 200, T("2024-01-04 00:00"), None),
        # persona 3, examen 100 con visit_occurrence_id de otra visita (vínculo roto): cuenta igual por persona
        (3, 99, 100, T("2024-06-01 07:00"), None),
        # examen 900 muy frecuente pero SOLO después del corte de entrenamiento
        (3, 30, 900, T("2024-06-01 08:00"), None),
        (4, 40, 900, T("2024-06-01 08:40"), None),
        (5, 50, 900, T("2024-06-01 08:50"), None),
        (6, 60, 900, T("2024-06-01 08:55"), None),
        # concepto 0 = sin mapeo OMOP -> se descarta
        (1, 10, 0, T("2024-01-02 09:00"), None),
    ]
    return pd.DataFrame(filas, columns=["person_id", "visit_occurrence_id", "measurement_concept_id",
                                        "measurement_datetime", "measurement_date"])


def _flags(cfg: ConfigMNAR) -> pd.DataFrame:
    m = _master()
    med = preparar_measurement(_measurement())
    top = labs_mas_frecuentes(m, med, cfg)
    return banderas_toma_muestra(m, med, top, cfg)


def test_fecha_efectiva_conservadora():
    med = preparar_measurement(_measurement())
    assert 0 not in set(med.concepto)
    assert "visit_occurrence_id" not in med.columns
    solo_fecha = med[(med.person_id == 2) & (med.concepto == 100)].t_efectiva.iloc[0]
    medianoche = med[(med.person_id == 2) & (med.concepto == 200)].t_efectiva.iloc[0]
    assert solo_fecha > T("2024-01-05 23:59:59")
    assert T("2024-01-04 23:59:59") < medianoche < T("2024-01-05")


def test_top_k_no_mira_el_periodo_de_prueba():
    top = labs_mas_frecuentes(_master(), preparar_measurement(_measurement()), ConfigMNAR(top_k=10))
    assert 900 not in set(top.concepto)  # solo aparece después del último τ de entrenamiento
    assert top.concepto.tolist() == [200, 100, 300]  # 200 lo tienen 2 personas; empate 100/300 por código
    assert top.bandera.is_unique and top.bandera.str.startswith("has_measurement_").all()


def test_regla_t_estrictamente_menor_que_tau_y_ventana():
    out = _flags(ConfigMNAR(ventana_horas=24))
    assert out.loc[55, "has_measurement_100"] == 0  # registro exactamente en τ -> no cuenta
    assert out.loc[55, "has_measurement_200"] == 0  # registro a las 11:00 -> futuro
    assert out.loc[7, "has_measurement_100"] == 1   # para τ = 12:00 el de las 10:00 ya es pasado
    assert out.loc[7, "has_measurement_200"] == 1
    assert out.loc[7, "has_measurement_300"] == 0   # 30 h antes, fuera de la ventana
    assert out.loc[103, "has_measurement_100"] == 0  # solo fecha del mismo día
    assert out.loc[103, "has_measurement_200"] == 1  # medianoche del día anterior
    assert out.loc[9, "has_measurement_100"] == 1   # cruce por persona, no por visita


def test_ventana_none_usa_toda_la_historia():
    out = _flags(ConfigMNAR(ventana_horas=None))
    assert out.loc[7, "has_measurement_300"] == 1
    assert out.loc[55, "has_measurement_200"] == 0  # el futuro sigue sin contar


def test_ninguna_bandera_usa_registros_de_tau_en_adelante():
    """Comprobación por fuerza bruta contra la definición."""
    cfg = ConfigMNAR(ventana_horas=None)
    m = _master()
    med = preparar_measurement(_measurement())
    top = labs_mas_frecuentes(m, med, cfg)
    out = banderas_toma_muestra(m, med, top, cfg)
    for i, fila in m.iterrows():
        for c, b in zip(top.concepto, top.bandera):
            esperado = ((med.person_id == fila.person_id) & (med.concepto == c) & (med.t_efectiva < fila.snapshot_time)).any()
            assert out.loc[i, b] == int(esperado)


def test_banderas_estructurales():
    out = banderas_estructurales(_master())
    assert out.loc[[103, 7, 55, 9], "sin_mediciones_24h"].tolist() == [1, 0, 1, 0]  # NaN en conteo = sin registro
    assert out.loc[[103, 7, 55, 9], "sin_dx_historia"].tolist() == [0, 1, 1, 0]
    assert out.loc[[103, 7, 55, 9], "nunca_hospitalizado"].tolist() == [1, 0, 0, 1]
    assert "sin_mediciones_historia" not in out  # su columna de origen no existe


def test_orquestador_no_toca_llaves_y_clasifica():
    m = _master()
    out, esq, top = construir_subfase_5_1(m, _measurement(), ConfigMNAR(top_k=2))
    assert out.index.equals(m.index)
    pd.testing.assert_frame_equal(out[m.columns], m)  # columnas originales intactas
    assert len(top) == 2

    todas = esq.num_cols + esq.cat_cols + esq.lab_cols
    assert len(todas) == len(set(todas))  # listas disjuntas
    assert set(esq.lab_cols) == {"n_mediciones_24h", "sin_mediciones_24h", *top.bandera}
    assert set(esq.cat_cols) == {"servicio", "sexo", "mes", "origen_urgencias", "fin_de_semana",
                                 "sin_dx_historia", "nunca_hospitalizado"}
    assert set(esq.num_cols) == {"horas_desde_ingreso", "edad", "n_dx_historia", "horas_desde_ultima_alta_hosp"}
    assert set(esq.excluidas["id"]) == {"snapshot_id", "episodio_id", "person_id"}
    assert set(esq.excluidas["target"]) == {"y_target_los_largo", "y_los_h"}
    assert set(esq.excluidas["tiempo"]) == {"snapshot_time", "inicio_episodio"}
    assert set(esq.excluidas["control"]) == {"es_entrenamiento", "en_periodo_estable", "en_riesgo_truncamiento",
                                             "umbral_los_h"}


def test_sin_measurement_solo_banderas_estructurales():
    m = _master()
    out, esq, top = construir_subfase_5_1(m, None)
    assert top.empty
    assert not any(c.startswith("has_measurement_") for c in out.columns)
    assert {"sin_mediciones_24h", "sin_dx_historia", "nunca_hospitalizado"} <= set(out.columns)


def test_sinonimos_se_fusionan_y_no_gastan_cupo():
    """El mismo examen con dos códigos (LOINC + SNOMED) da una sola bandera."""
    med_raw = _measurement()
    copia = med_raw[med_raw.measurement_concept_id == 200].assign(measurement_concept_id=777)
    med = preparar_measurement(pd.concat([med_raw, copia], ignore_index=True))
    m = _master()
    top = labs_mas_frecuentes(m, med, ConfigMNAR(top_k=3))
    assert 777 not in set(top.concepto) and top.concepto.tolist() == [200, 100, 300]
    assert top.set_index("concepto").loc[200, "sinonimos"] == "777"
    # sin fusión, el sinónimo ocuparía un cupo
    sin = labs_mas_frecuentes(m, med, ConfigMNAR(top_k=3, umbral_sinonimo=None))
    assert set(sin.concepto) == {200, 777, 100}
    # la bandera usa los registros de ambos códigos
    solo_777 = preparar_measurement(pd.concat([med_raw[med_raw.measurement_concept_id != 200], copia], ignore_index=True))
    out = banderas_toma_muestra(m, solo_777, top, ConfigMNAR())
    assert out.loc[7, "has_measurement_200"] == 1


def test_no_sobrescribe_columnas_existentes():
    m = _master().assign(has_measurement_100=1)
    med = preparar_measurement(_measurement())
    top = labs_mas_frecuentes(m, med, ConfigMNAR())
    with pytest.raises(ValueError):
        banderas_toma_muestra(m, med, top, ConfigMNAR())


def test_tau_nulo_falla():
    m = _master()
    m.loc[7, "snapshot_time"] = pd.NaT
    med = preparar_measurement(_measurement())
    top = labs_mas_frecuentes(m.drop(index=7), med, ConfigMNAR())
    with pytest.raises(ValueError):
        banderas_toma_muestra(m, med, top, ConfigMNAR())
