"""Pruebas de la subfase 7.2 con datos sintéticos (no usa datos reales).

    python -m pytest modulo_7/m7_2_calibracion_costo -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modulo_7.m7_1_multimetrica import ece_igual_frecuencia  # noqa: E402
from modulo_7.m7_2_calibracion_costo import (  # noqa: E402
    ajustar_calibrado_final,
    barrido_umbrales,
    calibrar_cruzado,
    comparar_umbrales,
    curva_fiabilidad,
    desempeno_umbral,
    graficar_costo,
    graficar_fiabilidad,
    metricas_calibracion,
    politicas_referencia,
    umbral_optimo,
)


def _calibrado(n=40000, semilla=0):
    rng = np.random.default_rng(semilla)
    p = rng.beta(2, 2, n)
    y = (rng.random(n) < p).astype(int)
    return y, p, rng.integers(0, 5, n)


def test_calibracion_cruzada_corrige_y_no_usa_el_propio_pliegue():
    y, p, pl = _calibrado()
    desc = p ** 3  # descalibrado (sobrestima el riesgo bajo), mismo orden
    assert ece_igual_frecuencia(y, desc) > 0.15
    for m in ("isotonic", "sigmoid"):
        c = calibrar_cruzado(desc, y, pl, m)
        assert ((c >= 0) & (c <= 1)).all()
        # Isotónica corrige cualquier distorsión monótona; Platt (logística lineal sobre el puntaje) solo una parte
        assert ece_igual_frecuencia(y, c) < (0.03 if m == "isotonic" else ece_igual_frecuencia(y, desc) / 2)
        y2 = y.copy()
        y2[pl == 0] = 1 - y2[pl == 0]
        assert np.allclose(calibrar_cruzado(desc, y2, pl, m)[pl == 0], c[pl == 0])
    with pytest.raises(ValueError):
        calibrar_cruzado(desc, y, pl, "beta")


def test_curva_fiabilidad_igual_frecuencia():
    y, p, _ = _calibrado()
    c = curva_fiabilidad(y, p, 15)
    assert len(c) == 15 and c.n.max() - c.n.min() <= 1
    assert np.allclose(c.p_media, c.fraccion_positivos, atol=0.03)


def test_barrido_igual_a_conteo_directo():
    y, p, _ = _calibrado(n=3000)
    b = barrido_umbrales(y, p)
    assert len(b) == 99 and b.umbral.iloc[0] == 0.01 and b.umbral.iloc[-1] == 0.99
    for t in (0.01, 0.17, 0.5, 0.83):
        d = desempeno_umbral(y, p, t)
        fila = b[np.isclose(b.umbral, t)].iloc[0]
        assert fila.costo_por_100 == pytest.approx(d["costo_por_100"])
        assert fila.fn == pytest.approx(d["fn_por_100"] * len(y) / 100)


def test_umbral_optimo_cerca_del_de_bayes_con_probabilidades_calibradas():
    y, p, _ = _calibrado(n=200000)
    t = umbral_optimo(barrido_umbrales(y, p, c_fn=5, c_fp=1))
    assert abs(t - 1 / 6) <= 0.03
    t10 = umbral_optimo(barrido_umbrales(y, p, c_fn=10, c_fp=1))
    assert abs(t10 - 1 / 11) <= 0.03 and t10 < t


def test_politicas_y_comparacion():
    y = np.array([1] * 30 + [0] * 70)
    ref = politicas_referencia(y)
    assert ref["alertar_a_todos"] == pytest.approx(70) and ref["no_alertar_nunca"] == pytest.approx(150)
    p = np.r_[np.full(30, 0.4), np.full(70, 0.1)]
    t = comparar_umbrales(y, p, {"θ = 0,5": 0.5, "θ*": 0.3})
    assert t.costo_por_100.tolist() == pytest.approx([150, 0])
    assert t.ahorro_vs_0_5.tolist() == pytest.approx([0, 150])
    assert t.ahorro_vs_mejor_sin_modelo.tolist() == pytest.approx([-80, 70])


def test_metricas_calibracion():
    y, p, _ = _calibrado()
    m = metricas_calibracion(y, p)
    assert m["ece"] < 0.02 and m["brier_skill"] > 0 and m["p_media"] == pytest.approx(m["prevalencia"], abs=0.01)


def test_calibrado_final_usa_pliegues_agrupados():
    rng = np.random.default_rng(1)
    n = 1500
    X = rng.normal(size=(n, 4))
    y = (X[:, 0] + rng.normal(0, 1, n) > 0).astype(int)
    pliegues = rng.integers(0, 5, n)
    for est, metodo in ((make_pipeline(StandardScaler(), LogisticRegression()), "isotonic"),
                        (make_pipeline(StandardScaler(), LinearSVC()), "sigmoid")):  # SVM: sin predict_proba
        cal = ajustar_calibrado_final(est, X, y, pliegues, metodo)
        assert len(cal.calibrated_classifiers_) == 5
        p = cal.predict_proba(X)[:, 1]
        assert ((p >= 0) & (p <= 1)).all()
        assert cal.method == metodo


def test_graficos(tmp_path):
    y, p, pl = _calibrado(n=5000)
    desc = p ** 3
    versiones = {"m": {"sin_calibrar": desc, "isotonic": calibrar_cruzado(desc, y, pl, "isotonic"),
                       "sigmoid": calibrar_cruzado(desc, y, pl, "sigmoid")}}
    graficar_fiabilidad(y, versiones, tmp_path / "f.png", "prueba")
    graficar_costo({"m": barrido_umbrales(y, p)}, politicas_referencia(y), tmp_path / "c.png", "prueba")
    assert (tmp_path / "f.png").stat().st_size > 20_000 and (tmp_path / "c.png").stat().st_size > 20_000
