"""Pruebas de la subfase 7.1 con datos sintéticos (no usa datos reales).

    python -m pytest modulo_7/m7_1_multimetrica -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modulo_7.m7_1_multimetrica import (  # noqa: E402
    CurvaPreordenada,
    bootstrap_ic,
    ece_igual_frecuencia,
    estratos_paciente,
    evaluar_modelos,
    graficar_roc_pr,
    metricas,
    metricas_por_estrato,
    metricas_umbral,
    platt_cruzado,
    replicas_bootstrap,
)


def _datos(n=3000, semilla=0, prev=0.3):
    rng = np.random.default_rng(semilla)
    g = rng.integers(0, n // 5, n)
    y = (rng.random(n) < prev).astype(int)
    bueno = 1 / (1 + np.exp(-(1.5 * (y - 0.5) + rng.normal(0, 1, n) - np.log((1 - prev) / prev) / 2)))
    malo = rng.random(n)
    return y, g, bueno, malo


def test_curva_preordenada_igual_a_sklearn_con_pesos_y_empates():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.random(500), 1)  # muchos empates
    w = rng.integers(0, 4, 500).astype(float)  # incluye pesos 0, como en el bootstrap
    c = CurvaPreordenada.desde(y, s)
    assert c.auc_roc(w) == pytest.approx(roc_auc_score(y, s, sample_weight=w))
    assert c.auc_pr(w) == pytest.approx(average_precision_score(y, s, sample_weight=w))
    uno = np.ones(500)
    assert c.auc_roc(uno) == pytest.approx(roc_auc_score(y, s))
    assert c.auc_pr(uno) == pytest.approx(average_precision_score(y, s))


def test_ece_igual_frecuencia():
    rng = np.random.default_rng(2)
    p = rng.random(30000)
    y = (rng.random(30000) < p).astype(int)
    assert ece_igual_frecuencia(y, p) < 0.02  # bien calibrado
    assert ece_igual_frecuencia(y, np.clip(p + 0.2, 0, 1)) > 0.15  # sesgo sistemático
    # cálculo a mano con 3 intervalos de igual frecuencia
    yy, pp = np.array([0, 0, 1, 1, 1, 0]), np.array([0.1, 0.2, 0.4, 0.5, 0.8, 0.9])
    esperado = (2 * abs(0 - 0.15) + 2 * abs(1 - 0.45) + 2 * abs(0.5 - 0.85)) / 6
    assert ece_igual_frecuencia(yy, pp, n_bins=3) == pytest.approx(esperado)


def test_metricas_umbral_a_mano():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 1])
    p = np.array([0.9, 0.6, 0.4, 0.7, 0.2, 0.1, 0.3, 0.5])  # pred: 1 1 0 1 0 0 0 1
    m = metricas_umbral(y, p)
    assert m["sensibilidad"] == pytest.approx(3 / 4)
    assert m["especificidad"] == pytest.approx(3 / 4)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(3 / 4)


def test_metricas_auc_pr_sin_informacion_es_la_prevalencia():
    y, g, bueno, malo = _datos(n=20000, prev=0.2)
    m = metricas(y, malo)
    assert m["auc_pr"] == pytest.approx(m["prevalencia"], abs=0.02)  # no 0,5
    assert m["auc_roc"] == pytest.approx(0.5, abs=0.02)
    assert metricas(y, bueno)["auc_pr"] > m["auc_pr"] + 0.1


def test_bootstrap_por_paciente_y_estratificado():
    y, g, *_ = _datos()
    inv, estrato = estratos_paciente(y, g)
    assert set(np.unique(estrato)) <= {0, 1, 2}
    for w in replicas_bootstrap(y, g, n_iter=5, semilla=3):
        assert pd.Series(w).groupby(inv).nunique().eq(1).all()  # todas las filas del paciente salen juntas
        cuenta = np.bincount(inv, weights=w) / np.bincount(inv)
        for e in np.unique(estrato):
            assert cuenta[estrato == e].sum() == pytest.approx((estrato == e).sum())  # tamaño de cada estrato fijo


def test_ic_contiene_el_valor_y_es_reproducible():
    y, g, bueno, malo = _datos()
    s = {"bueno": bueno, "malo": malo}
    ic, rep = bootstrap_ic(y, s, g, n_iter=200, semilla=7)
    ic2, _ = bootstrap_ic(y, s, g, n_iter=200, semilla=7)
    pd.testing.assert_frame_equal(ic, ic2)
    fila = ic.set_index("modelo").loc["bueno"]
    assert fila.auc_roc_ic95_inf < roc_auc_score(y, bueno) < fila.auc_roc_ic95_sup
    assert fila.auc_pr_ic95_inf < average_precision_score(y, bueno) < fila.auc_pr_ic95_sup
    assert len(rep) == 400 and rep.groupby("iteracion").size().eq(2).all()  # réplicas pareadas


def test_puntajes_con_nan_usan_solo_sus_filas():
    y, g, bueno, _ = _datos()
    parcial = bueno.copy()
    parcial[g % 5 != 0] = np.nan
    tabla, _ = evaluar_modelos(y, {"completo": bueno, "parcial": parcial}, g, n_iter=50)
    t = tabla.set_index("modelo")
    ok = ~np.isnan(parcial)
    assert t.loc["parcial", "n"] == ok.sum()
    assert t.loc["parcial", "auc_roc"] == pytest.approx(roc_auc_score(y[ok], bueno[ok]))
    assert tabla.auc_pr.is_monotonic_decreasing  # ordenada por AUC-PR


def test_platt_cruzado_no_usa_las_etiquetas_del_pliegue():
    y, g, bueno, _ = _datos()
    s = np.log(bueno / (1 - bueno)) * 3 + 1  # puntaje tipo decision_function
    pliegues = g % 5
    p = platt_cruzado(s, y, pliegues)
    assert ((p > 0) & (p < 1)).all()
    assert roc_auc_score(y, p) == pytest.approx(roc_auc_score(y, s), abs=0.01)
    assert ece_igual_frecuencia(y, p) < 0.05
    y_alterado = y.copy()
    y_alterado[pliegues == 0] = 1 - y_alterado[pliegues == 0]  # cambiar etiquetas del pliegue 0
    assert np.allclose(platt_cruzado(s, y_alterado, pliegues)[pliegues == 0], p[pliegues == 0])


def test_estratos_y_grafico(tmp_path):
    y, g, bueno, malo = _datos()
    estr = np.where(g % 2 == 0, "a", "b")
    t = metricas_por_estrato(y, {"bueno": bueno, "malo": malo}, estr)
    assert set(t.estrato) == {"a", "b"} and len(t) == 4
    graficar_roc_pr(y, {"bueno": bueno, "malo": malo}, tmp_path / "roc_pr.png", {"malo": " (aleatorio)"})
    assert (tmp_path / "roc_pr.png").stat().st_size > 20_000
