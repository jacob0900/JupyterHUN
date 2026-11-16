"""Pruebas de la subfase 7.4 con datos sintéticos (no usa datos reales).

    python -m pytest modulo_7/m7_4_pruebas_hipotesis -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modulo_7.m7_4_pruebas_hipotesis import (  # noqa: E402
    bootstrap_delta,
    comparar_ganador,
    delong,
    friedman_nemenyi,
    graficar_cd,
    graficar_heatmaps,
    ic_bca,
    jackknife_grupos,
    matriz_p_delong,
    p_bootstrap,
    veredicto,
    wilcoxon_metricas,
    wilcoxon_pareado,
)


def _datos(n=2000, semilla=0, tam_cong=1, señal=(1.0, 0.7)):
    """Pacientes con `tam_cong` filas casi idénticas (misma etiqueta y puntaje base)."""
    rng = np.random.default_rng(semilla)
    k = n // tam_cong
    y_p = rng.integers(0, 2, k)
    base1, base2 = rng.normal(size=k), rng.normal(size=k)
    y = np.repeat(y_p, tam_cong)
    g = np.repeat(np.arange(k), tam_cong)
    s1 = señal[0] * y + np.repeat(base1, tam_cong) + rng.normal(0, 0.1, len(y))
    s2 = señal[1] * y + np.repeat(base2, tam_cong) + rng.normal(0, 0.1, len(y))
    return y, s1, s2, g


def test_delong_auc_y_varianza_clasica():
    y, s1, s2, _ = _datos(n=3000)
    r = delong(y, s1)
    assert r["auc_1"] == pytest.approx(roc_auc_score(y, s1))
    rng = np.random.default_rng(5)
    boot = []
    for _ in range(400):
        i = rng.integers(0, len(y), len(y))
        boot.append(roc_auc_score(y[i], s1[i]))
    assert r["se"] == pytest.approx(np.std(boot, ddof=1), rel=0.15)
    d = delong(y, s1, s2)
    assert d["estimado"] == pytest.approx(roc_auc_score(y, s1) - roc_auc_score(y, s2))


def test_conglomerados_de_tamano_1_coinciden_con_el_clasico():
    y, s1, s2, _ = _datos(n=1500)
    cl, co = delong(y, s1, s2), delong(y, s1, s2, grupos=np.arange(len(y)))
    assert co["se"] == pytest.approx(cl["se"], rel=1e-10)
    assert co["p"] == pytest.approx(cl["p"], rel=1e-8)


def test_conglomerados_corrigen_la_correlacion_intra_paciente():
    y, s1, s2, g = _datos(n=4000, tam_cong=4)
    cl, co = delong(y, s1, s2), delong(y, s1, s2, grupos=g)
    assert co["se"] / cl["se"] == pytest.approx(2.0, rel=0.15)  # √4: 4 filas casi duplicadas por paciente


def test_error_tipo_i_cerca_de_alfa_con_conglomerados():
    rech_co, rech_cl = 0, 0
    for sem in range(200):
        y, s1, s2, g = _datos(n=600, semilla=sem, tam_cong=3, señal=(0.8, 0.8))  # misma discriminación
        rech_co += delong(y, s1, s2, grupos=g)["p"] < 0.05
        rech_cl += delong(y, s1, s2)["p"] < 0.05
    assert 0.01 <= rech_co / 200 <= 0.10
    assert rech_cl / 200 > rech_co / 200  # el clásico ignora la correlación y rechaza de más


def test_bca_y_p_bootstrap():
    y, s1, s2, g = _datos(n=2000, tam_cong=2)
    d = delong(y, s1, s2, grupos=g)["estimado"]
    boot = bootstrap_delta(y, s1, s2, g, n_iter=300, semilla=1)
    jack = jackknife_grupos(y, s1, s2, g, n_grupos=40)
    lo, hi = ic_bca(d, boot, jack)
    assert lo < d < hi
    assert lo == pytest.approx(np.quantile(boot, 0.025), abs=0.01)  # sin sesgo ni asimetría ≈ percentil
    assert p_bootstrap(boot) < 0.01  # s1 discrimina claramente mejor
    sim = np.random.default_rng(0).normal(0, 1, 999)
    assert 0.6 < p_bootstrap(sim) <= 1


def test_comparar_ganador_con_nan_y_holm():
    y, s1, s2, g = _datos(n=2400, tam_cong=2)
    s3 = s2.copy()
    s3[g % 4 != 0] = np.nan  # baseline evaluado en un subconjunto (como KNN)
    t = comparar_ganador(y, {"a": s1, "b": s2, "c": s3}, "a", g, n_iter=200, n_jack=30)
    assert set(t.baseline) == {"b", "c"}
    assert t.set_index("baseline").loc["c", "n"] == int((~np.isnan(s3)).sum())
    assert (t.p_holm >= t.p_delong_conglomerados).all() and (t.inflacion_se > 1).all()
    crudo, holm = matriz_p_delong(y, {"a": s1, "b": s2, "c": s3}, g, ["a", "b", "c"])
    assert np.allclose(crudo.to_numpy(), crudo.to_numpy().T, equal_nan=True) and (holm.fillna(1) >= crudo.fillna(1)).all().all()


def test_friedman_nemenyi_ranking_perfecto():
    k, nb = 7, 5
    rng = np.random.default_rng(0)
    m = pd.DataFrame(np.arange(k)[::-1][None, :] * 0.01 + rng.normal(0, 1e-4, (nb, k)) + 0.7,
                     columns=[f"m{j}" for j in range(k)])
    r = friedman_nemenyi(m)
    assert r["w_kendall"] == pytest.approx(1.0)
    assert r["chi2"] == pytest.approx(stats.friedmanchisquare(*[m[c] for c in m]).statistic)
    assert r["cd"] == pytest.approx(2.949 * np.sqrt(k * (k + 1) / (6 * nb)), rel=1e-3)
    assert list(r["rangos_medios"].index) == list(m.columns)
    assert r["p_nemenyi"].loc["m0", "m6"] < 0.05 and r["p_nemenyi"].loc["m0", "m1"] > 0.05


def test_wilcoxon_cinco_pares_no_alcanza_005():
    w = wilcoxon_pareado(np.array([0.74, 0.75, 0.73, 0.74, 0.76]), np.array([0.73, 0.74, 0.72, 0.73, 0.75]))
    assert w["p"] == pytest.approx(0.0625) and w["p_minimo_alcanzable"] == pytest.approx(0.0625)
    assert w["biserial_rangos"] == pytest.approx(1.0)
    assert "EQUIVALENCIA" in veredicto(w["p"]) and "SUPERIORIDAD" in veredicto(0.01)


def test_wilcoxon_metricas_orienta_las_de_menor_es_mejor():
    rng = np.random.default_rng(3)
    filas = []
    for b in range(30):
        filas += [{"bloque": b, "modelo": "top", "metrica": "auc", "valor": 0.75 + rng.normal(0, 0.003)},
                  {"bloque": b, "modelo": "seg", "metrica": "auc", "valor": 0.74 + rng.normal(0, 0.003)},
                  {"bloque": b, "modelo": "top", "metrica": "brier", "valor": 0.20 + rng.normal(0, 0.002)},
                  {"bloque": b, "modelo": "seg", "metrica": "brier", "valor": 0.21 + rng.normal(0, 0.002)}]
    t = wilcoxon_metricas(pd.DataFrame(filas), "top", "seg", {"auc": True, "brier": False}).set_index("metrica")
    assert t.top1_superior.all() and (t.mediana_diferencia > 0).all() and (t.p_holm >= t.p).all()


def test_graficos(tmp_path):
    m = pd.DataFrame(np.random.default_rng(1).normal(0.7, 0.01, (5, 7)), columns=list("abcdefg"))
    r = friedman_nemenyi(m)
    graficar_cd(r["rangos_medios"], r["cd"], tmp_path / "cd.png", "prueba")
    graficar_heatmaps({"Nemenyi": r["p_nemenyi"]}, tmp_path / "hm.png", "prueba")
    assert (tmp_path / "cd.png").stat().st_size > 10_000 and (tmp_path / "hm.png").stat().st_size > 10_000
