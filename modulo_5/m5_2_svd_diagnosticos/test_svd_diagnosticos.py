"""Pruebas de la subfase 5.2 con datos sintéticos (no usa datos reales).

    python -m pytest modulo_5/m5_2_svd_diagnosticos -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import parametrize_with_checks

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modulo_5.m5_2_svd_diagnosticos import (  # noqa: E402
    ReductorSVDClinico,
    construir_matriz_dispersa,
    preparar_eventos,
)

T = pd.Timestamp


# --------------------------------------------------------------------------- datos sintéticos
def _cohorte(n_personas: int = 400, seed: int = 0):
    """Personas con 3 perfiles latentes de comorbilidad (10 códigos cada uno) + ruido.

    El target depende del perfil 0, así que un buen reductor debe conservar esa señal.
    """
    rng = np.random.default_rng(seed)
    perfil = rng.integers(0, 3, n_personas)
    filas_dx, filas_snap = [], []
    base = T("2024-01-01")
    for p in range(n_personas):
        codigos = 1000 + 10 * perfil[p] + rng.choice(10, size=rng.integers(3, 8), replace=False)
        ruido = rng.integers(2000, 2200, size=2)
        for c in np.concatenate([codigos, ruido]):
            filas_dx.append((p, int(c), base + pd.Timedelta(hours=int(rng.integers(0, 24 * 300)))))
        tau = base + pd.Timedelta(days=330 + int(rng.integers(0, 30)))
        y = int(rng.random() < (0.7 if perfil[p] == 0 else 0.2))
        for h in (0, 2, 6):
            filas_snap.append((len(filas_snap), p, tau + pd.Timedelta(hours=h), y, bool(p < 0.7 * n_personas)))
    dx = pd.DataFrame(filas_dx, columns=["person_id", "condition_concept_id", "condition_start_datetime"])
    snap = pd.DataFrame(filas_snap, columns=["snapshot_id", "person_id", "snapshot_time", "y_target_los_largo", "es_entrenamiento"])
    return snap, dx


def _matriz():
    snap, dx = _cohorte()
    ev = preparar_eventos(dx, "condition_concept_id", "condition_start_datetime", prefijo="dx_")
    X, vocab = construir_matriz_dispersa(snap, ev)
    return snap, X, vocab


# --------------------------------------------------------------------------- construcción de la matriz
def test_preparar_eventos_primera_aparicion_y_exclusiones():
    dx = pd.DataFrame(
        {
            "person_id": [1, 1, 1, 2, 2],
            "condition_concept_id": [10, 10, 0, 20, 30],
            "condition_start_datetime": [T("2024-03-01 10:00"), T("2024-01-01 09:00"), T("2024-01-01 08:00"),
                                         T("2024-02-02 00:00"), pd.NaT],
            "condition_start_date": [None, None, None, None, T("2024-02-05")],
        }
    )
    ev = preparar_eventos(dx, "condition_concept_id", "condition_start_datetime", "condition_start_date", prefijo="dx_")
    ev = ev.set_index(["person_id", "codigo"]).t_efectiva
    assert ev[(1, "dx_10")] == T("2024-01-01 09:00")           # primera aparición
    assert (1, "dx_0") not in ev.index                          # concepto 0 fuera
    assert ev[(2, "dx_20")] > T("2024-02-02 23:59:59")          # 00:00 -> final del día
    assert ev[(2, "dx_30")] > T("2024-02-05 23:59:59")          # solo fecha -> final del día


def test_procedimientos_sin_fecha_quedan_fuera():
    px = pd.DataFrame({"person_id": [1, 2], "procedure_concept_id": [5, 6], "procedure_date": [None, None]})
    ev = preparar_eventos(px, "procedure_concept_id", None, "procedure_date", prefijo="px_")
    assert ev.empty


def test_matriz_respeta_t_estrictamente_menor_que_tau():
    snap = pd.DataFrame({"person_id": [1, 1, 2], "snapshot_time": [T("2024-01-01 10:00"), T("2024-01-01 12:00"), T("2024-01-01 10:00")]},
                        index=[50, 3, 7])
    ev = pd.DataFrame({"person_id": [1, 1, 2], "codigo": ["dx_a", "dx_b", "dx_a"],
                       "t_efectiva": [T("2024-01-01 10:00"), T("2024-01-01 08:00"), T("2024-01-02")]})
    X, vocab = construir_matriz_dispersa(snap, ev)
    assert sp.isspmatrix_csr(X) and X.shape == (3, 2)
    assert list(vocab) == ["dx_a", "dx_b"]
    np.testing.assert_array_equal(X.toarray(), [[0, 1],   # dx_a justo en τ no cuenta
                                                [1, 1],   # 2 h después ya cuenta
                                                [0, 0]])  # el de la persona 2 es futuro


def test_matriz_binaria_y_dispersa():
    _, X, _ = _matriz()
    assert set(np.unique(X.data)) == {1.0}
    assert X.nnz < 0.2 * X.shape[0] * X.shape[1]


# --------------------------------------------------------------------------- transformador
def test_fit_solo_entrenamiento_y_transform_no_reajusta():
    snap, X, vocab = _matriz()
    # un código que solo existe en filas de prueba
    tr = snap.es_entrenamiento.to_numpy()
    X = sp.hstack([X, sp.csr_matrix(np.where(~tr, 1.0, 0.0)[:, None])]).tocsr()
    red = ReductorSVDClinico(n_components=5, min_df=5).fit(X[tr])
    assert X.shape[1] - 1 not in red.columnas_
    antes = red.svd_.components_.copy()
    Z = red.transform(X[~tr])
    np.testing.assert_array_equal(antes, red.svd_.components_)
    assert Z.shape == (int((~tr).sum()), 5)


def test_varianza_acumulada(capsys):
    _, X, _ = _matriz()
    red = ReductorSVDClinico(n_components=20).fit(X)
    va = red.varianza_acumulada_
    assert np.all(np.diff(va) >= -1e-12) and 0 < va[-1] <= 1 + 1e-9
    red.imprimir_varianza()
    salida = capsys.readouterr().out
    assert "Varianza explicada acumulada" in salida and " 20 comp." in salida


def test_recorta_componentes_si_hay_pocos_codigos():
    X = sp.random(60, 8, density=0.5, format="csr", random_state=0)
    with pytest.warns(UserWarning):
        red = ReductorSVDClinico(n_components=50, min_df=1).fit(X)
    assert red.n_components_ == 8 and red.transform(X).shape == (60, 8)


def test_cargas_recuperan_los_perfiles():
    _, X, vocab = _matriz()
    red = ReductorSVDClinico(n_components=3, min_df=5).fit(X)
    c = red.cargas(vocab, top=5)
    # los códigos fuertes de cada componente pertenecen a los bloques de comorbilidad (10xx), no al ruido (2xxx)
    assert c.codigo.str.startswith("dx_10").mean() > 0.8


# --------------------------------------------------------------------------- compatibilidad scikit-learn
def test_api_basica_sklearn():
    _, X, _ = _matriz()
    red = ReductorSVDClinico(n_components=7)
    with pytest.raises(NotFittedError):
        red.transform(X)
    assert clone(red).get_params() == red.get_params()
    red.set_params(n_components=4)
    Z = red.fit_transform(X)
    assert Z.shape == (X.shape[0], 4) and red.n_features_in_ == X.shape[1]
    assert list(red.get_feature_names_out()) == ["svd_dx_00", "svd_dx_01", "svd_dx_02", "svd_dx_03"]
    with pytest.raises(ValueError):
        red.transform(X[:, :-1])  # otro número de columnas
    df = clone(red).set_output(transform="pandas").fit_transform(X)
    assert list(df.columns) == list(red.get_feature_names_out())


def test_pipeline_cv_agrupada_y_gridsearch():
    snap, X, _ = _matriz()
    y, grupos = snap.y_target_los_largo.to_numpy(), snap.person_id.to_numpy()
    pipe = Pipeline([("svd", ReductorSVDClinico(n_components=5)), ("lr", LogisticRegression(max_iter=1000))])
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    auc = cross_val_score(pipe, X, y, groups=grupos, cv=cv, scoring="roc_auc")
    assert auc.mean() > 0.7  # el perfil 0 sube el riesgo y la SVD lo conserva
    gs = GridSearchCV(pipe, {"svd__n_components": [2, 5]}, cv=cv, scoring="roc_auc").fit(X, y, groups=grupos)
    assert gs.best_params_["svd__n_components"] in (2, 5)


@parametrize_with_checks([ReductorSVDClinico(n_components=2, min_df=1)])
def test_check_estimator_sklearn(estimator, check):
    """Batería oficial de scikit-learn para transformadores."""
    check(estimator)
