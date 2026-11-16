"""Pruebas de la subfase 6.2 con datos sintéticos (no usa datos reales).

    python -m pytest modulo_6/m6_2_bateria_modelos -q
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modulo_5.m5_4_pipeline import EsquemaPipeline, construir_preprocesador  # noqa: E402
from modulo_6.m6_2_bateria_modelos import (  # noqa: E402
    PARAM_SPACES,
    SALIDA_PUNTAJE,
    Categorico,
    Entero,
    ModelFactory,
    MuestreadorAleatorio,
    Real,
    a_distribuciones_scipy,
    get_pipeline_and_space,
    listar_modelos,
    sugerir_parametros,
)

warnings.filterwarnings("ignore")
ESQ = EsquemaPipeline(num=["edad", "ocupacion"], num_log=["n_dx"], cat=["servicio", "sexo"], clinica="dx_codigos")
PRE = construir_preprocesador(ESQ, n_componentes_svd=4, min_df_svd=2)
ESPERADOS = {"logreg_ridge", "logreg_lasso", "knn", "naive_bayes", "cart", "random_forest", "lightgbm", "xgboost",
             "svm_lineal", "svm_rbf"}


def _datos(n=500, semilla=0):
    rng = np.random.default_rng(semilla)
    cod = np.array([f"dx_{i}" for i in range(25)])
    X = pd.DataFrame({
        "person_id": rng.integers(0, n // 3, n),
        "edad": rng.normal(45, 20, n),
        "ocupacion": rng.uniform(0.5, 1.1, n),
        "n_dx": rng.poisson(3, n).astype(float),
        "servicio": rng.choice(["hospitalizacion", "urgencias_observacion"], n),
        "sexo": rng.choice(["F", "M"], n),
        "dx_codigos": [" ".join(rng.choice(cod, rng.integers(0, 5), replace=False)) for _ in range(n)],
    })
    y = ((X.edad - 45) / 20 + X.dx_codigos.str.contains("dx_1 ").astype(int) + rng.normal(0, 0.7, n) > 0.3).astype(int)
    return X, y


def _puntaje(pipe, nombre, X):
    return pipe.decision_function(X) if SALIDA_PUNTAJE[nombre] == "decision_function" else pipe.predict_proba(X)[:, 1]


def _rapidos(params: dict) -> dict:
    """Recorta lo que solo cambia el tiempo (número de árboles / componentes) para que las pruebas sean ágiles."""
    p = dict(params)
    for k in ("modelo__n_estimators",):
        if k in p:
            p[k] = min(p[k], 60)
    if "modelo__nystroem__n_components" in p:
        p["modelo__nystroem__n_components"] = min(p["modelo__nystroem__n_components"], 120)
    if "modelo__n_neighbors" in p:
        p["modelo__n_neighbors"] = min(p["modelo__n_neighbors"], 30)
    return p


def test_bateria_completa_del_prompt():
    assert set(listar_modelos()) == ESPERADOS
    assert set(PARAM_SPACES) == ESPERADOS


@pytest.mark.parametrize("nombre", sorted(ESPERADOS))
def test_pipeline_sin_entrenar_con_preprocesador_clonado(nombre):
    pipe, espacio = get_pipeline_and_space(nombre, PRE)
    assert isinstance(pipe, Pipeline) and [n for n, _ in pipe.steps] == ["preprocesador", "modelo"]
    assert pipe.named_steps["preprocesador"] is not PRE  # se clona: el objeto del M5 nunca se toca
    assert not hasattr(pipe.named_steps["preprocesador"], "transformers_")  # sin entrenar
    assert not hasattr(PRE, "transformers_")
    otro, _ = get_pipeline_and_space(nombre, PRE)
    assert otro is not pipe and otro.named_steps["modelo"] is not pipe.named_steps["modelo"]
    assert espacio and espacio is not PARAM_SPACES[nombre]  # copia: modificarlo no altera el global


@pytest.mark.parametrize("nombre", sorted(ESPERADOS))
def test_semilla_42_en_todo_lo_aleatorio(nombre):
    pipe, _ = get_pipeline_and_space(nombre, PRE)
    semillas = {k: v for k, v in pipe.get_params().items() if k.endswith("random_state")}
    assert semillas, "debería haber al menos la semilla de la SVD del preprocesador"
    assert all(v == 42 for v in semillas.values()), semillas


def test_regularizacion_en_escala_logaritmica():
    claves = ("__C", "alpha", "gamma", "learning_rate", "var_smoothing", "reg_", "ccp_alpha", "min_child_weight")
    revisados = 0
    for nombre, espacio in PARAM_SPACES.items():
        for k, d in espacio.items():
            if any(c in k for c in claves):
                revisados += 1
                assert isinstance(d, Real) and d.log, f"{nombre}.{k} debe ser Real con log=True"
                assert d.low > 0
    assert revisados == 15
    assert PARAM_SPACES["logreg_ridge"]["modelo__C"] == Real(1e-4, 1e2, log=True)
    assert PARAM_SPACES["lightgbm"]["modelo__num_leaves"].log
    assert PARAM_SPACES["knn"]["modelo__n_neighbors"] == Entero(5, 201, log=True)


@pytest.mark.parametrize("nombre", sorted(ESPERADOS))
def test_espacio_valido_y_entrenable(nombre):
    """Cada punto muestreado del espacio es un parámetro válido del Pipeline, y el Pipeline entrena y predice."""
    X, y = _datos()
    pipe, espacio = get_pipeline_and_space(nombre, PRE)
    params = sugerir_parametros(MuestreadorAleatorio(semilla=3), espacio)
    for k, d in espacio.items():
        v = params[k]
        if isinstance(d, (Real, Entero)):
            assert d.low <= v <= d.high
        else:
            assert v in d.opciones
    pipe.set_params(**_rapidos(params))
    tr, va = np.arange(350), np.arange(350, 500)
    pipe.fit(X.iloc[tr], y.iloc[tr])
    s = _puntaje(pipe, nombre, X.iloc[va])
    assert s.shape == (150,) and np.isfinite(s).all()


@pytest.mark.parametrize("nombre", sorted(ESPERADOS))
def test_cada_modelo_aprende_la_senal(nombre):
    """Con sus valores por defecto, cada modelo de la batería supera el azar en datos sintéticos.
    (Un punto aleatorio del espacio puede no aprender con 350 filas, p. ej. Lasso con C = 1e-4:
    eso es legítimo y lo resuelve la búsqueda de la 6.3.)"""
    X, y = _datos(800)
    pipe, _ = get_pipeline_and_space(nombre, PRE)
    if nombre == "svm_rbf":
        pipe.set_params(modelo__nystroem__n_components=150)
    pipe.fit(X.iloc[:550], y.iloc[:550])
    assert roc_auc_score(y.iloc[550:], _puntaje(pipe, nombre, X.iloc[550:])) > 0.6


def test_fabrica_orientada_a_objetos_y_errores():
    f = ModelFactory(PRE)
    pipe, esp = f.get_pipeline_and_space("lightgbm")
    assert "modelo__num_leaves" in esp and type(pipe.named_steps["modelo"]).__name__ == "LGBMClassifier"
    with pytest.raises(KeyError, match="desconocido"):
        f.get_pipeline_and_space("modelo_inexistente")
    X, y = _datos()
    ajustado = construir_preprocesador(ESQ, n_componentes_svd=4, min_df_svd=2).fit(X, y)
    with pytest.raises(ValueError, match="ya está ajustado"):
        get_pipeline_and_space("cart", ajustado)


def test_preprocesador_por_defecto_del_modulo_5():
    pipe, _ = get_pipeline_and_space("logreg_ridge")
    ramas = {n for n, _, _ in pipe.named_steps["preprocesador"].transformers}
    assert ramas == {"num", "num_log", "cat", "clinica"}


def test_espacio_con_randomizedsearchcv_y_cv_agrupada():
    X, y = _datos(400)
    pipe, espacio = get_pipeline_and_space("logreg_ridge", PRE)
    busqueda = RandomizedSearchCV(pipe, a_distribuciones_scipy(espacio), n_iter=3, scoring="roc_auc",
                                  cv=StratifiedGroupKFold(3, shuffle=True, random_state=42), random_state=42)
    busqueda.fit(X, y, groups=X.person_id)
    assert 1e-4 <= busqueda.best_params_["modelo__C"] <= 1e2


def test_distribuciones_scipy_respetan_limites():
    rng = np.random.default_rng(0)
    for nombre, espacio in PARAM_SPACES.items():
        for k, dist in a_distribuciones_scipy(espacio).items():
            d = espacio[k]
            if isinstance(d, Categorico):
                assert dist == list(d.opciones)
                continue
            v = np.asarray(dist.rvs(size=200, random_state=rng))
            assert v.min() >= d.low and v.max() <= d.high, (nombre, k)


def test_compatible_con_optuna_si_esta_instalado():
    optuna = pytest.importorskip("optuna")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    estudio = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
    trial = estudio.ask()
    p = sugerir_parametros(trial, PARAM_SPACES["xgboost"])
    assert set(p) == set(PARAM_SPACES["xgboost"])
