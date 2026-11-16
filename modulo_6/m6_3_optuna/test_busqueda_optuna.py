"""Pruebas de la subfase 6.3 con datos sintéticos (no usa datos reales).

    python -m pytest modulo_6/m6_3_optuna -q
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modulo_5.m5_4_pipeline import EsquemaPipeline, construir_preprocesador  # noqa: E402
from modulo_6.m6_2_bateria_modelos import listar_modelos  # noqa: E402
from modulo_6.m6_3_optuna import (  # noqa: E402
    FAMILIAS,
    CachePliegues,
    guardar_resultados,
    objective,
    optimizar_modelo,
    resumen_estudio,
    tabla_por_familia,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
ESQ = EsquemaPipeline(num=["edad", "ocupacion"], num_log=["n_dx"], cat=["servicio", "sexo"], clinica="dx_codigos")
PRE = construir_preprocesador(ESQ, n_componentes_svd=4, min_df_svd=2)
CV = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)


def _datos(n=600, semilla=0):
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
    y = ((X.edad - 45) / 20 + X.dx_codigos.str.contains("dx_1 ").astype(int) + rng.normal(0, 0.8, n) > 0.3).astype(int)
    return X, y.to_numpy(), X.person_id.to_numpy()


def test_siete_familias_cubren_la_bateria():
    assert len(FAMILIAS) == 7
    assert sorted(m for ms in FAMILIAS.values() for m in ms) == sorted(listar_modelos())


def test_objective_devuelve_media_de_los_5_pliegues():
    X, y, g = _datos()
    trial = optuna.trial.FixedTrial({"modelo__C": 1.0, "modelo__class_weight": None})
    valor = objective(trial, "logreg_ridge", X, y, g, CV, preprocesador=PRE,
                      estratos=X.servicio.to_numpy())
    assert 0.6 < valor <= 1.0


@pytest.mark.parametrize("modelo, params", [
    ("logreg_ridge", {"modelo__C": 0.3, "modelo__class_weight": "balanced"}),
    ("lightgbm", {"modelo__n_estimators": 50, "modelo__learning_rate": 0.1, "modelo__num_leaves": 15,
                  "modelo__min_child_samples": 20, "modelo__subsample": 0.8, "modelo__colsample_bytree": 0.8,
                  "modelo__reg_alpha": 1e-3, "modelo__reg_lambda": 1e-3}),
    ("svm_lineal", {"modelo__C": 0.1, "modelo__class_weight": None}),
])
def test_cache_equivale_al_pipeline(modelo, params):
    """Ajustar el preprocesador una vez por pliegue (caché) da exactamente el mismo AUC que el Pipeline completo."""
    X, y, g = _datos()
    directo = objective(optuna.trial.FixedTrial(params), modelo, X, y, g, CV, preprocesador=PRE)
    cache = CachePliegues(PRE, X, y, CV, groups=g, dtype=np.float64)
    con_cache = objective(optuna.trial.FixedTrial(params), modelo, X, y, g, CV, cache=cache, preprocesador=PRE)
    assert con_cache == pytest.approx(directo, abs=1e-12)


def test_cache_ajusta_el_preprocesador_solo_con_el_entrenamiento_de_cada_pliegue():
    X, y, g = _datos()
    cache = CachePliegues(PRE, X, y, CV, groups=g)
    assert len(cache) == 5
    for p, (a, b) in zip(cache.pliegues, CV.split(X, y, groups=g)):
        assert len(p.Z_tr) == len(a) and len(p.Z_va) == len(b)
        # Las columnas numéricas escaladas tienen media ~0 en el entrenamiento del pliegue (no en la validación)
        assert abs(p.Z_tr[:, 0].mean()) < 1e-5


class _TrialQuePoda(optuna.trial.FixedTrial):
    def __init__(self, params, podar_en: int):
        super().__init__(params)
        self.reportes, self.podar_en = [], podar_en

    def report(self, value, step):
        self.reportes.append((step, value))

    def should_prune(self):
        return len(self.reportes) - 1 >= self.podar_en


def test_reporta_cada_pliegue_y_respeta_la_poda():
    X, y, g = _datos()
    t = _TrialQuePoda({"modelo__var_smoothing": 1e-9}, podar_en=1)
    with pytest.raises(optuna.TrialPruned):
        objective(t, "naive_bayes", X, y, g, CV, preprocesador=PRE)
    assert [s for s, _ in t.reportes] == [0, 1]  # cortó después del segundo pliegue


def test_estudio_tpe_median_pruner_y_reanudable(tmp_path):
    X, y, g = _datos(450)
    storage = f"sqlite:///{(tmp_path / 'estudio.db').as_posix()}"
    cache = CachePliegues(PRE, X, y, CV, groups=g)
    e1 = optimizar_modelo("naive_bayes", X, y, g, CV, n_trials=4, cache=cache, preprocesador=PRE, storage=storage)
    assert isinstance(e1.sampler, optuna.samplers.TPESampler)
    assert isinstance(e1.pruner, optuna.pruners.MedianPruner)
    assert e1.direction == optuna.study.StudyDirection.MAXIMIZE
    e2 = optimizar_modelo("naive_bayes", X, y, g, CV, n_trials=7, cache=cache, preprocesador=PRE, storage=storage)
    assert len(e2.trials) == 7  # reanudó: 4 previos + 3 nuevos
    fila = resumen_estudio("naive_bayes", e2, 1.0)
    assert fila["mejor_auc_cv"] == pytest.approx(e2.best_value)
    assert set(fila["hiperparametros"]) == {"modelo__var_smoothing"}


def test_cv_sin_agrupar_se_rechaza():
    X, y, g = _datos()
    with pytest.raises(ValueError, match="person_id"):
        optimizar_modelo("naive_bayes", X, y, g, KFold(5, shuffle=True, random_state=0), n_trials=1, preprocesador=PRE)


def test_json_y_tablas(tmp_path):
    filas = [
        {"modelo": "lightgbm", "familia": "Gradient boosting (XGBoost / LightGBM)", "mejor_auc_cv": 0.73,
         "auc_std_pliegues": 0.004, "auc_medio_servicio_tau": 0.64, "ensayos_completos": 40, "ensayos_podados": 10,
         "hiperparametros": {"modelo__learning_rate": 0.05}},
        {"modelo": "xgboost", "familia": "Gradient boosting (XGBoost / LightGBM)", "mejor_auc_cv": 0.72,
         "auc_std_pliegues": 0.004, "auc_medio_servicio_tau": 0.63, "ensayos_completos": 45, "ensayos_podados": 5,
         "hiperparametros": {"modelo__max_depth": np.int64(6)}},
        {"modelo": "cart", "familia": "Árbol de decisión (CART)", "mejor_auc_cv": 0.69, "auc_std_pliegues": 0.006,
         "auc_medio_servicio_tau": 0.6, "ensayos_completos": 50, "ensayos_podados": 0,
         "hiperparametros": {"modelo__max_depth": 8}},
    ]
    tabla = guardar_resultados(filas, tmp_path / "best_hyperparameters.json", tmp_path / "resumen.csv", {"n_trials": 50})
    j = json.loads((tmp_path / "best_hyperparameters.json").read_text(encoding="utf-8"))
    assert set(j["modelos"]) == {"lightgbm", "xgboost", "cart"} and j["n_trials"] == 50
    assert j["modelos"]["xgboost"]["hiperparametros"]["modelo__max_depth"] == 6
    assert tabla.modelo.tolist() == ["lightgbm", "xgboost", "cart"]
    fam = tabla_por_familia(tabla)
    assert fam.familia.is_unique and fam.iloc[0].modelo == "lightgbm" and len(fam) == 2
