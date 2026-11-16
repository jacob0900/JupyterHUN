"""Pruebas de la subfase 5.4 con datos sintéticos (no usa datos reales).

    python -m pytest modulo_5/m5_4_pipeline -q
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modulo_5.m5_4_pipeline import (  # noqa: E402
    EsquemaPipeline,
    codigos_desde_matriz,
    construir_pipeline,
    huella_parametros,
    verificar_aislamiento,
    verificar_estadisticas_solo_entrenamiento,
    verificar_pipeline_sin_fuga,
)

warnings.filterwarnings("ignore", message="solo hay")

ESQ = EsquemaPipeline(
    num=["edad", "ocupacion"], num_log=["n_dx"], cat=["servicio", "sexo", "bandera"], clinica="dx_codigos"
)


def _datos(n: int = 600, semilla: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(semilla)
    codigos = np.array([f"dx_{i}" for i in range(40)])
    dx = [" ".join(rng.choice(codigos, size=rng.integers(0, 6), replace=False)) for _ in range(n)]
    X = pd.DataFrame({
        "person_id": rng.integers(0, n // 3, n),
        "edad": rng.normal(45, 20, n).round(),
        "ocupacion": rng.uniform(0.5, 1.1, n),
        "n_dx": rng.poisson(3, n).astype(float),
        "servicio": rng.choice(["hospitalizacion", "urgencias_observacion"], n),
        "sexo": rng.choice(["F", "M"], n),
        "bandera": rng.integers(0, 2, n),
        "dx_codigos": dx,
        "y_target_los_largo": 0,  # presente en X para comprobar que se ignora
    })
    X.loc[rng.choice(n, n // 10, replace=False), "edad"] = np.nan
    X.loc[rng.choice(n, n // 20, replace=False), "sexo"] = np.nan
    logit = 0.04 * (X.edad.fillna(45) - 45) + 0.8 * X.dx_codigos.str.contains("dx_3 ").astype(int) + rng.normal(0, 1, n)
    y = pd.Series((logit > 0.3).astype(int), name="y")
    X["y_target_los_largo"] = y  # una etiqueta dentro de X: no debe usarse
    return X, y


def _partir(X, y, frac=0.7):
    corte = int(len(X) * frac)
    return X.iloc[:corte], y.iloc[:corte], X.iloc[corte:], y.iloc[corte:]


# ---------------------------------------------------------------- estructura
def test_tres_ramas_del_prompt_y_parametros():
    pipe = construir_pipeline(ESQ)
    pre = pipe.named_steps["preprocesador"]
    ramas = {n: p for n, p, _ in pre.transformers}
    assert set(ramas) == {"num", "num_log", "cat", "clinica"}
    assert ramas["num"].named_steps["imputar"].strategy == "median"
    assert type(ramas["num"].named_steps["escalar"]).__name__ == "StandardScaler"
    imp_cat = ramas["cat"].named_steps["imputar"]
    assert (imp_cat.strategy, imp_cat.fill_value) == ("constant", "Desconocido")
    ohe = ramas["cat"].named_steps["onehot"]
    assert (ohe.drop, ohe.handle_unknown) == ("first", "ignore")
    assert ramas["clinica"].named_steps["svd"].n_components == 50
    assert pre.remainder == "drop"
    assert type(pipe.named_steps["modelo"]).__name__ == "LogisticRegression"


def test_esquema_rechaza_etiquetas_y_llaves():
    with pytest.raises(ValueError, match="prohibidas"):
        EsquemaPipeline(num=["edad", "y_los_h"], num_log=[], cat=[], clinica=None).validar(
            pd.DataFrame({"edad": [1], "y_los_h": [2]}))
    with pytest.raises(ValueError, match="prohibidas"):
        EsquemaPipeline(num=[], num_log=[], cat=["person_id"], clinica=None).validar(pd.DataFrame({"person_id": [1]}))


def test_fit_predict_y_salida_sin_etiquetas():
    X, y = _datos()
    Xtr, ytr, Xva, _ = _partir(X, y)
    pipe = construir_pipeline(ESQ, n_componentes_svd=5).fit(Xtr, ytr)
    p = pipe.predict_proba(Xva)[:, 1]
    assert p.shape == (len(Xva),) and np.all((p >= 0) & (p <= 1))
    nombres = pipe.named_steps["preprocesador"].get_feature_names_out()
    assert not any("y_target" in n or "person_id" in n for n in nombres)
    assert any(n.startswith("clinica__svd_dx_") for n in nombres)


# ---------------------------------------------------------------- ANTI-FUGA (tarea 3 del prompt)
def test_fit_en_train_predict_en_val_sin_fuga():
    """pipeline.fit(X_train, y_train) + pipeline.predict(X_val): ninguna estadística de validación entra."""
    X, y = _datos()
    Xtr, ytr, Xva, _ = _partir(X, y)
    res = verificar_pipeline_sin_fuga(construir_pipeline(ESQ, n_componentes_svd=5), Xtr, ytr, Xva, ESQ)
    assert res["ok"], res


def test_estadisticas_aprendidas_son_las_de_entrenamiento_y_no_las_de_todo():
    X, y = _datos()
    Xtr, ytr, Xva, _ = _partir(X, y)
    Xva = Xva.copy()
    Xva["edad"] += 300  # validación muy distinta: si se filtrara, la media cambiaría
    pipe = construir_pipeline(ESQ, n_componentes_svd=5).fit(Xtr, ytr)
    r = verificar_estadisticas_solo_entrenamiento(pipe, Xtr, ESQ)
    assert all(r.values()), r
    media = pipe.named_steps["preprocesador"].named_transformers_["num"].named_steps["escalar"].mean_[0]
    imp = Xtr.edad.fillna(Xtr.edad.median())
    assert media == pytest.approx(imp.mean())
    assert media != pytest.approx(pd.concat([Xtr.edad, Xva.edad]).mean())


def test_categoria_y_codigo_nuevos_en_validacion_no_fallan():
    X, y = _datos()
    Xtr, ytr, Xva, _ = _partir(X, y)
    Xva = Xva.copy()
    Xva["servicio"] = "uci_nunca_vista"
    Xva["dx_codigos"] = "dx_999999 dx_888888"
    pipe = construir_pipeline(ESQ, n_componentes_svd=5).fit(Xtr, ytr)
    Z = pipe.named_steps["preprocesador"].transform(Xva)
    nombres = list(pipe.named_steps["preprocesador"].get_feature_names_out())
    col_srv = [i for i, n in enumerate(nombres) if n.startswith("cat__servicio_")]
    assert np.all(Z[:, col_srv] == 0)  # categoría desconocida → todo ceros (nivel de referencia)
    col_svd = [i for i, n in enumerate(nombres) if n.startswith("clinica__")]
    assert np.allclose(Z[:, col_svd], 0)  # códigos fuera del vocabulario → sin aporte
    assert np.isfinite(pipe.predict_proba(Xva)).all()


def test_la_etiqueta_dentro_de_X_no_cambia_nada():
    X, y = _datos()
    Xtr, ytr, Xva, _ = _partir(X, y)
    a = construir_pipeline(ESQ, n_componentes_svd=5).fit(Xtr, ytr)
    Xtr2 = Xtr.copy()
    Xtr2["y_target_los_largo"] = 1 - Xtr2["y_target_los_largo"]
    b = construir_pipeline(ESQ, n_componentes_svd=5).fit(Xtr2, ytr)
    assert huella_parametros(a) == huella_parametros(b)
    assert np.array_equal(a.predict_proba(Xva), b.predict_proba(Xva))


def test_prediccion_fila_a_fila_igual_que_en_lote():
    X, y = _datos()
    Xtr, ytr, Xva, _ = _partir(X, y)
    assert verificar_aislamiento(construir_pipeline(ESQ, n_componentes_svd=5).fit(Xtr, ytr), Xva)


def test_detector_si_detecta_un_preprocesador_con_fuga():
    """Control positivo: si las medianas se calcularan con train + validación, la verificación lo detecta."""
    X, y = _datos()
    Xtr, ytr, Xva, _ = _partir(X, y)
    pipe = construir_pipeline(ESQ, n_componentes_svd=5).fit(pd.concat([Xtr, Xva]), pd.concat([ytr, ytr.iloc[:len(Xva)]]))
    r = verificar_estadisticas_solo_entrenamiento(pipe, Xtr, ESQ)
    assert not all(r.values())


def test_validacion_cruzada_agrupada_reajusta_por_fold():
    """Con StratifiedGroupKFold por person_id: ningún paciente en ambos lados y cada fold con sus propias estadísticas."""
    X, y = _datos(900)
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=0)
    medias = []
    for tr, va in cv.split(X, y, groups=X.person_id):
        assert not set(X.person_id.iloc[tr]) & set(X.person_id.iloc[va])
        p = clone(construir_pipeline(ESQ, n_componentes_svd=5)).fit(X.iloc[tr], y.iloc[tr])
        r = verificar_estadisticas_solo_entrenamiento(p, X.iloc[tr], ESQ)
        assert all(r.values()), r
        medias.append(p.named_steps["preprocesador"].named_transformers_["num"].named_steps["escalar"].mean_[0])
    assert len(set(np.round(medias, 10))) == 3  # cada fold aprendió algo distinto


# ---------------------------------------------------------------- empaquetado para M6
def test_pipeline_se_serializa_y_reutiliza(tmp_path):
    X, y = _datos()
    Xtr, ytr, Xva, _ = _partir(X, y)
    sin_ajustar = construir_pipeline(ESQ, n_componentes_svd=5)
    joblib.dump(sin_ajustar, tmp_path / "plantilla.joblib")
    p1 = joblib.load(tmp_path / "plantilla.joblib").fit(Xtr, ytr)
    joblib.dump(p1, tmp_path / "ajustado.joblib")
    p2 = joblib.load(tmp_path / "ajustado.joblib")
    assert np.array_equal(p1.predict_proba(Xva), p2.predict_proba(Xva))


def test_cambiar_estimador_para_m6():
    from sklearn.ensemble import HistGradientBoostingClassifier

    X, y = _datos()
    Xtr, ytr, Xva, _ = _partir(X, y)
    pipe = construir_pipeline(ESQ, n_componentes_svd=5).set_params(modelo=HistGradientBoostingClassifier(max_iter=20))
    assert pipe.fit(Xtr, ytr).predict(Xva).shape == (len(Xva),)


def test_codigos_desde_matriz():
    m = sp.csr_matrix(np.array([[1, 0, 1], [0, 0, 0], [0, 1, 0]], dtype=np.float32))
    assert codigos_desde_matriz(m, np.array(["dx_a", "dx_b", "dx_c"])).tolist() == ["dx_a dx_c", "", "dx_b"]
