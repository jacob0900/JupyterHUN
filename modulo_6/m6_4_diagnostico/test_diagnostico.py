"""Pruebas de la subfase 6.4 con datos sintéticos (no usa datos reales).

    python -m pytest modulo_6/m6_4_diagnostico -q
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modulo_5.m5_4_pipeline import EsquemaPipeline, construir_preprocesador  # noqa: E402
from modulo_6.m6_4_diagnostico import (  # noqa: E402
    AdvertenciaSobreajuste,
    CacheDiagnostico,
    advertir_brechas,
    ajustar_final,
    cargar_mejores,
    curva_aprendizaje,
    diagnosticar,
    graficar_curvas_aprendizaje,
    graficar_roc,
    instanciar,
    puntaje,
)

ESQ = EsquemaPipeline(num=["edad", "ocupacion"], num_log=["n_dx"], cat=["servicio", "sexo"], clinica="dx_codigos")
PRE = construir_preprocesador(ESQ, n_componentes_svd=4, min_df_svd=2)
CV = StratifiedGroupKFold(5, shuffle=True, random_state=42)


def _datos(n=700, semilla=0):
    rng = np.random.default_rng(semilla)
    cod = np.array([f"dx_{i}" for i in range(25)])
    X = pd.DataFrame({
        "person_id": rng.integers(0, n // 3, n),
        "edad": rng.normal(45, 20, n), "ocupacion": rng.uniform(0.5, 1.1, n), "n_dx": rng.poisson(3, n).astype(float),
        "servicio": rng.choice(["hospitalizacion", "urgencias_observacion"], n), "sexo": rng.choice(["F", "M"], n),
        "dx_codigos": [" ".join(rng.choice(cod, rng.integers(0, 5), replace=False)) for _ in range(n)],
    })
    y = ((X.edad - 45) / 20 + X.dx_codigos.str.contains("dx_1 ").astype(int) + rng.normal(0, 0.8, n) > 0.3).astype(int)
    return X, y.to_numpy(), X.person_id.to_numpy()


def _json(tmp_path):
    modelos = {
        "logreg_ridge": {"mejor_auc_cv": 0.70, "hiperparametros": {"modelo__C": 1.0, "modelo__class_weight": None}},
        "logreg_lasso": {"mejor_auc_cv": 0.71, "hiperparametros": {"modelo__C": 0.5, "modelo__class_weight": None}},
        "cart": {"mejor_auc_cv": 0.65, "hiperparametros": {"modelo__max_depth": 4, "modelo__min_samples_leaf": 10,
                                                           "modelo__ccp_alpha": 1e-6, "modelo__criterion": "gini",
                                                           "modelo__class_weight": None}},
        "naive_bayes": {"mejor_auc_cv": 0.66, "hiperparametros": {"modelo__var_smoothing": 1e-9}},
    }
    ruta = tmp_path / "best.json"
    ruta.write_text(json.dumps({"modelos": modelos}), encoding="utf-8")
    return ruta


def test_cargar_mejor_de_cada_familia(tmp_path):
    m = cargar_mejores(_json(tmp_path))
    assert set(m) == {"logreg_lasso", "cart", "naive_bayes"}  # Ridge y Lasso son una familia: gana Lasso
    assert set(cargar_mejores(_json(tmp_path), por_familia=False)) == {"logreg_ridge", "logreg_lasso", "cart", "naive_bayes"}


def test_instanciar_aplica_hiperparametros(tmp_path):
    m = cargar_mejores(_json(tmp_path))
    pipe = instanciar("cart", m["cart"]["hiperparametros"], PRE)
    assert pipe.named_steps["modelo"].max_depth == 4
    assert not hasattr(pipe.named_steps["preprocesador"], "transformers_")


def test_cache_rechaza_pliegues_no_agrupados():
    X, y, g = _datos()
    with pytest.raises(ValueError, match="person_id"):
        CacheDiagnostico.construir(PRE, X, y, g, KFold(5, shuffle=True, random_state=0))


def test_diagnostico_oof_y_brecha_equivalen_al_pipeline():
    X, y, g = _datos()
    pipe = instanciar("logreg_lasso", {"modelo__C": 0.5, "modelo__class_weight": None}, PRE)
    cache = CacheDiagnostico.construir(PRE, X, y, g, CV, dtype=np.float64)
    d = diagnosticar("logreg_lasso", pipe, cache)
    assert not np.isnan(d["oof"]).any()  # cada fila validada exactamente una vez
    assert d["brecha"] == pytest.approx(d["train_auc"] - d["val_auc"])
    # Mismo resultado que reentrenar el Pipeline completo en cada pliegue
    for r, (a, b) in zip(d["por_pliegue"].itertuples(), CV.split(X, y, groups=g)):
        p = clone(pipe).fit(X.iloc[a], y[a])
        assert r.val_auc == pytest.approx(roc_auc_score(y[b], p.predict_proba(X.iloc[b])[:, 1]), abs=1e-10)
        assert np.allclose(d["oof"][b], p.predict_proba(X.iloc[b])[:, 1])


def test_advertencia_explicita_si_brecha_supera_005(capsys):
    X, y, g = _datos()
    sin_poda = instanciar("cart", {"modelo__max_depth": 30, "modelo__min_samples_leaf": 1, "modelo__ccp_alpha": 1e-9,
                                   "modelo__criterion": "gini", "modelo__class_weight": None}, PRE)
    d = diagnosticar("cart", sin_poda, CacheDiagnostico.construir(PRE, X, y, g, CV))
    assert d["brecha"] > 0.05  # un árbol sin poda memoriza
    resumen = pd.DataFrame([{k: d[k] for k in ("modelo", "train_auc", "val_auc", "brecha")},
                            {"modelo": "estable", "train_auc": 0.72, "val_auc": 0.70, "brecha": 0.02}])
    with pytest.warns(AdvertenciaSobreajuste, match="cart"):
        sobre = advertir_brechas(resumen)
    assert sobre == ["cart"]
    assert "SOBREAJUSTE" in capsys.readouterr().out


def test_curva_aprendizaje_por_pacientes_y_reutiliza_el_100():
    X, y, g = _datos()
    pipe = instanciar("naive_bayes", {"modelo__var_smoothing": 1e-9}, PRE)
    cache = CacheDiagnostico.construir(PRE, X, y, g, CV)
    d = diagnosticar("naive_bayes", pipe, cache)
    c = curva_aprendizaje("naive_bayes", pipe, cache, diagnostico=d)
    assert sorted(c.fraccion.unique()) == [0.1, 0.25, 0.5, 1.0]
    n = c.groupby("fraccion").n_train.mean()
    assert n.is_monotonic_increasing
    uno = c[c.fraccion == 1.0].reset_index(drop=True)
    assert np.allclose(uno.val_auc, d["por_pliegue"].val_auc)


def test_ajuste_final_equivale_a_pipeline_fit_y_se_serializa(tmp_path):
    X, y, g = _datos()
    pipe = instanciar("logreg_lasso", {"modelo__C": 0.5, "modelo__class_weight": None}, PRE)
    pre = clone(PRE).fit(X, y)
    final = ajustar_final("logreg_lasso", pipe, pre, pre.transform(X), y)
    directo = clone(pipe).fit(X, y)
    assert np.allclose(final.predict_proba(X), directo.predict_proba(X))
    joblib.dump(final, tmp_path / "logreg_lasso_pipeline.joblib")
    cargado = joblib.load(tmp_path / "logreg_lasso_pipeline.joblib")
    assert np.array_equal(cargado.predict_proba(X.iloc[:5]), final.predict_proba(X.iloc[:5]))


def test_graficos(tmp_path):
    X, y, g = _datos()
    cache = CacheDiagnostico.construir(PRE, X, y, g, CV)
    oofs, curvas = {}, []
    for nombre, hp in [("logreg_lasso", {"modelo__C": 0.5, "modelo__class_weight": None}),
                       ("naive_bayes", {"modelo__var_smoothing": 1e-9})]:
        pipe = instanciar(nombre, hp, PRE)
        d = diagnosticar(nombre, pipe, cache)
        oofs[nombre] = d["oof"]
        curvas.append(curva_aprendizaje(nombre, pipe, cache, diagnostico=d))
    graficar_curvas_aprendizaje(pd.concat(curvas), tmp_path / "curvas.png")
    roc = graficar_roc(y, oofs, tmp_path / "roc.png")
    assert (tmp_path / "curvas.png").stat().st_size > 10_000 and (tmp_path / "roc.png").stat().st_size > 10_000
    assert roc.modelo.iloc[0] == "logreg_lasso"  # ordenado por AUC


# ---------------------------------------------------------------- auditoría de la brecha
from modulo_6.m6_4_diagnostico.auditoria_brecha import (  # noqa: E402
    auditar_knn, auditar_knn_por_paciente, elegir_con_brecha, proba_knn_sin_mismo_paciente, proba_knn_sin_si_mismo,
    ruta_regularizacion,
)


def test_knn_resustitucion_es_artefacto_y_loo_lo_corrige():
    X, y, g = _datos()
    pipe = instanciar("knn", {"modelo__n_neighbors": 15, "modelo__weights": "distance", "modelo__leaf_size": 30}, PRE)
    df = auditar_knn(pipe, CacheDiagnostico.construir(PRE, X, y, g, CV))
    assert (df.train_auc_resustitucion > 0.99).all()  # cada punto es su propio vecino a distancia 0
    assert df.brecha_loo.abs().mean() < df.brecha_resustitucion.mean() / 3  # LOO ≈ validación


def test_proba_loo_coincide_con_reentrenar_sin_el_punto():
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(1)
    Z, yy = rng.normal(size=(120, 3)), rng.integers(0, 2, 120)
    for w in ("uniform", "distance"):
        knn = KNeighborsClassifier(n_neighbors=7, weights=w).fit(Z, yy)
        p = proba_knn_sin_si_mismo(knn)
        for i in (0, 17, 99):
            sin_i = KNeighborsClassifier(n_neighbors=7, weights=w).fit(np.delete(Z, i, 0), np.delete(yy, i))
            assert p[i] == pytest.approx(sin_i.predict_proba(Z[[i]])[0, 1])


def test_ruta_de_regularizacion_elige_dentro_del_umbral():
    X, y, g = _datos()
    pipe = instanciar("cart", {"modelo__max_depth": 30, "modelo__min_samples_leaf": 1, "modelo__ccp_alpha": 1e-9,
                               "modelo__criterion": "gini", "modelo__class_weight": None}, PRE)
    ruta = ruta_regularizacion("cart", pipe, CacheDiagnostico.construir(PRE, X, y, g, CV),
                               [{"max_depth": 30, "min_samples_leaf": 1}, {"max_depth": 3, "min_samples_leaf": 30}])
    assert ruta.brecha.iloc[0] > ruta.brecha.iloc[1]
    elegido = elegir_con_brecha(ruta, 0.05)
    assert elegido.brecha <= 0.05
    with pytest.raises(ValueError, match="Ninguna"):
        elegir_con_brecha(ruta.assign(brecha=0.2), 0.05)


def test_proba_lopo_coincide_con_reentrenar_sin_el_paciente():
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(2)
    grupos = rng.integers(0, 40, 150)
    Z, yy = rng.normal(size=(150, 3)), rng.integers(0, 2, 150)
    Z[1] = Z[0]  # duplicado exacto de otra fila → caso de distancia 0
    for w in ("uniform", "distance"):
        knn = KNeighborsClassifier(n_neighbors=7, weights=w).fit(Z, yy)
        p, incompletas = proba_knn_sin_mismo_paciente(knn, grupos)
        assert incompletas == 0
        for i in (0, 1, 17, 99):
            fuera = grupos != grupos[i]
            sin_pac = KNeighborsClassifier(n_neighbors=7, weights=w).fit(Z[fuera], yy[fuera])
            assert p[i] == pytest.approx(sin_pac.predict_proba(Z[[i]])[0, 1])


def test_lopo_corrige_la_fuga_entre_filas_del_mismo_paciente():
    # Cada paciente aporta 4 filas casi idénticas con su etiqueta: el LOO por fila sigue inflado, el LOPO no.
    rng = np.random.default_rng(3)
    base, yb = rng.normal(size=(150, 4)), rng.integers(0, 2, 150)
    X = pd.DataFrame(np.repeat(base, 4, 0) + rng.normal(0, 0.01, (600, 4)), columns=["edad", "ocupacion", "n_dx", "x"])
    X["n_dx"] = X.n_dx.abs()
    cod = [f"dx_{i}" for i in range(20)]
    dx = [" ".join(rng.choice(cod, 3, replace=False)) for _ in range(150)]
    X["servicio"], X["sexo"], X["dx_codigos"] = "hospitalizacion", "F", np.repeat(dx, 4)
    y, g = np.repeat(yb, 4), np.repeat(np.arange(150), 4)
    pipe = instanciar("knn", {"modelo__n_neighbors": 9, "modelo__weights": "distance", "modelo__leaf_size": 30}, PRE)
    cache = CacheDiagnostico.construir(PRE, X, y, g, CV, dtype=np.float64)
    loo, lopo = auditar_knn(pipe, cache), auditar_knn_por_paciente(pipe, cache)
    assert loo.train_auc_loo.mean() > 0.9  # los gemelos del mismo paciente delatan la etiqueta
    assert abs(lopo.train_auc_lopo.mean() - loo.val_auc.mean()) < 0.1  # LOPO ≈ validación agrupada
