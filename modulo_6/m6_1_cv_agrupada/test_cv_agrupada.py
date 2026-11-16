"""Pruebas de la subfase 6.1 con datos sintéticos (no usa datos reales).

    python -m pytest modulo_6/m6_1_cv_agrupada -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modulo_6.m6_1_cv_agrupada import (  # noqa: E402
    PlieguesPredefinidos,
    asignar_pliegues,
    crear_cv,
    verify_group_isolation,
)


def _datos(n_personas: int = 300, semilla: int = 0):
    """Cada persona tiene 1–4 episodios × 3 snapshots (τ = 0, 2, 6 h), como en el HUN."""
    rng = pd.Series(range(n_personas)).sample(frac=1, random_state=semilla)
    filas = []
    for p in rng:
        n_ep = 1 + (p * 7919) % 4
        for e in range(n_ep):
            pos = int((p + e) % 4 == 0)
            for tau in (0, 2, 6):
                filas.append({"person_id": 1000 + p, "episodio": e, "tau": tau, "x": p % 13, "y": pos})
    df = pd.DataFrame(filas)
    return df[["x", "tau"]], df["y"], df["person_id"]


def test_cinco_pliegues_sin_pacientes_compartidos():
    X, y, g = _datos()
    rep = verify_group_isolation(X, y, g, crear_cv(), n_splits=5)
    assert len(rep) == 5
    assert (rep.pacientes_compartidos == 0).all()
    assert rep.n_val.sum() == len(X)  # cada fila valida una sola vez
    assert (rep.n_train + rep.n_val == len(X)).all()
    assert rep.pacientes_val.sum() == g.nunique()


def test_proporcion_positiva_exacta_y_balanceada():
    X, y, g = _datos()
    cv = crear_cv()
    rep = verify_group_isolation(X, y, g, cv)
    for r, (i_tr, i_va) in zip(rep.itertuples(), cv.split(X, y, groups=g)):
        assert r.prop_pos_val == pytest.approx(y.iloc[i_va].mean())
        assert r.prop_pos_train == pytest.approx(y.iloc[i_tr].mean())
    assert rep.dif_prop_val_vs_global.abs().max() < 0.05


@pytest.mark.parametrize("cv_malo", [KFold(5, shuffle=True, random_state=0), StratifiedKFold(5, shuffle=True, random_state=0)])
def test_cv_no_agrupada_lanza_value_error(cv_malo):
    """Control positivo: una división que ignora person_id reparte snapshots de la misma persona → ValueError."""
    X, y, g = _datos()
    with pytest.raises(ValueError, match="person_id aparecen en entrenamiento y validación"):
        verify_group_isolation(X, y, g, cv_malo)


def test_errores_de_entrada():
    X, y, g = _datos(50)
    with pytest.raises(ValueError, match="mismo número de filas"):
        verify_group_isolation(X, y.iloc[:-1], g, crear_cv())
    g2 = g.astype("float").copy()
    g2.iloc[0] = float("nan")
    with pytest.raises(ValueError, match="sin person_id"):
        verify_group_isolation(X, y, g2, crear_cv())
    with pytest.raises(ValueError, match="Se esperaban 4"):
        verify_group_isolation(X, y, g, crear_cv(), n_splits=4)


def test_particion_incompleta_lanza_value_error():
    class Incompleta:
        def split(self, X, y=None, groups=None):
            n = len(X)
            yield list(range(n // 2, n)), list(range(0, n // 4))  # la otra cuarta parte nunca valida

    X, y, g = _datos(40)
    X, y = X.iloc[: len(X)], y
    g = pd.Series(range(len(X)))  # grupos únicos: no hay traslape, solo falta cobertura
    with pytest.raises(ValueError, match="exactamente una vez"):
        verify_group_isolation(X, y, g, Incompleta())


def test_reproducible_e_independiente_del_orden():
    X, y, g = _datos()
    a = asignar_pliegues(g, y, crear_cv())
    b = asignar_pliegues(g, y, crear_cv())
    assert a.equals(b)
    # Todas las filas de una persona tienen el mismo pliegue
    assert pd.DataFrame({"g": g, "k": a}).groupby("g").k.nunique().max() == 1


def test_pliegues_predefinidos_reproducen_la_cv():
    X, y, g = _datos()
    cv = crear_cv()
    pl = asignar_pliegues(g, y, cv)
    pre = PlieguesPredefinidos(pl)
    assert pre.get_n_splits() == 5
    for (a_tr, a_va), (b_tr, b_va) in zip(cv.split(X, y, groups=g), pre.split(X, y)):
        assert sorted(a_va) == sorted(b_va) and sorted(a_tr) == sorted(b_tr)
    verify_group_isolation(X, y, g, pre, n_splits=5)
    assert len(cross_val_score(LogisticRegression(), X, y, cv=pre)) == 5  # compatible con scikit-learn
