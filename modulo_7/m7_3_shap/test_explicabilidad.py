"""Pruebas de la subfase 7.3 con datos sintéticos (no usa datos reales).

    python -m pytest modulo_7/m7_3_shap -q
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("shap")
from lightgbm import LGBMClassifier  # noqa: E402

from modulo_5.m5_4_pipeline import EsquemaPipeline, construir_preprocesador  # noqa: E402
from modulo_7.m7_3_shap import (  # noqa: E402
    ETIQUETAS,
    agrupar_shap,
    efecto_diagnosticos,
    formatear_valor,
    explicacion,
    explicar,
    graficar_barras,
    graficar_barras_por_servicio,
    graficar_beeswarm,
    graficar_waterfall,
    importancia_global,
    mapa_grupos,
    perfil,
    sintesis_medica,
    valores_color,
)

ESQ = EsquemaPipeline(num=["edad", "hour_sin", "hour_cos"], num_log=["n_dx_historia"],
                      cat=["servicio", "sexo", "es_festivo", "es_fin_de_semana"], clinica="dx_codigos")


def _ajustado(n=1500, semilla=0):
    rng = np.random.default_rng(semilla)
    hora = rng.integers(0, 24, n)
    cod = np.array([f"dx_{i}" for i in range(30)])
    X = pd.DataFrame({
        "edad": rng.normal(50, 18, n), "hora_del_dia": hora,
        "hour_sin": np.sin(2 * np.pi * hora / 24), "hour_cos": np.cos(2 * np.pi * hora / 24),
        "n_dx_historia": rng.poisson(4, n).astype(float),
        "servicio": rng.choice(["hospitalizacion", "urgencias_observacion"], n), "sexo": rng.choice(["F", "M"], n),
        "es_festivo": rng.random(n) < 0.1, "es_fin_de_semana": rng.random(n) < 0.3,
        "dx_codigos": [" ".join(rng.choice(cod, rng.integers(0, 6), replace=False)) for _ in range(n)],
    })
    logit = 0.05 * (X.edad - 50) + 1.2 * (X.servicio == "hospitalizacion") + 0.8 * X.dx_codigos.str.contains("dx_3 ")
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    pre = construir_preprocesador(ESQ, n_componentes_svd=5, min_df_svd=2).fit(X, y)
    Z = pre.transform(X)
    m = LGBMClassifier(n_estimators=60, num_leaves=15, random_state=42, verbose=-1).fit(Z, y)
    return X, y, pre, Z, m


def test_mapa_grupos_fusiona_columnas_de_una_variable():
    X, y, pre, Z, m = _ajustado()
    nombres = list(pre.get_feature_names_out())
    g = mapa_grupos(nombres, ESQ)
    assert len(g) == len(nombres)
    assert set(g) == {"edad", "hora_del_dia", "n_dx_historia", "servicio", "sexo", "es_festivo", "es_fin_de_semana",
                      "dx_codigos"}
    assert g.count("hora_del_dia") == 2 and g.count("dx_codigos") == 5  # seno+coseno · 5 componentes SVD
    assert g[nombres.index("cat__es_fin_de_semana_True")] == "es_fin_de_semana"  # no se confunde con es_festivo
    with pytest.raises(ValueError):
        mapa_grupos(["otra__x"], ESQ)


def test_shap_agrupado_es_aditivo_y_reproduce_la_salida_del_modelo():
    X, y, pre, Z, m = _ajustado()
    phi, base = explicar(m, Z)
    assert phi.shape == Z.shape
    phi_g = agrupar_shap(phi, mapa_grupos(list(pre.get_feature_names_out()), ESQ))
    assert np.allclose(phi_g.sum(axis=1), phi.sum(axis=1))
    assert np.allclose(base + phi_g.sum(axis=1), m.predict(Z, raw_score=True), atol=1e-5)


def test_importancia_recupera_las_variables_y_su_direccion():
    X, y, pre, Z, m = _ajustado(n=4000)
    phi, base = explicar(m, Z)
    phi_g = agrupar_shap(phi, mapa_grupos(list(pre.get_feature_names_out()), ESQ))
    col = valores_color(X, list(phi_g.columns), clinica="dx_codigos")
    imp = importancia_global(phi_g, col)
    assert set(imp.variable.head(3)) >= {"edad", "servicio"}
    assert imp.porcentaje.sum() == pytest.approx(100)
    fila = imp.set_index("variable")
    assert fila.loc["edad", "spearman_valor_shap"] > 0.5 and fila.loc["servicio", "spearman_valor_shap"] > 0.5
    texto = sintesis_medica(imp)
    assert texto.count("\n") == 4
    assert not re.search(r"[a-z]+_[a-z]+", texto)  # sin nombres de columnas
    assert "mayor edad se asocia con un riesgo mayor" in texto.lower() or "hospitalización se asocia" in texto.lower()


def test_valores_color_y_perfil():
    X, y, pre, Z, m = _ajustado()
    col = valores_color(X, ["servicio", "sexo", "es_festivo", "hora_del_dia", "dx_codigos"], clinica="dx_codigos")
    assert set(col.servicio.unique()) <= {0.0, 1.0} and col.hora_del_dia.max() <= 23
    assert (col.dx_codigos == X.dx_codigos.str.split().str.len()).all()
    phi_g = pd.DataFrame({"servicio": np.ones(len(X)), "sexo": np.zeros(len(X))})
    medio, tipico = perfil(phi_g, col[["servicio", "sexo"]], np.arange(len(X)) < 10)
    assert medio.servicio == 1 and 0 <= tipico.servicio <= 1  # proporción en binarias


def test_todas_las_variables_reales_tienen_etiqueta_clinica():
    raiz = Path(__file__).resolve().parents[2]
    man = raiz / "models" / "manifest.json"
    if man.exists():
        cols = json.loads(man.read_text(encoding="utf-8"))["columnas_entrada"]
        esq = EsquemaPipeline().con_banderas_lab(cols)
    else:
        esq = EsquemaPipeline()
    variables = {v for v in esq.todas() if v not in ("hour_sin", "hour_cos", "dow_sin", "dow_cos")}
    variables |= {"hora_del_dia", "dia_semana"}
    assert not (variables - set(ETIQUETAS))


def test_graficos_a_300_dpi(tmp_path):
    from PIL import Image
    X, y, pre, Z, m = _ajustado()
    phi, base = explicar(m, Z)
    phi_g = agrupar_shap(phi, mapa_grupos(list(pre.get_feature_names_out()), ESQ))
    col = valores_color(X, list(phi_g.columns), clinica="dx_codigos")
    exp = explicacion(phi_g, base, col)
    graficar_beeswarm(exp, tmp_path / "b.png", "prueba")
    graficar_barras(exp, tmp_path / "bar.png", "prueba")
    medio, tipico = perfil(phi_g, col, np.arange(len(X)) < 50)
    graficar_waterfall(medio, tipico, base, tmp_path / "w.png", "prueba")
    imp = importancia_global(phi_g, col)
    graficar_barras_por_servicio({"a": imp, "b": imp}, tmp_path / "s.png")
    for f in ("b", "bar", "w", "s"):
        dpi = Image.open(tmp_path / f"{f}.png").info.get("dpi", (0, 0))[0]
        assert round(dpi) == 300


def test_efecto_diagnosticos_identifica_los_codigos_de_riesgo():
    rng = np.random.default_rng(4)
    cod = np.array([f"dx_{i}" for i in range(12)])
    textos = pd.Series([" ".join(rng.choice(cod, rng.integers(1, 4), replace=False)) for _ in range(3000)])
    lista = textos.str.split()
    phi_dx = (lista.apply(lambda l: "dx_3" in l) * 1.0 - lista.apply(lambda l: "dx_7" in l) * 0.8
              + rng.normal(0, 0.1, 3000)).to_numpy()
    dx = efecto_diagnosticos(phi_dx, textos, {"dx_3": "Diagnóstico tres", "dx_7": "Diagnóstico siete"}, min_n=50, top=2)
    assert dx[dx.efecto == "aumenta"].diagnostico.iloc[0] == "Diagnóstico tres"
    assert dx[dx.efecto == "reduce"].diagnostico.iloc[0] == "Diagnóstico siete"
    imp = pd.DataFrame([{"variable": "dx_codigos", "etiqueta": "Perfil de diagnósticos (historia clínica)",
                         "dominio": "historia clínica", "porcentaje": 30.0, "spearman_valor_shap": 0.0}])
    texto = sintesis_medica(imp, 1, dx)
    assert "Diagnóstico tres" in texto and "Diagnóstico siete" in texto


def test_formatear_valor_sin_codigos_internos():
    assert formatear_valor("servicio", 0.0) == "urgencias" and formatear_valor("servicio", 1.0) == "hospitalización"
    assert formatear_valor("edad", 48.5) == "48 años" and formatear_valor("horas_desde_ingreso", 6) == "6 h"
    assert formatear_valor("es_festivo", 0.25) == "25% sí" and formatear_valor("edad", float("nan")) == "—"
