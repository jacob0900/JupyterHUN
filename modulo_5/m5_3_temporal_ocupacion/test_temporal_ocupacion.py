"""Pruebas de la subfase 5.3 con datos sintéticos (no usa datos reales).

    python -m pytest modulo_5/m5_3_temporal_ocupacion -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from modulo_5.m5_3_temporal_ocupacion import (  # noqa: E402
    ConfigTemporal,
    build_temporal_and_occupancy_features,
    calendario_colombiano,
    unir_ocupacion_a_snapshots,
    variables_temporales,
    verificar_sin_fuga,
)

T = pd.Timestamp
SERVICIOS = ("hospitalizacion", "urgencias_observacion")


def _serie(horas: int = 400, inicio: str = "2024-01-01 00:00", semilla: int = 1) -> pd.DataFrame:
    """Serie servicio × hora con el contrato de `ocupacion_servicio_hora` y etiqueta y_t = 1{ocup(t+4) ≥ u}."""
    rng = np.random.default_rng(semilla)
    filas = []
    for s, u in zip(SERVICIOS, (0.95, 0.85)):
        h = pd.date_range(inicio, periods=horas, freq="h")
        censo = np.round(250 + 40 * np.sin(2 * np.pi * np.arange(horas) / 24) + rng.normal(0, 8, horas))
        cap = np.full(horas, 300.0)
        ocup = censo / cap
        y_fut = pd.Series(ocup).shift(-4)
        filas.append(pd.DataFrame({
            "servicio": s, "hora": h, "censo": censo, "llegadas": rng.poisson(10, horas), "capacidad": cap,
            "ocupacion": ocup, "umbral_ocupacion": u,
            "y_ocupacion_futura": y_fut, "y_target_saturacion": (y_fut >= u).astype(int),
        }))
    df = pd.concat(filas, ignore_index=True)
    return df.sample(frac=1, random_state=3).set_axis(np.arange(len(df)) * 7 + 11)  # desordenado, índice raro


def _cal() -> pd.DataFrame:
    # 2024-01-08 lunes festivo (Reyes, Emiliani); 2024-01-01 lunes festivo (Año Nuevo)
    d = pd.date_range("2023-12-25", "2024-02-29", freq="D")
    return pd.DataFrame({"fecha": d, "festivo": d.isin([T("2024-01-01"), T("2024-01-08")])})


# ---------------------------------------------------------------- cíclicas
def test_ciclicas_valores_exactos():
    t = pd.Series([T("2024-01-01 00:00"), T("2024-01-01 06:00"), T("2024-01-03 12:00"), T("2024-01-07 18:00")])
    v = variables_temporales(t, calendario_dia=_cal())
    np.testing.assert_allclose(v.hour_sin, [0, 1, 0, -1], atol=1e-12)
    np.testing.assert_allclose(v.hour_cos, [1, 0, -1, 0], atol=1e-12)
    # lunes = 0, miércoles = 2, domingo = 6
    np.testing.assert_allclose(v.dow_sin, np.sin(2 * np.pi * np.array([0, 0, 2, 6]) / 7))
    np.testing.assert_allclose(v.dow_cos, np.cos(2 * np.pi * np.array([0, 0, 2, 6]) / 7))


def test_ciclicas_circulo_unitario_y_continuidad_medianoche():
    t = pd.Series(pd.date_range("2024-01-01", periods=48, freq="30min"))
    v = variables_temporales(t, calendario_dia=_cal())
    np.testing.assert_allclose(v.hour_sin**2 + v.hour_cos**2, 1)
    np.testing.assert_allclose(v.dow_sin**2 + v.dow_cos**2, 1)
    # 23:30 y 00:00 quedan juntos en el círculo (distancia de media hora, no de 23.5 h)
    a = v.loc[47, ["hour_sin", "hour_cos"]].to_numpy()
    b = v.loc[0, ["hour_sin", "hour_cos"]].to_numpy()
    assert np.linalg.norm(a - b) < 0.14


def test_snapshot_usa_minutos():
    v = variables_temporales(pd.Series([T("2024-01-02 10:30")]), calendario_dia=_cal())
    assert v.hour_sin.iloc[0] == pytest.approx(np.sin(2 * np.pi * 10.5 / 24))


# ---------------------------------------------------------------- banderas
@pytest.mark.parametrize("h, pico", [(6, 0), (7, 1), (9, 1), (10, 0), (16, 0), (17, 1), (19, 1), (20, 0)])
def test_hora_pico_limites(h, pico):
    v = variables_temporales(pd.Series([T("2024-01-03") + pd.Timedelta(hours=h)]), calendario_dia=_cal())
    assert v.es_hora_pico_urgencias.iloc[0] == pico


def test_hora_pico_configurable():
    cfg = ConfigTemporal(horas_pico=((8, 12),))
    v = variables_temporales(pd.Series([T("2024-01-03 07:00"), T("2024-01-03 11:00")]), cfg, _cal())
    assert v.es_hora_pico_urgencias.tolist() == [0, 1]


def test_turnos():
    t = pd.Series([T("2024-01-03") + pd.Timedelta(hours=h) for h in (6, 7, 12, 13, 18, 19, 23)])
    assert variables_temporales(t, calendario_dia=_cal()).turno.tolist() == [
        "noche", "mañana", "mañana", "tarde", "tarde", "noche", "noche"]


def test_festivos_fin_de_semana_y_dia_habil():
    dias = ["2024-01-06", "2024-01-07", "2024-01-08", "2024-01-09", "2024-01-15", "2024-01-16"]
    #          sábado      domingo      lunes fest.  martes       lunes normal  martes normal
    v = variables_temporales(pd.Series(pd.to_datetime(dias) + pd.Timedelta(hours=10)), calendario_dia=_cal())
    assert v.es_fin_de_semana.tolist() == [1, 1, 0, 0, 0, 0]
    assert v.es_festivo.tolist() == [0, 0, 1, 0, 0, 0]
    assert v.es_dia_no_habil.tolist() == [1, 1, 1, 0, 0, 0]
    assert v.tras_festivo.tolist() == [0, 0, 0, 1, 0, 0]  # solo tras un festivo real
    assert v.primer_dia_habil.tolist() == [0, 0, 0, 1, 1, 0]  # incluye el lunes normal


def test_festivos_sin_calendario_usa_libreria_holidays():
    pytest.importorskip("holidays")
    cal = calendario_colombiano(pd.Series(pd.to_datetime(["2024-01-01", "2024-07-20", "2024-07-22"])))
    f = cal.set_index("fecha").es_festivo
    assert f[T("2024-01-01")] and f[T("2024-07-20")] and not f[T("2024-07-22")]


# ---------------------------------------------------------------- ocupación t − 1
def test_ocupacion_t1_es_exactamente_la_hora_anterior_del_mismo_servicio():
    df = _serie()
    out = build_temporal_and_occupancy_features(df, calendario_dia=_cal())
    for s in SERVICIOS:
        g = out[out.servicio == s].sort_values("hora")
        esperado = g.ocupacion.shift(1)
        pd.testing.assert_series_equal(g.ocupacion_t1, esperado, check_names=False)
        pd.testing.assert_series_equal(g.censo_t1, g.censo.shift(1).astype(float), check_names=False)
        assert np.isnan(g.ocupacion_t1.iloc[0])  # la primera hora no tiene pasado (no hereda del otro servicio)


def test_ventanas_moviles_solo_pasado():
    df = _serie()
    out = build_temporal_and_occupancy_features(df, calendario_dia=_cal())
    g = out[out.servicio == "hospitalizacion"].sort_values("hora").reset_index(drop=True)
    i = 100
    assert g.ocupacion_media_6h[i] == pytest.approx(g.ocupacion[i - 6:i].mean())  # horas t−6 … t−1
    assert g.ocupacion_max_24h[i] == pytest.approx(g.ocupacion[i - 24:i].max())
    assert g.llegadas_suma_4h[i] == pytest.approx(g.llegadas[i - 4:i].sum())
    assert g.ocupacion_tendencia_3h[i] == pytest.approx(g.ocupacion[i - 1] - g.ocupacion[i - 4])
    assert g.ocupacion_semana_previa[200] == pytest.approx(g.ocupacion[200 - 168])
    assert g.frac_horas_sobre_umbral_24h[i] == pytest.approx((g.ocupacion[i - 24:i] >= 0.95).mean())
    assert g.ocupacion_media_24h[:24].isna().all()  # ventana incompleta → NaN, no media parcial


def test_hueco_en_la_serie_no_toma_la_fila_anterior():
    df = _serie(horas=50)
    df = df[~((df.servicio == "hospitalizacion") & (df.hora == T("2024-01-01 10:00")))]
    out = build_temporal_and_occupancy_features(df, calendario_dia=_cal())
    fila = out[(out.servicio == "hospitalizacion") & (out.hora == T("2024-01-01 11:00"))]
    assert fila.ocupacion_t1.isna().all()  # la hora 10 falta: no se usa la de las 09


def test_ocupacion_se_calcula_como_censo_sobre_capacidad():
    df = _serie().drop(columns="ocupacion")
    out = build_temporal_and_occupancy_features(df, calendario_dia=_cal())
    g = out[out.servicio == "urgencias_observacion"].sort_values("hora")
    pd.testing.assert_series_equal(g.ocupacion_t1, (g.censo / g.capacidad).shift(1), check_names=False)


# ---------------------------------------------------------------- ANTI-FUGA (restricción del prompt)
def test_sin_fuga_del_objetivo_y_t():
    """Si se altera todo lo observado en t ≥ corte y se barajan las etiquetas y_*, las variables
    de las filas t ≤ corte no cambian. En particular la fila del corte no depende de ocupacion(t)
    (mismo instante que y_t) ni de ocupacion(t+4) (de donde sale y_t)."""
    df = _serie()
    res = verificar_sin_fuga(df, corte=T("2024-01-09 13:00"), calendario_dia=_cal())
    assert res["ok"], res["columnas_con_diferencias"]
    assert res["filas_comparadas"] > 300


def test_detector_de_fuga_si_detecta_una_variable_contaminada(monkeypatch):
    """Control positivo: si alguien olvida el shift(1), la verificación debe fallar."""
    import modulo_5.m5_3_temporal_ocupacion.temporal_ocupacion as m

    original = m.variables_ocupacion_t1

    def con_fuga(df, col_tiempo="hora", cfg=m.ConfigTemporal()):
        out = original(df, col_tiempo, cfg)
        out["ocupacion_sin_shift"] = df["censo"] / df["capacidad"]  # usa el instante t
        return out

    monkeypatch.setattr(m, "variables_ocupacion_t1", con_fuga)
    res = m.verificar_sin_fuga(_serie(), corte=T("2024-01-09 13:00"), calendario_dia=_cal())
    assert not res["ok"] and res["columnas_con_diferencias"] == ["ocupacion_sin_shift"]


def test_ninguna_variable_lee_columnas_y():
    df = _serie()
    con_y = build_temporal_and_occupancy_features(df, calendario_dia=_cal())
    sin_y = build_temporal_and_occupancy_features(df.drop(columns=["y_ocupacion_futura", "y_target_saturacion"]),
                                                  calendario_dia=_cal())
    nuevas = [c for c in sin_y.columns if c not in df.columns]
    pd.testing.assert_frame_equal(con_y[nuevas], sin_y[nuevas])


def test_variables_no_reproducen_la_etiqueta():
    """Ninguna variable nueva es una copia (ni una función monótona exacta) de y_ocupacion_futura."""
    df = _serie()
    out = build_temporal_and_occupancy_features(df, calendario_dia=_cal())
    num = [c for c in out.columns if c not in df.columns and out[c].dtype.kind == "f"]
    rho = out[num + ["y_ocupacion_futura"]].corr(method="spearman")["y_ocupacion_futura"].drop("y_ocupacion_futura")
    assert (rho.abs() < 0.999).all(), rho[rho.abs() >= 0.999]


# ---------------------------------------------------------------- forma
def test_no_modifica_entrada_y_conserva_indice():
    df = _serie()
    copia = df.copy()
    out = build_temporal_and_occupancy_features(df, calendario_dia=_cal())
    pd.testing.assert_frame_equal(df, copia)
    assert out.index.equals(df.index)
    assert len(out) == len(df)


def test_rechaza_filas_duplicadas():
    df = _serie(horas=30)
    with pytest.raises(ValueError, match="repetidas"):
        build_temporal_and_occupancy_features(pd.concat([df, df.iloc[:1]]), calendario_dia=_cal())


def test_snapshots_solo_parte_temporal_y_union_con_servicio():
    serie = build_temporal_and_occupancy_features(_serie(), calendario_dia=_cal())
    snaps = pd.DataFrame({
        "snapshot_id": [1, 2],
        "servicio": ["hospitalizacion", "urgencias_observacion"],
        "snapshot_time": [T("2024-01-05 10:37"), T("2024-01-06 00:05")],
    }, index=[50, 9])
    out = build_temporal_and_occupancy_features(snaps, calendario_dia=_cal())
    assert "ocupacion_t1" not in out.columns and "hour_sin" in out.columns
    srv = unir_ocupacion_a_snapshots(snaps, serie)
    assert srv.index.equals(snaps.index)
    # τ = 10:37 → fila de las 10:00 → ocupación de las 09:00 (estrictamente antes de τ)
    esperado = serie[(serie.servicio == "hospitalizacion") & (serie.hora == T("2024-01-05 09:00"))].ocupacion.iloc[0]
    assert srv.loc[50, "srv_ocupacion_t1"] == pytest.approx(esperado)
    esperado2 = serie[(serie.servicio == "urgencias_observacion") & (serie.hora == T("2024-01-05 23:00"))].ocupacion.iloc[0]
    assert srv.loc[9, "srv_ocupacion_t1"] == pytest.approx(esperado2)
