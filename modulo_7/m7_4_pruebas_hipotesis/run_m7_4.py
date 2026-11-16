"""Subfase 7.4 sobre los datos reales: tres pruebas formales de superioridad sobre el OOF (desarrollo).

Entrada: `data/m7/probabilidades_oof_m7_1.parquet` (OOF de la 6.4; la SVM ya en probabilidad, con Platt cruzado).
La prueba temporal no se usa (es de evaluación única, en la 7.2).

Reglas fijadas ANTES de ver los resultados:
- Ganador y top 2 = los dos primeros por AUC-PR en la 7.1 (lightgbm, random_forest).
- Prueba 1: DeLong por conglomerados de paciente (principal), Holm sobre las 6 comparaciones; IC BCa por bootstrap
  pareado por paciente (1 000 réplicas) y jackknife de 100 grupos. Relevancia práctica: ΔAUC con IC BCa ≥ 0,01.
- Prueba 2: Friedman + Nemenyi sobre AUC-ROC por pliegue (5 × 7) en las filas comunes a los 7 modelos (las de KNN).
- Prueba 3: Wilcoxon top 1 vs top 2 en 4 métricas (AUC-ROC, AUC-PR, Brier, log-loss), Holm entre métricas.
  Decisión sobre 50 particiones disjuntas de pacientes (10 por pliegue). Con los 5 pliegues se reporta también,
  pero su p exacto mínimo bilateral es 0,0625: no puede llegar a p < 0,05.

Uso (desde la raíz del proyecto):
    python modulo_7/m7_4_pruebas_hipotesis/run_m7_4.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modulo_7.m7_4_pruebas_hipotesis import (  # noqa: E402
    ALFA,
    DELTA_RELEVANTE,
    comparar_ganador,
    friedman_nemenyi,
    graficar_cd,
    graficar_heatmaps,
    matriz_p_delong,
    veredicto,
    wilcoxon_metricas,
)

TARGET = "y_target_los_largo"
OUT, TAB, FIG = ROOT / "outputs" / "m7", ROOT / "outputs" / "m7" / "tablas", ROOT / "outputs" / "m7" / "figuras"
METRICAS = {"auc_roc": True, "auc_pr": True, "brier": False, "log_loss": False}  # True: mayor es mejor
SUBBLOQUES = 10


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def _metricas(y, p) -> dict:
    return {"auc_roc": roc_auc_score(y, p), "auc_pr": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))}


def bloques_largos(df: pd.DataFrame, modelos: list[str], col_bloque: str) -> pd.DataFrame:
    filas = []
    for b, d in df.groupby(col_bloque):
        for m in modelos:
            ok = d[f"p_{m}"].notna()
            for met, v in _metricas(d[TARGET][ok], d[f"p_{m}"][ok]).items():
                filas.append({"bloque": b, "modelo": m, "metrica": met, "valor": v})
    return pd.DataFrame(filas)


def main() -> None:
    for d in (TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ROOT / "data" / "m7" / "probabilidades_oof_m7_1.parquet")
    resumen = pd.read_csv(OUT / "multimetric_evaluation_summary.csv")
    orden = resumen.modelo.tolist()
    completos = resumen[resumen.n == len(df)].modelo.tolist()
    ganador, top2 = completos[0], completos[1]
    y, g = df[TARGET].to_numpy(), df.person_id.to_numpy()
    p = {m: df[f"p_{m}"].to_numpy(float) for m in orden}
    log(f"OOF: {len(df):,} snapshots · {df.person_id.nunique():,} pacientes · ganador {ganador} · top 2 {top2}")
    res = {"alfa": ALFA, "delta_relevante": DELTA_RELEVANTE, "ganador": ganador, "top2": top2}

    # ------------------------------------------------------------------ Prueba 1: DeLong + BCa
    t0 = time.time()
    p1 = comparar_ganador(y, p, ganador, g, n_iter=1000, semilla=42, n_jack=100)
    p1.to_csv(TAB / "m7_4_prueba1_delong.csv", index=False)
    crudo, holm = matriz_p_delong(y, p, g, orden)
    crudo.to_csv(TAB / "m7_4_matriz_p_delong_crudo.csv")
    holm.to_csv(TAB / "m7_4_matriz_p_delong_holm.csv")
    log(f"Prueba 1 lista ({(time.time() - t0) / 60:.1f} min)")

    # ------------------------------------------------------------------ Prueba 2: Friedman + Nemenyi
    comun = df[df[[f"p_{m}" for m in orden]].notna().all(axis=1)]
    mat = pd.DataFrame({m: comun.groupby("pliegue")[[TARGET, f"p_{m}"]].apply(lambda d, m=m: roc_auc_score(d[TARGET], d[f"p_{m}"]))
                        for m in orden})
    fr = friedman_nemenyi(mat)
    mat.to_csv(TAB / "m7_4_prueba2_auc_por_pliegue.csv")
    fr["p_nemenyi"].to_csv(TAB / "m7_4_prueba2_nemenyi_p.csv")
    graficar_cd(fr["rangos_medios"], fr["cd"], FIG / "m7_4_diagrama_diferencia_critica.png",
                f"Diferencia crítica de Nemenyi (α = 0,05) · AUC-ROC en {fr['n_bloques']} pliegues · "
                f"Friedman p = {fr['p']:.2g}")
    # sensibilidad: 6 modelos en todas las filas (sin KNN)
    mat6 = pd.DataFrame({m: df.groupby("pliegue")[[TARGET, f"p_{m}"]].apply(lambda d, m=m: roc_auc_score(d[TARGET], d[f"p_{m}"]))
                         for m in completos})
    fr6 = friedman_nemenyi(mat6)

    # ------------------------------------------------------------------ Prueba 3: Wilcoxon top1 vs top2
    df["bloque50"] = df.pliegue.astype(int) * SUBBLOQUES + (
        pd.util.hash_pandas_object(df.person_id, index=False).to_numpy() % SUBBLOQUES).astype(int)
    b5 = bloques_largos(df, [ganador, top2], "pliegue")
    b50 = bloques_largos(df, [ganador, top2], "bloque50")
    w5 = wilcoxon_metricas(b5, ganador, top2, METRICAS).assign(particiones="5 pliegues")
    w50 = wilcoxon_metricas(b50, ganador, top2, METRICAS).assign(particiones="50 particiones de pacientes (decisión)")
    w = pd.concat([w50, w5])
    w.to_csv(TAB / "m7_4_prueba3_wilcoxon.csv", index=False)
    pd.concat([b5.assign(particiones="5"), b50.assign(particiones="50")]).to_csv(
        TAB / "m7_4_prueba3_metricas_por_particion.csv", index=False)

    auc = {m: roc_auc_score(y[~np.isnan(p[m])], p[m][~np.isnan(p[m])]) for m in orden}
    auc_c = {m: roc_auc_score(comun[TARGET], comun[f"p_{m}"]) for m in orden}
    delta = pd.DataFrame([[(f"Δ={(auc_c if 'knn' in (a, b) else auc)[a] - (auc_c if 'knn' in (a, b) else auc)[b]:+.3f}"
                            if a != b else "") for b in orden] for a in orden], index=orden, columns=orden)
    rmed = fr["rangos_medios"]
    drango = pd.DataFrame([[(f"ΔR={rmed[a] - rmed[b]:+.1f}" if a != b else "") for b in orden] for a in orden],
                          index=orden, columns=orden)
    t_delong = "DeLong por conglomerados (ΔAUC-ROC fila − columna), Holm sobre 21 pares"
    t_nem = f"Nemenyi post-hoc (rango medio fila − columna; {fr['n_bloques']} pliegues, filas comunes)"
    graficar_heatmaps({t_delong: holm, t_nem: fr["p_nemenyi"]}, FIG / "m7_4_heatmap_pvalores.png",
                      "Matrices de p-valores entre los 7 modelos", anotaciones={t_delong: delta, t_nem: drango})

    res.update({
        "prueba1": p1.to_dict(orient="records"),
        "prueba2": {k: (v.to_dict() if isinstance(v, (pd.Series, pd.DataFrame)) else v) for k, v in fr.items()},
        "prueba2_sensibilidad_6_modelos_todas_las_filas": {"chi2": fr6["chi2"], "p": fr6["p"], "w_kendall": fr6["w_kendall"],
                                                          "rangos_medios": fr6["rangos_medios"].to_dict()},
        "prueba3": w.to_dict(orient="records"), "fecha": datetime.now().isoformat(timespec="seconds")})
    (OUT / "m7_4_resultados.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float),
                                              encoding="utf-8")

    # ------------------------------------------------------------------ consola: resultados y veredictos
    r4 = lambda x: x.round(4).to_string(index=False)  # noqa: E731
    print("\n=== PRUEBA 1 · DeLong pareado (ΔAUC-ROC = ganador − baseline) ===")
    print(r4(p1[["baseline", "n", "auc_ganador", "auc_baseline", "delta_auc", "ic95_bca_inf", "ic95_bca_sup",
                 "z_conglomerados", "log10_p_conglomerados", "inflacion_se", "p_holm"]]))
    print("  (p = 0,0 en pantalla es desbordamiento numérico: ver z y log10 p; el SE por conglomerados es "
          f"{p1.inflacion_se.mean():.2f}× el del DeLong clásico, que ignora la correlación entre snapshots del paciente)")
    for _, r in p1.iterrows():
        rel = ("relevante (IC BCa ≥ 0,01)" if r.relevante else
               f"magnitud pequeña: IC BCa [{r.ic95_bca_inf:.4f}; {r.ic95_bca_sup:.4f}] no supera 0,01")
        print(f"  {ganador} vs {r.baseline}: {veredicto(r.p_holm, r.delta_auc > 0)} · ΔAUC = {r.delta_auc:+.4f} · {rel}")

    print(f"\n=== PRUEBA 2 · Friedman + Nemenyi (AUC-ROC, {fr['n_bloques']} pliegues × {fr['k']} modelos, "
          f"{len(comun):,} filas comunes) ===")
    print(mat.round(4).to_string())
    idf = ("∞ (concordancia perfecta entre pliegues)" if not np.isfinite(fr["iman_davenport_F"])
           else f"{fr['iman_davenport_F']:.2f}, p = {fr['p_iman_davenport']:.3g}")
    print(f"Friedman χ² = {fr['chi2']:.2f}, p = {fr['p']:.3g} · Iman-Davenport F = {idf} · "
          f"W de Kendall = {fr['w_kendall']:.3f} · CD Nemenyi = {fr['cd']:.2f}")
    print("Rangos medios: " + ", ".join(f"{m} {v:.1f}" for m, v in fr["rangos_medios"].items()))
    print(f"  Friedman: {veredicto(fr['p'])}")
    if fr["p"] < ALFA:
        pn = fr["p_nemenyi"].loc[ganador].drop(ganador)
        for m, v in pn.items():
            print(f"  Nemenyi {ganador} vs {m}: p = {v:.3g} → {veredicto(v)}")
    print(f"  (Sensibilidad, 6 modelos en todas las filas: Friedman p = {fr6['p']:.3g}, W = {fr6['w_kendall']:.3f})")

    print(f"\n=== PRUEBA 3 · Wilcoxon de rangos con signo, {ganador} vs {top2} (Holm entre 4 métricas) ===")
    print(r4(w[["particiones", "metrica", "media_top1", "media_top2", "mediana_diferencia", "a_mejor_en", "n_pares",
                "p", "p_holm", "biserial_rangos", "p_minimo_alcanzable"]]))
    for _, r in w.iterrows():
        nota = " (con 5 pares el p mínimo posible es 0,0625)" if r.n_pares <= 5 else ""
        print(f"  [{r.particiones}] {r.metrica}: {veredicto(r.p_holm, r.mediana_diferencia > 0)}{nota}")

    print("\n=== VEREDICTO FINAL ===")
    sup = p1[p1.superior]
    print(f"Prueba 1 (DeLong por conglomerados + Holm): {ganador} supera estadísticamente a "
          f"{len(sup)}/{len(p1)} baselines" + (f" ({', '.join(sup.baseline)})" if len(sup) else "") + ".")
    print(f"Prueba 2 (Friedman + Nemenyi): {veredicto(fr['p'])}; el post-hoc separa a {ganador} de "
          f"{int((fr['p_nemenyi'].loc[ganador].drop(ganador) < ALFA).sum())}/{fr['k'] - 1} modelos (potencia limitada con 5 pliegues).")
    dec = w50.set_index("metrica")
    print(f"Prueba 3 (Wilcoxon, 50 particiones): {ganador} vs {top2} → "
          + "; ".join(f"{m}: {'superior' if dec.loc[m, 'top1_superior'] else 'sin diferencia demostrada'} "
                      f"(p_Holm = {dec.loc[m, 'p_holm']:.3g})" for m in METRICAS))
    log("Listo")


if __name__ == "__main__":
    main()
