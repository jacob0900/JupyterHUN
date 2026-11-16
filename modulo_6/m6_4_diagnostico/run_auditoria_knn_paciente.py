"""Auditoría de KNN, parte 2: AUC de entrenamiento leave-one-patient-out (se corre después de run_auditoria_m6_4.py).

El LOO por fila (0,762) seguía 0,057 por encima de la validación: el 99 % de las filas comparte paciente con
otras filas casi idénticas (snapshots τ = 0/2/6 h) y con la misma etiqueta. La validación agrupada nunca ve
al mismo paciente, así que el AUC de entrenamiento comparable debe excluir **al paciente completo**.

Si aun así la brecha supera 0,05, regulariza KNN (más vecinos y pesos uniformes), elige la mejor validación
con brecha ≤ 0,05, lo reempaqueta y actualiza el benchmark, el manifiesto, el OOF y la ROC.

Uso (desde la raíz del proyecto):
    python modulo_6/m6_4_diagnostico/run_auditoria_knn_paciente.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from modulo_5.m5_4_pipeline import construir_preprocesador, preparar_X  # noqa: E402
from modulo_6.m6_1_cv_agrupada import PlieguesPredefinidos  # noqa: E402
from modulo_6.m6_4_diagnostico import UMBRAL_BRECHA, CacheDiagnostico, ajustar_final, cargar_mejores, graficar_roc, instanciar  # noqa: E402
from modulo_6.m6_4_diagnostico.auditoria_brecha import proba_knn_sin_mismo_paciente  # noqa: E402

TARGET = "y_target_los_largo"
# Ruta de regularización de KNN si la brecha LOPO supera el umbral (de menos a más suavizado).
RUTA_KNN = [{"weights": "uniform"}, {"n_neighbors": 300, "weights": "uniform"}, {"n_neighbors": 500, "weights": "uniform"}]


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def evaluar(modelo, cache):
    """AUC train (leave-one-patient-out) y val por pliegue, más el OOF de validación."""
    filas, oof = [], {}
    for k, p in enumerate(cache.pliegues, 1):
        m = clone(modelo).fit(p.Z_tr, p.y_tr)
        proba, inc = proba_knn_sin_mismo_paciente(m, p.g_tr)
        s_va = m.predict_proba(p.Z_va)[:, 1]
        oof[k] = (p.i_va, s_va)
        filas.append({"pliegue": k, "train_auc_lopo": roc_auc_score(p.y_tr, proba),
                      "val_auc": roc_auc_score(p.y_va, s_va), "filas_con_menos_de_k_vecinos": inc})
    df = pd.DataFrame(filas)
    df["brecha_lopo"] = df.train_auc_lopo - df.val_auc
    return df, oof


def main() -> None:
    warnings.filterwarnings("ignore")
    log("Cargando datos")
    X, esquema = preparar_X(ROOT)
    tr = X[X.en_periodo_estable.astype(bool) & X.es_entrenamiento.astype(bool)].reset_index(drop=True)
    tr["pliegue"] = pd.read_parquet(ROOT / "data" / "m6" / "pliegues_m6_1.parquet").pliegue.to_numpy()
    cols = esquema.todas()
    pre = construir_preprocesador(esquema)
    mejores = cargar_mejores(ROOT / "outputs" / "m6" / "best_hyperparameters.json")
    hp = dict(mejores["knn"]["hiperparametros"])
    tab = ROOT / "outputs" / "m6" / "tablas"

    personas = tr.person_id.drop_duplicates().sample(frac=0.2, random_state=42)  # misma submuestra que la 6.3/6.4
    sub = tr[tr.person_id.isin(personas)].reset_index(drop=True)
    cache = CacheDiagnostico.construir(pre, sub[cols], sub[TARGET].to_numpy(), sub.person_id.to_numpy(),
                                       PlieguesPredefinidos(sub.pliegue))
    log(f"Submuestra 20 %: {len(sub):,} filas")

    t0 = time.time()
    base = instanciar("knn", hp, pre).named_steps["modelo"]
    df, _ = evaluar(base, cache)
    aud = pd.read_csv(tab / "m6_4_auditoria_knn.csv").drop(columns=["train_auc_lopo", "brecha_lopo",
                                                                     "filas_con_menos_de_k_vecinos"], errors="ignore")
    aud = aud.merge(df[["pliegue", "train_auc_lopo", "brecha_lopo", "filas_con_menos_de_k_vecinos"]], on="pliegue")
    aud.to_csv(tab / "m6_4_auditoria_knn.csv", index=False)
    log(f"KNN de la 6.3 ({(time.time() - t0) / 60:.1f} min): train LOPO {df.train_auc_lopo.mean():.4f} · "
        f"val {df.val_auc.mean():.4f} · brecha LOPO {df.brecha_lopo.mean():+.4f} "
        f"(filas con < k vecinos: {df.filas_con_menos_de_k_vecinos.sum()})")

    hp_final, d_final, ruta = hp, df, [{"config": "6.3", **{k.replace('modelo__', ''): v for k, v in hp.items()},
                                         "train_auc_lopo": df.train_auc_lopo.mean(), "val_auc": df.val_auc.mean(),
                                         "brecha_lopo": df.brecha_lopo.mean()}]
    oof_final = None
    if df.brecha_lopo.mean() > UMBRAL_BRECHA:
        log("Brecha LOPO > 0,05: ruta de regularización de KNN")
        mejor = None
        for i, cfg in enumerate(RUTA_KNN, 1):
            t0 = time.time()
            d, oof = evaluar(clone(base).set_params(**cfg), cache)
            fila = {"config": f"r{i}", **{**{k.replace('modelo__', ''): v for k, v in hp.items()}, **cfg},
                    "train_auc_lopo": d.train_auc_lopo.mean(), "val_auc": d.val_auc.mean(), "brecha_lopo": d.brecha_lopo.mean()}
            ruta.append(fila)
            log(f"  {cfg} ({(time.time() - t0) / 60:.1f} min): train LOPO {fila['train_auc_lopo']:.4f} · "
                f"val {fila['val_auc']:.4f} · brecha {fila['brecha_lopo']:+.4f}")
            if fila["brecha_lopo"] <= UMBRAL_BRECHA and (mejor is None or fila["val_auc"] > mejor[0]["val_auc"]):
                mejor = (fila, cfg, d, oof)
        if mejor is None:
            raise SystemExit("Ninguna configuración de KNN cumple brecha ≤ 0,05: ampliar RUTA_KNN")
        _, cfg, d_final, oof_final = mejor
        hp_final = {**hp, **{f"modelo__{k}": v for k, v in cfg.items()}}
    pd.DataFrame(ruta).to_csv(tab / "m6_4_ruta_regularizacion_knn.csv", index=False)

    # -------- actualización de artefactos
    bench_p = ROOT / "outputs" / "m6" / "model_benchmark_summary.csv"
    bench = pd.read_csv(bench_p)
    j = bench.index[bench.modelo == "knn"][0]
    bench.loc[j, "train_auc"] = d_final.train_auc_lopo.mean()
    bench.loc[j, "val_auc"] = d_final.val_auc.mean()
    bench.loc[j, "val_auc_std"] = d_final.val_auc.std(ddof=1)
    bench.loc[j, "brecha_train_val"] = d_final.brecha_lopo.mean()
    bench.loc[j, "metodo_train_auc"] = "leave-one-patient-out (excluye todas las filas del mismo paciente)"
    man_p = ROOT / "models" / "manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8"))

    if oof_final is not None:  # KNN regularizado: reempaquetar, OOF y ROC
        log("Reempaquetando KNN regularizado")
        oof_p = ROOT / "data" / "m6" / "oof_m6_4.parquet"
        oof = pd.read_parquet(oof_p)
        y_full = tr[TARGET].to_numpy()
        s = np.full(len(tr), np.nan)
        idx_sub = np.flatnonzero(tr.person_id.isin(personas).to_numpy())  # filas de la submuestra en orden
        for i_va, s_va in oof_final.values():
            s[idx_sub[i_va]] = s_va
        oof["oof_knn"] = s
        oof.to_parquet(oof_p, index=False)
        ok = ~np.isnan(s)
        estr = (tr.servicio + "_t" + tr.horas_desde_ingreso.astype(str)).to_numpy()
        dfe = pd.DataFrame({"y": y_full[ok], "s": s[ok], "e": estr[ok]})
        bench.loc[j, "auc_oof"] = roc_auc_score(dfe.y, dfe.s)
        bench.loc[j, "auc_oof_servicio_tau"] = float(np.mean([roc_auc_score(g.y, g.s) for _, g in dfe.groupby("e")
                                                              if g.y.nunique() == 2]))
        pre_sub = clone(pre).fit(sub[cols], sub[TARGET].to_numpy())
        final = ajustar_final("knn", instanciar("knn", hp_final, pre), pre_sub,
                              pre_sub.transform(sub[cols]).astype(np.float32), sub[TARGET].to_numpy())
        ruta_m = ROOT / "models" / "knn_pipeline.joblib"
        joblib.dump(final, ruta_m, compress=3)
        joblib.load(ruta_m).predict(sub[cols].iloc[:5])
        bench.loc[j, "mb"] = round(ruta_m.stat().st_size / 1e6, 1)
        bench.loc[j, "hiperparametros"] = json.dumps(hp_final, default=str)
        man["modelos"]["knn"]["hiperparametros"] = hp_final
        graficar_roc(y_full, {c.replace("oof_", ""): oof[c].to_numpy() for c in oof.columns if c.startswith("oof_")},
                     ROOT / "outputs" / "m6" / "figuras" / "m6_4_roc_comparativa.png")
        fin_p = ROOT / "outputs" / "m6" / "hiperparametros_finales.json"
        fin = json.loads(fin_p.read_text(encoding="utf-8"))
        fin["modelos"]["knn"] = hp_final
        fin["nota"] = ("Hiperparámetros de la 6.3, salvo random_forest y knn regularizados en la auditoría de la 6.4 "
                       "(brecha ≤ 0,05)")
        fin_p.write_text(json.dumps(fin, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    bench["sobreajuste_gt_005"] = bench.brecha_train_val > UMBRAL_BRECHA
    bench = bench.sort_values("val_auc", ascending=False)
    bench.to_csv(bench_p, index=False)
    for _, r in bench.iterrows():
        if r.modelo in man["modelos"]:
            for k in ("train_auc", "val_auc", "val_auc_std", "brecha_train_val", "sobreajuste_gt_005", "auc_oof",
                      "auc_oof_servicio_tau", "mb", "metodo_train_auc"):
                v = r[k]
                man["modelos"][r.modelo][k] = v.item() if hasattr(v, "item") else v
    man.setdefault("auditoria_brecha", {})["knn"] = (
        "AUC de entrenamiento leave-one-patient-out (el de resustitución ≈ 1 es un artefacto de weights='distance' y el "
        "LOO por fila sigue inflado por las filas gemelas del mismo paciente)"
        + ("; regularizado: " + json.dumps({k: v for k, v in hp_final.items() if k not in hp or hp[k] != v})
           if oof_final is not None else ""))
    man_p.write_text(json.dumps(man, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\nResumen auditado (model_benchmark_summary.csv):")
    print(bench[["modelo", "train_auc", "val_auc", "brecha_train_val", "sobreajuste_gt_005", "auc_oof",
                 "metodo_train_auc"]].round(4).to_string(index=False))
    log("Listo")


if __name__ == "__main__":
    main()
