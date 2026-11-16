"""Subfase 6.4 · Diagnóstico de sobreajuste, curvas de aprendizaje y empaquetado (.joblib).

1. **Brecha de generalización.** Para cada modelo optimizado (6.3) se reentrena el Pipeline en
   los 5 pliegues de la 6.1, ahora con **todo** el entrenamiento estable (la búsqueda usó el
   20 % de los pacientes). En cada pliegue se mide el AUC-ROC sobre el entrenamiento del pliegue
   y sobre su validación; la brecha es `train_auc − val_auc` (media de los 5 pliegues). Una
   brecha > 0,05 se advierte explícitamente: indica memorización del entrenamiento.
   Ojo: en ensambles de árboles una brecha alta es habitual (cada árbol ve sus datos) y no
   implica peor validación. Lo que decide es el AUC de validación; la brecha es un diagnóstico.
2. **Predicciones fuera de pliegue (OOF).** Las predicciones de validación de los 5 pliegues
   cubren cada fila del entrenamiento una sola vez. Con ellas se dibuja la ROC comparativa y se
   alimenta M7 (calibración y pruebas de hipótesis pareadas sobre los mismos pacientes).
3. **Curvas de aprendizaje.** Para los 3 mejores modelos: AUC de entrenamiento y validación
   con 10 %, 25 %, 50 % y 100 % del entrenamiento de cada pliegue. El submuestreo se hace por
   pacientes completos, y la validación siempre es el pliegue completo.
4. **Empaquetado.** Pipeline final (preprocesador + modelo) ajustado con todo el entrenamiento,
   serializado en `models/<modelo>_pipeline.joblib`.

Detalles anti-fuga
------------------
- El preprocesador del M5 se ajusta una vez por pliegue, solo con el entrenamiento del pliegue
  (`CacheDiagnostico`, mismo principio que la caché de la 6.3). Así es idéntico al Pipeline.
- En la curva de aprendizaje el preprocesador se ajusta con todo el entrenamiento del pliegue
  y el modelo con la fracción. Es pasado del mismo pliegue: no hay fuga hacia la validación.
  Rehacer la SVD para cada fracción multiplicaría el costo sin cambiar la lectura de la curva.
- La prueba temporal (`es_entrenamiento = False`) no se usa en ningún paso de la 6.4.
"""
from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline

from modulo_6.m6_2_bateria_modelos import SALIDA_PUNTAJE, get_pipeline_and_space
from modulo_6.m6_3_optuna import FAMILIAS

UMBRAL_BRECHA = 0.05
FRACCIONES = (0.10, 0.25, 0.50, 1.00)
SEMILLA = 42


class AdvertenciaSobreajuste(UserWarning):
    """Brecha train − val del AUC-ROC por encima del umbral."""


# --------------------------------------------------------------------------------------
# Carga de los mejores hiperparámetros
# --------------------------------------------------------------------------------------
def cargar_mejores(ruta_json: Path, modelos: list[str] | None = None, por_familia: bool = True) -> dict[str, dict]:
    """Lee `best_hyperparameters.json` (6.3).

    Con `por_familia=True` devuelve el mejor modelo de cada una de las 7 familias del prompt.
    """
    datos = json.loads(Path(ruta_json).read_text(encoding="utf-8"))["modelos"]
    if modelos is None and por_familia:
        modelos = []
        for miembros in FAMILIAS.values():
            presentes = [m for m in miembros if m in datos]
            if presentes:
                modelos.append(max(presentes, key=lambda m: datos[m]["mejor_auc_cv"]))
    modelos = modelos or list(datos)
    return {m: datos[m] for m in modelos}


def instanciar(nombre: str, hiperparametros: dict, preprocesador) -> Pipeline:
    """Pipeline optimizado sin entrenar (preprocesador del M5 + modelo con los hiperparámetros ganadores)."""
    pipe, _ = get_pipeline_and_space(nombre, preprocesador)
    return pipe.set_params(**hiperparametros)


def puntaje(modelo, nombre: str, Z) -> np.ndarray:
    return modelo.decision_function(Z) if SALIDA_PUNTAJE[nombre] == "decision_function" else modelo.predict_proba(Z)[:, 1]


# --------------------------------------------------------------------------------------
# Caché de pliegues
# --------------------------------------------------------------------------------------
@dataclass
class _Pliegue:
    Z_tr: np.ndarray
    y_tr: np.ndarray
    g_tr: np.ndarray
    Z_va: np.ndarray
    y_va: np.ndarray
    i_va: np.ndarray


@dataclass
class CacheDiagnostico:
    """Preprocesador ajustado por pliegue (solo con su entrenamiento) + índices de validación para el OOF."""

    pliegues: list[_Pliegue] = field(default_factory=list)
    n: int = 0

    @classmethod
    def construir(cls, preprocesador, X: pd.DataFrame, y, groups, cv, dtype=np.float32) -> "CacheDiagnostico":
        y, g = np.asarray(y), np.asarray(groups)
        c = cls(n=len(X))
        for i_tr, i_va in cv.split(X, y, groups=g):
            if set(g[i_tr]) & set(g[i_va]):
                raise ValueError("Un person_id está en entrenamiento y validación del mismo pliegue")
            pre = clone(preprocesador).fit(X.iloc[i_tr], y[i_tr])
            c.pliegues.append(_Pliegue(pre.transform(X.iloc[i_tr]).astype(dtype), y[i_tr], g[i_tr],
                                       pre.transform(X.iloc[i_va]).astype(dtype), y[i_va], np.asarray(i_va)))
        return c


# --------------------------------------------------------------------------------------
# Diagnóstico de brecha + OOF
# --------------------------------------------------------------------------------------
def diagnosticar(nombre: str, pipe: Pipeline, cache: CacheDiagnostico) -> dict:
    """AUC de entrenamiento y validación por pliegue, brecha y predicciones OOF."""
    modelo_base = pipe.named_steps["modelo"]
    oof = np.full(cache.n, np.nan)
    filas = []
    t0 = time.time()
    for k, p in enumerate(cache.pliegues, 1):
        m = clone(modelo_base)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(p.Z_tr, p.y_tr)
            s_tr, s_va = puntaje(m, nombre, p.Z_tr), puntaje(m, nombre, p.Z_va)
        oof[p.i_va] = s_va
        filas.append({"pliegue": k, "train_auc": roc_auc_score(p.y_tr, s_tr), "val_auc": roc_auc_score(p.y_va, s_va)})
    df = pd.DataFrame(filas)
    df["brecha"] = df.train_auc - df.val_auc
    return {"modelo": nombre, "por_pliegue": df, "oof": oof, "segundos_cv": time.time() - t0,
            "train_auc": df.train_auc.mean(), "val_auc": df.val_auc.mean(), "val_auc_std": df.val_auc.std(ddof=1),
            "brecha": df.brecha.mean()}


def advertir_brechas(resumen: pd.DataFrame, umbral: float = UMBRAL_BRECHA) -> list[str]:
    """Advertencia explícita en consola (y `warnings`) por cada modelo con brecha > umbral."""
    sobre = resumen[resumen.brecha > umbral]
    for r in sobre.itertuples():
        msg = (f"⚠️  SOBREAJUSTE · {r.modelo}: train_auc {r.train_auc:.4f} − val_auc {r.val_auc:.4f} = "
               f"brecha {r.brecha:.4f} > {umbral}")
        print(msg, flush=True)
        warnings.warn(msg, AdvertenciaSobreajuste, stacklevel=2)
    return sobre.modelo.tolist()


# --------------------------------------------------------------------------------------
# Curva de aprendizaje
# --------------------------------------------------------------------------------------
def curva_aprendizaje(nombre: str, pipe: Pipeline, cache: CacheDiagnostico, fracciones=FRACCIONES,
                      diagnostico: dict | None = None, semilla: int = SEMILLA) -> pd.DataFrame:
    """AUC train/val por fracción del entrenamiento (submuestreo por pacientes completos) y pliegue.

    Si se pasa el `diagnostico` del mismo modelo, la fracción 1,0 reutiliza esos ajustes.
    """
    modelo_base = pipe.named_steps["modelo"]
    rng = np.random.default_rng(semilla)
    filas = []
    for k, p in enumerate(cache.pliegues, 1):
        personas = np.unique(p.g_tr)
        orden = rng.permutation(personas)
        for f in fracciones:
            if f >= 1 and diagnostico is not None:
                r = diagnostico["por_pliegue"].iloc[k - 1]
                filas.append({"modelo": nombre, "fraccion": 1.0, "pliegue": k, "n_train": len(p.y_tr),
                              "train_auc": r.train_auc, "val_auc": r.val_auc})
                continue
            elegidas = set(orden[: max(1, int(round(f * len(personas))))])
            mask = np.fromiter((g in elegidas for g in p.g_tr), bool, len(p.g_tr))
            m = clone(modelo_base)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.fit(p.Z_tr[mask], p.y_tr[mask])
                filas.append({"modelo": nombre, "fraccion": f, "pliegue": k, "n_train": int(mask.sum()),
                              "train_auc": roc_auc_score(p.y_tr[mask], puntaje(m, nombre, p.Z_tr[mask])),
                              "val_auc": roc_auc_score(p.y_va, puntaje(m, nombre, p.Z_va))})
    return pd.DataFrame(filas)


# --------------------------------------------------------------------------------------
# Ajuste final y empaquetado
# --------------------------------------------------------------------------------------
def ajustar_final(nombre: str, pipe: Pipeline, preprocesador_ajustado, Z, y) -> Pipeline:
    """Pipeline final: preprocesador ya ajustado con todo el entrenamiento + modelo ajustado sobre su salida.

    Es equivalente a `pipe.fit(X, y)`, pero el preprocesador se ajusta una sola vez para los 7 modelos.
    """
    m = clone(pipe.named_steps["modelo"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Z, y)
    return Pipeline([("preprocesador", preprocesador_ajustado), ("modelo", m)])


# --------------------------------------------------------------------------------------
# Gráficos
# --------------------------------------------------------------------------------------
def graficar_curvas_aprendizaje(curvas: pd.DataFrame, ruta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modelos = list(dict.fromkeys(curvas.modelo))
    fig, axes = plt.subplots(1, len(modelos), figsize=(5.2 * len(modelos), 4.2), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], modelos):
        d = curvas[curvas.modelo == m].groupby("fraccion").agg(
            n=("n_train", "mean"), tr=("train_auc", "mean"), tr_sd=("train_auc", "std"),
            va=("val_auc", "mean"), va_sd=("val_auc", "std")).reset_index()
        for col, sd, color, etq in (("tr", "tr_sd", "#c0504d", "Entrenamiento"), ("va", "va_sd", "#1f4e79", "Validación")):
            ax.plot(d.n, d[col], "-o", color=color, label=etq)
            ax.fill_between(d.n, d[col] - d[sd].fillna(0), d[col] + d[sd].fillna(0), color=color, alpha=0.15)
        ax.set_xscale("log")
        ax.set_title(m, fontweight="bold")
        ax.set_xlabel("Filas de entrenamiento (escala log)")
        ax.grid(alpha=0.3)
        final = d.iloc[-1]
        ax.annotate(f"brecha {final.tr - final.va:.3f}", (final.n, (final.tr + final.va) / 2), fontsize=8,
                    ha="right")
    axes[0][0].set_ylabel("AUC-ROC (media ± DE de 5 pliegues)")
    axes[0][0].legend()
    fig.suptitle("Curvas de aprendizaje · 3 mejores modelos (pliegues agrupados por paciente)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)


def graficar_roc(y, oofs: dict[str, np.ndarray], ruta: Path) -> pd.DataFrame:
    """ROC comparativa con las predicciones OOF de todos los modelos."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 6))
    filas = []
    for nombre, s in sorted(oofs.items(), key=lambda kv: -roc_auc_score(y[~np.isnan(kv[1])], kv[1][~np.isnan(kv[1])])):
        ok = ~np.isnan(s)
        fpr, tpr, _ = roc_curve(y[ok], s[ok])
        auc = roc_auc_score(y[ok], s[ok])
        filas.append({"modelo": nombre, "auc_oof": auc, "n": int(ok.sum())})
        ax.plot(fpr, tpr, lw=1.6, label=f"{nombre} · AUC {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("Tasa de falsos positivos (1 − especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (sensibilidad)")
    ax.set_title("ROC comparativa (predicciones fuera de pliegue)", fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(filas)
