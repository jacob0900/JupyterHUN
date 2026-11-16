"""Subfase 6.1 · Validación cruzada agrupada por paciente (StratifiedGroupKFold).

Por qué agrupar
---------------
En el HUN una misma persona aparece muchas veces: tiene varios episodios a lo largo de
los años y cada episodio genera 3 snapshots (τ = 0, 2 y 6 h). Esas filas están muy
correlacionadas (misma edad, mismos diagnósticos previos, misma estancia futura). Si
se repartieran al azar, el modelo vería en entrenamiento casi la misma fila que luego
"predice" en validación, y el rendimiento estimado sería irrealmente optimista.

Con `StratifiedGroupKFold(n_splits=5)` y `groups = person_id`, todas las filas de una
persona caen en el mismo pliegue (entrenamiento **o** validación, nunca ambos) y se
intenta conservar en cada pliegue la proporción de la clase positiva.

Reglas del proyecto (CLAUDE.md)
-------------------------------
- La validación cruzada se hace **solo dentro de entrenamiento**
  (`es_entrenamiento & en_periodo_estable`). La prueba temporal queda apartada para la
  evaluación final (M7).
- El target del modelo de paciente es `y_target_los_largo`. Las columnas `y_*` nunca son
  variables de entrada; aquí solo se usan para estratificar y reportar.

Solo usa pandas y scikit-learn.
"""
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

N_SPLITS = 5
SEMILLA = 42


def crear_cv(n_splits: int = N_SPLITS, shuffle: bool = True, random_state: int | None = SEMILLA) -> StratifiedGroupKFold:
    """Estrategia de CV del modelo de paciente: `StratifiedGroupKFold` (grupos = person_id).

    Con `shuffle=True` y semilla fija, los pliegues son reproducibles y no dependen del
    orden de las filas. Es la misma configuración que usó la 5.4.
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def verify_group_isolation(X, y, groups, cv, n_splits: int | None = None) -> pd.DataFrame:
    """Recorre los pliegues de `cv` y comprueba que ningún grupo (person_id) quede en ambos lados.

    Además verifica que cada fila esté en validación exactamente una vez (los pliegues
    de validación forman una partición de los datos).

    Lanza:
        ValueError: si un pliegue comparte algún `person_id` entre entrenamiento y
            validación, o si los pliegues de validación no cubren cada fila una sola vez.

    Devuelve:
        Reporte con una fila por pliegue: muestras, pacientes únicos y proporción de la
        clase positiva en entrenamiento y validación.
    """
    y = pd.Series(y).reset_index(drop=True)
    g = pd.Series(groups).reset_index(drop=True)
    if len(y) != len(g) or len(y) != len(X):
        raise ValueError(f"X ({len(X)}), y ({len(y)}) y groups ({len(g)}) deben tener el mismo número de filas")
    if g.isna().any():
        raise ValueError("Hay filas sin person_id: no se pueden agrupar")

    veces_en_validacion = pd.Series(0, index=y.index)
    filas = []
    for k, (i_tr, i_va) in enumerate(cv.split(X, y, groups=g), start=1):
        g_tr, g_va = set(g.iloc[i_tr]), set(g.iloc[i_va])
        traslape = g_tr.intersection(g_va)
        # Comprobación explícita con ValueError (un `assert` se desactiva con `python -O`).
        if len(traslape) != 0:
            ejemplo = sorted(traslape)[:5]
            raise ValueError(
                f"Pliegue {k}: {len(traslape)} person_id aparecen en entrenamiento y validación "
                f"(p. ej. {ejemplo}). Hay fuga por observaciones agrupadas."
            )
        assert len(g_tr.intersection(g_va)) == 0  # invariante documentada del prompt
        veces_en_validacion.iloc[i_va] += 1
        filas.append({
            "pliegue": k,
            "n_train": len(i_tr),
            "n_val": len(i_va),
            "pacientes_train": len(g_tr),
            "pacientes_val": len(g_va),
            "pacientes_compartidos": 0,
            "prop_pos_train": float(y.iloc[i_tr].mean()),
            "prop_pos_val": float(y.iloc[i_va].mean()),
        })

    if n_splits is not None and len(filas) != n_splits:
        raise ValueError(f"Se esperaban {n_splits} pliegues y la estrategia generó {len(filas)}")
    if not (veces_en_validacion == 1).all():
        mal = int((veces_en_validacion != 1).sum())
        raise ValueError(f"{mal} filas no están en validación exactamente una vez: los pliegues no son una partición")

    rep = pd.DataFrame(filas)
    rep["dif_prop_val_vs_global"] = rep["prop_pos_val"] - float(y.mean())
    return rep


def asignar_pliegues(groups, y, cv) -> pd.Series:
    """Número de pliegue de validación (1…k) de cada fila, para reutilizar los mismos pliegues en todo M6."""
    y = pd.Series(y).reset_index(drop=True)
    g = pd.Series(groups).reset_index(drop=True)
    pliegue = pd.Series(0, index=y.index, name="pliegue", dtype="int8")
    for k, (_, i_va) in enumerate(cv.split(g.to_frame(), y, groups=g), start=1):
        pliegue.iloc[i_va] = k
    return pliegue


class PlieguesPredefinidos:
    """Divisor compatible con scikit-learn que reproduce pliegues ya asignados (columna `pliegue`).

    Sirve para que M6.2+ (modelos, Optuna) use exactamente los pliegues verificados en la 6.1,
    aunque cambie la versión de scikit-learn: `cross_validate(pipe, X, y, cv=PlieguesPredefinidos(pliegue))`.
    """

    def __init__(self, pliegue: Iterable[int]):
        self.pliegue = pd.Series(list(pliegue)).reset_index(drop=True)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return int(self.pliegue.nunique())

    def split(self, X=None, y=None, groups=None):
        idx = pd.RangeIndex(len(self.pliegue))
        for k in sorted(self.pliegue.unique()):
            es_val = (self.pliegue == k).to_numpy()
            yield idx[~es_val].to_numpy(), idx[es_val].to_numpy()


def imprimir_reporte(rep: pd.DataFrame, titulo: str = "Validación cruzada agrupada por person_id") -> None:
    """Reporte legible por pliegue."""
    print(f"\n{titulo}")
    print("=" * len(titulo))
    for r in rep.itertuples(index=False):
        print(
            f"Pliegue {r.pliegue}: "
            f"train {r.n_train:>9,} filas · {r.pacientes_train:>8,} pacientes · {100 * r.prop_pos_train:6.2f} % positivos | "
            f"val {r.n_val:>8,} filas · {r.pacientes_val:>7,} pacientes · {100 * r.prop_pos_val:6.2f} % positivos | "
            f"compartidos {r.pacientes_compartidos}"
        )
    print(
        f"→ Ningún person_id compartido en los {len(rep)} pliegues. "
        f"Prop. positiva en validación entre {100 * rep.prop_pos_val.min():.2f} % y {100 * rep.prop_pos_val.max():.2f} % "
        f"(máx. desvío vs global {100 * rep.dif_prop_val_vs_global.abs().max():.2f} pp)."
    )
