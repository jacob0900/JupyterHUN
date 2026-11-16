"""Subfase 6.3 · Optimización bayesiana de hiperparámetros con Optuna (TPE + MedianPruner)."""
from .busqueda_optuna import (
    FAMILIAS,
    CachePliegues,
    crear_estudio,
    guardar_resultados,
    objective,
    optimizar_modelo,
    resumen_estudio,
    tabla_por_familia,
)

__all__ = ["FAMILIAS", "CachePliegues", "crear_estudio", "guardar_resultados", "objective", "optimizar_modelo",
           "resumen_estudio", "tabla_por_familia"]
