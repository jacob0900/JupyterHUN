"""Subfase 6.1 · Validación cruzada agrupada por paciente (StratifiedGroupKFold)."""
from .cv_agrupada import (
    N_SPLITS,
    SEMILLA,
    PlieguesPredefinidos,
    asignar_pliegues,
    crear_cv,
    imprimir_reporte,
    verify_group_isolation,
)

__all__ = ["N_SPLITS", "SEMILLA", "PlieguesPredefinidos", "asignar_pliegues", "crear_cv", "imprimir_reporte",
           "verify_group_isolation"]
