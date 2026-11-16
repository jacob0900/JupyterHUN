"""Subfase 5.4 · Ensamblaje estricto anti-fuga (ColumnTransformer + Pipeline)."""
from .pipeline_antifuga import (
    COL_DX,
    EsquemaPipeline,
    construir_pipeline,
    construir_preprocesador,
    codigos_desde_matriz,
    huella_parametros,
    preparar_X,
    verificar_aislamiento,
    verificar_estadisticas_solo_entrenamiento,
    verificar_pipeline_sin_fuga,
)

__all__ = [
    "COL_DX",
    "EsquemaPipeline",
    "construir_pipeline",
    "construir_preprocesador",
    "codigos_desde_matriz",
    "huella_parametros",
    "preparar_X",
    "verificar_aislamiento",
    "verificar_estadisticas_solo_entrenamiento",
    "verificar_pipeline_sin_fuga",
]
