"""Subfase 6.2 · Batería de modelos supervisados y espacios de búsqueda."""
from .model_factory import (
    COSTO,
    PARAM_SPACES,
    SALIDA_PUNTAJE,
    SEMILLA,
    Categorico,
    Entero,
    ModelFactory,
    MuestreadorAleatorio,
    Real,
    a_distribuciones_scipy,
    construir_modelo,
    get_pipeline_and_space,
    listar_modelos,
    sugerir_parametros,
)

__all__ = [
    "COSTO", "PARAM_SPACES", "SALIDA_PUNTAJE", "SEMILLA", "Categorico", "Entero", "ModelFactory",
    "MuestreadorAleatorio", "Real", "a_distribuciones_scipy", "construir_modelo", "get_pipeline_and_space",
    "listar_modelos", "sugerir_parametros",
]
