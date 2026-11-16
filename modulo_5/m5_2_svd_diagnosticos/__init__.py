"""Subfase 5.2 · Reducción de dimensionalidad de códigos clínicos (TruncatedSVD sobre diagnósticos)."""
from .svd_diagnosticos import ReductorSVDClinico, construir_matriz_dispersa, preparar_eventos

__all__ = ["ReductorSVDClinico", "construir_matriz_dispersa", "preparar_eventos"]
