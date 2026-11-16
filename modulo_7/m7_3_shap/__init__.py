"""Subfase 7.3 · Explicabilidad global y local con SHAP."""
from .explicabilidad import (
    ETIQUETAS,
    NOTAS,
    TRADUCCION_DX,
    agrupar_shap,
    efecto_diagnosticos,
    formatear_valor,
    etiqueta,
    explicacion,
    explicar,
    graficar_barras,
    graficar_barras_por_servicio,
    graficar_beeswarm,
    graficar_waterfall,
    importancia_global,
    mapa_grupos,
    perfil,
    sintesis_medica,
    valores_color,
)

__all__ = ["ETIQUETAS", "NOTAS", "TRADUCCION_DX", "agrupar_shap", "efecto_diagnosticos", "formatear_valor", "etiqueta", "explicacion", "explicar", "graficar_barras",
           "graficar_barras_por_servicio", "graficar_beeswarm", "graficar_waterfall", "importancia_global",
           "mapa_grupos", "perfil", "sintesis_medica", "valores_color"]
