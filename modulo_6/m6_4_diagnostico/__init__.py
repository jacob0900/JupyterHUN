"""Subfase 6.4 · Diagnóstico de sobreajuste, curvas de aprendizaje y empaquetado (.joblib)."""
from .diagnostico import (
    FRACCIONES,
    UMBRAL_BRECHA,
    AdvertenciaSobreajuste,
    CacheDiagnostico,
    advertir_brechas,
    ajustar_final,
    cargar_mejores,
    curva_aprendizaje,
    diagnosticar,
    graficar_curvas_aprendizaje,
    graficar_roc,
    instanciar,
    puntaje,
)

__all__ = ["FRACCIONES", "UMBRAL_BRECHA", "AdvertenciaSobreajuste", "CacheDiagnostico", "advertir_brechas",
           "ajustar_final", "cargar_mejores", "curva_aprendizaje", "diagnosticar", "graficar_curvas_aprendizaje",
           "graficar_roc", "instanciar", "puntaje"]
