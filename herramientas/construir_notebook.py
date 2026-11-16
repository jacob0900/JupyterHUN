"""Arma un único notebook (un solo capítulo) con todo el proyecto y su versión HTML.

    python herramientas/construir_notebook.py

Genera, en la raíz del proyecto:
    Data_challenge_HUN.ipynb   el documento completo: contexto, M1–M4 (datos y EDA), M5, M6 y M7
                               con sus informes, figuras y el código fuente de cada subfase
    index.html                 el mismo notebook renderizado, con las figuras incrustadas
                               (una sola página para GitHub Pages y para NotebookLM)
    Data_challenge_HUN.md      el mismo contenido en un solo Markdown (fuente única para NotebookLM)

Las celdas de código muestran el código fuente; no se ejecutan (los datos de pacientes no
se publican). Para correr el proyecto se usan los scripts `run_*.py` de cada carpeta.
"""
from __future__ import annotations

import posixpath
import re
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "Data_challenge_HUN.ipynb"
HTML = ROOT / "index.html"
MD = ROOT / "Data_challenge_HUN.md"

celdas: list = []


# --------------------------------------------------------------------------- utilidades
def leer(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def md(texto: str) -> None:
    texto = texto.strip("\n")
    if texto.strip():
        celdas.append(new_markdown_cell(texto))


def codigo(rel: str) -> None:
    fuente = leer(rel).rstrip() + "\n"
    celdas.append(new_code_cell(f"# Archivo: {rel}\n{fuente}", metadata={"tags": ["skip-execution"]}))


def bajar_titulos(texto: str, niveles: int = 1, quitar_h1: bool = True) -> str:
    """Baja el nivel de los títulos para que el documento tenga un solo H1."""
    salida, en_bloque = [], False
    for linea in texto.splitlines():
        if linea.lstrip().startswith("```"):
            en_bloque = not en_bloque
        elif not en_bloque and re.match(r"^#{1,6} ", linea):
            if quitar_h1 and linea.startswith("# "):
                continue
            hashes = len(linea) - len(linea.lstrip("#"))
            linea = "#" * min(6, hashes + niveles) + linea[hashes:]
        salida.append(linea)
    return "\n".join(salida)


def arreglar_enlaces(texto: str, carpeta_origen: str) -> str:
    """Reescribe enlaces e imágenes relativos para que apunten desde la raíz del proyecto."""

    def cambio(m: re.Match) -> str:
        etiqueta, destino = m.group(1), m.group(2)
        if re.match(r"^(https?:|mailto:|#|/)", destino):
            return m.group(0)
        nuevo = posixpath.normpath(posixpath.join(carpeta_origen, destino)) if carpeta_origen else destino
        return f"{etiqueta}({nuevo})"

    return re.sub(r"(!?\[[^\]]*\])\(([^)\s]+)\)", cambio, texto)


def documento(rel: str, niveles: int = 1, quitar_h1: bool = True) -> str:
    carpeta = posixpath.dirname(rel)
    return arreglar_enlaces(bajar_titulos(leer(rel), niveles, quitar_h1), carpeta)


def secciones(texto: str) -> list[tuple[str, str]]:
    """Parte un markdown por títulos `## ` (fuera de bloques de código): [(titulo, cuerpo)]."""
    partes, actual, titulo, en_bloque = [], [], "", False
    for linea in texto.splitlines():
        if linea.lstrip().startswith("```"):
            en_bloque = not en_bloque
        if not en_bloque and linea.startswith("## "):
            partes.append((titulo, "\n".join(actual)))
            titulo, actual = linea[3:].strip(), [linea]
        else:
            actual.append(linea)
    partes.append((titulo, "\n".join(actual)))
    return partes


def figuras(patrones: list[str], ya_citadas: str = "") -> None:
    """Inserta las figuras que coinciden con los patrones y que el texto aún no cita."""
    rutas = []
    for patron in patrones:
        rutas += sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob(patron))
    mostradas = set(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", ya_citadas))  # solo imágenes ya visibles
    rutas = [r for r in dict.fromkeys(rutas) if r not in mostradas]
    if rutas:
        md("\n\n".join(f"![{Path(r).stem.replace('_', ' ')}]({r})" for r in rutas))


def codigo_subfase(carpeta: Path) -> None:
    """Código de una subfase: módulo principal, script de ejecución y pruebas."""
    py = sorted(p for p in carpeta.glob("*.py") if p.name != "__init__.py")
    principales = [p for p in py if not p.name.startswith(("run_", "test_"))]
    runs = [p for p in py if p.name.startswith("run_")]
    tests = [p for p in py if p.name.startswith("test_")]
    rel = lambda p: p.relative_to(ROOT).as_posix()  # noqa: E731
    if principales:
        md(f"**Código fuente · `{rel(carpeta)}/`**")
        for p in principales:
            codigo(rel(p))
    if runs:
        md("**Ejecución sobre los datos reales**")
        for p in runs:
            codigo(rel(p))
    if tests:
        md("**Pruebas con datos sintéticos**")
        for p in tests:
            codigo(rel(p))


def modulo(n: int, titulo: str, extra: dict[str, str] | None = None) -> None:
    """Un módulo M5–M7: su README partido por subfases, con figuras y código tras cada una."""
    base = ROOT / f"modulo_{n}"
    md(f"## Módulo {n} · {titulo}")
    subcarpetas = {p.name: p for p in sorted(base.glob(f"m{n}_*")) if p.is_dir()}
    usadas = set()
    # Primero se parte por "## " (títulos originales) y después se baja el nivel de cada parte.
    for encabezado, cuerpo in secciones(leer(f"modulo_{n}/README.md")):
        cuerpo = arreglar_enlaces(bajar_titulos(cuerpo, 1), f"modulo_{n}")
        md(cuerpo)
        m = re.search(rf"Subfase {n}\.(\d+)", encabezado)
        if not m:
            continue
        k = m.group(1)
        if k == "1":  # lo previo a la subfase 1 (p. ej. m5_0_extraccion) va antes de ella
            for nombre, carpeta in subcarpetas.items():
                if nombre.startswith(f"m{n}_0_") and nombre not in usadas:
                    celdas.insert(len(celdas) - 1, new_markdown_cell(f"### Paso previo · `{nombre}`"))
                    pos = len(celdas) - 1
                    tmp = celdas[pos:]
                    del celdas[pos:]
                    codigo_subfase(carpeta)
                    celdas.extend(tmp)
                    usadas.add(nombre)
        for nombre, carpeta in subcarpetas.items():
            if nombre.startswith(f"m{n}_{k}_"):
                figuras([f"outputs/m{n}/figuras/m{n}_{k}_*.png", f"reports/figures/m{n}_{k}_*.png"], cuerpo)
                if extra and nombre in extra:
                    md(extra[nombre])
                codigo_subfase(carpeta)
                usadas.add(nombre)
    for nombre, carpeta in subcarpetas.items():
        if nombre not in usadas:
            md(f"### Código de apoyo · `{nombre}`")
            codigo_subfase(carpeta)


# --------------------------------------------------------------------------- documento
def construir() -> None:
    md(
        """# Data Challenge HUN 2026 · Predicción de estancia prolongada y saturación hospitalaria

**Hospital Universidad del Norte (HUN)** · base OMOP CDM `uninorte_omop` · Data Challenge Pro 2026, Universidad del Norte.

Este documento reúne en un solo capítulo todo el proyecto: el contexto y los datos (M1), el análisis exploratorio (M2–M4), la ingeniería de variables (M5), el modelado con validación cruzada (M6) y la evaluación, calibración, explicabilidad y pruebas estadísticas (M7). Cada módulo trae su informe, sus figuras y el código fuente de cada subfase.

**Dos tareas de predicción**

1. **Paciente**: ¿la estancia superará el percentil 75 de su servicio? Se predice en τ = ingreso + 0, 2 y 6 h.
2. **Servicio**: ¿la ocupación llegará al umbral de saturación dentro de 4 h?

**Cómo leer este documento.** Las celdas de código muestran el código fuente tal como está en el repositorio; no se ejecutan en esta página porque los datos de pacientes no se publican. Las tablas y figuras son resultados agregados de las corridas con los datos reales."""
    )
    md(_tabla_equipo())

    # ---- contexto
    md("## Contexto del proyecto y reglas de trabajo")
    claude = dict(secciones(leer("CLAUDE.md")))
    for clave in ("Lo que el PDF supone y NO es cierto en los datos", "Reglas no negociables",
                  "Particularidades de cada servicio", "Cifras de referencia (para verificar que lees bien los datos)"):
        if clave not in claude:
            raise KeyError(f"CLAUDE.md ya no tiene la sección '{clave}'")
        md(bajar_titulos(claude[clave], 1))

    # ---- M1
    md("## Módulo 1 · Datos: extracción, hallazgos y contrato")
    md(documento("docs/M1_hallazgos_datos.md"))
    md(documento("docs/M1_contrato_datos.md"))
    md("### Validaciones del ETL")
    md(documento("outputs/m1_validaciones.md", 2))

    # ---- M2–M4
    md("## Módulo 2 · Análisis univariado, calidad y censura")
    md(documento("outputs/m2/M2_informe.md"))
    md("**Código fuente · utilidades compartidas del EDA y script del M2**")
    codigo("eda_comun/__init__.py")
    codigo("m2_eda/run_m2.py")

    md("## Módulo 3 · Análisis bivariado y teoría de colas")
    md(documento("outputs/m3/M3_informe.md"))
    md("**Código fuente**")
    codigo("m3_bivariado/run_m3.py")

    md("## Módulo 4 · Análisis espacio-temporal")
    md("### Parte SQL (transiciones, calendario y colas)")
    md(documento("outputs/m4_resumen.md", 2))
    md(documento("docs/M4_entrega_persona3.md", 2))
    md("### Parte visual")
    md(documento("outputs/m4_visual/M4_visual_informe.md", 2))
    htmls = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "outputs/m4_visual/html").glob("*.html"))
    if htmls:
        md("Versiones interactivas (plotly): " + " · ".join(f"[{Path(h).stem}]({h})" for h in htmls))
    md("**Código fuente**")
    codigo("m4_visual/run_m4_visual.py")
    md("### Entrega del EDA a los módulos de modelado")
    md(documento("docs/P3_entrega_eda.md", 2))

    # ---- M5–M7
    modulo(5, "Ingeniería de variables")
    modulo(6, "Modelado supervisado y validación cruzada")
    sintesis = documento("reports/m7_3_sintesis_medica.md", 3)
    modulo(7, "Evaluación, calibración y pruebas estadísticas",
           extra={"m7_3_shap": "#### Síntesis clínica para el equipo médico\n\n" + sintesis})

    nb = new_notebook(cells=celdas, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "title": "Data Challenge HUN 2026",
    })
    nbformat.validate(nb)
    nbformat.write(nb, NB)
    print(f"✓ {NB.name}: {len(celdas)} celdas · {NB.stat().st_size / 2**20:.1f} MB")


def _tabla_equipo() -> str:
    for titulo, cuerpo in secciones(leer("README.md")):
        if titulo.startswith("Módulos y responsables"):
            return cuerpo.replace("## Módulos y responsables", "## Equipo y módulos")
    return ""


def html() -> None:
    from nbconvert import HTMLExporter

    exp = HTMLExporter(template_name="lab")
    exp.embed_images = True  # figuras dentro del HTML: una sola página autosuficiente
    cuerpo, _ = exp.from_notebook_node(nbformat.read(NB, as_version=4), resources={"metadata": {"path": str(ROOT)}})
    HTML.write_text(cuerpo, encoding="utf-8")
    (ROOT / ".nojekyll").write_text("", encoding="utf-8")
    print(f"✓ {HTML.name}: {HTML.stat().st_size / 2**20:.1f} MB")


def markdown() -> None:
    """Todo el documento en un solo .md (una sola fuente para NotebookLM)."""
    from nbconvert import MarkdownExporter

    cuerpo, _ = MarkdownExporter().from_notebook_node(nbformat.read(NB, as_version=4))
    MD.write_text(cuerpo, encoding="utf-8")
    print(f"✓ {MD.name}: {MD.stat().st_size / 2**20:.1f} MB · {len(cuerpo.split()):,} palabras")


if __name__ == "__main__":
    construir()
    html()
    markdown()
