# Data Challenge HUN — Predicción de flujo hospitalario

Base: PostgreSQL `uhn_i`, esquema `uninorte_omop` (OMOP CDM recortado), vía VPN FortiClient.
**El servidor es de solo lectura**: el pipeline solo hace `SELECT` y procesa en local con DuckDB.

## Puesta en marcha

```bash
pip install -r requirements.txt
cp .env.example .env        # llenar host, usuario y contraseña
python m1_etl/run_etl.py probar
python m1_etl/run_etl.py extraer
python m1_etl/run_etl.py construir
```

## Estructura

| Ruta | Contenido |
|---|---|
| `m1_etl/run_etl.py` | runner del M1: `probar`, `explorar`, `extraer`, `construir` |
| `m1_etl/sql/` | consultas al servidor (solo lectura): exploración `00*` y extracción `10` |
| `m1_etl/sql_local/` | ETL local en DuckDB: base, calidad, ocupación, episodios, snapshots, validaciones |
| `m4_espaciotemporal/run_m4.py` | M4 parte SQL: transiciones, Sankey, heatmaps, calendario colombiano, colas (`--paquete` arma la entrega a Persona 3) |
| `docs/M4_entrega_persona3.md` | guía de qué archivo usar en M2, M3 y M4 visual |
| `docs/M1_hallazgos_datos.md` | qué tienen (y qué no) los datos, y por qué el diseño es como es |
| `docs/M1_contrato_datos.md` | **tablas que entrega el M1 y reglas para usarlas** (léelo antes de M2–M7) |
| `modulo_5/` | M5 feature engineering (Persona 2): ver `modulo_5/README.md`; prompt para retomarlo en otra sesión en `modulo_5/PROMPT_SESION.md` |
| `data/raw/`, `data/m1/` | parquets extraídos y tablas finales (no van a git) |
| `outputs/` | resultados de exploración y reporte de validaciones |

## Documento único (GitHub Pages y NotebookLM)

Todo el proyecto (contexto, M1–M7, informes, figuras y código de cada subfase) está en un solo capítulo:

| Archivo | Uso |
|---|---|
| `Data_challenge_HUN.ipynb` | el notebook completo (las celdas de código muestran el código; no se ejecutan) |
| `index.html` | el notebook renderizado con las figuras incrustadas: es la página de GitHub Pages |
| `Data_challenge_HUN.md` | el mismo contenido en un solo Markdown, para subirlo como una única fuente a NotebookLM |

Se regeneran los tres con:

```bash
python herramientas/construir_notebook.py
```

Para publicar: subir esta carpeta a un repositorio de GitHub y en **Settings → Pages** elegir la rama `main` y la carpeta `/ (root)`. La página queda en `https://<usuario>.github.io/<repositorio>/`, y esa URL también sirve como fuente única en NotebookLM.

Esta copia no incluye `data/`, `models/`, `entregas/` ni `.env`: tienen datos de pacientes, modelos entrenados con ellos o credenciales.

## Módulos y responsables

| Persona | Módulos |
|---|---|
| Persona 1 (SQL & MLOps) | M1 ETL y targets, M4 (SQL de transiciones), M8 API + dashboard |
| Persona 2 (ML) | M5 features, M6 modelos y CV, M7 evaluación y calibración |
| Persona 3 (EDA) | M2 univariado y censura, M3 bivariado y colas, M4 (visual) |
