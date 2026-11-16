"""Subfase 7.3 · Explicabilidad global y local con SHAP (TreeExplainer sobre LightGBM).

Decisiones:
- **SHAP agrupado por variable clínica.** El modelo ve ~120 columnas transformadas: one-hot, escalado y 50
  componentes SVD del perfil de diagnósticos, que no son interpretables una por una. Los valores SHAP son
  aditivos, así que la contribución de una variable original es la **suma exacta** de las de sus columnas:
  las 50 SVD → "perfil de diagnósticos"; seno y coseno de la hora → "hora del día"; las categorías de un
  one-hot → la variable categórica.
- **Escala:** log-odds del modelo (`TreeExplainer`, salida cruda). E[f(X)] es el valor base en log-odds; las
  figuras anotan su equivalente en probabilidad.
- **Casos locales sin exponer pacientes** (regla 7 de CLAUDE.md: solo agregados en informes). Cada waterfall
  muestra el **perfil SHAP medio** de un grupo (5 % de mayor / menor riesgo), que también es aditivo:
  E[f(X)] + Σ φ̄_j = f̄ del grupo. Los valores que acompañan a cada variable son medianas (o proporciones)
  del grupo. Las explicaciones individuales se calculan a demanda en el despliegue (M8).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------- vocabulario clínico
LAB = {
    "4016928": "hemograma completo", "4324383": "creatinina", "4074649": "nitrógeno ureico (BUN)",
    "46272908": "proteína C reactiva", "3019550": "sodio", "4245152": "potasio", "4099313": "uroanálisis",
    "3014576": "cloro", "4245261": "tiempo de protrombina", "44809202": "tiempo de tromboplastina (TTP)",
    "3013826": "glucosa", "4146380": "transaminasa ALT", "4263457": "transaminasa AST",
    "3045524": "bilirrubinas", "4239236": "gases arteriales", "40547237": "antibiograma",
    "4213582": "ácido láctico", "4246879": "TSH (tiroides)", "4184637": "hemoglobina glicosilada",
    "42529218": "prueba de VIH",
}

# variable → (etiqueta clínica, dominio, descripción del valor ALTO / de la categoría marcada)
ETIQUETAS: dict[str, tuple[str, str, str]] = {
    "horas_desde_ingreso": ("Momento de la evaluación (0, 2 o 6 h tras el ingreso)", "admisión",
                            "evaluar al paciente más horas después del ingreso"),
    "edad": ("Edad", "admisión", "mayor edad"),
    "sexo": ("Sexo", "admisión", "sexo masculino"),
    "servicio": ("Servicio (hospitalización vs urgencias)", "admisión", "estar en hospitalización"),
    "origen_urgencias": ("Ingreso por urgencias", "admisión", "haber ingresado por urgencias"),
    "horas_desde_ultima_alta_hosp": ("Tiempo desde la última alta hospitalaria", "historia clínica",
                                     "más tiempo desde la última alta"),
    "n_hospitalizaciones_previas": ("Hospitalizaciones previas", "historia clínica", "más hospitalizaciones previas"),
    "nunca_hospitalizado": ("Nunca hospitalizado antes", "historia clínica", "no tener hospitalizaciones previas"),
    "n_visitas_365d": ("Visitas al hospital en el último año", "historia clínica", "más visitas en el último año"),
    "n_dx_historia": ("Diagnósticos registrados en la historia", "historia clínica", "más diagnósticos registrados"),
    "n_dx_distintos_historia": ("Diagnósticos distintos en la historia", "historia clínica",
                                "más diagnósticos distintos (multimorbilidad)"),
    "n_dx_30d": ("Diagnósticos en los últimos 30 días", "historia clínica", "más diagnósticos recientes"),
    "sin_dx_historia": ("Sin diagnósticos previos", "historia clínica", "no tener diagnósticos previos"),
    "dx_codigos": ("Perfil de diagnósticos (historia clínica)", "historia clínica",
                   "un perfil de diagnósticos más cargado"),
    "n_mediciones_24h": ("Laboratorios tomados en las últimas 24 h", "laboratorios", "más laboratorios en 24 h"),
    "n_mediciones_historia": ("Laboratorios en la historia", "laboratorios", "más laboratorios en la historia"),
    "sin_mediciones_24h": ("Sin laboratorios en las últimas 24 h", "laboratorios", "no tener laboratorios en 24 h"),
    "sin_mediciones_historia": ("Sin laboratorios previos", "laboratorios", "no tener laboratorios previos"),
    "censo_servicio_t_menos_1h": ("Pacientes en el servicio (hora previa)", "presión del servicio",
                                  "más pacientes en el servicio"),
    "ocupacion_servicio_t_menos_1h": ("Ocupación del servicio (hora previa)", "presión del servicio",
                                      "mayor ocupación del servicio"),
    "llegadas_servicio_t_menos_1h": ("Llegadas al servicio (hora previa)", "presión del servicio", "más llegadas"),
    "srv_llegadas_suma_4h": ("Llegadas al servicio en las últimas 4 h", "presión del servicio", "más llegadas en 4 h"),
    "srv_llegadas_suma_24h": ("Llegadas al servicio en las últimas 24 h", "presión del servicio", "más llegadas en 24 h"),
    "srv_ocupacion_media_3h": ("Ocupación media del servicio (3 h)", "presión del servicio", "mayor ocupación"),
    "srv_ocupacion_media_6h": ("Ocupación media del servicio (6 h)", "presión del servicio", "mayor ocupación"),
    "srv_ocupacion_media_12h": ("Ocupación media del servicio (12 h)", "presión del servicio", "mayor ocupación"),
    "srv_ocupacion_media_24h": ("Ocupación media del servicio (24 h)", "presión del servicio", "mayor ocupación"),
    "srv_ocupacion_max_24h": ("Ocupación máxima del servicio (24 h)", "presión del servicio", "mayor pico de ocupación"),
    "srv_ocupacion_std_24h": ("Variabilidad de la ocupación (24 h)", "presión del servicio", "ocupación más inestable"),
    "srv_ocupacion_tendencia_3h": ("Tendencia de la ocupación (3 h)", "presión del servicio", "ocupación en aumento"),
    "srv_frac_horas_sobre_umbral_24h": ("Horas del servicio por encima del umbral de saturación (24 h)",
                                        "presión del servicio", "más horas saturado"),
    "srv_ocupacion_semana_previa": ("Ocupación del servicio la semana anterior", "presión del servicio",
                                    "mayor ocupación la semana anterior"),
    "hora_del_dia": ("Hora del día", "calendario", "una hora más tardía"),
    "dia_semana": ("Día de la semana", "calendario", "más cerca del fin de semana"),
    "turno": ("Turno (mañana / tarde / noche)", "calendario", "un turno más tardío"),
    "mes": ("Mes del año", "calendario", "meses más avanzados del año"),
    "es_fin_de_semana": ("Fin de semana", "calendario", "ser fin de semana"),
    "es_festivo": ("Día festivo", "calendario", "ser festivo"),
    "es_hora_pico_urgencias": ("Hora pico de urgencias", "calendario", "estar en hora pico"),
    "tras_festivo": ("Día siguiente a un festivo", "calendario", "ser el día siguiente a un festivo"),
    "primer_dia_habil": ("Primer día hábil", "calendario", "ser primer día hábil"),
}
for _cid, _nom in LAB.items():
    ETIQUETAS[f"has_measurement_{_cid}"] = (f"Laboratorio registrado: {_nom}", "laboratorios",
                                            f"tener registrado {_nom}")

# Traducción al español de los diagnósticos (nombres OMOP en inglés) que aparecen en los informes
TRADUCCION_DX = {
    "Localized infection of skin AND/OR subcutaneous tissue": "infección localizada de piel o tejido subcutáneo",
    "Acute lower respiratory tract infection": "infección aguda de vías respiratorias bajas",
    "Upper abdominal pain": "dolor abdominal superior",
    "Pneumonia": "neumonía",
    "Essential hypertension": "hipertensión esencial",
    "Renal colic": "cólico renal",
    "Congestive heart failure": "insuficiencia cardíaca congestiva",
    "Urinary tract infectious disease": "infección de vías urinarias",
    "Gastrointestinal hemorrhage": "hemorragia gastrointestinal",
    "False labor at or after 37 completed weeks of gestation": "falso trabajo de parto a término (≥ 37 semanas)",
    "False labor before 37 completed weeks of gestation": "falso trabajo de parto pretérmino (< 37 semanas)",
    "False labor": "falso trabajo de parto",
    "Deliveries by cesarean": "parto por cesárea",
    "Delivery by emergency cesarean section": "cesárea de urgencia",
    "Pelvic and perineal pain": "dolor pélvico y perineal",
    "Threatened miscarriage": "amenaza de aborto",
    "Genitourinary tract hemorrhage": "hemorragia del tracto genitourinario",
}

# Notas para variables cuyo efecto viene de la definición del objetivo, no de la fisiología
NOTAS = {
    "servicio": ("Este efecto refleja sobre todo cómo se define la estancia prolongada: el umbral es propio de cada "
                 "servicio y, en urgencias, quien sigue allí a las 2 o 6 horas ya es un paciente de estancia larga. "
                 "No significa que hospitalizar acorte la estancia."),
    "horas_desde_ingreso": ("Que el paciente siga en el servicio horas después del ingreso es en sí información "
                            "pronóstica: cuanto más tiempo lleva, más probable es que su estancia sea larga."),
}

FUSIONES = {"hour_sin": "hora_del_dia", "hour_cos": "hora_del_dia", "dow_sin": "dia_semana", "dow_cos": "dia_semana"}
TURNO_ORDEN = {"manana": 0, "mañana": 0, "tarde": 1, "noche": 2}


def etiqueta(var: str) -> str:
    return ETIQUETAS.get(var, (var, "otro", var))[0]


# ---------------------------------------------------------------------------- agrupación
def mapa_grupos(nombres: list[str], esquema) -> list[str]:
    """Variable clínica original de cada columna transformada (salida de `get_feature_names_out`)."""
    cats = sorted(esquema.cat, key=len, reverse=True)  # el prefijo más largo gana (p. ej. es_festivo vs es_...)
    grupos = []
    for n in nombres:
        rama, _, resto = n.partition("__")
        if rama in ("num", "num_log"):
            grupos.append(FUSIONES.get(resto, resto))
        elif rama == "cat":
            g = next((c for c in cats if resto == c or resto.startswith(c + "_")), None)
            if g is None:
                raise ValueError(f"No se reconoce la variable categórica de la columna {n}")
            grupos.append(g)
        elif rama == "clinica":
            grupos.append(esquema.clinica)
        else:
            raise ValueError(f"Rama desconocida en la columna {n}")
    return grupos


def agrupar_shap(phi: np.ndarray, grupos: list[str]) -> pd.DataFrame:
    """Suma exacta de los SHAP de las columnas de cada variable original (n × variables)."""
    g = pd.Index(grupos)
    orden = list(dict.fromkeys(grupos))
    return pd.DataFrame({v: phi[:, np.flatnonzero(g == v)].sum(axis=1) for v in orden})


def valores_color(X: pd.DataFrame, variables: list[str], clinica: str | None = None) -> pd.DataFrame:
    """Valor numérico (en unidades originales) de cada variable, para colorear el beeswarm y describir perfiles."""
    out = {}
    for v in variables:
        if v == clinica:
            out[v] = X[v].fillna("").astype(str).str.split().str.len().astype(float)
        elif v == "servicio":
            out[v] = (X[v] == "hospitalizacion").astype(float)
        elif v == "sexo":
            out[v] = (X[v] == "M").astype(float)
        elif v == "turno":
            out[v] = X[v].astype(str).str.lower().map(TURNO_ORDEN).astype(float)
        elif v in X.columns:
            out[v] = pd.to_numeric(X[v].astype(float) if X[v].dtype == bool else X[v], errors="coerce")
        else:
            out[v] = np.nan
    return pd.DataFrame(out, index=X.index).reset_index(drop=True)


# ---------------------------------------------------------------------------- SHAP
def explicar(modelo_lgbm, Z: np.ndarray) -> tuple[np.ndarray, float]:
    """φ (n × p) en log-odds y valor base E[f(X)] con TreeExplainer (exacto para árboles)."""
    import shap
    ex = shap.TreeExplainer(modelo_lgbm)
    phi = ex.shap_values(Z)
    base = ex.expected_value
    if isinstance(phi, list):  # versiones antiguas: una matriz por clase
        phi, base = phi[1], base[1]
    phi = np.asarray(phi)
    if phi.ndim == 3:
        phi = phi[:, :, 1]
    base = float(np.ravel(base)[-1])
    return phi, base


def importancia_global(phi_g: pd.DataFrame, colores: pd.DataFrame) -> pd.DataFrame:
    """Impacto medio |φ| por variable, su porcentaje del total y la dirección (Spearman valor ↔ φ)."""
    filas = []
    total = phi_g.abs().mean().sum()
    for v in phi_g.columns:
        c = colores[v]
        ok = c.notna()
        rho = (pd.Series(c[ok]).rank().corr(pd.Series(phi_g[v][ok]).rank())
               if ok.sum() > 2 and c[ok].nunique() > 1 else np.nan)
        filas.append({"variable": v, "etiqueta": etiqueta(v), "dominio": ETIQUETAS.get(v, ("", "otro", ""))[1],
                      "media_abs_shap": phi_g[v].abs().mean(), "porcentaje": 100 * phi_g[v].abs().mean() / total,
                      "spearman_valor_shap": rho})
    return pd.DataFrame(filas).sort_values("media_abs_shap", ascending=False).reset_index(drop=True)


def perfil(phi_g: pd.DataFrame, colores: pd.DataFrame, mascara) -> tuple[pd.Series, pd.Series]:
    """SHAP medio y valor típico (mediana; proporción en binarias) de un grupo de snapshots."""
    m = np.asarray(mascara)
    c = colores[m]
    tipico = pd.Series({v: (c[v].mean() if c[v].dropna().isin([0, 1]).all() else c[v].median()) for v in c.columns})
    return phi_g[m].mean(), tipico


def _direccion(rho: float) -> str | None:
    if np.isnan(rho) or abs(rho) < 0.2:
        return None
    return "mayor" if rho > 0 else "menor"


def efecto_diagnosticos(phi_dx, codigos, nombres: dict[str, str], min_n: int = 100, top: int = 5) -> pd.DataFrame:
    """Diagnósticos (frecuencia ≥ min_n) cuya presencia más sube o baja la contribución del perfil de diagnósticos.

    diferencia = SHAP medio del perfil de diagnósticos con el código presente − con el código ausente (log-odds).
    """
    phi_dx = np.asarray(phi_dx, float)
    listas = pd.Series(codigos).fillna("").astype(str).str.split()
    cuenta = listas.explode().dropna().value_counts()
    filas = []
    for c in cuenta[cuenta >= min_n].index:
        pres = listas.apply(lambda l, c=c: c in l).to_numpy()
        if pres.all():
            continue
        filas.append({"codigo": c, "diagnostico": nombres.get(c, c), "n_snapshots": int(pres.sum()),
                      "shap_con": phi_dx[pres].mean(), "shap_sin": phi_dx[~pres].mean(),
                      "diferencia": phi_dx[pres].mean() - phi_dx[~pres].mean()})
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    df = df.sort_values("diferencia", ascending=False)
    return pd.concat([df.head(top).assign(efecto="aumenta"), df.tail(top).iloc[::-1].assign(efecto="reduce")])


def sintesis_medica(imp: pd.DataFrame, n: int = 5, dx: pd.DataFrame | None = None) -> str:
    """Texto en lenguaje clínico para las n variables con mayor impacto (sin términos de programación)."""
    lineas = []
    for i, (_, r) in enumerate(imp.head(n).iterrows()):
        alto = ETIQUETAS.get(r.variable, (r.etiqueta, "", r.etiqueta))[2]
        if alto.startswith("más "):
            alto = "tener " + alto  # concordancia: "tener más X se asocia…"
        d = _direccion(r.spearman_valor_shap)
        if r.variable == "dx_codigos" and dx is not None and len(dx):
            sube = "; ".join(dx[dx.efecto == "aumenta"].diagnostico.head(3))
            baja = "; ".join(dx[dx.efecto == "reduce"].diagnostico.head(3))
            cuerpo = (f"Importa más qué diagnósticos tiene el paciente que cuántos. Entre los más frecuentes, los que más "
                      f"elevan el riesgo son: {sube}. Los que más lo reducen: {baja}.")
        elif d:
            cuerpo = (f"{alto[0].upper() + alto[1:]} se asocia con un riesgo {d} de que la estancia supere lo habitual "
                      "para su servicio.")
        else:
            cuerpo = ("Su efecto no es lineal: según el rango de valores puede aumentar o disminuir el riesgo, por lo que "
                      "debe leerse junto con el resto del cuadro.")
        nota = f" {NOTAS[r.variable]}" if r.variable in NOTAS else ""
        lineas.append(f"{i + 1}. {r.etiqueta} ({r.dominio}; {r.porcentaje:.0f} % de la influencia total). {cuerpo}{nota}")
    return "\n".join(lineas)


def formatear_valor(var: str, v) -> str:
    """Valor típico legible para el waterfall (sin códigos internos)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if var == "servicio":
        return "hospitalización" if v >= 0.5 else "urgencias"
    if var == "sexo":
        return f"{v:.0%} hombres"
    if var == "horas_desde_ingreso":
        return f"{v:.0f} h"
    if var == "edad":
        return f"{v:.0f} años"
    if var == "hora_del_dia":
        return f"{v:.0f}:00"
    if var == "dx_codigos":
        return f"{v:.0f} diagnóstico" + ("" if round(v) == 1 else "s")
    if var.startswith(("ocupacion", "srv_ocupacion", "srv_frac")):
        return f"{v:.2f}"
    if 0 <= v <= 1 and var.startswith(("es_", "sin_", "has_", "nunca", "origen", "tras", "primer")):
        return f"{v:.0%} sí"
    return f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}"


# ---------------------------------------------------------------------------- gráficos
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _traducir(fig) -> None:
    """Pasa al español los textos fijos que dibuja shap."""
    reemplazos = [(r"^Sum of (\d+) other features$", r"Suma de otras \1 variables"),
                  (r"^(\d+) other features$", r"Otras \1 variables"), (r"^High$", "Alto"), (r"^Low$", "Bajo")]

    def tr(t):
        for pat, rep in reemplazos:
            t = re.sub(pat, rep, t)
        return t

    for ax in fig.axes:
        for t in ax.texts:
            t.set_text(tr(t.get_text()))
        ys = ax.get_yticklabels()
        if ys:
            ax.set_yticks(ax.get_yticks(), [tr(l.get_text()) for l in ys])


def explicacion(phi_g: pd.DataFrame, base: float, colores: pd.DataFrame):
    import shap
    return shap.Explanation(values=phi_g.to_numpy(), base_values=np.full(len(phi_g), base),
                            data=colores[phi_g.columns].to_numpy(), feature_names=[etiqueta(v) for v in phi_g.columns])


def graficar_beeswarm(exp, ruta, titulo: str, max_display: int = 15) -> None:
    import shap
    plt = _plt()
    shap.plots.beeswarm(exp, max_display=max_display, show=False, plot_size=(10, 7.5),
                        color_bar_label="Valor de la variable (bajo → alto)")
    plt.title(titulo, fontweight="bold", fontsize=11)
    plt.xlabel("Contribución al riesgo de estancia prolongada (valor SHAP, log-odds)")
    _traducir(plt.gcf())
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close("all")


def graficar_barras(exp, ruta, titulo: str, max_display: int = 15) -> None:
    import shap
    plt = _plt()
    shap.plots.bar(exp, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.title(titulo, fontweight="bold", fontsize=11)
    plt.xlabel("Impacto medio absoluto |φ| (log-odds)")
    _traducir(plt.gcf())
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close("all")


def graficar_waterfall(phi_medio: pd.Series, tipico: pd.Series, base: float, ruta, titulo: str,
                       max_display: int = 12) -> None:
    import shap
    plt = _plt()
    nombres = [f"{etiqueta(v)} [{formatear_valor(v, tipico.get(v))}]" for v in phi_medio.index]
    exp = shap.Explanation(values=phi_medio.to_numpy(), base_values=base, data=None, feature_names=nombres)
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    f = base + phi_medio.sum()
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    plt.title(f"{titulo}\nE[f(X)] = {base:.2f} (p = {sig(base):.2f}) → f = {f:.2f} (p = {sig(f):.2f})",
              fontweight="bold", fontsize=11)
    _traducir(fig)
    plt.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close("all")


def graficar_barras_por_servicio(imps: dict[str, pd.DataFrame], ruta, n: int = 12) -> None:
    plt = _plt()
    fig, ejes = plt.subplots(1, len(imps), figsize=(7.5 * len(imps), 6.5))
    for a, (srv, imp) in zip(np.atleast_1d(ejes), imps.items()):
        t = imp.head(n).iloc[::-1]
        a.barh([re.sub(r" \(.*\)$", "", e) for e in t.etiqueta], t.media_abs_shap, color="#1f77b4")
        a.set(title=srv, xlabel="Impacto medio absoluto |φ| (log-odds)")
        a.grid(alpha=0.3, axis="x")
    fig.suptitle("Variables con mayor impacto dentro de cada servicio", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ruta, dpi=300, bbox_inches="tight")
    plt.close(fig)
