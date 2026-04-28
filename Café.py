from __future__ import annotations

import io
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Dashboard | Precios del Café Mexicano",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.data_transformers.disable_max_rows()

APP_TITLE = "Estructura de precios del café mexicano"
APP_SUBTITLE = (
    "Exploración interactiva de precios convencionales, especiales, "
    "brechas, primas, correlaciones y segmentación territorial."
)

LOCAL_CSV_CANDIDATES = [
    "Base_Cafe.csv",
    "Base_Cafe(3).csv",
    "base_cafe.csv",
    "data/Base_Cafe.csv",
    "data/Base_Cafe(3).csv",
]

STATE_COLS = ["Veracruz", "Puebla", "Chiapas", "Oaxaca", "Guerrero"]

META_COLS = [
    "Estado",
    "Tipo_cafe",
    "Municipio",
    "Nombre de localidad más cercana a la finca",
]

CONV_MIN = [
    "Precio mínimo por Kilo de fruto o cereza convencional",
    "Precio mínimo por Kilo de pergamino lavado convencional",
    "Precio mínimo por Kilo de natural convencional",
    "Precio mínimo por Kilo de verde, oro, morteado convencional",
]

CONV_MAX = [
    "Precio máximo por Kilo de fruto o cereza convencional",
    "Precio máximo por Kilo de pergamino lavado convencional",
    "Precio máximo por Kilo de natural convencional",
    "Precio máximo por Kilo de verde, oro, morteado convencional",
]

ESP_MIN = [
    "Precio mínimo por Kilo de pergamino lavado especial",
    "Precio mínimo por Kilo de pergamino honey especial",
    "Precio mínimo por Kilo de pergamino semilavado especial",
    "Precio mínimo por Kilo de natural especial",
    "Precio mínimo por Kilo de café verde, oro, morteado especial",
]

ESP_MAX = [
    "Precio máximo por Kilo de Pergamino lavado especial",
    "Precio máximo por Kilo de pergamino honey especial",
    "Precio máximo por Kilo de pergamino semilavado especial",
    "Precio máximo por Kilo de natural especial",
    "Precio máximo por Kilo de café verde, oro o morteado especial",
]

SHORT_LABELS = {
    "Precio mínimo por Kilo de fruto o cereza convencional": "Cereza conv. mín.",
    "Precio mínimo por Kilo de pergamino lavado convencional": "Perg. lavado conv. mín.",
    "Precio mínimo por Kilo de natural convencional": "Natural conv. mín.",
    "Precio mínimo por Kilo de verde, oro, morteado convencional": "Verde/oro conv. mín.",
    "Precio máximo por Kilo de fruto o cereza convencional": "Cereza conv. máx.",
    "Precio máximo por Kilo de pergamino lavado convencional": "Perg. lavado conv. máx.",
    "Precio máximo por Kilo de natural convencional": "Natural conv. máx.",
    "Precio máximo por Kilo de verde, oro, morteado convencional": "Verde/oro conv. máx.",
    "Precio mínimo por Kilo de pergamino lavado especial": "Perg. lavado esp. mín.",
    "Precio mínimo por Kilo de pergamino honey especial": "Honey esp. mín.",
    "Precio mínimo por Kilo de pergamino semilavado especial": "Semilavado esp. mín.",
    "Precio mínimo por Kilo de natural especial": "Natural esp. mín.",
    "Precio mínimo por Kilo de café verde, oro, morteado especial": "Verde/oro esp. mín.",
    "Precio máximo por Kilo de Pergamino lavado especial": "Perg. lavado esp. máx.",
    "Precio máximo por Kilo de pergamino honey especial": "Honey esp. máx.",
    "Precio máximo por Kilo de pergamino semilavado especial": "Semilavado esp. máx.",
    "Precio máximo por Kilo de natural especial": "Natural esp. máx.",
    "Precio máximo por Kilo de café verde, oro o morteado especial": "Verde/oro esp. máx.",
}

THEMES = {
    "Café editorial": {
        "bg": "#F7F1E8",
        "surface": "#FFFDF8",
        "surface_2": "#F2E8DA",
        "ink": "#1F2933",
        "muted": "#6B5E53",
        "grid": "#E7DCCB",
        "primary": "#7C2D12",
        "secondary": "#0F766E",
        "accent": "#D97706",
        "danger": "#BE123C",
        "blue": "#2563EB",
    },
    "UP sobrio": {
        "bg": "#F8F7F4",
        "surface": "#FFFFFF",
        "surface_2": "#EFECE6",
        "ink": "#172033",
        "muted": "#68707E",
        "grid": "#E4E1DA",
        "primary": "#8A1538",
        "secondary": "#00685E",
        "accent": "#C99700",
        "danger": "#B91C1C",
        "blue": "#1D4ED8",
    },
    "Nocturno elegante": {
        "bg": "#111827",
        "surface": "#182235",
        "surface_2": "#243044",
        "ink": "#F9FAFB",
        "muted": "#CBD5E1",
        "grid": "#334155",
        "primary": "#F59E0B",
        "secondary": "#2DD4BF",
        "accent": "#F472B6",
        "danger": "#FB7185",
        "blue": "#60A5FA",
    },
}

# ============================================================
# CARGA Y LIMPIEZA
# ============================================================

def _read_csv_with_fallback(raw: bytes) -> pd.DataFrame:
    for enc in ("utf-8-sig", "utf-8", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(raw))


@st.cache_data(show_spinner=False)
def read_uploaded_csv(raw: bytes, filename: str) -> pd.DataFrame:
    return _read_csv_with_fallback(raw)


@st.cache_data(show_spinner=False)
def read_local_csv(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        return _read_csv_with_fallback(f.read())


def cell_has_value(x) -> bool:
    if pd.isna(x):
        return False
    s = str(x).strip()
    return s != "" and s.lower() not in {"nan", "none", "0", "0.0", "no"}


def clean_numeric(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    cleaned = (
        s.astype(str)
        .str.replace(r"[$\s]", "", regex=True)
        .str.replace(",", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "—": np.nan, "-": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def infer_estado(row: pd.Series) -> str:
    for estado in STATE_COLS:
        if estado in row.index and cell_has_value(row[estado]):
            return estado

    otro = row.get("Otro (especifique)", np.nan)
    if cell_has_value(otro):
        return str(otro).strip().title()

    return "Otro"


def infer_tipo_cafe(row: pd.Series) -> str:
    arabica = cell_has_value(row.get("Arábica", np.nan))
    robusta = cell_has_value(row.get("Robusta", np.nan))

    if arabica and robusta:
        return "Mixto"
    if arabica:
        return "Arábica"
    if robusta:
        return "Robusta"

    return "No especificado"


def infer_mercado(col: str) -> str:
    return "Especial" if "especial" in col.lower() else "Convencional"


def infer_rango(col: str) -> str:
    return "Mínimo" if "mínimo" in col.lower() else "Máximo"


def infer_proceso(col: str) -> str:
    low = col.lower()

    if "honey" in low:
        return "Honey"
    if "semilavado" in low:
        return "Semilavado"
    if "pergamino" in low and "lavado" in low:
        return "Pergamino lavado"
    if "fruto" in low or "cereza" in low:
        return "Fruto/cereza"
    if "natural" in low:
        return "Natural"
    if "verde" in low or "oro" in low or "morteado" in low:
        return "Verde/oro/morteado"

    return "Otro proceso"


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    df["Estado"] = df.apply(infer_estado, axis=1)
    df["Tipo_cafe"] = df.apply(infer_tipo_cafe, axis=1)

    price_cols = [c for c in df.columns if "precio" in c.lower()]
    for col in price_cols:
        df[col] = clean_numeric(df[col])

    if "Municipio" in df.columns:
        df["Municipio"] = (
            df["Municipio"]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "": np.nan})
        )

    if "Nombre de localidad más cercana a la finca" in df.columns:
        df["Nombre de localidad más cercana a la finca"] = (
            df["Nombre de localidad más cercana a la finca"]
            .astype(str)
            .str.strip()
            .replace({"nan": np.nan, "": np.nan})
        )

    return df


def make_long_prices(df: pd.DataFrame, price_cols: list[str]) -> pd.DataFrame:
    meta = [c for c in META_COLS if c in df.columns]
    frames = []

    for col in price_cols:
        if col not in df.columns:
            continue

        tmp = df[meta + [col]].copy()
        tmp = tmp.rename(columns={col: "Precio"})
        tmp["Precio"] = clean_numeric(tmp["Precio"])
        tmp["Columna_original"] = col
        tmp["Variable"] = SHORT_LABELS.get(col, col)
        tmp["Mercado"] = infer_mercado(col)
        tmp["Rango"] = infer_rango(col)
        tmp["Proceso"] = infer_proceso(col)
        frames.append(tmp)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["Precio"])
    out = out[out["Precio"] > 0]

    return out


def build_gap_df(df: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    pairs.extend(zip(CONV_MIN, CONV_MAX, ["Convencional"] * len(CONV_MIN)))
    pairs.extend(zip(ESP_MIN, ESP_MAX, ["Especial"] * len(ESP_MIN)))

    meta = [c for c in META_COLS if c in df.columns]
    rows = []

    for col_min, col_max, mercado in pairs:
        if col_min not in df.columns or col_max not in df.columns:
            continue

        tmp = df[meta + [col_min, col_max]].copy()
        tmp[col_min] = clean_numeric(tmp[col_min])
        tmp[col_max] = clean_numeric(tmp[col_max])

        tmp = tmp.dropna(subset=[col_min, col_max], how="any")
        tmp = tmp[(tmp[col_min] > 0) & (tmp[col_max] > 0)]

        if tmp.empty:
            continue

        tmp["Precio_min"] = tmp[col_min]
        tmp["Precio_max"] = tmp[col_max]
        tmp["Brecha"] = tmp["Precio_max"] - tmp["Precio_min"]
        tmp["Mercado"] = mercado
        tmp["Proceso"] = infer_proceso(col_min)
        tmp["Producto"] = SHORT_LABELS.get(col_min, col_min).replace(" mín.", "")
        tmp = tmp[tmp["Brecha"] >= 0]

        rows.append(
            tmp[
                meta
                + [
                    "Mercado",
                    "Proceso",
                    "Producto",
                    "Precio_min",
                    "Precio_max",
                    "Brecha",
                ]
            ]
        )

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def add_row_medians(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    conv_cols = [c for c in CONV_MIN + CONV_MAX if c in df.columns]
    esp_cols = [c for c in ESP_MIN + ESP_MAX if c in df.columns]

    if conv_cols:
        df["_med_conv"] = df[conv_cols].replace(0, np.nan).median(axis=1)
    else:
        df["_med_conv"] = np.nan

    if esp_cols:
        df["_med_esp"] = df[esp_cols].replace(0, np.nan).median(axis=1)
    else:
        df["_med_esp"] = np.nan

    df["_premium_abs"] = df["_med_esp"] - df["_med_conv"]
    df["_premium_pct"] = (df["_med_esp"] / df["_med_conv"] - 1) * 100
    df.loc[~np.isfinite(df["_premium_pct"]), "_premium_pct"] = np.nan

    return df


def safe_median(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").replace(0, np.nan).dropna()
    return float(s.median()) if len(s) else np.nan


def money(x, digits: int = 0) -> str:
    if pd.isna(x) or not np.isfinite(x):
        return "N/D"
    return f"${x:,.{digits}f}"


def pct(x, digits: int = 1) -> str:
    if pd.isna(x) or not np.isfinite(x):
        return "N/D"
    return f"{x:,.{digits}f}%"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## ☕ Datos")

uploaded = st.sidebar.file_uploader(
    "Carga tu CSV de café",
    type=["csv"],
    help="También puedes dejar Base_Cafe.csv en la misma carpeta del app.",
)

if uploaded is not None:
    df_raw = read_uploaded_csv(uploaded.getvalue(), uploaded.name)
    source_name = uploaded.name
else:
    found_path = None

    for candidate in LOCAL_CSV_CANDIDATES:
        if Path(candidate).exists():
            found_path = candidate
            break

    if found_path is None:
        st.error(
            "No encontré la base. Sube un CSV o coloca Base_Cafe.csv junto al archivo .py."
        )
        st.stop()

    df_raw = read_local_csv(found_path)
    source_name = found_path

df = prepare_data(df_raw)
PRICE_COLS = [c for c in df.columns if "precio" in c.lower()]
df = add_row_medians(df)

long_all = make_long_prices(df, PRICE_COLS)
gap_all = build_gap_df(df)

st.sidebar.caption(f"Fuente activa: `{source_name}`")

theme_name = st.sidebar.selectbox(
    "Paleta visual",
    options=list(THEMES.keys()),
    index=0,
)

C = THEMES[theme_name]

STATE_PALETTE = {
    "Chiapas": C["secondary"],
    "Veracruz": C["blue"],
    "Puebla": C["accent"],
    "Oaxaca": C["danger"],
    "Guerrero": "#7C3AED",
    "Otro": "#64748B",
}

TYPE_PALETTE = {
    "Arábica": C["primary"],
    "Robusta": C["accent"],
    "Mixto": C["secondary"],
    "No especificado": "#94A3B8",
}

MARKET_PALETTE = {
    "Convencional": C["blue"],
    "Especial": C["accent"],
}

estado_options = sorted(df["Estado"].dropna().unique().tolist())
tipo_options = sorted(df["Tipo_cafe"].dropna().unique().tolist())

municipio_options = (
    sorted(df["Municipio"].dropna().unique().tolist())
    if "Municipio" in df.columns
    else []
)

st.sidebar.markdown("## Filtros")

estados_sel = st.sidebar.multiselect(
    "Estado",
    options=estado_options,
    default=estado_options,
)

tipos_sel = st.sidebar.multiselect(
    "Tipo de café",
    options=tipo_options,
    default=tipo_options,
)

if municipio_options:
    municipios_sel = st.sidebar.multiselect(
        "Municipio",
        options=municipio_options,
        default=municipio_options,
    )
else:
    municipios_sel = []

market_sel_side = st.sidebar.multiselect(
    "Mercado",
    options=["Convencional", "Especial"],
    default=["Convencional", "Especial"],
)

df_f = df[df["Estado"].isin(estados_sel) & df["Tipo_cafe"].isin(tipos_sel)].copy()

if municipio_options:
    df_f = df_f[df_f["Municipio"].isin(municipios_sel)]

long_f = make_long_prices(df_f, PRICE_COLS)

if not long_f.empty:
    long_f = long_f[long_f["Mercado"].isin(market_sel_side)]

gap_f = build_gap_df(df_f)

if not gap_f.empty:
    gap_f = gap_f[gap_f["Mercado"].isin(market_sel_side)]


# ============================================================
# ESTILO
# ============================================================

FONT = "Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif"

STATE_DOMAIN = list(STATE_PALETTE.keys())
STATE_RANGE = list(STATE_PALETTE.values())

TYPE_DOMAIN = list(TYPE_PALETTE.keys())
TYPE_RANGE = list(TYPE_PALETTE.values())

MARKET_DOMAIN = list(MARKET_PALETTE.keys())
MARKET_RANGE = list(MARKET_PALETTE.values())

st.markdown(
    f"""
    <style>
    :root {{
        --bg: {C["bg"]};
        --surface: {C["surface"]};
        --surface-2: {C["surface_2"]};
        --ink: {C["ink"]};
        --muted: {C["muted"]};
        --primary: {C["primary"]};
        --secondary: {C["secondary"]};
        --accent: {C["accent"]};
        --grid: {C["grid"]};
    }}

    .stApp {{
        background:
          radial-gradient(circle at top left, rgba(217,119,6,0.12), transparent 28rem),
          radial-gradient(circle at top right, rgba(15,118,110,0.11), transparent 30rem),
          var(--bg);
        color: var(--ink);
        font-family: {FONT};
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, var(--surface), var(--surface-2));
        border-right: 1px solid var(--grid);
    }}

    .hero {{
        padding: 1.45rem 1.55rem;
        border: 1px solid var(--grid);
        border-radius: 28px;
        background:
          linear-gradient(135deg, rgba(255,255,255,0.84), rgba(255,255,255,0.50)),
          linear-gradient(135deg, rgba(124,45,18,0.10), rgba(15,118,110,0.08));
        box-shadow: 0 18px 45px rgba(31, 41, 51, 0.08);
        margin-bottom: 1.1rem;
    }}

    .eyebrow {{
        color: var(--primary);
        font-size: 0.78rem;
        letter-spacing: .11em;
        text-transform: uppercase;
        font-weight: 800;
        margin-bottom: .35rem;
    }}

    .hero-title {{
        font-size: clamp(2.0rem, 4.4vw, 3.4rem);
        line-height: 1.02;
        font-weight: 850;
        letter-spacing: -0.055em;
        color: var(--ink);
        margin: 0;
    }}

    .hero-subtitle {{
        max-width: 1020px;
        margin-top: .75rem;
        font-size: 1.02rem;
        line-height: 1.55;
        color: var(--muted);
    }}

    .metric-card {{
        padding: 1.1rem 1.15rem;
        border-radius: 24px;
        border: 1px solid var(--grid);
        background: var(--surface);
        box-shadow: 0 12px 32px rgba(31, 41, 51, 0.07);
        min-height: 122px;
    }}

    .metric-label {{
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 800;
        font-size: .72rem;
    }}

    .metric-value {{
        margin-top: .35rem;
        color: var(--ink);
        font-size: 1.9rem;
        font-weight: 850;
        letter-spacing: -0.04em;
    }}

    .metric-help {{
        margin-top: .25rem;
        color: var(--muted);
        font-size: .86rem;
        line-height: 1.35;
    }}

    .section-note {{
        color: var(--muted);
        font-size: .95rem;
        line-height: 1.55;
        margin: -.25rem 0 1rem 0;
    }}

    div[data-testid="stMetric"] {{
        background: var(--surface);
        border: 1px solid var(--grid);
        padding: 1rem;
        border-radius: 22px;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: .45rem;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 999px;
        padding: .55rem .95rem;
        background: rgba(255,255,255,0.45);
        border: 1px solid var(--grid);
    }}

    .stTabs [aria-selected="true"] {{
        background: var(--surface);
        color: var(--primary);
        box-shadow: 0 8px 20px rgba(31, 41, 51, 0.08);
    }}

    h1, h2, h3 {{
        color: var(--ink);
        letter-spacing: -0.025em;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def polish(chart: alt.Chart, height: int = 360) -> alt.Chart:
    return (
        chart.properties(height=height)
        .configure_axis(
            labelFont=FONT,
            titleFont=FONT,
            labelColor=C["muted"],
            titleColor=C["ink"],
            gridColor=C["grid"],
            domainColor=C["grid"],
            tickColor=C["grid"],
            labelFontSize=12,
            titleFontSize=13,
        )
        .configure_title(
            font=FONT,
            color=C["ink"],
            fontSize=17,
            fontWeight=700,
            anchor="start",
            dy=-4,
        )
        .configure_legend(
            labelFont=FONT,
            titleFont=FONT,
            labelColor=C["ink"],
            titleColor=C["muted"],
            orient="bottom",
            padding=8,
        )
        .configure_view(strokeWidth=0)
    )


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HEADER + KPIs
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">Dashboard analítico · Café · México</div>
        <h1 class="hero-title">{APP_TITLE}</h1>
        <div class="hero-subtitle">{APP_SUBTITLE}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if long_f.empty:
    st.warning("No hay observaciones positivas de precio con los filtros actuales.")
    st.stop()

conv_med = safe_median(long_f.loc[long_f["Mercado"] == "Convencional", "Precio"])
esp_med = safe_median(long_f.loc[long_f["Mercado"] == "Especial", "Precio"])
premium_med = safe_median(df_f["_premium_abs"])
premium_pct_med = safe_median(df_f["_premium_pct"])

k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    metric_card("Productores", f"{len(df_f):,}", "Registros después de filtros.")

with k2:
    metric_card("Estados", f"{df_f['Estado'].nunique():,}", "Cobertura territorial filtrada.")

with k3:
    metric_card("Mediana conv.", f"{money(conv_med)}", "Precio convencional mediano.")

with k4:
    metric_card("Mediana esp.", f"{money(esp_med)}", "Precio especial mediano.")

with k5:
    metric_card("Prima esp.", f"{money(premium_med)}", f"Mediana fila a fila · {pct(premium_pct_med)}")

st.caption(
    "Lectura sugerida: la prima especial se calcula como la diferencia entre la mediana de precios especiales "
    "y la mediana de precios convencionales por productor, cuando ambas existen."
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "01 · Panorama",
        "02 · Precios y brechas",
        "03 · Prima especial",
        "04 · Correlaciones",
        "05 · PCA y clusters",
        "06 · Tabla",
    ]
)


# ------------------------------------------------------------
# TAB 1
# ------------------------------------------------------------

with tab1:
    st.subheader("Panorama territorial")

    st.markdown(
        '<div class="section-note">Una vista compacta para entender la composición de la muestra, '
        "la distribución por tipo de café y las diferencias iniciales entre mercados.</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.25, 1])

    cnt = (
        df_f.groupby(["Estado", "Tipo_cafe"], dropna=False)
        .size()
        .reset_index(name="Productores")
    )

    with c1:
        chart = (
            alt.Chart(cnt, title="Productores por estado y tipo de café")
            .mark_bar(cornerRadiusEnd=7)
            .encode(
                x=alt.X("Estado:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Productores:Q", title="Productores"),
                color=alt.Color(
                    "Tipo_cafe:N",
                    title="Tipo",
                    scale=alt.Scale(domain=TYPE_DOMAIN, range=TYPE_RANGE),
                ),
                tooltip=[
                    alt.Tooltip("Estado:N"),
                    alt.Tooltip("Tipo_cafe:N", title="Tipo"),
                    alt.Tooltip("Productores:Q", format=","),
                ],
            )
        )

        st.altair_chart(polish(chart, 390), use_container_width=True)

    with c2:
        total_estado = (
            df_f["Estado"]
            .value_counts()
            .rename_axis("Estado")
            .reset_index(name="Productores")
        )

        donut = (
            alt.Chart(total_estado, title="Participación por estado")
            .mark_arc(innerRadius=72, outerRadius=118, cornerRadius=7)
            .encode(
                theta=alt.Theta("Productores:Q"),
                color=alt.Color(
                    "Estado:N",
                    scale=alt.Scale(domain=STATE_DOMAIN, range=STATE_RANGE),
                    legend=alt.Legend(title="Estado"),
                ),
                tooltip=[
                    alt.Tooltip("Estado:N"),
                    alt.Tooltip("Productores:Q", format=","),
                ],
            )
        )

        st.altair_chart(polish(donut, 390), use_container_width=True)

    st.markdown("### Precios medianos por estado")

    avg_market = (
        long_f.groupby(["Estado", "Mercado"], as_index=False)["Precio"]
        .median()
        .rename(columns={"Precio": "Precio_mediano"})
    )

    chart_avg = (
        alt.Chart(avg_market, title="Mediana de precios por mercado")
        .mark_bar(cornerRadiusEnd=7)
        .encode(
            x=alt.X("Estado:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Precio_mediano:Q", title="MXN/kg"),
            xOffset=alt.XOffset("Mercado:N"),
            color=alt.Color(
                "Mercado:N",
                scale=alt.Scale(domain=MARKET_DOMAIN, range=MARKET_RANGE),
            ),
            tooltip=[
                "Estado:N",
                "Mercado:N",
                alt.Tooltip("Precio_mediano:Q", title="Mediana", format=",.1f"),
            ],
        )
    )

    st.altair_chart(polish(chart_avg, 405), use_container_width=True)

    st.markdown("### Distribución global de precios")

    hist = (
        alt.Chart(long_f, title="Distribución de precios positivos")
        .mark_area(opacity=0.55, interpolate="monotone")
        .encode(
            x=alt.X("Precio:Q", bin=alt.Bin(maxbins=38), title="MXN/kg"),
            y=alt.Y("count():Q", title="Frecuencia"),
            color=alt.Color(
                "Mercado:N",
                scale=alt.Scale(domain=MARKET_DOMAIN, range=MARKET_RANGE),
            ),
            tooltip=[
                "Mercado:N",
                alt.Tooltip("count():Q", title="Frecuencia", format=","),
            ],
        )
    )

    st.altair_chart(polish(hist, 320), use_container_width=True)


# ------------------------------------------------------------
# TAB 2
# ------------------------------------------------------------

with tab2:
    st.subheader("Precios y brechas de negociación")

    st.markdown(
        '<div class="section-note">La brecha se define como precio máximo menos precio mínimo para el mismo '
        "producto. Captura dispersión, margen de negociación y posible heterogeneidad de calidad.</div>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.1, 1])

    mercado_box = st.radio(
        "Mercado para boxplots",
        options=["Convencional", "Especial", "Ambos"],
        horizontal=True,
        key="market_box",
    )

    if mercado_box == "Ambos":
        d_box = long_f.copy()
    else:
        d_box = long_f[long_f["Mercado"] == mercado_box].copy()

    order_var = (
        d_box.groupby("Variable")["Precio"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    with c1:
        box = (
            alt.Chart(d_box, title=f"Distribución por producto · {mercado_box}")
            .mark_boxplot(extent=1.5, size=32)
            .encode(
                x=alt.X("Variable:N", sort=order_var, title=None, axis=alt.Axis(labelAngle=32)),
                y=alt.Y("Precio:Q", title="MXN/kg"),
                color=alt.Color(
                    "Mercado:N",
                    scale=alt.Scale(domain=MARKET_DOMAIN, range=MARKET_RANGE),
                ),
                tooltip=[
                    "Mercado:N",
                    "Proceso:N",
                    "Variable:N",
                    alt.Tooltip("Precio:Q", aggregate="median", title="Mediana", format=",.1f"),
                ],
            )
        )

        st.altair_chart(polish(box, 450), use_container_width=True)

    with c2:
        med_state_proc = (
            d_box.groupby(["Estado", "Proceso"], as_index=False)["Precio"]
            .median()
            .rename(columns={"Precio": "Precio_mediano"})
        )

        heat = (
            alt.Chart(med_state_proc, title="Mapa territorial por proceso")
            .mark_rect(cornerRadius=4)
            .encode(
                x=alt.X("Proceso:N", title=None, axis=alt.Axis(labelAngle=30)),
                y=alt.Y("Estado:N", title=None),
                color=alt.Color(
                    "Precio_mediano:Q",
                    title="MXN/kg",
                    scale=alt.Scale(scheme="goldorange"),
                ),
                tooltip=[
                    "Estado:N",
                    "Proceso:N",
                    alt.Tooltip("Precio_mediano:Q", title="Mediana", format=",.1f"),
                ],
            )
        )

        text = heat.mark_text(fontWeight=700).encode(
            text=alt.Text("Precio_mediano:Q", format=".0f"),
            color=alt.condition(
                "datum.Precio_mediano > 100",
                alt.value("white"),
                alt.value(C["ink"]),
            ),
        )

        st.altair_chart(polish(heat + text, 450), use_container_width=True)

    st.markdown("### Brechas mínimo–máximo")

    if gap_f.empty:
        st.info("No hay pares mínimo–máximo suficientes con los filtros actuales.")
    else:
        mercado_gap = st.radio(
            "Mercado para brechas",
            options=["Convencional", "Especial", "Ambos"],
            horizontal=True,
            key="market_gap",
        )

        d_gap = gap_f if mercado_gap == "Ambos" else gap_f[gap_f["Mercado"] == mercado_gap]

        c3, c4 = st.columns([1.1, 1])

        with c3:
            order_gap = (
                d_gap.groupby("Producto")["Brecha"]
                .median()
                .sort_values(ascending=False)
                .index.tolist()
            )

            gap_box = (
                alt.Chart(d_gap, title=f"Brecha por producto · {mercado_gap}")
                .mark_boxplot(extent=1.5, size=32)
                .encode(
                    x=alt.X("Producto:N", sort=order_gap, title=None, axis=alt.Axis(labelAngle=32)),
                    y=alt.Y("Brecha:Q", title="Brecha MXN/kg"),
                    color=alt.Color(
                        "Mercado:N",
                        scale=alt.Scale(domain=MARKET_DOMAIN, range=MARKET_RANGE),
                    ),
                    tooltip=[
                        "Estado:N",
                        "Mercado:N",
                        "Producto:N",
                        alt.Tooltip("Brecha:Q", format=",.1f"),
                    ],
                )
            )

            st.altair_chart(polish(gap_box, 410), use_container_width=True)

        with c4:
            gap_state = (
                d_gap.groupby(["Estado", "Mercado"], as_index=False)["Brecha"]
                .median()
                .rename(columns={"Brecha": "Brecha_mediana"})
            )

            gap_bar = (
                alt.Chart(gap_state, title="Brecha mediana por estado")
                .mark_bar(cornerRadiusEnd=7)
                .encode(
                    x=alt.X("Estado:N", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Brecha_mediana:Q", title="MXN/kg"),
                    xOffset=alt.XOffset("Mercado:N"),
                    color=alt.Color(
                        "Mercado:N",
                        scale=alt.Scale(domain=MARKET_DOMAIN, range=MARKET_RANGE),
                    ),
                    tooltip=[
                        "Estado:N",
                        "Mercado:N",
                        alt.Tooltip("Brecha_mediana:Q", title="Brecha mediana", format=",.1f"),
                    ],
                )
            )

            st.altair_chart(polish(gap_bar, 410), use_container_width=True)


# ------------------------------------------------------------
# TAB 3
# ------------------------------------------------------------

with tab3:
    st.subheader("Prima especial")

    st.markdown(
        '<div class="section-note">Compara, por productor, la mediana de precios especiales contra la mediana '
        "de precios convencionales. Esta vista es útil para hablar de calidad, diferenciación y captura de valor.</div>",
        unsafe_allow_html=True,
    )

    premium_df = df_f.dropna(subset=["_med_conv", "_med_esp"]).copy()
    premium_df = premium_df[(premium_df["_med_conv"] > 0) & (premium_df["_med_esp"] > 0)]

    if premium_df.empty:
        st.info("No hay suficientes observaciones con precio convencional y especial en la misma fila.")
    else:
        c1, c2 = st.columns([1.15, 1])

        with c1:
            max_lim = float(
                np.nanmax([premium_df["_med_conv"].max(), premium_df["_med_esp"].max()])
            )
            max_lim = max(max_lim * 1.08, 1.0)

            line_df = pd.DataFrame({"x": [0, max_lim], "y": [0, max_lim]})

            scatter = (
                alt.Chart(premium_df, title="Especial vs convencional")
                .mark_circle(size=92, opacity=0.78, stroke="white", strokeWidth=0.7)
                .encode(
                    x=alt.X(
                        "_med_conv:Q",
                        title="Mediana convencional · MXN/kg",
                        scale=alt.Scale(domain=[0, max_lim]),
                    ),
                    y=alt.Y(
                        "_med_esp:Q",
                        title="Mediana especial · MXN/kg",
                        scale=alt.Scale(domain=[0, max_lim]),
                    ),
                    color=alt.Color(
                        "Estado:N",
                        scale=alt.Scale(domain=STATE_DOMAIN, range=STATE_RANGE),
                    ),
                    shape=alt.Shape("Tipo_cafe:N", title="Tipo"),
                    tooltip=[
                        "Estado:N",
                        "Tipo_cafe:N",
                        alt.Tooltip("_med_conv:Q", title="Convencional", format=",.1f"),
                        alt.Tooltip("_med_esp:Q", title="Especial", format=",.1f"),
                        alt.Tooltip("_premium_abs:Q", title="Prima", format=",.1f"),
                        alt.Tooltip("_premium_pct:Q", title="Prima %", format=",.1f"),
                    ],
                )
            )

            diag = (
                alt.Chart(line_df)
                .mark_line(strokeDash=[6, 6], color=C["muted"])
                .encode(x="x:Q", y="y:Q")
            )

            st.altair_chart(polish(diag + scatter, 470), use_container_width=True)

        with c2:
            premium_state = (
                premium_df.groupby("Estado", as_index=False)
                .agg(
                    Prima_mediana=("_premium_abs", "median"),
                    Prima_pct_mediana=("_premium_pct", "median"),
                    n=("_premium_abs", "size"),
                )
            )

            bar = (
                alt.Chart(premium_state, title="Prima mediana por estado")
                .mark_bar(cornerRadiusEnd=7)
                .encode(
                    x=alt.X("Estado:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Prima_mediana:Q", title="MXN/kg"),
                    color=alt.Color(
                        "Estado:N",
                        scale=alt.Scale(domain=STATE_DOMAIN, range=STATE_RANGE),
                        legend=None,
                    ),
                    tooltip=[
                        "Estado:N",
                        alt.Tooltip("Prima_mediana:Q", title="Prima", format=",.1f"),
                        alt.Tooltip("Prima_pct_mediana:Q", title="Prima %", format=",.1f"),
                        alt.Tooltip("n:Q", title="Observaciones", format=","),
                    ],
                )
            )

            rule = (
                alt.Chart(pd.DataFrame({"y": [0]}))
                .mark_rule(color=C["muted"])
                .encode(y="y:Q")
            )

            st.altair_chart(polish(rule + bar, 470), use_container_width=True)

        st.markdown("### Ranking de prima por productor/localidad")

        rank_cols = [
            c
            for c in [
                "Estado",
                "Municipio",
                "Nombre de localidad más cercana a la finca",
                "Tipo_cafe",
                "_med_conv",
                "_med_esp",
                "_premium_abs",
                "_premium_pct",
            ]
            if c in premium_df.columns
        ]

        rank = premium_df[rank_cols].sort_values("_premium_abs", ascending=False).head(25)

        rename_map = {
            "_med_conv": "Mediana convencional",
            "_med_esp": "Mediana especial",
            "_premium_abs": "Prima MXN/kg",
            "_premium_pct": "Prima %",
        }

        st.dataframe(
            rank.rename(columns=rename_map),
            use_container_width=True,
            height=360,
        )


# ------------------------------------------------------------
# TAB 4
# ------------------------------------------------------------

with tab4:
    st.subheader("Correlaciones de Spearman")

    st.markdown(
        '<div class="section-note">Spearman captura relaciones monótonas y es más estable cuando hay colas, '
        "outliers o escalas de precio heterogéneas.</div>",
        unsafe_allow_html=True,
    )

    default_corr = [c for c in CONV_MIN + CONV_MAX + ESP_MIN + ESP_MAX if c in PRICE_COLS][:10]

    corr_cols = st.multiselect(
        "Variables de precio",
        options=PRICE_COLS,
        default=default_corr,
        format_func=lambda c: SHORT_LABELS.get(c, c),
    )

    if len(corr_cols) < 2:
        st.info("Selecciona al menos dos variables.")
    else:
        corr_data = df_f[corr_cols].replace(0, np.nan)
        corr = corr_data.corr(method="spearman", min_periods=4)
        corr = corr.rename(index=SHORT_LABELS, columns=SHORT_LABELS)

        corr_long = (
            corr.reset_index()
            .melt(id_vars="index", var_name="Variable_y", value_name="rho")
            .rename(columns={"index": "Variable_x"})
            .dropna(subset=["rho"])
        )

        heat = (
            alt.Chart(corr_long, title="Matriz de correlación")
            .mark_rect(cornerRadius=4)
            .encode(
                x=alt.X("Variable_y:N", title=None, axis=alt.Axis(labelAngle=35)),
                y=alt.Y("Variable_x:N", title=None),
                color=alt.Color(
                    "rho:Q",
                    title="ρ",
                    scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
                ),
                tooltip=[
                    alt.Tooltip("Variable_x:N", title="Variable X"),
                    alt.Tooltip("Variable_y:N", title="Variable Y"),
                    alt.Tooltip("rho:Q", title="ρ Spearman", format=".2f"),
                ],
            )
        )

        text = heat.mark_text(fontSize=11, fontWeight=700).encode(
            text=alt.Text("rho:Q", format=".2f"),
            color=alt.condition(
                "abs(datum.rho) > 0.55",
                alt.value("white"),
                alt.value(C["ink"]),
            ),
        )

        st.altair_chart(polish(heat + text, 560), use_container_width=True)

        st.markdown("### Relación bivariada")

        c1, c2 = st.columns(2)

        with c1:
            var_x = st.selectbox(
                "Eje X",
                options=corr_cols,
                index=0,
                format_func=lambda c: SHORT_LABELS.get(c, c),
            )

        with c2:
            y_options = [c for c in corr_cols if c != var_x]

            var_y = st.selectbox(
                "Eje Y",
                options=y_options,
                index=0,
                format_func=lambda c: SHORT_LABELS.get(c, c),
            )

        scatter_df = df_f[[var_x, var_y, "Estado", "Tipo_cafe"]].replace(0, np.nan).dropna()

        if scatter_df.empty:
            st.info("No hay suficientes observaciones para ese par.")
        else:
            scatter = (
                alt.Chart(scatter_df, title="Dispersión con tendencia lineal")
                .mark_circle(size=82, opacity=0.76, stroke="white", strokeWidth=0.6)
                .encode(
                    x=alt.X(f"{var_x}:Q", title=SHORT_LABELS.get(var_x, var_x)),
                    y=alt.Y(f"{var_y}:Q", title=SHORT_LABELS.get(var_y, var_y)),
                    color=alt.Color(
                        "Estado:N",
                        scale=alt.Scale(domain=STATE_DOMAIN, range=STATE_RANGE),
                    ),
                    shape=alt.Shape("Tipo_cafe:N", title="Tipo"),
                    tooltip=[
                        "Estado:N",
                        "Tipo_cafe:N",
                        alt.Tooltip(
                            f"{var_x}:Q",
                            title=SHORT_LABELS.get(var_x, var_x),
                            format=",.1f",
                        ),
                        alt.Tooltip(
                            f"{var_y}:Q",
                            title=SHORT_LABELS.get(var_y, var_y),
                            format=",.1f",
                        ),
                    ],
                )
            )

            trend = (
                alt.Chart(scatter_df)
                .transform_regression(var_x, var_y)
                .mark_line(color=C["primary"], strokeWidth=3)
                .encode(
                    x=alt.X(f"{var_x}:Q"),
                    y=alt.Y(f"{var_y}:Q"),
                )
            )

            st.altair_chart(polish(scatter + trend, 430), use_container_width=True)


# ------------------------------------------------------------
# TAB 5
# ------------------------------------------------------------

with tab5:
    st.subheader("PCA y clustering")

    st.markdown(
        '<div class="section-note">Segmenta productores/localidades según su perfil de precios. '
        "PCA resume la estructura multivariada; K-means sugiere grupos interpretables.</div>",
        unsafe_allow_html=True,
    )

    candidate_features = [
        c for c in PRICE_COLS if df_f[c].replace(0, np.nan).notna().sum() >= 8
    ]

    default_features = [
        c for c in CONV_MIN + CONV_MAX + ESP_MIN + ESP_MAX if c in candidate_features
    ]

    if len(default_features) > 12:
        default_features = default_features[:12]

    feat_cols = st.multiselect(
        "Variables para clustering",
        options=candidate_features,
        default=default_features,
        format_func=lambda c: SHORT_LABELS.get(c, c),
    )

    k_val = st.slider("Número de clusters", min_value=2, max_value=6, value=3)

    if len(feat_cols) < 2:
        st.info("Selecciona al menos dos variables de precio.")
    else:
        X = df_f[feat_cols].replace(0, np.nan).copy()
        X = X.dropna(how="all")

        valid_index = X.index

        X = X.fillna(X.median(numeric_only=True))
        X = X.loc[:, X.nunique(dropna=True) > 1]

        if X.shape[1] < 2 or X.shape[0] <= k_val:
            st.warning("No hay suficientes observaciones o variabilidad para PCA/K-means con esos filtros.")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            km = KMeans(n_clusters=k_val, random_state=42, n_init=20)
            labels = km.fit_predict(X_scaled)

            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X_scaled)

            meta_cols = [c for c in ["Estado", "Tipo_cafe", "Municipio"] if c in df_f.columns]

            cluster_df = df_f.loc[valid_index, meta_cols].copy()
            cluster_df["Cluster"] = [f"C{int(x) + 1}" for x in labels]
            cluster_df["PC1"] = coords[:, 0]
            cluster_df["PC2"] = coords[:, 1]

            explained = pca.explained_variance_ratio_

            sil = np.nan
            if len(set(labels)) > 1 and X_scaled.shape[0] > k_val:
                sil = silhouette_score(X_scaled, labels)

            c1, c2, c3 = st.columns(3)

            c1.metric("Varianza PC1", f"{explained[0] * 100:.1f}%")
            c2.metric("Varianza PC2", f"{explained[1] * 100:.1f}%")
            c3.metric("Silhouette", "N/D" if pd.isna(sil) else f"{sil:.2f}")

            col1, col2 = st.columns([1.2, 1])

            with col1:
                pca_chart = (
                    alt.Chart(cluster_df, title=f"PCA + K-means · k={k_val}")
                    .mark_circle(size=98, opacity=0.78, stroke="white", strokeWidth=0.7)
                    .encode(
                        x=alt.X("PC1:Q", title=f"PC1 ({explained[0] * 100:.1f}%)"),
                        y=alt.Y("PC2:Q", title=f"PC2 ({explained[1] * 100:.1f}%)"),
                        color=alt.Color("Cluster:N", scale=alt.Scale(scheme="tableau20")),
                        shape=alt.Shape("Estado:N", title="Estado"),
                        tooltip=[
                            "Cluster:N",
                            "Estado:N",
                            "Tipo_cafe:N",
                            alt.Tooltip("PC1:Q", format=".2f"),
                            alt.Tooltip("PC2:Q", format=".2f"),
                        ],
                    )
                )

                st.altair_chart(polish(pca_chart, 470), use_container_width=True)

            with col2:
                comp = cluster_df.groupby(["Cluster", "Estado"], as_index=False).size()
                comp = comp.rename(columns={"size": "Productores"})

                comp_chart = (
                    alt.Chart(comp, title="Composición territorial")
                    .mark_bar(cornerRadiusEnd=6)
                    .encode(
                        x=alt.X("Cluster:N", title=None),
                        y=alt.Y("Productores:Q", title="Productores"),
                        color=alt.Color(
                            "Estado:N",
                            scale=alt.Scale(domain=STATE_DOMAIN, range=STATE_RANGE),
                        ),
                        tooltip=[
                            "Cluster:N",
                            "Estado:N",
                            alt.Tooltip("Productores:Q", format=","),
                        ],
                    )
                )

                st.altair_chart(polish(comp_chart, 470), use_container_width=True)

            st.markdown("### Perfil mediano de precios por cluster")

            X_profile = X.copy()
            X_profile["Cluster"] = cluster_df["Cluster"].values

            profile = X_profile.groupby("Cluster").median(numeric_only=True).T
            profile.index = [SHORT_LABELS.get(i, i) for i in profile.index]

            profile_long = (
                profile.reset_index()
                .melt(id_vars="index", var_name="Cluster", value_name="Precio_mediano")
                .rename(columns={"index": "Variable"})
            )

            profile_heat = (
                alt.Chart(profile_long, title="Heatmap de perfil por cluster")
                .mark_rect(cornerRadius=4)
                .encode(
                    x=alt.X("Cluster:N", title=None),
                    y=alt.Y("Variable:N", title=None, sort="-x"),
                    color=alt.Color(
                        "Precio_mediano:Q",
                        title="MXN/kg",
                        scale=alt.Scale(scheme="goldorange"),
                    ),
                    tooltip=[
                        "Cluster:N",
                        "Variable:N",
                        alt.Tooltip("Precio_mediano:Q", title="Mediana", format=",.1f"),
                    ],
                )
            )

            profile_text = profile_heat.mark_text(fontSize=11, fontWeight=700).encode(
                text=alt.Text("Precio_mediano:Q", format=".0f"),
                color=alt.condition(
                    "datum.Precio_mediano > 100",
                    alt.value("white"),
                    alt.value(C["ink"]),
                ),
            )

            st.altair_chart(polish(profile_heat + profile_text, 540), use_container_width=True)

            st.markdown("### Método del codo")

            max_k = min(8, X_scaled.shape[0] - 1)

            if max_k >= 2:
                inertia_rows = []

                for k in range(2, max_k + 1):
                    km_i = KMeans(n_clusters=k, random_state=42, n_init=20)
                    km_i.fit(X_scaled)
                    inertia_rows.append({"k": k, "Inercia": km_i.inertia_})

                inertia_df = pd.DataFrame(inertia_rows)

                line = (
                    alt.Chart(inertia_df, title="Inercia por número de clusters")
                    .mark_line(
                        point=alt.OverlayMarkDef(size=80, filled=True),
                        strokeWidth=3,
                    )
                    .encode(
                        x=alt.X("k:O", title="k"),
                        y=alt.Y("Inercia:Q", title="Inercia"),
                        tooltip=[
                            alt.Tooltip("k:O"),
                            alt.Tooltip("Inercia:Q", format=",.1f"),
                        ],
                    )
                )

                st.altair_chart(polish(line, 310), use_container_width=True)


# ------------------------------------------------------------
# TAB 6
# ------------------------------------------------------------

with tab6:
    st.subheader("Tabla analítica y descargas")

    st.markdown(
        '<div class="section-note">Resumen estadístico en formato largo. Útil para validar, exportar o insertar '
        "en un reporte técnico.</div>",
        unsafe_allow_html=True,
    )

    summary = (
        long_f.groupby(["Mercado", "Proceso", "Variable"], as_index=False)
        .agg(
            n=("Precio", "size"),
            media=("Precio", "mean"),
            mediana=("Precio", "median"),
            desviacion=("Precio", "std"),
            p10=("Precio", lambda x: np.nanpercentile(x, 10)),
            p90=("Precio", lambda x: np.nanpercentile(x, 90)),
            minimo=("Precio", "min"),
            maximo=("Precio", "max"),
        )
    )

    numeric_cols = [
        "media",
        "mediana",
        "desviacion",
        "p10",
        "p90",
        "minimo",
        "maximo",
    ]

    summary[numeric_cols] = summary[numeric_cols].round(2)

    st.dataframe(summary, use_container_width=True, height=420)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "Descargar precios en formato largo",
            data=long_f.to_csv(index=False).encode("utf-8-sig"),
            file_name="cafe_precios_largo.csv",
            mime="text/csv",
        )

    with c2:
        st.download_button(
            "Descargar resumen estadístico",
            data=summary.to_csv(index=False).encode("utf-8-sig"),
            file_name="cafe_resumen_estadistico.csv",
            mime="text/csv",
        )

    with c3:
        if not gap_f.empty:
            st.download_button(
                "Descargar brechas",
                data=gap_f.to_csv(index=False).encode("utf-8-sig"),
                file_name="cafe_brechas.csv",
                mime="text/csv",
            )

    with st.expander("Ver base filtrada"):
        st.dataframe(df_f, use_container_width=True, height=380)


st.caption(
    "Dashboard construido con Streamlit + Altair. Nota: los ceros se tratan como ausencia de precio en los cálculos estadísticos."
)
