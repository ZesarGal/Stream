# ============================================================
# COLORFUL DATA GALLERY — BUSINESS CASE KPI VERSION
# Streamlit + Altair + Pandas
# Sin streamlit-echarts · Sin xlsxwriter
#
# Instalar:
#   python -m pip install streamlit pandas numpy altair openpyxl xlrd
#
# Ejecutar:
#   python -m streamlit run stats.py
# ============================================================

from __future__ import annotations

from io import BytesIO
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(
    page_title="Colorful Data Gallery",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.data_transformers.disable_max_rows()


# ============================================================
# ESTILO CSS — SOBRIO
# ============================================================

CSS = """
<style>
.main .block-container {
    max-width: 1620px;
    padding-top: 1.1rem;
    padding-bottom: 3rem;
}

h1, h2, h3 {
    letter-spacing: -0.035em;
}

.hero {
    padding: 1.45rem 1.6rem;
    border-radius: 26px;
    background:
        linear-gradient(135deg, #ffffff 0%, #f7f9fc 55%, #f3f6fb 100%);
    border: 1px solid rgba(15,23,42,.08);
    box-shadow: 0 16px 42px rgba(15,23,42,.06);
    margin-bottom: 1.1rem;
}

.hero-title {
    font-size: 2.35rem;
    font-weight: 800;
    line-height: 1.05;
    margin-bottom: .45rem;
    color: #101828;
}

.hero-subtitle {
    font-size: 1.02rem;
    color: rgba(15,23,42,.70);
    max-width: 1080px;
}

.metric-card {
    min-height: 116px;
    padding: 1.05rem 1.08rem;
    border-radius: 22px;
    border: 1px solid rgba(15,23,42,.08);
    background:
        linear-gradient(180deg, rgba(255,255,255,.96), rgba(255,255,255,.82));
    box-shadow: 0 12px 30px rgba(15,23,42,.055);
}

.metric-label {
    color: rgba(15,23,42,.56);
    font-size: .77rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .075em;
    margin-bottom: .26rem;
}

.metric-value {
    color: rgba(15,23,42,.96);
    font-size: 1.48rem;
    font-weight: 850;
    letter-spacing: -.045em;
}

.small-note {
    font-size: .83rem;
    color: rgba(15,23,42,.61);
    margin-top: .14rem;
}

.section-label {
    color: rgba(15,23,42,.54);
    font-size: .80rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: .095em;
    margin-top: .35rem;
    margin-bottom: .15rem;
}

div[data-testid="stTabs"] button {
    font-weight: 700;
}

div[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

.stDownloadButton button {
    border-radius: 16px;
    font-weight: 700;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ============================================================
# PALETAS
# ============================================================

PALETTES = {
    "Aurora Board": {
        "colors": [
            "#4F46E5", "#0891B2", "#059669", "#D97706", "#E11D48",
            "#7C3AED", "#2563EB", "#0D9488", "#9333EA", "#EA580C",
            "#65A30D", "#0284C7", "#BE123C", "#A16207", "#4338CA",
        ],
        "seq": "blues",
        "div": "redblue",
        "accent": "#4F46E5",
        "accent2": "#0891B2",
    },
    "Executive Bright": {
        "colors": [
            "#2563EB", "#7C3AED", "#0891B2", "#059669", "#D97706",
            "#DC2626", "#9333EA", "#0D9488", "#EA580C", "#0284C7",
            "#65A30D", "#BE123C", "#4338CA", "#0F766E", "#A16207",
        ],
        "seq": "viridis",
        "div": "purplegreen",
        "accent": "#2563EB",
        "accent2": "#7C3AED",
    },
    "Tropical Finance": {
        "colors": [
            "#00A6FB", "#00C49A", "#F59E0B", "#EF476F", "#845EC2",
            "#2C73D2", "#84CC16", "#008F7A", "#F97316", "#C34A36",
            "#4D8076", "#B39CD0", "#06B6D4", "#A855F7", "#FF8066",
        ],
        "seq": "plasma",
        "div": "redblue",
        "accent": "#00A6FB",
        "accent2": "#EF476F",
    },
    "Sunset Strategy": {
        "colors": [
            "#E11D48", "#F97316", "#FACC15", "#7C3AED", "#2563EB",
            "#BE185D", "#9333EA", "#6D28D9", "#1D4ED8", "#06B6D4",
            "#EA580C", "#14B8A6", "#DC2626", "#A855F7", "#22D3EE",
        ],
        "seq": "orangered",
        "div": "brownbluegreen",
        "accent": "#F97316",
        "accent2": "#7C3AED",
    },
}


# ============================================================
# FORMATO
# ============================================================

def money(x) -> str:
    if x is None or pd.isna(x):
        return "—"
    x = float(x)
    sign = "-" if x < 0 else ""
    x = abs(x)

    if x >= 1_000_000_000:
        return f"{sign}${x / 1_000_000_000:,.2f}B"
    if x >= 1_000_000:
        return f"{sign}${x / 1_000_000:,.2f}M"
    if x >= 1_000:
        return f"{sign}${x / 1_000:,.1f}K"
    return f"{sign}${x:,.2f}"


def num(x) -> str:
    if x is None or pd.isna(x):
        return "—"

    x = float(x)

    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:,.2f}M"
    if abs(x) >= 1_000:
        return f"{x:,.0f}"
    if x.is_integer():
        return f"{x:,.0f}"
    return f"{x:,.2f}"


def pct(x) -> str:
    if x is None or pd.isna(x):
        return "—"

    x = float(x)

    if abs(x) <= 1.5:
        x *= 100

    return f"{x:,.2f}%"


def metric_card(label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="small-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# UTILIDADES DE DATOS
# ============================================================

def safe_sum(df: pd.DataFrame, col: Optional[str]) -> float:
    if not col or col not in df.columns:
        return np.nan
    return pd.to_numeric(df[col], errors="coerce").sum()


def safe_mean(df: pd.DataFrame, col: Optional[str]) -> float:
    if not col or col not in df.columns:
        return np.nan
    return pd.to_numeric(df[col], errors="coerce").mean()


def first_existing(cols: list[str], candidates: list[str]) -> Optional[str]:
    low = {c.lower(): c for c in cols}
    norm = {c.lower().replace(" ", "_"): c for c in cols}

    for cand in candidates:
        c = cand.lower()

        if c in low:
            return low[c]
        if c in norm:
            return norm[c]

    for col in cols:
        col_low = col.lower()

        for cand in candidates:
            if cand.lower() in col_low:
                return col

    return None


@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes, sep: str, encoding: str) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes), sep=sep, encoding=encoding)


@st.cache_data(show_spinner=False)
def read_excel_cached(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)


def excel_sheet_names(file_bytes: bytes) -> list[str]:
    return pd.ExcelFile(BytesIO(file_bytes)).sheet_names


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="datos_filtrados")

    return output.getvalue()


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out.columns = (
        pd.Series(out.columns.astype(str))
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^\wáéíóúÁÉÍÓÚñÑüÜ]+", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
        .tolist()
    )

    return out


def detect_numeric_text_columns(df: pd.DataFrame, threshold: float = 0.55) -> list[str]:
    detected = []

    for col in df.columns:
        if df[col].dtype == "object":
            cleaned = (
                df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
            )

            ratio = pd.to_numeric(cleaned, errors="coerce").notna().mean()

            if ratio > threshold:
                detected.append(col)

    return detected


def parse_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()

    for col in cols:
        if col in out.columns:
            cleaned = (
                out[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("%", "", regex=False)
                .str.strip()
            )

            out[col] = pd.to_numeric(cleaned, errors="coerce")

    return out


def parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()

    for col in cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    return out


def infer_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    temporal = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric and c not in temporal]

    return {
        "numeric": numeric,
        "temporal": temporal,
        "categorical": categorical,
        "all": df.columns.tolist(),
    }


# ============================================================
# BUSINESS CASE DE EJEMPLO
# ============================================================

def make_sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260428)
    n = 2160
    dates = pd.date_range("2025-01-01", periods=360, freq="D")

    regions = ["Norte", "Centro", "Sur", "Occidente", "Bajío", "Golfo"]

    cities = {
        "Norte": ["Monterrey", "Saltillo", "Chihuahua"],
        "Centro": ["CDMX", "Toluca", "Puebla"],
        "Sur": ["Oaxaca", "Mérida", "Tuxtla"],
        "Occidente": ["Guadalajara", "León", "Morelia"],
        "Bajío": ["Querétaro", "Aguascalientes", "San Luis Potosí"],
        "Golfo": ["Veracruz", "Tampico", "Villahermosa"],
    }

    channels = ["Retail", "Mayorista", "E-commerce", "Marketplace", "B2B"]
    segments = ["Nuevo", "Recurrente", "Premium", "PyME", "Corporativo"]
    categories = ["Software", "Hardware", "Servicios", "Capacitación", "Soporte"]

    campaigns = [
        "Awareness Q1",
        "Performance Ads",
        "Referral Program",
        "Enterprise Push",
        "Retention Boost",
        "Seasonal Promo",
    ]

    rows = []

    for _ in range(n):
        region = rng.choice(regions, p=[.16, .28, .16, .17, .14, .09])
        city = rng.choice(cities[region])
        channel = rng.choice(channels, p=[.28, .18, .25, .16, .13])
        segment = rng.choice(segments, p=[.28, .26, .16, .18, .12])
        category = rng.choice(categories, p=[.24, .18, .22, .18, .18])
        campaign = rng.choice(campaigns)
        date = rng.choice(dates)

        reg_eff = {
            "Norte": 1.08,
            "Centro": 1.16,
            "Sur": .88,
            "Occidente": 1.11,
            "Bajío": 1.02,
            "Golfo": .93,
        }[region]

        chan_eff = {
            "Retail": 1.02,
            "Mayorista": .94,
            "E-commerce": 1.12,
            "Marketplace": 1.05,
            "B2B": 1.18,
        }[channel]

        seg_eff = {
            "Nuevo": .92,
            "Recurrente": 1.03,
            "Premium": 1.25,
            "PyME": 1.08,
            "Corporativo": 1.35,
        }[segment]

        cat_price = {
            "Software": 520,
            "Hardware": 780,
            "Servicios": 640,
            "Capacitación": 420,
            "Soporte": 360,
        }[category]

        camp_eff = {
            "Awareness Q1": .95,
            "Performance Ads": 1.18,
            "Referral Program": 1.10,
            "Enterprise Push": 1.26,
            "Retention Boost": 1.06,
            "Seasonal Promo": 1.15,
        }[campaign]

        leads = int(max(20, rng.normal(360, 110) * camp_eff * chan_eff))
        conv = float(np.clip(rng.normal(.08, .025) * seg_eff * chan_eff, .015, .28))
        clients = int(max(1, leads * conv + rng.normal(0, 4)))
        units = int(max(1, clients * rng.normal(2.4, .55) * seg_eff))

        price = cat_price * reg_eff * seg_eff * rng.normal(1.0, .09)
        discount = float(np.clip(rng.normal(.08, .04), 0, .28))

        revenue = units * price * (1 - discount)
        marketing_cost = leads * rng.normal(8.5, 2.2) * camp_eff
        cogs_rate = float(np.clip(rng.normal(.54, .07), .32, .78))
        operating_cost = revenue * rng.normal(.18, .04)

        gross_margin = revenue * (1 - cogs_rate)
        profit = gross_margin - marketing_cost - operating_cost
        roi = profit / marketing_cost if marketing_cost > 0 else np.nan
        cac = marketing_cost / clients if clients > 0 else np.nan

        nps = float(
            np.clip(
                rng.normal(48, 17)
                + 10 * (segment == "Premium")
                + 5 * (channel == "B2B"),
                -40,
                90,
            )
        )

        churn = float(
            np.clip(
                rng.normal(.075, .025)
                - .02 * (segment == "Premium")
                - .015 * (segment == "Recurrente"),
                .005,
                .22,
            )
        )

        rows.append(
            {
                "fecha": date,
                "region": region,
                "ciudad": city,
                "canal": channel,
                "segmento_cliente": segment,
                "categoria_producto": category,
                "campaña": campaign,
                "leads": leads,
                "tasa_conversion": round(conv, 4),
                "clientes_nuevos": clients,
                "unidades_vendidas": units,
                "precio_unitario": round(price, 2),
                "descuento_pct": round(discount, 4),
                "ingresos_netos": round(revenue, 2),
                "costo_marketing": round(marketing_cost, 2),
                "CAC": round(cac, 2),
                "NPS": round(nps, 2),
                "churn_rate": round(churn, 4),
                "margen_bruto": round(gross_margin, 2),
                "utilidad_operativa": round(profit, 2),
                "ROI_marketing": round(roi, 4),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# MAPEO KPI
# ============================================================

def mapping_defaults(cols: list[str]) -> dict[str, Optional[str]]:
    return {
        "date": first_existing(cols, ["fecha", "date", "periodo", "mes"]),
        "sales": first_existing(cols, ["ingresos_netos", "ventas", "sales", "revenue", "ingresos", "facturacion"]),
        "units": first_existing(cols, ["unidades_vendidas", "unidades", "units", "quantity", "qty"]),
        "price": first_existing(cols, ["precio_unitario", "price", "precio", "unit_price"]),
        "discount": first_existing(cols, ["descuento_pct", "discount", "descuento"]),
        "profit": first_existing(cols, ["utilidad_operativa", "utilidad", "profit", "operating_profit", "ganancia"]),
        "gross_margin": first_existing(cols, ["margen_bruto", "gross_margin", "gross_profit", "margen"]),
        "marketing_cost": first_existing(cols, ["costo_marketing", "marketing_cost", "ad_spend", "costo_campaña", "cost"]),
        "roi": first_existing(cols, ["roi_marketing", "roi", "return_on_investment"]),
        "leads": first_existing(cols, ["leads", "prospectos"]),
        "conversion": first_existing(cols, ["tasa_conversion", "conversion_rate", "conversion", "conv_rate"]),
        "customers": first_existing(cols, ["clientes_nuevos", "new_customers", "clientes", "customers"]),
        "cac": first_existing(cols, ["cac", "customer_acquisition_cost"]),
        "nps": first_existing(cols, ["nps"]),
        "churn": first_existing(cols, ["churn_rate", "churn", "tasa_churn"]),
        "region": first_existing(cols, ["region", "región", "zona", "territorio"]),
        "city": first_existing(cols, ["ciudad", "city", "plaza"]),
        "channel": first_existing(cols, ["canal", "channel", "sales_channel"]),
        "campaign": first_existing(cols, ["campaña", "campaign", "campaign_name"]),
        "segment": first_existing(cols, ["segmento_cliente", "segmento", "segment", "customer_segment"]),
        "category": first_existing(cols, ["categoria_producto", "categoría_producto", "categoria", "category", "product_category"]),
    }


def select_col(label: str, options: list[str], default: Optional[str], key: str) -> Optional[str]:
    choices = ["Ninguna"] + options
    idx = choices.index(default) if default in options else 0
    val = st.selectbox(label, choices, index=idx, key=key)
    return None if val == "Ninguna" else val


# ============================================================
# ALTAIR HELPERS
# ============================================================

def scale(colors: list[str]) -> alt.Scale:
    return alt.Scale(range=colors)


def final(chart):
    return (
        chart
        .configure_title(
            fontSize=19,
            fontWeight=800,
            anchor="start",
            color="#101828",
            subtitleColor="#475467",
            subtitleFontSize=12,
        )
        .configure_axis(
            labelFontSize=12,
            titleFontSize=13,
            grid=True,
            gridOpacity=.15,
            domainOpacity=.18,
            tickOpacity=.18,
            labelColor="#344054",
            titleColor="#344054",
        )
        .configure_legend(
            titleFontSize=12,
            labelFontSize=12,
            orient="top",
            padding=8,
            symbolSize=120,
        )
        .configure_view(strokeWidth=0)
        .configure_concat(spacing=24)
    )


def render(chart):
    if chart is not None:
        st.altair_chart(final(chart), use_container_width=True, theme=None)


def bar(
    df: pd.DataFrame,
    group: Optional[str],
    value: Optional[str],
    agg: str,
    colors: list[str],
    title: str,
    top_n: int = 20,
    horizontal: bool = False,
):
    if not group or not value or group not in df.columns or value not in df.columns:
        return None

    data = (
        df.groupby(group, dropna=False)[value]
        .agg(agg)
        .reset_index(name="valor")
        .sort_values("valor", ascending=False)
        .head(top_n)
    )

    data[group] = data[group].astype("object").fillna("NA").astype(str)

    if horizontal:
        return (
            alt.Chart(data)
            .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8)
            .encode(
                y=alt.Y(f"{group}:N", sort="-x", title=None),
                x=alt.X("valor:Q", title=agg),
                color=alt.Color(f"{group}:N", scale=scale(colors), legend=None),
                tooltip=[
                    alt.Tooltip(f"{group}:N"),
                    alt.Tooltip("valor:Q", format=",.2f"),
                ],
            )
            .properties(title=title, height=430)
        )

    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X(f"{group}:N", sort="-y", title=None),
            y=alt.Y("valor:Q", title=agg),
            color=alt.Color(f"{group}:N", scale=scale(colors), legend=None),
            tooltip=[
                alt.Tooltip(f"{group}:N"),
                alt.Tooltip("valor:Q", format=",.2f"),
            ],
        )
        .properties(title=title, height=430)
    )


def timeseries(
    df: pd.DataFrame,
    date_col: Optional[str],
    value_col: Optional[str],
    freq: str,
    accent: str,
    title: str,
    agg: str = "sum",
):
    if not date_col or not value_col or date_col not in df.columns or value_col not in df.columns:
        return None

    data = df[[date_col, value_col]].dropna().copy()

    if data.empty:
        return None

    data["periodo"] = data[date_col].dt.to_period(freq).dt.to_timestamp()

    if agg == "mean":
        grouped = data.groupby("periodo", as_index=False)[value_col].mean()
    else:
        grouped = data.groupby("periodo", as_index=False)[value_col].sum()

    area = (
        alt.Chart(grouped)
        .mark_area(opacity=.22, color=accent)
        .encode(
            x=alt.X("periodo:T", title="Periodo"),
            y=alt.Y(f"{value_col}:Q", title=value_col),
        )
    )

    line = (
        alt.Chart(grouped)
        .mark_line(color=accent, strokeWidth=3.2, point=True)
        .encode(
            x=alt.X("periodo:T", title="Periodo"),
            y=alt.Y(f"{value_col}:Q", title=value_col),
            tooltip=[
                alt.Tooltip("periodo:T"),
                alt.Tooltip(f"{value_col}:Q", format=",.2f"),
            ],
        )
    )

    return (area + line).properties(title=title, height=420).interactive()


def heatmap(
    df: pd.DataFrame,
    x: Optional[str],
    y: Optional[str],
    value: Optional[str],
    agg: str,
    scheme: str,
    title: str,
):
    if not x or not y or not value:
        return None

    if x not in df.columns or y not in df.columns or value not in df.columns:
        return None

    data = df.copy()
    data[x] = data[x].astype("object").fillna("NA").astype(str)
    data[y] = data[y].astype("object").fillna("NA").astype(str)

    grouped = data.groupby([x, y], dropna=False)[value].agg(agg).reset_index(name="valor")

    rect = (
        alt.Chart(grouped)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X(f"{x}:N", title=x),
            y=alt.Y(f"{y}:N", title=y),
            color=alt.Color("valor:Q", scale=alt.Scale(scheme=scheme), title=value),
            tooltip=[
                alt.Tooltip(f"{x}:N"),
                alt.Tooltip(f"{y}:N"),
                alt.Tooltip("valor:Q", format=",.2f"),
            ],
        )
    )

    text = (
        alt.Chart(grouped)
        .mark_text(fontSize=10, fontWeight="bold")
        .encode(
            x=alt.X(f"{x}:N"),
            y=alt.Y(f"{y}:N"),
            text=alt.Text("valor:Q", format=".2s"),
            color=alt.value("#101828"),
        )
    )

    return (rect + text).properties(title=title, height=460)


def stacked(
    df: pd.DataFrame,
    x: Optional[str],
    stack: Optional[str],
    value: Optional[str],
    agg: str,
    colors: list[str],
    normalize: bool,
    title: str,
):
    if not x or not stack or not value:
        return None

    if x not in df.columns or stack not in df.columns or value not in df.columns:
        return None

    data = df.copy()
    data[x] = data[x].astype("object").fillna("NA").astype(str)
    data[stack] = data[stack].astype("object").fillna("NA").astype(str)

    grouped = data.groupby([x, stack], dropna=False)[value].agg(agg).reset_index(name="valor")

    if normalize:
        grouped["share"] = grouped["valor"] / grouped.groupby(x)["valor"].transform("sum")
        y_enc = alt.Y("share:Q", stack="normalize", title="Participación")
        tip = alt.Tooltip("share:Q", format=".1%")
    else:
        y_enc = alt.Y("valor:Q", stack="zero", title=value)
        tip = alt.Tooltip("valor:Q", format=",.2f")

    return (
        alt.Chart(grouped)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{x}:N", title=None),
            y=y_enc,
            color=alt.Color(f"{stack}:N", scale=scale(colors), title=stack),
            tooltip=[
                alt.Tooltip(f"{x}:N"),
                alt.Tooltip(f"{stack}:N"),
                tip,
            ],
        )
        .properties(title=title, height=430)
    )


def scatter(
    df: pd.DataFrame,
    x: Optional[str],
    y: Optional[str],
    color: Optional[str],
    size: Optional[str],
    colors: list[str],
    accent: str,
    title: str,
):
    if not x or not y or x not in df.columns or y not in df.columns:
        return None

    cols = [x, y] + ([color] if color else []) + ([size] if size else [])
    data = df[cols].dropna().copy()

    enc = {
        "x": alt.X(f"{x}:Q", title=x),
        "y": alt.Y(f"{y}:Q", title=y),
        "tooltip": [
            alt.Tooltip(f"{x}:Q", format=",.3f"),
            alt.Tooltip(f"{y}:Q", format=",.3f"),
        ],
    }

    if color:
        enc["color"] = alt.Color(f"{color}:N", scale=scale(colors), title=color)
        enc["tooltip"].append(alt.Tooltip(f"{color}:N"))
    else:
        enc["color"] = alt.value(colors[0])

    if size:
        enc["size"] = alt.Size(f"{size}:Q", title=size)
        enc["tooltip"].append(alt.Tooltip(f"{size}:Q", format=",.2f"))

    points = alt.Chart(data).mark_circle(opacity=.72).encode(**enc)

    if len(data) > 2:
        reg = (
            alt.Chart(data)
            .transform_regression(x, y)
            .mark_line(color=accent, strokeWidth=3.1)
            .encode(
                x=alt.X(f"{x}:Q"),
                y=alt.Y(f"{y}:Q"),
            )
        )
        chart = points + reg
    else:
        chart = points

    return chart.properties(title=title, height=440).interactive()


def histogram(
    df: pd.DataFrame,
    col: Optional[str],
    scheme: str,
    accent: str,
    title: str,
):
    if not col or col not in df.columns:
        return None

    data = df[[col]].dropna()

    hist = (
        alt.Chart(data)
        .mark_bar(opacity=.78, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=44), title=col),
            y=alt.Y("count():Q", title="Frecuencia"),
            color=alt.Color("count():Q", scale=alt.Scale(scheme=scheme), legend=None),
            tooltip=[alt.Tooltip("count():Q")],
        )
    )

    density = (
        alt.Chart(data)
        .transform_density(col, as_=[col, "densidad"], counts=True)
        .mark_line(strokeWidth=3.3, color=accent)
        .encode(
            x=alt.X(f"{col}:Q"),
            y=alt.Y("densidad:Q", title="Frecuencia / densidad"),
        )
    )

    return (hist + density).properties(title=title, height=410)


def boxplot(
    df: pd.DataFrame,
    x: Optional[str],
    y: Optional[str],
    colors: list[str],
    title: str,
):
    if not x or not y or x not in df.columns or y not in df.columns:
        return None

    data = df[[x, y]].dropna().copy()
    data[x] = data[x].astype(str)

    return (
        alt.Chart(data)
        .mark_boxplot(size=42)
        .encode(
            x=alt.X(f"{x}:N", sort="-y", title=x),
            y=alt.Y(f"{y}:Q", title=y),
            color=alt.Color(f"{x}:N", scale=scale(colors), legend=None),
            tooltip=[
                alt.Tooltip(x),
                alt.Tooltip(y, format=",.2f"),
            ],
        )
        .properties(title=title, height=430)
    )


def funnel(
    df: pd.DataFrame,
    leads_col: Optional[str],
    conv_col: Optional[str],
    customers_col: Optional[str],
    colors: list[str],
):
    leads = safe_sum(df, leads_col)
    customers = safe_sum(df, customers_col)

    if (
        pd.isna(customers)
        and leads_col
        and conv_col
        and leads_col in df.columns
        and conv_col in df.columns
    ):
        customers = (df[leads_col] * df[conv_col]).sum()

    leads = 0 if pd.isna(leads) else leads
    customers = 0 if pd.isna(customers) else customers

    qualified = max(customers, leads * .45)
    opportunities = max(customers, leads * .25)

    data = pd.DataFrame(
        {
            "etapa": ["Leads", "Leads calificados", "Oportunidades", "Clientes nuevos"],
            "valor": [leads, qualified, opportunities, customers],
        }
    )

    order = ["Leads", "Leads calificados", "Oportunidades", "Clientes nuevos"]

    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopRight=12, cornerRadiusBottomRight=12)
        .encode(
            y=alt.Y("etapa:N", sort=order, title=None),
            x=alt.X("valor:Q", title="Volumen"),
            color=alt.Color("etapa:N", scale=scale(colors), legend=None),
            tooltip=[
                alt.Tooltip("etapa:N"),
                alt.Tooltip("valor:Q", format=",.0f"),
            ],
        )
        .properties(title="Embudo comercial", height=330)
    )


def corr_chart(df: pd.DataFrame, cols: list[str], div_scheme: str):
    cols = [c for c in cols if c and c in df.columns]
    cols = list(dict.fromkeys(cols))

    if len(cols) < 2:
        return None

    corr = df[cols].corr(method="pearson")

    long = corr.reset_index().melt(
        id_vars="index",
        var_name="variable_y",
        value_name="correlacion",
    )

    long = long.rename(columns={"index": "variable_x"})

    rect = (
        alt.Chart(long)
        .mark_rect(cornerRadius=5)
        .encode(
            x=alt.X("variable_x:N", title=None),
            y=alt.Y("variable_y:N", title=None),
            color=alt.Color(
                "correlacion:Q",
                scale=alt.Scale(scheme=div_scheme, domain=[-1, 1]),
                title="ρ",
            ),
            tooltip=[
                alt.Tooltip("variable_x:N"),
                alt.Tooltip("variable_y:N"),
                alt.Tooltip("correlacion:Q", format=".3f"),
            ],
        )
    )

    text = (
        alt.Chart(long)
        .mark_text(fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("variable_x:N"),
            y=alt.Y("variable_y:N"),
            text=alt.Text("correlacion:Q", format=".2f"),
            color=alt.condition(
                "abs(datum.correlacion) > 0.55",
                alt.value("white"),
                alt.value("#101828"),
            ),
        )
    )

    return (rect + text).properties(title="Mapa de correlaciones KPI", height=560)


# ============================================================
# HERO SOBRIO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">Colorful Data Gallery</div>
        <div class="hero-subtitle">
            Dashboard de Streamlit para cargar datos limpios en CSV o Excel,
            explorar estadística descriptiva, diagnosticar calidad de datos
            y generar visualizaciones interactivas con Altair.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR — CARGA
# ============================================================

with st.sidebar:
    st.markdown("### 1. Cargar datos")

    uploaded_file = st.file_uploader(
        "Archivo CSV o Excel",
        type=["csv", "xlsx", "xls"],
        help="Usa un archivo limpio: encabezados en la primera fila y columnas consistentes.",
    )

    use_sample = st.toggle(
        "Usar Business Case de ejemplo",
        value=uploaded_file is None,
    )

    st.markdown("### 2. Paleta")

    palette_name = st.selectbox(
        "Color palette",
        list(PALETTES.keys()),
        index=0,
    )

    palette = PALETTES[palette_name]
    colors = palette["colors"]
    seq_scheme = palette["seq"]
    div_scheme = palette["div"]
    accent = palette["accent"]
    accent2 = palette["accent2"]


# ============================================================
# CARGA DE DATOS
# ============================================================

source_label = "Business Case sintético"

if uploaded_file is None or use_sample:
    df_raw = make_sample_data()
else:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
    source_label = filename

    try:
        if filename.lower().endswith(".csv"):
            with st.sidebar.expander("Opciones CSV", expanded=True):
                sep = st.selectbox(
                    "Separador",
                    [",", ";", "\t", "|"],
                    index=0,
                )

                encoding = st.selectbox(
                    "Codificación",
                    ["utf-8", "latin-1", "cp1252"],
                    index=0,
                )

            df_raw = read_csv_cached(file_bytes, sep=sep, encoding=encoding)

        else:
            sheets = excel_sheet_names(file_bytes)

            sheet = st.sidebar.selectbox(
                "Hoja de Excel",
                sheets,
                index=0,
            )

            df_raw = read_excel_cached(file_bytes, sheet_name=sheet)

    except Exception as exc:
        st.error("No pude leer el archivo. Revisa separador, codificación, hoja o formato.")
        st.exception(exc)
        st.stop()


df = clean_column_names(df_raw)


with st.sidebar:
    st.markdown("### 3. Tipos de columnas")

    possible_dates = [
        c for c in df.columns
        if "fecha" in c.lower()
        or "date" in c.lower()
        or "periodo" in c.lower()
        or pd.api.types.is_datetime64_any_dtype(df[c])
    ]

    date_cols = st.multiselect(
        "Columnas de fecha",
        df.columns.tolist(),
        default=possible_dates,
    )

    possible_numeric = detect_numeric_text_columns(df)

    numeric_text_cols = st.multiselect(
        "Convertir texto a número",
        df.columns.tolist(),
        default=possible_numeric,
    )


df = parse_dates(df, date_cols)
df = parse_numeric(df, numeric_text_cols)

groups = infer_groups(df)

if df.empty:
    st.warning("El archivo no contiene datos.")
    st.stop()


# ============================================================
# MAPEO KPI
# ============================================================

defaults = mapping_defaults(df.columns.tolist())

with st.sidebar:
    st.markdown("### 4. Mapeo KPI")

    with st.expander("Finanzas", expanded=True):
        date_col = select_col("Fecha", groups["temporal"], defaults["date"], "map_date")
        sales_col = select_col("Ventas / ingresos netos", groups["numeric"], defaults["sales"], "map_sales")
        profit_col = select_col("Utilidad operativa", groups["numeric"], defaults["profit"], "map_profit")
        gross_margin_col = select_col("Margen bruto", groups["numeric"], defaults["gross_margin"], "map_margin")
        cost_col = select_col("Costo marketing", groups["numeric"], defaults["marketing_cost"], "map_cost")
        roi_col = select_col("ROI", groups["numeric"], defaults["roi"], "map_roi")

    with st.expander("Comercial", expanded=False):
        leads_col = select_col("Leads", groups["numeric"], defaults["leads"], "map_leads")
        conversion_col = select_col("Conversión", groups["numeric"], defaults["conversion"], "map_conversion")
        customers_col = select_col("Clientes nuevos", groups["numeric"], defaults["customers"], "map_customers")
        units_col = select_col("Unidades vendidas", groups["numeric"], defaults["units"], "map_units")
        price_col = select_col("Precio unitario", groups["numeric"], defaults["price"], "map_price")
        discount_col = select_col("Descuento", groups["numeric"], defaults["discount"], "map_discount")
        cac_col = select_col("CAC", groups["numeric"], defaults["cac"], "map_cac")
        nps_col = select_col("NPS", groups["numeric"], defaults["nps"], "map_nps")
        churn_col = select_col("Churn", groups["numeric"], defaults["churn"], "map_churn")

    with st.expander("Dimensiones", expanded=False):
        region_col = select_col("Región", groups["categorical"], defaults["region"], "map_region")
        city_col = select_col("Ciudad", groups["categorical"], defaults["city"], "map_city")
        channel_col = select_col("Canal", groups["categorical"], defaults["channel"], "map_channel")
        campaign_col = select_col("Campaña", groups["categorical"], defaults["campaign"], "map_campaign")
        segment_col = select_col("Segmento", groups["categorical"], defaults["segment"], "map_segment")
        category_col = select_col("Categoría", groups["categorical"], defaults["category"], "map_category")


# ============================================================
# FILTROS
# ============================================================

filtered = df.copy()

with st.sidebar:
    st.markdown("### 5. Filtros")

    if date_col:
        s = filtered[date_col].dropna()

        if len(s) > 0:
            min_d = s.min().date()
            max_d = s.max().date()

            date_range = st.date_input(
                f"Rango: {date_col}",
                value=(min_d, max_d),
                min_value=min_d,
                max_value=max_d,
            )

            if isinstance(date_range, tuple) and len(date_range) == 2:
                start, end = date_range

                filtered = filtered[
                    (filtered[date_col].dt.date >= start)
                    & (filtered[date_col].dt.date <= end)
                ]

    cat_candidates = [
        c for c in [
            region_col,
            city_col,
            channel_col,
            campaign_col,
            segment_col,
            category_col,
        ]
        if c
    ]

    cat_candidates = list(dict.fromkeys(cat_candidates))

    cat_filters = st.multiselect(
        "Filtros categóricos",
        cat_candidates,
        default=[],
    )

    for c in cat_filters:
        values = (
            filtered[c]
            .astype("object")
            .fillna("NA")
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        selected = st.multiselect(
            f"Valores: {c}",
            values,
            default=values,
        )

        filtered = filtered[
            filtered[c]
            .astype("object")
            .fillna("NA")
            .astype(str)
            .isin(selected)
        ]

    num_candidates = [
        c for c in [
            sales_col,
            profit_col,
            gross_margin_col,
            cost_col,
            roi_col,
            conversion_col,
            cac_col,
        ]
        if c
    ]

    num_candidates = list(dict.fromkeys(num_candidates))

    num_filters = st.multiselect(
        "Filtros numéricos",
        num_candidates,
        default=[],
    )

    for c in num_filters:
        s = filtered[c].dropna()

        if len(s) > 0:
            lo = float(s.min())
            hi = float(s.max())

            if lo < hi:
                a, b = st.slider(
                    f"Rango: {c}",
                    min_value=lo,
                    max_value=hi,
                    value=(lo, hi),
                )

                filtered = filtered[
                    (filtered[c] >= a)
                    & (filtered[c] <= b)
                ]

    st.divider()
    st.caption(f"Fuente: {source_label}")
    st.write(f"Filas filtradas: **{len(filtered):,} / {len(df):,}**")


# ============================================================
# KPI
# ============================================================

total_sales = safe_sum(filtered, sales_col)
total_profit = safe_sum(filtered, profit_col)
total_margin = safe_sum(filtered, gross_margin_col)
total_cost = safe_sum(filtered, cost_col)
total_leads = safe_sum(filtered, leads_col)
total_customers = safe_sum(filtered, customers_col)
total_units = safe_sum(filtered, units_col)

operating_margin = (
    total_profit / total_sales
    if pd.notna(total_profit)
    and pd.notna(total_sales)
    and total_sales != 0
    else np.nan
)

gross_margin_rate = (
    total_margin / total_sales
    if pd.notna(total_margin)
    and pd.notna(total_sales)
    and total_sales != 0
    else np.nan
)

roi_calc = (
    total_profit / total_cost
    if pd.notna(total_profit)
    and pd.notna(total_cost)
    and total_cost != 0
    else safe_mean(filtered, roi_col)
)

conversion_calc = (
    total_customers / total_leads
    if pd.notna(total_customers)
    and pd.notna(total_leads)
    and total_leads != 0
    else safe_mean(filtered, conversion_col)
)

avg_ticket = (
    total_sales / total_customers
    if pd.notna(total_sales)
    and pd.notna(total_customers)
    and total_customers != 0
    else np.nan
)

cac_calc = (
    total_cost / total_customers
    if pd.notna(total_cost)
    and pd.notna(total_customers)
    and total_customers != 0
    else safe_mean(filtered, cac_col)
)


st.markdown('<div class="section-label">Executive KPI Panel</div>', unsafe_allow_html=True)

kpi_cols = st.columns(8)

kpis = [
    ("Ventas", money(total_sales), "ingresos netos"),
    ("Utilidad", money(total_profit), f"margen op. {pct(operating_margin)}"),
    ("Costo MKT", money(total_cost), "campañas / adquisición"),
    ("ROI", pct(roi_calc), "utilidad / costo MKT"),
    ("Conversión", pct(conversion_calc), "clientes / leads"),
    ("Leads", num(total_leads), "prospectos"),
    ("Clientes", num(total_customers), f"CAC {money(cac_calc)}"),
    ("Ticket prom.", money(avg_ticket), f"margen bruto {pct(gross_margin_rate)}"),
]

for col, (label, value, note) in zip(kpi_cols, kpis):
    with col:
        metric_card(label, value, note)

st.write("")


# ============================================================
# TABS
# ============================================================

tab_exec, tab_sales, tab_campaigns, tab_channels, tab_costs, tab_roi, tab_quality, tab_export = st.tabs(
    [
        "Executive",
        "Ventas",
        "Campañas",
        "Canales y regiones",
        "Costos y margen",
        "ROI y conversión",
        "Calidad",
        "Exportar",
    ]
)


# ============================================================
# TAB EXECUTIVE
# ============================================================

with tab_exec:
    st.subheader("Dashboard ejecutivo")

    c1, c2 = st.columns([1.25, 1])

    with c1:
        freq = st.selectbox(
            "Frecuencia temporal",
            ["D", "W", "M", "Q"],
            index=2,
            key="exec_freq",
        )

        render(
            timeseries(
                filtered,
                date_col,
                sales_col,
                freq,
                accent,
                "Ventas en el tiempo",
            )
        )

    with c2:
        render(
            bar(
                filtered,
                channel_col,
                sales_col,
                "sum",
                colors,
                "Ventas por canal",
                top_n=15,
            )
        )

    c3, c4 = st.columns(2)

    with c3:
        render(
            funnel(
                filtered,
                leads_col,
                conversion_col,
                customers_col,
                colors,
            )
        )

    with c4:
        render(
            bar(
                filtered,
                region_col,
                profit_col,
                "sum",
                colors,
                "Utilidad por región",
                top_n=15,
            )
        )

    st.markdown("#### Vista previa")
    st.dataframe(filtered.head(160), use_container_width=True)


# ============================================================
# TAB VENTAS
# ============================================================

with tab_sales:
    st.subheader("Ventas, unidades, categorías y segmentos")

    a, b = st.columns(2)

    with a:
        render(
            bar(
                filtered,
                category_col,
                sales_col,
                "sum",
                colors,
                "Ventas por categoría",
                top_n=20,
            )
        )

    with b:
        render(
            bar(
                filtered,
                segment_col,
                sales_col,
                "sum",
                colors,
                "Ventas por segmento",
                top_n=20,
            )
        )

    c, d = st.columns(2)

    with c:
        render(
            heatmap(
                filtered,
                region_col,
                channel_col,
                sales_col,
                "sum",
                seq_scheme,
                "Heatmap de ventas: región × canal",
            )
        )

    with d:
        render(
            stacked(
                filtered,
                channel_col,
                category_col,
                sales_col,
                "sum",
                colors,
                True,
                "Mix de categoría por canal",
            )
        )

    render(
        scatter(
            filtered,
            units_col,
            sales_col,
            category_col,
            price_col,
            colors,
            accent,
            "Precio–volumen: ventas vs unidades",
        )
    )


# ============================================================
# TAB CAMPAÑAS
# ============================================================

with tab_campaigns:
    st.subheader("Campañas: inversión, conversión, ROI y utilidad")

    a, b = st.columns(2)

    with a:
        if roi_col:
            render(
                bar(
                    filtered,
                    campaign_col,
                    roi_col,
                    "mean",
                    colors,
                    "ROI promedio por campaña",
                    top_n=20,
                    horizontal=True,
                )
            )
        elif profit_col and cost_col:
            temp = filtered.copy()
            temp["_roi_calculado"] = temp[profit_col] / temp[cost_col].replace(0, np.nan)

            render(
                bar(
                    temp,
                    campaign_col,
                    "_roi_calculado",
                    "mean",
                    colors,
                    "ROI calculado por campaña",
                    top_n=20,
                    horizontal=True,
                )
            )

    with b:
        render(
            bar(
                filtered,
                campaign_col,
                conversion_col,
                "mean",
                colors,
                "Conversión promedio por campaña",
                top_n=20,
                horizontal=True,
            )
        )

    c, d = st.columns(2)

    with c:
        render(
            scatter(
                filtered,
                cost_col,
                profit_col,
                campaign_col,
                sales_col,
                colors,
                accent,
                "Eficiencia de campañas: costo vs utilidad",
            )
        )

    with d:
        render(
            bar(
                filtered,
                campaign_col,
                cost_col,
                "sum",
                colors,
                "Inversión de marketing por campaña",
                top_n=20,
                horizontal=True,
            )
        )

    st.markdown("#### Tabla ejecutiva de campañas")

    if campaign_col:
        agg = {}

        if sales_col:
            agg["ventas"] = (sales_col, "sum")
        if profit_col:
            agg["utilidad"] = (profit_col, "sum")
        if cost_col:
            agg["costo_marketing"] = (cost_col, "sum")
        if leads_col:
            agg["leads"] = (leads_col, "sum")
        if customers_col:
            agg["clientes"] = (customers_col, "sum")
        if conversion_col:
            agg["conversion_prom"] = (conversion_col, "mean")
        if roi_col:
            agg["roi_prom"] = (roi_col, "mean")
        if cac_col:
            agg["cac_prom"] = (cac_col, "mean")

        if agg:
            table = filtered.groupby(campaign_col, dropna=False).agg(**agg).reset_index()

            if "utilidad" in table.columns and "costo_marketing" in table.columns:
                table["roi_calculado"] = table["utilidad"] / table["costo_marketing"].replace(0, np.nan)

            if "clientes" in table.columns and "leads" in table.columns:
                table["conversion_calculada"] = table["clientes"] / table["leads"].replace(0, np.nan)

            st.dataframe(table.round(4), use_container_width=True)
        else:
            st.info("Mapea variables de campaña para generar la tabla.")
    else:
        st.info("Mapea campaña para activar la tabla ejecutiva.")


# ============================================================
# TAB CANALES Y REGIONES
# ============================================================

with tab_channels:
    st.subheader("Canales, regiones y territorios")

    a, b = st.columns(2)

    with a:
        render(
            bar(
                filtered,
                region_col,
                sales_col,
                "sum",
                colors,
                "Ventas por región",
                top_n=20,
            )
        )

    with b:
        render(
            bar(
                filtered,
                channel_col,
                profit_col,
                "sum",
                colors,
                "Utilidad por canal",
                top_n=20,
            )
        )

    c, d = st.columns(2)

    with c:
        render(
            heatmap(
                filtered,
                region_col,
                channel_col,
                profit_col,
                "sum",
                seq_scheme,
                "Heatmap de utilidad: región × canal",
            )
        )

    with d:
        render(
            bar(
                filtered,
                city_col,
                sales_col,
                "sum",
                colors,
                "Top ciudades por ventas",
                top_n=20,
                horizontal=True,
            )
        )

    render(
        stacked(
            filtered,
            region_col,
            channel_col,
            sales_col,
            "sum",
            colors,
            True,
            "Mix de canales por región",
        )
    )


# ============================================================
# TAB COSTOS Y MARGEN
# ============================================================

with tab_costs:
    st.subheader("Costos, margen y rentabilidad")

    a, b = st.columns(2)

    with a:
        render(
            histogram(
                filtered,
                cost_col,
                seq_scheme,
                accent,
                "Distribución de costo marketing",
            )
        )

    with b:
        render(
            histogram(
                filtered,
                profit_col,
                seq_scheme,
                accent2,
                "Distribución de utilidad operativa",
            )
        )

    c, d = st.columns(2)

    with c:
        render(
            boxplot(
                filtered,
                channel_col,
                gross_margin_col,
                colors,
                "Margen bruto por canal",
            )
        )

    with d:
        render(
            boxplot(
                filtered,
                category_col,
                profit_col,
                colors,
                "Utilidad por categoría",
            )
        )

    e, f = st.columns(2)

    with e:
        render(
            scatter(
                filtered,
                cost_col,
                sales_col,
                channel_col,
                profit_col,
                colors,
                accent,
                "Costo marketing vs ventas",
            )
        )

    with f:
        render(
            scatter(
                filtered,
                sales_col,
                profit_col,
                region_col,
                cost_col,
                colors,
                accent2,
                "Ventas vs utilidad",
            )
        )


# ============================================================
# TAB ROI Y CONVERSIÓN
# ============================================================

with tab_roi:
    st.subheader("ROI, conversión, CAC y eficiencia comercial")

    a, b = st.columns(2)

    with a:
        if roi_col:
            render(
                scatter(
                    filtered,
                    conversion_col,
                    roi_col,
                    campaign_col,
                    sales_col,
                    colors,
                    accent,
                    "ROI vs conversión",
                )
            )
        elif profit_col and cost_col:
            temp = filtered.copy()
            temp["_roi_calculado"] = temp[profit_col] / temp[cost_col].replace(0, np.nan)

            render(
                scatter(
                    temp,
                    conversion_col,
                    "_roi_calculado",
                    campaign_col,
                    sales_col,
                    colors,
                    accent,
                    "ROI calculado vs conversión",
                )
            )

    with b:
        render(
            scatter(
                filtered,
                cac_col,
                profit_col,
                channel_col,
                sales_col,
                colors,
                accent2,
                "CAC vs utilidad",
            )
        )

    c, d = st.columns(2)

    with c:
        render(
            bar(
                filtered,
                channel_col,
                conversion_col,
                "mean",
                colors,
                "Conversión por canal",
                top_n=20,
            )
        )

    with d:
        render(
            bar(
                filtered,
                segment_col,
                roi_col,
                "mean",
                colors,
                "ROI por segmento",
                top_n=20,
            )
        )

    kpi_numeric = [
        sales_col,
        profit_col,
        gross_margin_col,
        cost_col,
        roi_col,
        leads_col,
        conversion_col,
        customers_col,
        units_col,
        cac_col,
        nps_col,
        churn_col,
    ]

    render(
        corr_chart(
            filtered,
            kpi_numeric,
            div_scheme,
        )
    )


# ============================================================
# TAB CALIDAD
# ============================================================

with tab_quality:
    st.subheader("Calidad de datos y estructura")

    missing = pd.DataFrame(
        {
            "variable": filtered.columns,
            "tipo": [str(filtered[c].dtype) for c in filtered.columns],
            "faltantes": [int(filtered[c].isna().sum()) for c in filtered.columns],
            "faltantes_%": [100 * filtered[c].isna().mean() for c in filtered.columns],
            "unicos": [int(filtered[c].nunique(dropna=True)) for c in filtered.columns],
        }
    ).sort_values("faltantes_%", ascending=False)

    a, b = st.columns(2)

    with a:
        chart = (
            alt.Chart(missing.head(45))
            .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
            .encode(
                x=alt.X("variable:N", sort="-y", title=None),
                y=alt.Y("faltantes_%:Q", title="% faltante"),
                color=alt.Color("faltantes_%:Q", scale=alt.Scale(scheme=seq_scheme), legend=None),
                tooltip=[
                    alt.Tooltip("variable:N"),
                    alt.Tooltip("faltantes:Q"),
                    alt.Tooltip("faltantes_%:Q", format=".2f"),
                ],
            )
            .properties(title="Faltantes por variable", height=400)
        )

        render(chart)

    with b:
        dup_count = int(filtered.duplicated().sum())

        metric_card(
            "Duplicados",
            num(dup_count),
            "filas duplicadas exactas",
        )

        st.write("")
        st.dataframe(missing.round(2), use_container_width=True)

    st.markdown("#### Diccionario KPI detectado")

    mapping_table = pd.DataFrame(
        {
            "KPI / dimensión": [
                "fecha",
                "ventas",
                "utilidad",
                "margen bruto",
                "costo marketing",
                "ROI",
                "leads",
                "conversión",
                "clientes",
                "unidades",
                "precio",
                "descuento",
                "CAC",
                "NPS",
                "churn",
                "región",
                "ciudad",
                "canal",
                "campaña",
                "segmento",
                "categoría",
            ],
            "columna_mapeada": [
                date_col,
                sales_col,
                profit_col,
                gross_margin_col,
                cost_col,
                roi_col,
                leads_col,
                conversion_col,
                customers_col,
                units_col,
                price_col,
                discount_col,
                cac_col,
                nps_col,
                churn_col,
                region_col,
                city_col,
                channel_col,
                campaign_col,
                segment_col,
                category_col,
            ],
        }
    )

    st.dataframe(mapping_table, use_container_width=True)


# ============================================================
# TAB EXPORTAR
# ============================================================

with tab_export:
    st.subheader("Exportar datos, KPI y tablas")

    a, b = st.columns(2)

    with a:
        st.download_button(
            "Descargar datos filtrados CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="business_case_filtrado.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with b:
        try:
            st.download_button(
                "Descargar datos filtrados Excel",
                data=to_excel_bytes(filtered),
                file_name="business_case_filtrado.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning("No se pudo exportar a Excel. Verifica openpyxl.")
            st.exception(exc)

    kpi_table = pd.DataFrame(
        {
            "KPI": [
                "ventas",
                "utilidad",
                "costo_marketing",
                "roi",
                "conversion",
                "leads",
                "clientes",
                "ticket_promedio",
                "cac",
                "margen_operativo",
                "margen_bruto",
            ],
            "valor": [
                total_sales,
                total_profit,
                total_cost,
                roi_calc,
                conversion_calc,
                total_leads,
                total_customers,
                avg_ticket,
                cac_calc,
                operating_margin,
                gross_margin_rate,
            ],
        }
    )

    st.markdown("#### Tabla KPI")
    st.dataframe(kpi_table, use_container_width=True)

    st.download_button(
        "Descargar KPI CSV",
        data=kpi_table.to_csv(index=False).encode("utf-8"),
        file_name="business_case_kpis.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.divider()

st.caption(
    "Colorful Data Gallery · Streamlit + Altair · ventas · campañas · canales · regiones · costos · margen · conversión · ROI · utilidad."
)
