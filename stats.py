# ============================================================
# COLORFUL STREAMLIT DATA GALLERY
# ------------------------------------------------------------
# Sin streamlit-echarts.
# Sin xlsxwriter.
# Con Altair + Streamlit + Pandas.
#
# Instalar:
#   python -m pip install streamlit pandas numpy altair openpyxl xlrd
#
# Ejecutar:
#   python -m streamlit run stats.py
# ============================================================

from __future__ import annotations

from io import BytesIO
from statistics import NormalDist
from typing import Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Colorful Data Gallery",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.data_transformers.disable_max_rows()


CSS = """
<style>
    .main .block-container {
        max-width: 1580px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.045em;
    }

    .hero {
        padding: 1.55rem 1.7rem;
        border-radius: 32px;
        background:
            radial-gradient(circle at 5% 10%, rgba(255, 64, 129, 0.28), transparent 30%),
            radial-gradient(circle at 33% 0%, rgba(124, 77, 255, 0.24), transparent 30%),
            radial-gradient(circle at 70% 0%, rgba(0, 200, 255, 0.22), transparent 28%),
            radial-gradient(circle at 92% 88%, rgba(0, 230, 118, 0.22), transparent 32%),
            linear-gradient(135deg, #ffffff 0%, #f6f8ff 50%, #fff6fa 100%);
        border: 1px solid rgba(15, 23, 42, 0.08);
        box-shadow: 0 24px 64px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 2.55rem;
        font-weight: 800;
        line-height: 1.02;
        margin-bottom: 0.45rem;
        color: #101828;
    }

    .hero-subtitle {
        font-size: 1.04rem;
        color: rgba(15, 23, 42, 0.72);
        max-width: 1080px;
    }

    .chip-row {
        display: flex;
        gap: 0.48rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }

    .chip {
        border-radius: 999px;
        padding: 0.36rem 0.78rem;
        background: rgba(255, 255, 255, 0.74);
        border: 1px solid rgba(15, 23, 42, 0.08);
        font-size: 0.83rem;
        font-weight: 700;
        color: rgba(15, 23, 42, 0.72);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.045);
    }

    .metric-card {
        min-height: 112px;
        padding: 1.05rem 1.1rem;
        border-radius: 24px;
        border: 1px solid rgba(15, 23, 42, 0.08);
        background:
            radial-gradient(circle at top left, rgba(124, 77, 255, 0.12), transparent 38%),
            linear-gradient(180deg, rgba(255,255,255,0.94), rgba(255,255,255,0.78));
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
    }

    .metric-label {
        color: rgba(15, 23, 42, 0.55);
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.075em;
        margin-bottom: 0.26rem;
    }

    .metric-value {
        color: rgba(15, 23, 42, 0.96);
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.04em;
    }

    .small-note {
        font-size: 0.84rem;
        color: rgba(15, 23, 42, 0.60);
        margin-top: 0.14rem;
    }

    .section-label {
        color: rgba(15, 23, 42, 0.54);
        font-size: 0.80rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.095em;
        margin-top: 0.35rem;
        margin-bottom: 0.15rem;
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
    "Candy neon": {
        "colors": [
            "#FF2E88", "#7C4DFF", "#00B8D9", "#00E676", "#FFD600",
            "#FF6D00", "#00BFA5", "#536DFE", "#F50057", "#64DD17",
            "#AA00FF", "#00C853", "#FFAB00", "#18FFFF", "#FF4081",
        ],
        "sequential": "turbo",
        "diverging": "redblue",
        "accent": "#7C4DFF",
        "accent2": "#FF2E88",
    },
    "Tropical": {
        "colors": [
            "#00C2FF", "#00E5A8", "#FFB000", "#FF477E", "#845EC2",
            "#2C73D2", "#F9F871", "#008F7A", "#FFC75F", "#C34A36",
            "#4D8076", "#B39CD0", "#00D2FC", "#FBEAFF", "#FF8066",
        ],
        "sequential": "viridis",
        "diverging": "purplegreen",
        "accent": "#00B8D9",
        "accent2": "#FF477E",
    },
    "Editorial bright": {
        "colors": [
            "#2962FF", "#D500F9", "#00BFA5", "#FF6D00", "#FF1744",
            "#00B0FF", "#64DD17", "#FFD600", "#651FFF", "#1DE9B6",
            "#C51162", "#304FFE", "#AEEA00", "#FF9100", "#00E5FF",
        ],
        "sequential": "blues",
        "diverging": "redblue",
        "accent": "#2962FF",
        "accent2": "#D500F9",
    },
    "Sunset": {
        "colors": [
            "#FF006E", "#FB5607", "#FFBE0B", "#8338EC", "#3A86FF",
            "#F72585", "#B5179E", "#7209B7", "#4361EE", "#4CC9F0",
            "#FF9F1C", "#2EC4B6", "#E71D36", "#9B5DE5", "#00F5D4",
        ],
        "sequential": "orangered",
        "diverging": "brownbluegreen",
        "accent": "#FB5607",
        "accent2": "#8338EC",
    },
}


# ============================================================
# UTILIDADES
# ============================================================

@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes: bytes, sep: str, encoding: str) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes), sep=sep, encoding=encoding)


@st.cache_data(show_spinner=False)
def read_excel_cached(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)


def excel_sheet_names(file_bytes: bytes) -> list[str]:
    return pd.ExcelFile(BytesIO(file_bytes)).sheet_names


def to_excel_openpyxl_bytes(df: pd.DataFrame) -> bytes:
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


def maybe_parse_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def maybe_parse_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
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


def infer_column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    temporal = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns.tolist()
    categorical = [c for c in df.columns if c not in numeric and c not in temporal]

    return {
        "numeric": numeric,
        "temporal": temporal,
        "categorical": categorical,
        "all": df.columns.tolist(),
    }


def make_sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260428)
    n = 760

    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    regions = rng.choice(
        ["Norte", "Centro", "Sur", "Occidente", "Bajío", "Golfo"],
        n,
        p=[0.16, 0.28, 0.18, 0.17, 0.13, 0.08],
    )
    categories = rng.choice(
        ["Premium", "Estándar", "Experimental", "Especialidad"],
        n,
        p=[0.28, 0.42, 0.12, 0.18],
    )
    channels = rng.choice(
        ["Retail", "Mayorista", "Exportación", "Online"],
        n,
        p=[0.38, 0.25, 0.22, 0.15],
    )

    region_effect = pd.Series(regions).map(
        {"Norte": 5, "Centro": 2, "Sur": -3, "Occidente": 8, "Bajío": 1, "Golfo": 4}
    ).to_numpy()

    category_effect = pd.Series(categories).map(
        {"Premium": 17, "Estándar": 0, "Experimental": 24, "Especialidad": 31}
    ).to_numpy()

    channel_effect = pd.Series(channels).map(
        {"Retail": 4, "Mayorista": -2, "Exportación": 10, "Online": 6}
    ).to_numpy()

    trend = np.linspace(0, 18, n)
    season = 7.5 * np.sin(np.linspace(0, 12 * np.pi, n))

    price = 86 + region_effect + category_effect + channel_effect + trend + season + rng.normal(0, 8, n)
    quality = 68 + 0.22 * price + rng.normal(0, 5.2, n)
    volume = rng.gamma(shape=4.2, scale=18.0, size=n) + rng.normal(0, 4, n)
    margin = 0.18 + 0.04 * (categories == "Especialidad") + 0.02 * (channels == "Online") + rng.normal(0, 0.025, n)
    income = price * volume
    profit = income * margin
    risk_score = 100 - quality + rng.normal(0, 4, n) + 0.03 * volume

    df = pd.DataFrame(
        {
            "fecha": dates,
            "region": regions,
            "categoria": categories,
            "canal": channels,
            "precio": np.round(price, 2),
            "calidad": np.round(quality, 2),
            "volumen": np.round(volume, 2),
            "ingreso": np.round(income, 2),
            "margen": np.round(margin, 4),
            "utilidad": np.round(profit, 2),
            "riesgo": np.round(risk_score, 2),
        }
    )

    miss_idx = rng.choice(df.index, size=38, replace=False)
    df.loc[miss_idx[:12], "calidad"] = np.nan
    df.loc[miss_idx[12:24], "precio"] = np.nan
    df.loc[miss_idx[24:], "canal"] = np.nan

    return df


def metric_card(label: str, value: str, note: str) -> None:
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
# ESTADÍSTICA
# ============================================================

def numeric_profile(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []

    for col in numeric_cols:
        s = df[col]
        x = s.dropna()

        if len(x) == 0:
            rows.append(
                {
                    "variable": col,
                    "n": 0,
                    "faltantes": int(s.isna().sum()),
                    "faltantes_%": round(100 * s.isna().mean(), 2),
                    "media": np.nan,
                    "mediana": np.nan,
                    "desv_est": np.nan,
                    "varianza": np.nan,
                    "min": np.nan,
                    "p05": np.nan,
                    "q1": np.nan,
                    "q3": np.nan,
                    "p95": np.nan,
                    "max": np.nan,
                    "iqr": np.nan,
                    "rango": np.nan,
                    "asimetria": np.nan,
                    "curtosis": np.nan,
                    "cv": np.nan,
                    "ceros": int((s == 0).sum()),
                    "unicos": int(s.nunique(dropna=True)),
                }
            )
            continue

        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        mean = x.mean()
        std = x.std(ddof=1)

        rows.append(
            {
                "variable": col,
                "n": int(x.shape[0]),
                "faltantes": int(s.isna().sum()),
                "faltantes_%": round(100 * s.isna().mean(), 2),
                "media": mean,
                "mediana": x.median(),
                "desv_est": std,
                "varianza": x.var(ddof=1),
                "min": x.min(),
                "p05": x.quantile(0.05),
                "q1": q1,
                "q3": q3,
                "p95": x.quantile(0.95),
                "max": x.max(),
                "iqr": q3 - q1,
                "rango": x.max() - x.min(),
                "asimetria": x.skew(),
                "curtosis": x.kurtosis(),
                "cv": std / mean if mean != 0 else np.nan,
                "ceros": int((s == 0).sum()),
                "unicos": int(s.nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows).round(4)


def categorical_profile(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    rows = []

    for col in categorical_cols:
        s = df[col]
        vc = s.astype("object").fillna("NA").value_counts(dropna=False)
        mode_value = vc.index[0] if len(vc) else np.nan
        mode_count = int(vc.iloc[0]) if len(vc) else 0

        entropy = 0.0
        if len(vc) > 0:
            p = vc / vc.sum()
            entropy = float(-(p * np.log2(p)).sum())

        rows.append(
            {
                "variable": col,
                "n": int(s.notna().sum()),
                "faltantes": int(s.isna().sum()),
                "faltantes_%": round(100 * s.isna().mean(), 2),
                "unicos": int(s.nunique(dropna=True)),
                "moda": mode_value,
                "frecuencia_moda": mode_count,
                "concentracion_moda_%": round(100 * mode_count / len(s), 2) if len(s) else np.nan,
                "entropia": round(entropy, 4),
            }
        )

    return pd.DataFrame(rows)


def missing_profile(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "variable": df.columns,
            "faltantes": [int(df[c].isna().sum()) for c in df.columns],
            "faltantes_%": [100 * df[c].isna().mean() for c in df.columns],
            "no_faltantes": [int(df[c].notna().sum()) for c in df.columns],
        }
    )

    return out.sort_values("faltantes_%", ascending=False).round(2)


def outlier_profile_iqr(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []

    for col in numeric_cols:
        x = df[col].dropna()

        if len(x) == 0:
            continue

        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        n_out = int(((x < lo) | (x > hi)).sum())

        rows.append(
            {
                "variable": col,
                "limite_inferior": lo,
                "limite_superior": hi,
                "outliers_iqr": n_out,
                "outliers_%": 100 * n_out / len(x),
            }
        )

    return pd.DataFrame(rows).sort_values("outliers_%", ascending=False).round(4)


def grouped_numeric_table(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col, dropna=False)[value_col]
        .agg(
            n="count",
            media="mean",
            mediana="median",
            desv_est="std",
            min="min",
            q1=lambda x: x.quantile(0.25),
            q3=lambda x: x.quantile(0.75),
            max="max",
        )
        .reset_index()
        .round(4)
    )


# ============================================================
# ALTAIR HELPERS
# ============================================================

def cat_scale(colors: list[str]) -> alt.Scale:
    return alt.Scale(range=colors)


def finalize_chart(chart):
    """
    IMPORTANTE:
    Altair/Vega-Lite solo acepta fontWeight:
    normal, bold, lighter, bolder, 100, 200, ..., 900.
    Por eso aquí usamos 800, NO 760.
    """
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
            gridOpacity=0.16,
            domainOpacity=0.18,
            tickOpacity=0.18,
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


def render(chart) -> None:
    st.altair_chart(finalize_chart(chart), use_container_width=True, theme=None)


def category_counts(df: pd.DataFrame, col: str, top_n: int = 30) -> pd.DataFrame:
    return (
        df[col]
        .astype("object")
        .fillna("NA")
        .value_counts()
        .head(top_n)
        .rename_axis(col)
        .reset_index(name="conteo")
    )


def chart_category_bar(df: pd.DataFrame, col: str, colors: list[str], top_n: int = 30):
    data = category_counts(df, col, top_n)

    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
        .encode(
            x=alt.X(f"{col}:N", sort="-y", title=None),
            y=alt.Y("conteo:Q", title="Conteo"),
            color=alt.Color(f"{col}:N", scale=cat_scale(colors), legend=None),
            tooltip=[alt.Tooltip(f"{col}:N"), alt.Tooltip("conteo:Q")],
        )
    )

    text = (
        alt.Chart(data)
        .mark_text(dy=-7, fontSize=11, fontWeight="bold", color="#344054")
        .encode(
            x=alt.X(f"{col}:N", sort="-y"),
            y=alt.Y("conteo:Q"),
            text=alt.Text("conteo:Q", format=".0f"),
        )
    )

    return (bars + text).properties(title=f"Distribución de {col}", height=410)


def chart_bar_agg(df: pd.DataFrame, x: str, y: Optional[str], agg: str, top_n: int, colors: list[str]):
    if y is None:
        data = (
            df.groupby(x, dropna=False)
            .size()
            .reset_index(name="valor")
            .sort_values("valor", ascending=False)
            .head(top_n)
        )
        label = "conteo"
    else:
        data = (
            df.groupby(x, dropna=False)[y]
            .agg(agg)
            .reset_index(name="valor")
            .sort_values("valor", ascending=False)
            .head(top_n)
        )
        label = f"{agg}({y})"

    data[x] = data[x].astype("object").fillna("NA").astype(str)

    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=9, cornerRadiusTopRight=9)
        .encode(
            x=alt.X(f"{x}:N", sort="-y", title=None),
            y=alt.Y("valor:Q", title=label),
            color=alt.Color(f"{x}:N", scale=cat_scale(colors), legend=None),
            tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip("valor:Q", format=",.3f")],
        )
    )

    text = (
        alt.Chart(data)
        .mark_text(dy=-7, fontSize=11, fontWeight="bold", color="#344054")
        .encode(
            x=alt.X(f"{x}:N", sort="-y"),
            y=alt.Y("valor:Q"),
            text=alt.Text("valor:Q", format=".2s"),
        )
    )

    return (bars + text).properties(title=f"{label} por {x}", height=470)


def chart_lollipop(df: pd.DataFrame, x: str, y: Optional[str], agg: str, top_n: int, colors: list[str]):
    if y is None:
        data = (
            df.groupby(x, dropna=False)
            .size()
            .reset_index(name="valor")
            .sort_values("valor", ascending=False)
            .head(top_n)
        )
        label = "conteo"
    else:
        data = (
            df.groupby(x, dropna=False)[y]
            .agg(agg)
            .reset_index(name="valor")
            .sort_values("valor", ascending=False)
            .head(top_n)
        )
        label = f"{agg}({y})"

    data[x] = data[x].astype("object").fillna("NA").astype(str)

    rules = (
        alt.Chart(data)
        .mark_rule(strokeWidth=4, opacity=0.35)
        .encode(
            y=alt.Y(f"{x}:N", sort="-x", title=None),
            x=alt.X("valor:Q", title=label),
            color=alt.Color(f"{x}:N", scale=cat_scale(colors), legend=None),
        )
    )

    points = (
        alt.Chart(data)
        .mark_circle(size=210, opacity=0.92)
        .encode(
            y=alt.Y(f"{x}:N", sort="-x", title=None),
            x=alt.X("valor:Q", title=label),
            color=alt.Color(f"{x}:N", scale=cat_scale(colors), legend=None),
            tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip("valor:Q", format=",.3f")],
        )
    )

    return (rules + points).properties(title=f"Lollipop: {label} por {x}", height=460)


def chart_pareto(df: pd.DataFrame, x: str, y: Optional[str], agg: str, top_n: int, colors: list[str], accent: str):
    if y is None:
        data = df.groupby(x, dropna=False).size().reset_index(name="valor")
        label = "conteo"
    else:
        data = df.groupby(x, dropna=False)[y].agg(agg).reset_index(name="valor")
        label = f"{agg}({y})"

    data[x] = data[x].astype("object").fillna("NA").astype(str)
    data = data.sort_values("valor", ascending=False).head(top_n).reset_index(drop=True)
    total = data["valor"].sum()
    data["cum_pct"] = 100 * data["valor"].cumsum() / total if total else 0

    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8, opacity=0.9)
        .encode(
            x=alt.X(f"{x}:N", sort=None, title=None),
            y=alt.Y("valor:Q", title=label),
            color=alt.Color(f"{x}:N", scale=cat_scale(colors), legend=None),
            tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip("valor:Q", format=",.3f")],
        )
    )

    line = (
        alt.Chart(data)
        .mark_line(color=accent, strokeWidth=3.5, point=True)
        .encode(
            x=alt.X(f"{x}:N", sort=None),
            y=alt.Y("cum_pct:Q", title="% acumulado"),
            tooltip=[alt.Tooltip("cum_pct:Q", format=".2f")],
        )
    )

    return (bars + line).resolve_scale(y="independent").properties(title=f"Pareto: {label} por {x}", height=470)


def chart_donut(df: pd.DataFrame, col: str, colors: list[str], top_n: int = 12):
    data = category_counts(df, col, top_n)

    return (
        alt.Chart(data)
        .mark_arc(innerRadius=82, outerRadius=150)
        .encode(
            theta=alt.Theta("conteo:Q"),
            color=alt.Color(f"{col}:N", scale=cat_scale(colors), title=col),
            tooltip=[alt.Tooltip(f"{col}:N"), alt.Tooltip("conteo:Q")],
        )
        .properties(title=f"Donut: distribución de {col}", height=430)
    )


def chart_stacked_bar(
    df: pd.DataFrame,
    x: str,
    stack: str,
    y: Optional[str],
    agg: str,
    normalize: bool,
    colors: list[str],
):
    data = df.copy()
    data[x] = data[x].astype("object").fillna("NA").astype(str)
    data[stack] = data[stack].astype("object").fillna("NA").astype(str)

    if y is None:
        grouped = data.groupby([x, stack], dropna=False).size().reset_index(name="valor")
        label = "conteo"
    else:
        grouped = data.groupby([x, stack], dropna=False)[y].agg(agg).reset_index(name="valor")
        label = f"{agg}({y})"

    if normalize:
        grouped["valor_pct"] = grouped["valor"] / grouped.groupby(x)["valor"].transform("sum")
        y_enc = alt.Y("valor_pct:Q", stack="normalize", title="Proporción")
        tooltip_value = alt.Tooltip("valor_pct:Q", format=".1%")
        title = f"Barras apiladas 100%: {label}"
    else:
        y_enc = alt.Y("valor:Q", stack="zero", title=label)
        tooltip_value = alt.Tooltip("valor:Q", format=",.3f")
        title = f"Barras apiladas: {label}"

    return (
        alt.Chart(grouped)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{x}:N", title=None),
            y=y_enc,
            color=alt.Color(f"{stack}:N", scale=cat_scale(colors), title=stack),
            tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{stack}:N"), tooltip_value],
        )
        .properties(title=title, height=470)
    )


def chart_line(df: pd.DataFrame, x: str, y: str, color: Optional[str], colors: list[str]):
    data = df.dropna(subset=[x, y]).copy()
    x_type = "T" if pd.api.types.is_datetime64_any_dtype(data[x]) else "N"

    enc = {
        "x": alt.X(f"{x}:{x_type}", title=x),
        "y": alt.Y(f"{y}:Q", title=y),
        "tooltip": [alt.Tooltip(x), alt.Tooltip(f"{y}:Q", format=",.3f")],
    }

    if color:
        enc["color"] = alt.Color(f"{color}:N", scale=cat_scale(colors), title=color)
    else:
        enc["color"] = alt.value(colors[0])

    return (
        alt.Chart(data)
        .mark_line(point=True, strokeWidth=2.8)
        .encode(**enc)
        .properties(title=f"Línea: {y} respecto a {x}", height=470)
        .interactive()
    )


def chart_area(df: pd.DataFrame, x: str, y: str, color: Optional[str], colors: list[str]):
    data = df.dropna(subset=[x, y]).copy()
    x_type = "T" if pd.api.types.is_datetime64_any_dtype(data[x]) else "N"

    enc = {
        "x": alt.X(f"{x}:{x_type}", title=x),
        "y": alt.Y(f"{y}:Q", title=y),
        "tooltip": [alt.Tooltip(x), alt.Tooltip(f"{y}:Q", format=",.3f")],
    }

    if color:
        enc["color"] = alt.Color(f"{color}:N", scale=cat_scale(colors), title=color)
    else:
        enc["color"] = alt.value(colors[1])

    return (
        alt.Chart(data)
        .mark_area(opacity=0.65, line=True)
        .encode(**enc)
        .properties(title=f"Área: {y} respecto a {x}", height=470)
        .interactive()
    )


def chart_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str],
    size: Optional[str],
    regression: bool,
    colors: list[str],
    accent: str,
):
    data = df.dropna(subset=[x, y]).copy()

    enc = {
        "x": alt.X(f"{x}:Q", title=x),
        "y": alt.Y(f"{y}:Q", title=y),
        "tooltip": [
            alt.Tooltip(f"{x}:Q", format=",.3f"),
            alt.Tooltip(f"{y}:Q", format=",.3f"),
        ],
    }

    if color:
        enc["color"] = alt.Color(f"{color}:N", scale=cat_scale(colors), title=color)
        enc["tooltip"].append(alt.Tooltip(f"{color}:N"))
    else:
        enc["color"] = alt.value(colors[0])

    if size:
        enc["size"] = alt.Size(f"{size}:Q", title=size)
        enc["tooltip"].append(alt.Tooltip(f"{size}:Q", format=",.3f"))

    points = alt.Chart(data).mark_circle(opacity=0.72).encode(**enc)

    if regression and len(data) > 2:
        reg = (
            alt.Chart(data)
            .transform_regression(x, y)
            .mark_line(color=accent, strokeWidth=3.4)
            .encode(x=alt.X(f"{x}:Q"), y=alt.Y(f"{y}:Q"))
        )
        chart = points + reg
    else:
        chart = points

    return chart.properties(title=f"Dispersión: {y} vs {x}", height=470).interactive()


def chart_hist_density(df: pd.DataFrame, col: str, seq_scheme: str, accent: str):
    data = df[[col]].dropna()

    hist = (
        alt.Chart(data)
        .mark_bar(opacity=0.78, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=46), title=col),
            y=alt.Y("count():Q", title="Frecuencia"),
            color=alt.Color("count():Q", scale=alt.Scale(scheme=seq_scheme), legend=None),
            tooltip=[alt.Tooltip("count():Q")],
        )
    )

    density = (
        alt.Chart(data)
        .transform_density(col, as_=[col, "densidad"], counts=True)
        .mark_line(strokeWidth=3.4, color=accent)
        .encode(
            x=alt.X(f"{col}:Q", title=col),
            y=alt.Y("densidad:Q", title="Frecuencia / densidad"),
            tooltip=[
                alt.Tooltip(f"{col}:Q", format=",.3f"),
                alt.Tooltip("densidad:Q", format=",.3f"),
            ],
        )
    )

    if len(data) > 0:
        mean = data[col].mean()
    else:
        mean = np.nan

    rule_df = pd.DataFrame({col: [mean], "etiqueta": ["media"]})

    rule = (
        alt.Chart(rule_df)
        .mark_rule(strokeDash=[7, 5], strokeWidth=2.8, color="#101828")
        .encode(
            x=alt.X(f"{col}:Q"),
            tooltip=[alt.Tooltip(f"{col}:Q", format=",.3f")],
        )
    )

    return (hist + density + rule).properties(title=f"Histograma + densidad: {col}", height=440)


def chart_ecdf(df: pd.DataFrame, col: str, accent: str):
    x = df[col].dropna().sort_values().to_numpy()

    if len(x) == 0:
        data = pd.DataFrame({col: [], "F": []})
    else:
        data = pd.DataFrame({col: x, "F": np.arange(1, len(x) + 1) / len(x)})

    return (
        alt.Chart(data)
        .mark_line(strokeWidth=3.2, color=accent)
        .encode(
            x=alt.X(f"{col}:Q", title=col),
            y=alt.Y("F:Q", title="F(x)", scale=alt.Scale(domain=[0, 1])),
            tooltip=[
                alt.Tooltip(f"{col}:Q", format=",.3f"),
                alt.Tooltip("F:Q", format=".3f"),
            ],
        )
        .properties(title=f"ECDF: distribución acumulada de {col}", height=430)
        .interactive()
    )


def chart_qq(df: pd.DataFrame, col: str, colors: list[str], accent: str):
    x = df[col].dropna().sort_values().to_numpy()

    if len(x) < 3:
        data = pd.DataFrame({"teorico": [], "observado": []})
    else:
        n = len(x)
        probs = (np.arange(1, n + 1) - 0.5) / n
        nd = NormalDist()
        theoretical = np.array([nd.inv_cdf(float(p)) for p in probs])
        std = np.std(x, ddof=1)
        observed = (x - np.mean(x)) / std if std != 0 else x * 0
        data = pd.DataFrame({"teorico": theoretical, "observado": observed})

    pts = (
        alt.Chart(data)
        .mark_circle(size=55, opacity=0.72, color=colors[0])
        .encode(
            x=alt.X("teorico:Q", title="Cuantiles normales teóricos"),
            y=alt.Y("observado:Q", title="Cuantiles observados estandarizados"),
            tooltip=[
                alt.Tooltip("teorico:Q", format=".3f"),
                alt.Tooltip("observado:Q", format=".3f"),
            ],
        )
    )

    line = (
        alt.Chart(pd.DataFrame({"x": [-3.5, 3.5], "y": [-3.5, 3.5]}))
        .mark_line(strokeWidth=3, color=accent)
        .encode(x="x:Q", y="y:Q")
    )

    return (pts + line).properties(title=f"QQ plot normal: {col}", height=430).interactive()


def chart_boxplot(df: pd.DataFrame, x: str, y: str, colors: list[str]):
    data = df[[x, y]].dropna().copy()
    data[x] = data[x].astype("object").fillna("NA").astype(str)

    return (
        alt.Chart(data)
        .mark_boxplot(size=42)
        .encode(
            x=alt.X(f"{x}:N", sort="-y", title=x),
            y=alt.Y(f"{y}:Q", title=y),
            color=alt.Color(f"{x}:N", scale=cat_scale(colors), legend=None),
            tooltip=[alt.Tooltip(x), alt.Tooltip(y, format=",.3f")],
        )
        .properties(title=f"Boxplot de {y} por {x}", height=450)
    )


def chart_violin_density(df: pd.DataFrame, x: str, y: str, colors: list[str]):
    data = df[[x, y]].dropna().copy()
    data[x] = data[x].astype("object").fillna("NA").astype(str)

    top = data[x].value_counts().head(10).index.tolist()
    data = data[data[x].isin(top)]

    return (
        alt.Chart(data)
        .transform_density(
            y,
            as_=[y, "density"],
            groupby=[x],
            counts=True,
        )
        .mark_area(orient="horizontal", opacity=0.72)
        .encode(
            y=alt.Y(f"{y}:Q", title=y),
            x=alt.X("density:Q", stack="center", impute=None, title="Densidad"),
            color=alt.Color(f"{x}:N", scale=cat_scale(colors), title=x),
            column=alt.Column(f"{x}:N", title=None, spacing=8),
            tooltip=[alt.Tooltip(f"{x}:N"), alt.Tooltip(f"{y}:Q", format=",.3f")],
        )
        .properties(title=f"Violin/densidad por {x}: {y}", height=360)
    )


def chart_heatmap_2d(df: pd.DataFrame, x: str, y: str, seq_scheme: str):
    data = df[[x, y]].dropna()

    return (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X(f"{x}:Q", bin=alt.Bin(maxbins=38), title=x),
            y=alt.Y(f"{y}:Q", bin=alt.Bin(maxbins=38), title=y),
            color=alt.Color("count():Q", scale=alt.Scale(scheme=seq_scheme), title="Conteo"),
            tooltip=[alt.Tooltip("count():Q", title="conteo")],
        )
        .properties(title=f"Heatmap bivariado: {y} vs {x}", height=470)
    )


def chart_corr_heatmap(corr: pd.DataFrame, div_scheme: str):
    corr_long = corr.reset_index().melt(
        id_vars="index",
        var_name="variable_y",
        value_name="correlacion",
    )

    corr_long = corr_long.rename(columns={"index": "variable_x"})

    rect = (
        alt.Chart(corr_long)
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
        alt.Chart(corr_long)
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

    return (rect + text).properties(title="Heatmap de correlaciones", height=620)


def chart_category_heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    value: Optional[str],
    agg: str,
    seq_scheme: str,
):
    data = df.copy()
    data[x] = data[x].astype("object").fillna("NA").astype(str)
    data[y] = data[y].astype("object").fillna("NA").astype(str)

    if value is None:
        grouped = data.groupby([x, y], dropna=False).size().reset_index(name="valor")
        label = "conteo"
    else:
        grouped = data.groupby([x, y], dropna=False)[value].agg(agg).reset_index(name="valor")
        label = f"{agg}({value})"

    return (
        alt.Chart(grouped)
        .mark_rect(cornerRadius=4)
        .encode(
            x=alt.X(f"{x}:N", title=x),
            y=alt.Y(f"{y}:N", title=y),
            color=alt.Color("valor:Q", scale=alt.Scale(scheme=seq_scheme), title=label),
            tooltip=[
                alt.Tooltip(f"{x}:N"),
                alt.Tooltip(f"{y}:N"),
                alt.Tooltip("valor:Q", format=",.3f"),
            ],
        )
        .properties(title=f"Heatmap categórico: {label}", height=470)
    )


def chart_calendar_heatmap(df: pd.DataFrame, date_col: str, value_col: Optional[str], agg: str, seq_scheme: str):
    cols = [date_col] + ([] if value_col is None else [value_col])
    data = df[cols].dropna(subset=[date_col]).copy()

    if data.empty:
        return alt.Chart(pd.DataFrame({"week": [], "weekday": [], "valor": []})).mark_rect()

    data["date_day"] = data[date_col].dt.floor("D")

    if value_col is None:
        grouped = data.groupby("date_day").size().reset_index(name="valor")
        label = "conteo"
    else:
        grouped = data.groupby("date_day")[value_col].agg(agg).reset_index(name="valor")
        label = f"{agg}({value_col})"

    grouped["week"] = grouped["date_day"].dt.isocalendar().week.astype(int)
    grouped["weekday_num"] = grouped["date_day"].dt.weekday

    names = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
    grouped["weekday"] = grouped["weekday_num"].map(dict(enumerate(names)))

    return (
        alt.Chart(grouped)
        .mark_rect(cornerRadius=3)
        .encode(
            x=alt.X("week:O", title="Semana ISO"),
            y=alt.Y("weekday:N", sort=names, title=None),
            color=alt.Color("valor:Q", scale=alt.Scale(scheme=seq_scheme), title=label),
            tooltip=[
                alt.Tooltip("date_day:T", title="Fecha"),
                alt.Tooltip("weekday:N", title="Día"),
                alt.Tooltip("valor:Q", format=",.3f", title=label),
            ],
        )
        .properties(title=f"Calendar heatmap: {label}", height=300)
    )


def chart_control(df: pd.DataFrame, date_col: str, y_col: str, accent: str, colors: list[str]):
    data = df[[date_col, y_col]].dropna().sort_values(date_col).copy()

    if data.empty:
        return alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()

    data["periodo"] = data[date_col].dt.floor("D")
    grouped = data.groupby("periodo", as_index=False)[y_col].mean()

    mu = grouped[y_col].mean()
    sd = grouped[y_col].std(ddof=1)

    grouped["media"] = mu
    grouped["ucl2"] = mu + 2 * sd
    grouped["lcl2"] = mu - 2 * sd
    grouped["ucl3"] = mu + 3 * sd
    grouped["lcl3"] = mu - 3 * sd

    line = (
        alt.Chart(grouped)
        .mark_line(strokeWidth=2.8, color=accent, point=True)
        .encode(
            x=alt.X("periodo:T", title="Periodo"),
            y=alt.Y(f"{y_col}:Q", title=y_col),
            tooltip=[
                alt.Tooltip("periodo:T"),
                alt.Tooltip(f"{y_col}:Q", format=",.3f"),
            ],
        )
    )

    media = (
        alt.Chart(grouped)
        .mark_line(strokeDash=[8, 5], strokeWidth=2.2, color="#101828")
        .encode(x="periodo:T", y="media:Q")
    )

    ucl2 = (
        alt.Chart(grouped)
        .mark_line(strokeDash=[8, 5], strokeWidth=2, color=colors[2])
        .encode(x="periodo:T", y="ucl2:Q")
    )

    lcl2 = (
        alt.Chart(grouped)
        .mark_line(strokeDash=[8, 5], strokeWidth=2, color=colors[2])
        .encode(x="periodo:T", y="lcl2:Q")
    )

    ucl3 = (
        alt.Chart(grouped)
        .mark_line(strokeDash=[8, 5], strokeWidth=2, color=colors[4])
        .encode(x="periodo:T", y="ucl3:Q")
    )

    lcl3 = (
        alt.Chart(grouped)
        .mark_line(strokeDash=[8, 5], strokeWidth=2, color=colors[4])
        .encode(x="periodo:T", y="lcl3:Q")
    )

    return (
        line + media + ucl2 + lcl2 + ucl3 + lcl3
    ).properties(title=f"Gráfica de control: {y_col}", height=450).interactive()


def chart_slope(df: pd.DataFrame, date_col: str, cat_col: str, value_col: str, agg: str, colors: list[str]):
    data = df[[date_col, cat_col, value_col]].dropna().copy()

    if data.empty:
        return alt.Chart(pd.DataFrame({"periodo": [], cat_col: [], "valor": []})).mark_line()

    data["periodo"] = data[date_col].dt.to_period("M").dt.to_timestamp()

    grouped = (
        data.groupby(["periodo", cat_col], as_index=False)[value_col]
        .agg(agg)
        .rename(columns={value_col: "valor"})
    )

    first = grouped["periodo"].min()
    last = grouped["periodo"].max()

    slope = grouped[grouped["periodo"].isin([first, last])].copy()

    top_cats = (
        slope.groupby(cat_col)["valor"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .index
        .tolist()
    )

    slope = slope[slope[cat_col].isin(top_cats)]

    return (
        alt.Chart(slope)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("periodo:T", title="Periodo"),
            y=alt.Y("valor:Q", title=f"{agg}({value_col})"),
            color=alt.Color(f"{cat_col}:N", scale=cat_scale(colors), title=cat_col),
            tooltip=[
                alt.Tooltip("periodo:T"),
                alt.Tooltip(f"{cat_col}:N"),
                alt.Tooltip("valor:Q", format=",.3f"),
            ],
        )
        .properties(title=f"Slope chart: primer vs último periodo ({value_col})", height=450)
    )


def chart_time_overview(df: pd.DataFrame, date_col: str, y_col: str, freq: str, accent: str):
    data = df[[date_col, y_col]].dropna().copy()

    if data.empty:
        return alt.Chart(pd.DataFrame({"periodo": [], y_col: []})).mark_line()

    data["periodo"] = data[date_col].dt.to_period(freq).dt.to_timestamp()
    grouped = data.groupby("periodo", as_index=False)[y_col].mean()

    area = (
        alt.Chart(grouped)
        .mark_area(opacity=0.22, color=accent)
        .encode(
            x=alt.X("periodo:T", title="Periodo"),
            y=alt.Y(f"{y_col}:Q", title=f"Promedio de {y_col}"),
        )
    )

    line = (
        alt.Chart(grouped)
        .mark_line(strokeWidth=3.3, color=accent, point=True)
        .encode(
            x=alt.X("periodo:T", title="Periodo"),
            y=alt.Y(f"{y_col}:Q", title=f"Promedio de {y_col}"),
            tooltip=[
                alt.Tooltip("periodo:T"),
                alt.Tooltip(f"{y_col}:Q", format=",.3f"),
            ],
        )
    )

    return (area + line).properties(title=f"Evolución temporal de {y_col}", height=420).interactive()


def chart_pair_summary(
    df: pd.DataFrame,
    x: str,
    y: str,
    category: Optional[str],
    colors: list[str],
    seq_scheme: str,
    accent: str,
):
    data = df.dropna(subset=[x, y]).copy()

    scatter = chart_scatter(
        data,
        x=x,
        y=y,
        color=category,
        size=None,
        regression=True,
        colors=colors,
        accent=accent,
    ).properties(height=335)

    hist_x = (
        alt.Chart(data)
        .mark_bar(opacity=0.80, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{x}:Q", bin=alt.Bin(maxbins=35), title=x),
            y=alt.Y("count():Q", title="Frecuencia"),
            color=alt.Color("count():Q", scale=alt.Scale(scheme=seq_scheme), legend=None),
            tooltip=[alt.Tooltip("count():Q")],
        )
        .properties(title=f"Histograma de {x}", height=230)
    )

    hist_y = (
        alt.Chart(data)
        .mark_bar(opacity=0.80, cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{y}:Q", bin=alt.Bin(maxbins=35), title=y),
            y=alt.Y("count():Q", title="Frecuencia"),
            color=alt.Color("count():Q", scale=alt.Scale(scheme=seq_scheme), legend=None),
            tooltip=[alt.Tooltip("count():Q")],
        )
        .properties(title=f"Histograma de {y}", height=230)
    )

    return alt.vconcat(scatter, alt.hconcat(hist_x, hist_y), spacing=24)


def chart_pair_matrix(df: pd.DataFrame, numeric_cols: list[str], color_col: Optional[str], colors: list[str]):
    cols = numeric_cols[:4]
    data_cols = cols + ([] if color_col is None else [color_col])
    data = df[data_cols].dropna().copy()

    rows = []

    for y in cols:
        row_charts = []

        for x in cols:
            if x == y:
                ch = (
                    alt.Chart(data)
                    .mark_bar(opacity=0.78)
                    .encode(
                        x=alt.X(f"{x}:Q", bin=alt.Bin(maxbins=24), title=x),
                        y=alt.Y("count():Q", title=None),
                        color=alt.value(colors[0]),
                    )
                    .properties(width=215, height=175)
                )
            else:
                enc = {
                    "x": alt.X(f"{x}:Q", title=x),
                    "y": alt.Y(f"{y}:Q", title=y),
                    "tooltip": [
                        alt.Tooltip(f"{x}:Q", format=",.3f"),
                        alt.Tooltip(f"{y}:Q", format=",.3f"),
                    ],
                }

                if color_col:
                    enc["color"] = alt.Color(f"{color_col}:N", scale=cat_scale(colors), title=color_col)
                else:
                    enc["color"] = alt.value(colors[1])

                ch = (
                    alt.Chart(data)
                    .mark_circle(size=35, opacity=0.56)
                    .encode(**enc)
                    .properties(width=215, height=175)
                )

            row_charts.append(ch)

        rows.append(alt.hconcat(*row_charts))

    return alt.vconcat(*rows, spacing=14).properties(title="Matriz de pares")


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🌈 Colorful Data Gallery</div>
        <div class="hero-subtitle">
            Dashboard de Streamlit para cargar datos limpios en CSV o Excel, explorar estadística descriptiva,
            diagnosticar calidad de datos y generar visualizaciones interactivas con Altair.
        </div>
        <div class="chip-row">
            <span class="chip">CSV / Excel</span>
            <span class="chip">Altair nativo</span>
            <span class="chip">Sin streamlit-echarts</span>
            <span class="chip">Sin xlsxwriter</span>
            <span class="chip">Más gráficos</span>
            <span class="chip">Más color</span>
            <span class="chip">Exportable</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("### 1. Cargar datos")

    uploaded_file = st.file_uploader(
        "Archivo CSV o Excel",
        type=["csv", "xlsx", "xls"],
        help="Usa un archivo limpio: encabezados en la primera fila y columnas consistentes.",
    )

    use_sample = st.toggle("Usar datos de ejemplo", value=uploaded_file is None)

    st.markdown("### 2. Color")

    palette_name = st.selectbox("Paleta", list(PALETTES.keys()), index=0)
    palette = PALETTES[palette_name]

    colors = palette["colors"]
    seq_scheme = palette["sequential"]
    div_scheme = palette["diverging"]
    accent = palette["accent"]
    accent2 = palette["accent2"]


# ============================================================
# CARGA DE DATOS
# ============================================================

source_label = "datos de ejemplo"

if uploaded_file is None or use_sample:
    df_raw = make_sample_data()
else:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
    source_label = filename

    try:
        if filename.lower().endswith(".csv"):
            with st.sidebar.expander("Opciones CSV", expanded=True):
                sep = st.selectbox("Separador", [",", ";", "\t", "|"], index=0)
                encoding = st.selectbox("Codificación", ["utf-8", "latin-1", "cp1252"], index=0)

            df_raw = read_csv_cached(file_bytes, sep=sep, encoding=encoding)

        else:
            sheets = excel_sheet_names(file_bytes)
            sheet = st.sidebar.selectbox("Hoja de Excel", sheets, index=0)
            df_raw = read_excel_cached(file_bytes, sheet_name=sheet)

    except Exception as exc:
        st.error("No pude leer el archivo. Revisa formato, separador, codificación u hoja de Excel.")
        st.exception(exc)
        st.stop()

df = clean_column_names(df_raw)

with st.sidebar:
    st.markdown("### 3. Interpretar columnas")

    possible_dates = [
        c for c in df.columns
        if "fecha" in c.lower()
        or "date" in c.lower()
        or "periodo" in c.lower()
        or pd.api.types.is_datetime64_any_dtype(df[c])
    ]

    date_cols = st.multiselect(
        "Columnas de fecha",
        options=df.columns.tolist(),
        default=possible_dates,
    )

    possible_numeric_strings = detect_numeric_text_columns(df)

    numeric_string_cols = st.multiselect(
        "Convertir texto a número",
        options=df.columns.tolist(),
        default=possible_numeric_strings,
    )

df = maybe_parse_dates(df, date_cols)
df = maybe_parse_numeric(df, numeric_string_cols)
groups = infer_column_groups(df)

if df.empty:
    st.warning("El archivo no contiene datos.")
    st.stop()


# ============================================================
# FILTROS
# ============================================================

filtered = df.copy()

with st.sidebar:
    st.markdown("### 4. Filtros")

    if groups["temporal"]:
        with st.expander("Fechas", expanded=True):
            for c in groups["temporal"]:
                s = filtered[c].dropna()

                if len(s) > 0:
                    min_d = s.min().date()
                    max_d = s.max().date()

                    value = st.date_input(
                        f"Rango: {c}",
                        value=(min_d, max_d),
                        min_value=min_d,
                        max_value=max_d,
                    )

                    if isinstance(value, tuple) and len(value) == 2:
                        start, end = value
                        filtered = filtered[
                            (filtered[c].dt.date >= start)
                            & (filtered[c].dt.date <= end)
                        ]

    cat_filter_cols = st.multiselect(
        "Variables categóricas para filtrar",
        options=groups["categorical"],
        default=[],
    )

    for c in cat_filter_cols:
        values = (
            filtered[c]
            .astype("object")
            .fillna("NA")
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )

        selected = st.multiselect(f"Valores: {c}", values, default=values)

        filtered = filtered[
            filtered[c]
            .astype("object")
            .fillna("NA")
            .astype(str)
            .isin(selected)
        ]

    num_filter_cols = st.multiselect(
        "Variables numéricas para filtrar",
        options=groups["numeric"],
        default=[],
    )

    for c in num_filter_cols:
        s = filtered[c].dropna()

        if len(s) > 0:
            lo = float(s.min())
            hi = float(s.max())

            if lo < hi:
                selected_lo, selected_hi = st.slider(
                    f"Rango: {c}",
                    min_value=lo,
                    max_value=hi,
                    value=(lo, hi),
                )

                filtered = filtered[
                    (filtered[c] >= selected_lo)
                    & (filtered[c] <= selected_hi)
                ]

    st.divider()
    st.caption(f"Fuente: {source_label}")
    st.write(f"Filas filtradas: **{len(filtered):,} / {len(df):,}**")


# ============================================================
# KPI CARDS
# ============================================================

missing_total = int(filtered.isna().sum().sum())
missing_pct = (
    100 * missing_total / (filtered.shape[0] * filtered.shape[1])
    if filtered.shape[0] and filtered.shape[1]
    else 0
)

duplicated_total = int(filtered.duplicated().sum())
numeric_count = len(groups["numeric"])
categorical_count = len(groups["categorical"])
temporal_count = len(groups["temporal"])

cols_metric = st.columns(7)

metric_data = [
    ("Filas", f"{len(filtered):,}", "registros activos"),
    ("Columnas", f"{filtered.shape[1]:,}", "variables"),
    ("Numéricas", f"{numeric_count:,}", "detectadas"),
    ("Categóricas", f"{categorical_count:,}", "detectadas"),
    ("Fechas", f"{temporal_count:,}", "detectadas"),
    ("Faltantes", f"{missing_pct:.2f}%", f"{missing_total:,} celdas"),
    ("Duplicados", f"{duplicated_total:,}", "filas"),
]

for c, (label, value, note) in zip(cols_metric, metric_data):
    with c:
        metric_card(label, value, note)

st.write("")


# ============================================================
# TABS
# ============================================================

tab_dash, tab_desc, tab_gallery, tab_bivar, tab_multi, tab_quality, tab_export = st.tabs(
    [
        "Dashboard",
        "Descriptiva",
        "Galería",
        "Bivariado",
        "Multivariado",
        "Calidad",
        "Exportar",
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

with tab_dash:
    st.markdown('<div class="section-label">Executive overview</div>', unsafe_allow_html=True)
    st.subheader("Vista ejecutiva")

    left, right = st.columns([1.08, 1])

    with left:
        if groups["temporal"] and groups["numeric"]:
            date_col = st.selectbox("Fecha", groups["temporal"], key="dash_date")
            y_col = st.selectbox("Variable numérica", groups["numeric"], key="dash_y")
            freq = st.selectbox("Frecuencia", ["D", "W", "M", "Q"], index=2, key="dash_freq")
            render(chart_time_overview(filtered, date_col, y_col, freq, accent))
        elif groups["numeric"]:
            y_col = st.selectbox("Variable numérica", groups["numeric"], key="dash_num")
            render(chart_hist_density(filtered, y_col, seq_scheme, accent))
        else:
            st.info("No hay variables numéricas o temporales suficientes.")

    with right:
        if groups["categorical"]:
            cat = st.selectbox("Variable categórica", groups["categorical"], key="dash_cat")
            render(chart_category_bar(filtered, cat, colors, top_n=28))
        else:
            st.info("No hay variables categóricas.")

    a, b = st.columns(2)

    with a:
        miss = missing_profile(filtered)

        chart = (
            alt.Chart(miss.head(35))
            .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7)
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
            .properties(title="Faltantes por variable", height=360)
        )

        render(chart)

    with b:
        if len(groups["numeric"]) >= 2:
            corr_cols = groups["numeric"][: min(8, len(groups["numeric"]))]
            corr = filtered[corr_cols].corr(method="pearson")
            render(chart_corr_heatmap(corr, div_scheme))
        else:
            st.info("Se necesitan al menos dos variables numéricas para correlaciones.")

    st.markdown("#### Vista previa")
    st.dataframe(filtered.head(180), use_container_width=True)


# ============================================================
# DESCRIPTIVA
# ============================================================

with tab_desc:
    st.markdown('<div class="section-label">Descriptive statistics</div>', unsafe_allow_html=True)
    st.subheader("Estadística descriptiva ampliada")

    nprof = numeric_profile(filtered, groups["numeric"])
    cprof = categorical_profile(filtered, groups["categorical"])

    a, b = st.columns([1.28, 1])

    with a:
        st.markdown("#### Perfil numérico")
        if not nprof.empty:
            st.dataframe(nprof, use_container_width=True)
        else:
            st.info("No hay variables numéricas.")

    with b:
        st.markdown("#### Perfil categórico")
        if not cprof.empty:
            st.dataframe(cprof, use_container_width=True)
        else:
            st.info("No hay variables categóricas.")

    st.divider()

    d1, d2, d3 = st.columns(3)

    with d1:
        if groups["numeric"]:
            selected_num = st.selectbox("Histograma", groups["numeric"], key="desc_hist")
            render(chart_hist_density(filtered, selected_num, seq_scheme, accent))

    with d2:
        if groups["numeric"]:
            selected_ecdf = st.selectbox("ECDF", groups["numeric"], key="desc_ecdf")
            render(chart_ecdf(filtered, selected_ecdf, accent2))

    with d3:
        if groups["numeric"]:
            selected_qq = st.selectbox("QQ plot", groups["numeric"], key="desc_qq")
            render(chart_qq(filtered, selected_qq, colors, accent))

    if groups["categorical"] and groups["numeric"]:
        st.markdown("#### Distribución por grupos")

        g1, g2 = st.columns(2)

        with g1:
            group_col = st.selectbox("Grupo", groups["categorical"], key="desc_group")
            value_col = st.selectbox("Variable", groups["numeric"], key="desc_value")
            render(chart_boxplot(filtered, group_col, value_col, colors))

        with g2:
            render(chart_violin_density(filtered, group_col, value_col, colors))
            st.caption("La gráfica tipo violin/densidad se limita a los 10 grupos más frecuentes.")

        st.markdown("#### Tabla por grupo")
        st.dataframe(grouped_numeric_table(filtered, group_col, value_col), use_container_width=True)


# ============================================================
# GALERÍA
# ============================================================

with tab_gallery:
    st.markdown('<div class="section-label">Colorful chart gallery</div>', unsafe_allow_html=True)
    st.subheader("Galería de visualizaciones")

    chart_type = st.selectbox(
        "Tipo de gráfico",
        [
            "Barras agregadas",
            "Lollipop",
            "Pareto",
            "Donut",
            "Barras apiladas",
            "Barras 100%",
            "Línea",
            "Área",
            "Dispersión",
            "Bubble plot",
            "Histograma + densidad",
            "ECDF",
            "QQ plot",
            "Boxplot",
            "Violin/densidad por grupo",
            "Heatmap 2D",
            "Heatmap categórico",
            "Heatmap de correlación",
            "Calendar heatmap",
            "Gráfica de control",
            "Slope chart",
        ],
    )

    chart = None

    if chart_type in ["Barras agregadas", "Lollipop", "Pareto"]:
        if groups["categorical"]:
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                x = st.selectbox("Categoría", groups["categorical"], key="g_bar_x")

            with c2:
                y_choice = st.selectbox("Valor", ["conteo"] + groups["numeric"], key="g_bar_y")

            with c3:
                agg = st.selectbox("Agregación", ["sum", "mean", "median", "min", "max", "count"], key="g_bar_agg")

            with c4:
                top_n = st.number_input("Top N", 3, 100, 25, key="g_bar_top")

            y = None if y_choice == "conteo" else y_choice

            if chart_type == "Barras agregadas":
                chart = chart_bar_agg(filtered, x, y, agg, int(top_n), colors)
            elif chart_type == "Lollipop":
                chart = chart_lollipop(filtered, x, y, agg, int(top_n), colors)
            else:
                chart = chart_pareto(filtered, x, y, agg, int(top_n), colors, accent)
        else:
            st.info("Se necesita una variable categórica.")

    elif chart_type == "Donut":
        if groups["categorical"]:
            c1, c2 = st.columns(2)

            with c1:
                col = st.selectbox("Categoría", groups["categorical"], key="donut_col")

            with c2:
                top_n = st.number_input("Top N", 3, 30, 12, key="donut_top")

            chart = chart_donut(filtered, col, colors, int(top_n))
        else:
            st.info("Se necesita una variable categórica.")

    elif chart_type in ["Barras apiladas", "Barras 100%"]:
        if len(groups["categorical"]) >= 2:
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                x = st.selectbox("Eje X", groups["categorical"], key="stack_x")

            with c2:
                stack = st.selectbox("Apilar por", [c for c in groups["categorical"] if c != x], key="stack_stack")

            with c3:
                y_choice = st.selectbox("Valor", ["conteo"] + groups["numeric"], key="stack_y")

            with c4:
                agg = st.selectbox("Agregación", ["sum", "mean", "median", "count"], key="stack_agg")

            chart = chart_stacked_bar(
                filtered,
                x=x,
                stack=stack,
                y=None if y_choice == "conteo" else y_choice,
                agg=agg,
                normalize=(chart_type == "Barras 100%"),
                colors=colors,
            )
        else:
            st.info("Se necesitan al menos dos variables categóricas.")

    elif chart_type == "Línea":
        if groups["numeric"]:
            c1, c2, c3 = st.columns(3)

            with c1:
                x = st.selectbox("Eje X", groups["temporal"] + groups["categorical"] + groups["numeric"], key="line_x")

            with c2:
                y = st.selectbox("Eje Y", groups["numeric"], key="line_y")

            with c3:
                color = st.selectbox("Color", ["Ninguno"] + groups["categorical"], key="line_color")

            chart = chart_line(filtered, x, y, None if color == "Ninguno" else color, colors)
        else:
            st.info("Se necesita una variable numérica.")

    elif chart_type == "Área":
        if groups["numeric"]:
            c1, c2, c3 = st.columns(3)

            with c1:
                x = st.selectbox("Eje X", groups["temporal"] + groups["categorical"] + groups["numeric"], key="area_x")

            with c2:
                y = st.selectbox("Eje Y", groups["numeric"], key="area_y")

            with c3:
                color = st.selectbox("Color", ["Ninguno"] + groups["categorical"], key="area_color")

            chart = chart_area(filtered, x, y, None if color == "Ninguno" else color, colors)
        else:
            st.info("Se necesita una variable numérica.")

    elif chart_type in ["Dispersión", "Bubble plot"]:
        if len(groups["numeric"]) >= 2:
            c1, c2, c3, c4, c5 = st.columns(5)

            with c1:
                x = st.selectbox("X", groups["numeric"], key="scatter_x")

            with c2:
                y = st.selectbox("Y", groups["numeric"], index=1, key="scatter_y")

            with c3:
                color = st.selectbox("Color", ["Ninguno"] + groups["categorical"], key="scatter_color")

            with c4:
                if chart_type == "Bubble plot":
                    size = st.selectbox("Tamaño", groups["numeric"], key="scatter_size")
                else:
                    size = st.selectbox("Tamaño", ["Ninguno"] + groups["numeric"], key="scatter_size2")

            with c5:
                regression = st.toggle("Regresión", value=True, key="scatter_reg")

            chart = chart_scatter(
                filtered,
                x=x,
                y=y,
                color=None if color == "Ninguno" else color,
                size=None if size == "Ninguno" else size,
                regression=regression,
                colors=colors,
                accent=accent,
            )
        else:
            st.info("Se necesitan al menos dos variables numéricas.")

    elif chart_type == "Histograma + densidad":
        if groups["numeric"]:
            x = st.selectbox("Variable", groups["numeric"], key="hist_x")
            chart = chart_hist_density(filtered, x, seq_scheme, accent)
        else:
            st.info("Se necesita una variable numérica.")

    elif chart_type == "ECDF":
        if groups["numeric"]:
            x = st.selectbox("Variable", groups["numeric"], key="ecdf_x")
            chart = chart_ecdf(filtered, x, accent2)
        else:
            st.info("Se necesita una variable numérica.")

    elif chart_type == "QQ plot":
        if groups["numeric"]:
            x = st.selectbox("Variable", groups["numeric"], key="qq_x")
            chart = chart_qq(filtered, x, colors, accent)
        else:
            st.info("Se necesita una variable numérica.")

    elif chart_type == "Boxplot":
        if groups["categorical"] and groups["numeric"]:
            c1, c2 = st.columns(2)

            with c1:
                x = st.selectbox("Grupo", groups["categorical"], key="box_x")

            with c2:
                y = st.selectbox("Variable", groups["numeric"], key="box_y")

            chart = chart_boxplot(filtered, x, y, colors)
        else:
            st.info("Se necesita una variable categórica y una numérica.")

    elif chart_type == "Violin/densidad por grupo":
        if groups["categorical"] and groups["numeric"]:
            c1, c2 = st.columns(2)

            with c1:
                x = st.selectbox("Grupo", groups["categorical"], key="vio_x")

            with c2:
                y = st.selectbox("Variable", groups["numeric"], key="vio_y")

            chart = chart_violin_density(filtered, x, y, colors)
        else:
            st.info("Se necesita una variable categórica y una numérica.")

    elif chart_type == "Heatmap 2D":
        if len(groups["numeric"]) >= 2:
            c1, c2 = st.columns(2)

            with c1:
                x = st.selectbox("X", groups["numeric"], key="heat_x")

            with c2:
                y = st.selectbox("Y", groups["numeric"], index=1, key="heat_y")

            chart = chart_heatmap_2d(filtered, x, y, seq_scheme)
        else:
            st.info("Se necesitan dos variables numéricas.")

    elif chart_type == "Heatmap categórico":
        if len(groups["categorical"]) >= 2:
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                x = st.selectbox("X", groups["categorical"], key="catheat_x")

            with c2:
                y = st.selectbox("Y", [c for c in groups["categorical"] if c != x], key="catheat_y")

            with c3:
                value = st.selectbox("Valor", ["conteo"] + groups["numeric"], key="catheat_value")

            with c4:
                agg = st.selectbox("Agregación", ["sum", "mean", "median", "count"], key="catheat_agg")

            chart = chart_category_heatmap(
                filtered,
                x=x,
                y=y,
                value=None if value == "conteo" else value,
                agg=agg,
                seq_scheme=seq_scheme,
            )
        else:
            st.info("Se necesitan al menos dos variables categóricas.")

    elif chart_type == "Heatmap de correlación":
        if len(groups["numeric"]) >= 2:
            corr_cols = st.multiselect(
                "Variables",
                groups["numeric"],
                default=groups["numeric"][: min(10, len(groups["numeric"]))],
                key="corr_cols",
            )

            method = st.selectbox("Método", ["pearson", "spearman", "kendall"], key="corr_method")

            if len(corr_cols) >= 2:
                corr = filtered[corr_cols].corr(method=method)
                chart = chart_corr_heatmap(corr, div_scheme)
            else:
                st.info("Selecciona al menos dos variables.")
        else:
            st.info("Se necesitan dos variables numéricas.")

    elif chart_type == "Calendar heatmap":
        if groups["temporal"]:
            c1, c2, c3 = st.columns(3)

            with c1:
                date_col = st.selectbox("Fecha", groups["temporal"], key="cal_date")

            with c2:
                value = st.selectbox("Valor", ["conteo"] + groups["numeric"], key="cal_value")

            with c3:
                agg = st.selectbox("Agregación", ["sum", "mean", "median", "count"], key="cal_agg")

            chart = chart_calendar_heatmap(
                filtered,
                date_col=date_col,
                value_col=None if value == "conteo" else value,
                agg=agg,
                seq_scheme=seq_scheme,
            )
        else:
            st.info("Se necesita una variable de fecha.")

    elif chart_type == "Gráfica de control":
        if groups["temporal"] and groups["numeric"]:
            c1, c2 = st.columns(2)

            with c1:
                date_col = st.selectbox("Fecha", groups["temporal"], key="control_date")

            with c2:
                y_col = st.selectbox("Variable", groups["numeric"], key="control_y")

            chart = chart_control(filtered, date_col, y_col, accent, colors)
        else:
            st.info("Se necesita una variable temporal y una numérica.")

    elif chart_type == "Slope chart":
        if groups["temporal"] and groups["categorical"] and groups["numeric"]:
            c1, c2, c3, c4 = st.columns(4)

            with c1:
                date_col = st.selectbox("Fecha", groups["temporal"], key="slope_date")

            with c2:
                cat_col = st.selectbox("Categoría", groups["categorical"], key="slope_cat")

            with c3:
                value_col = st.selectbox("Valor", groups["numeric"], key="slope_value")

            with c4:
                agg = st.selectbox("Agregación", ["mean", "sum", "median"], key="slope_agg")

            chart = chart_slope(filtered, date_col, cat_col, value_col, agg, colors)
        else:
            st.info("Se necesita fecha, categoría y variable numérica.")

    if chart is not None:
        render(chart)

        with st.expander("Ver especificación Vega-Lite / Altair"):
            st.json(finalize_chart(chart).to_dict())


# ============================================================
# BIVARIADO
# ============================================================

with tab_bivar:
    st.markdown('<div class="section-label">Bivariate analysis</div>', unsafe_allow_html=True)
    st.subheader("Análisis bivariado")

    if len(groups["numeric"]) >= 2:
        b1, b2, b3 = st.columns(3)

        with b1:
            x = st.selectbox("Variable X", groups["numeric"], key="bivar_x")

        with b2:
            y = st.selectbox("Variable Y", groups["numeric"], index=1, key="bivar_y")

        with b3:
            category = st.selectbox("Segmentar por", ["Ninguno"] + groups["categorical"], key="bivar_cat")

        data_pair = filtered[[x, y] + ([] if category == "Ninguno" else [category])].dropna()

        corr_p = data_pair[[x, y]].corr(method="pearson").iloc[0, 1] if len(data_pair) > 1 else np.nan
        corr_s = data_pair[[x, y]].corr(method="spearman").iloc[0, 1] if len(data_pair) > 1 else np.nan

        c1, c2, c3 = st.columns(3)

        c1.metric("Pearson", "—" if pd.isna(corr_p) else f"{corr_p:.4f}")
        c2.metric("Spearman", "—" if pd.isna(corr_s) else f"{corr_s:.4f}")
        c3.metric("Observaciones", f"{len(data_pair):,}")

        chart = chart_pair_summary(
            filtered,
            x=x,
            y=y,
            category=None if category == "Ninguno" else category,
            colors=colors,
            seq_scheme=seq_scheme,
            accent=accent,
        )

        render(chart)

        if category != "Ninguno":
            st.markdown("#### Tabla por segmento")

            grouped = (
                data_pair.groupby(category)
                .agg(
                    n=(x, "size"),
                    media_x=(x, "mean"),
                    media_y=(y, "mean"),
                    mediana_x=(x, "median"),
                    mediana_y=(y, "median"),
                    desv_x=(x, "std"),
                    desv_y=(y, "std"),
                )
                .reset_index()
                .round(4)
            )

            st.dataframe(grouped, use_container_width=True)
    else:
        st.info("Se necesitan al menos dos variables numéricas.")


# ============================================================
# MULTIVARIADO
# ============================================================

with tab_multi:
    st.markdown('<div class="section-label">Multivariate exploration</div>', unsafe_allow_html=True)
    st.subheader("Exploración multivariada")

    if len(groups["numeric"]) >= 2:
        selected = st.multiselect(
            "Variables para matriz de pares",
            groups["numeric"],
            default=groups["numeric"][: min(4, len(groups["numeric"]))],
            key="pair_selected",
        )

        color_col = st.selectbox(
            "Color por categoría",
            ["Ninguno"] + groups["categorical"],
            key="pair_color",
        )

        if 2 <= len(selected) <= 4:
            render(chart_pair_matrix(filtered, selected, None if color_col == "Ninguno" else color_col, colors))
        elif len(selected) > 4:
            st.warning("Para mantener legibilidad, selecciona máximo 4 variables.")
        else:
            st.info("Selecciona al menos dos variables.")
    else:
        st.info("Se necesitan al menos dos variables numéricas.")

    st.divider()

    if len(groups["numeric"]) >= 2:
        st.markdown("#### Ranking de correlaciones absolutas")

        corr_cols = st.multiselect(
            "Variables para ranking",
            groups["numeric"],
            default=groups["numeric"][: min(8, len(groups["numeric"]))],
            key="rank_corr_cols",
        )

        if len(corr_cols) >= 2:
            corr = filtered[corr_cols].corr().abs()

            pairs = []

            for i, a in enumerate(corr_cols):
                for b in corr_cols[i + 1:]:
                    pairs.append({"par": f"{a} · {b}", "corr_abs": corr.loc[a, b]})

            corr_rank = pd.DataFrame(pairs).sort_values("corr_abs", ascending=False)

            chart = (
                alt.Chart(corr_rank)
                .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7)
                .encode(
                    x=alt.X("par:N", sort="-y", title=None),
                    y=alt.Y("corr_abs:Q", title="|correlación|"),
                    color=alt.Color("corr_abs:Q", scale=alt.Scale(scheme=seq_scheme), legend=None),
                    tooltip=[
                        alt.Tooltip("par:N"),
                        alt.Tooltip("corr_abs:Q", format=".3f"),
                    ],
                )
                .properties(title="Ranking de correlaciones absolutas", height=410)
            )

            render(chart)
            st.dataframe(corr_rank.round(4), use_container_width=True)


# ============================================================
# CALIDAD
# ============================================================

with tab_quality:
    st.markdown('<div class="section-label">Data quality</div>', unsafe_allow_html=True)
    st.subheader("Calidad de datos")

    miss = missing_profile(filtered)
    outliers = outlier_profile_iqr(filtered, groups["numeric"])

    q1, q2 = st.columns([1.1, 1])

    with q1:
        st.markdown("#### Faltantes")

        chart = (
            alt.Chart(miss.head(45))
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
            .properties(title="Faltantes por variable", height=390)
        )

        render(chart)
        st.dataframe(miss, use_container_width=True)

    with q2:
        st.markdown("#### Outliers IQR")

        if not outliers.empty:
            chart = (
                alt.Chart(outliers)
                .mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8)
                .encode(
                    x=alt.X("variable:N", sort="-y", title=None),
                    y=alt.Y("outliers_%:Q", title="% outliers"),
                    color=alt.Color("outliers_%:Q", scale=alt.Scale(scheme=seq_scheme), legend=None),
                    tooltip=[
                        alt.Tooltip("variable:N"),
                        alt.Tooltip("outliers_iqr:Q"),
                        alt.Tooltip("outliers_%:Q", format=".2f"),
                    ],
                )
                .properties(title="Outliers por regla IQR", height=390)
            )

            render(chart)
            st.dataframe(outliers, use_container_width=True)
        else:
            st.info("No hay variables numéricas.")

    st.markdown("#### Cardinalidad y duplicados")

    card = pd.DataFrame(
        {
            "variable": filtered.columns,
            "tipo": [str(filtered[c].dtype) for c in filtered.columns],
            "unicos": [filtered[c].nunique(dropna=True) for c in filtered.columns],
            "unicos_%": [
                100 * filtered[c].nunique(dropna=True) / len(filtered) if len(filtered) else 0
                for c in filtered.columns
            ],
        }
    ).sort_values("unicos_%", ascending=False)

    c1, c2 = st.columns(2)

    with c1:
        chart = (
            alt.Chart(card)
            .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7)
            .encode(
                x=alt.X("variable:N", sort="-y", title=None),
                y=alt.Y("unicos_%:Q", title="% únicos"),
                color=alt.Color("unicos_%:Q", scale=alt.Scale(scheme=seq_scheme), legend=None),
                tooltip=[
                    alt.Tooltip("variable:N"),
                    alt.Tooltip("unicos:Q"),
                    alt.Tooltip("unicos_%:Q", format=".2f"),
                ],
            )
            .properties(title="Cardinalidad relativa", height=380)
        )

        render(chart)

    with c2:
        dup_count = int(filtered.duplicated().sum())
        st.metric("Filas duplicadas", f"{dup_count:,}")

        if dup_count > 0:
            st.dataframe(filtered[filtered.duplicated(keep=False)].head(250), use_container_width=True)
        else:
            st.success("No se detectaron filas duplicadas exactas.")

    st.dataframe(card.round(2), use_container_width=True)


# ============================================================
# EXPORTAR
# ============================================================

with tab_export:
    st.markdown('<div class="section-label">Export</div>', unsafe_allow_html=True)
    st.subheader("Exportar datos y tablas")

    c1, c2 = st.columns(2)

    with c1:
        st.download_button(
            "Descargar datos filtrados en CSV",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="datos_filtrados.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with c2:
        try:
            excel_bytes = to_excel_openpyxl_bytes(filtered)

            st.download_button(
                "Descargar datos filtrados en Excel",
                data=excel_bytes,
                file_name="datos_filtrados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning("No se pudo crear el Excel. Verifica que openpyxl esté instalado.")
            st.exception(exc)

    st.markdown("#### Perfiles exportables")

    profile_num = numeric_profile(filtered, groups["numeric"])
    profile_cat = categorical_profile(filtered, groups["categorical"])
    profile_miss = missing_profile(filtered)
    profile_out = outlier_profile_iqr(filtered, groups["numeric"])

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.download_button(
            "Perfil numérico CSV",
            data=profile_num.to_csv(index=False).encode("utf-8"),
            file_name="perfil_numerico.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with p2:
        st.download_button(
            "Perfil categórico CSV",
            data=profile_cat.to_csv(index=False).encode("utf-8"),
            file_name="perfil_categorico.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with p3:
        st.download_button(
            "Faltantes CSV",
            data=profile_miss.to_csv(index=False).encode("utf-8"),
            file_name="faltantes.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with p4:
        st.download_button(
            "Outliers CSV",
            data=profile_out.to_csv(index=False).encode("utf-8"),
            file_name="outliers_iqr.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("#### Diccionario de variables")

    dictionary = pd.DataFrame(
        {
            "variable": filtered.columns,
            "tipo": [str(filtered[c].dtype) for c in filtered.columns],
            "no_nulos": [int(filtered[c].notna().sum()) for c in filtered.columns],
            "faltantes": [int(filtered[c].isna().sum()) for c in filtered.columns],
            "unicos": [int(filtered[c].nunique(dropna=True)) for c in filtered.columns],
        }
    )

    st.dataframe(dictionary, use_container_width=True)

    st.download_button(
        "Descargar diccionario CSV",
        data=dictionary.to_csv(index=False).encode("utf-8"),
        file_name="diccionario_variables.csv",
        mime="text/csv",
        use_container_width=True,
    )


st.divider()
st.caption(
    "Colorful Data Gallery · Streamlit + Altair · Sin streamlit-echarts · Sin xlsxwriter · Excel exportado con openpyxl."
)
