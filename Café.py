import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy import stats

# ── Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Precios del Café Mexicano",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "Chiapas":  "#1B6CA8",
    "Veracruz": "#BE123C",
    "Puebla":   "#D97706",
    "Oaxaca":   "#0D9488",
    "Guerrero": "#7C3AED",
    "Otro":     "#92400E",
}

# ── Data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Base_Cafe.csv", encoding="utf-8-sig")
    estados = ["Veracruz", "Puebla", "Chiapas", "Oaxaca", "Guerrero"]
    def get_estado(row):
        for e in estados:
            if pd.notna(row[e]) and str(row[e]).strip() != "":
                return e
        return "Otro"
    df["Estado"] = df.apply(get_estado, axis=1)
    df["Tipo_cafe"] = df["Arábica"].apply(
        lambda x: "Arábica" if pd.notna(x) and str(x).strip() != "" else "Robusta"
    )
    return df

df = load_data()

PRICE_COLS = [c for c in df.columns if "Precio" in c]

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
    "Precio mínimo por Kilo de fruto o cereza convencional": "Cereza Conv. Min",
    "Precio mínimo por Kilo de pergamino lavado convencional": "Perg. Lav. Conv. Min",
    "Precio mínimo por Kilo de natural convencional": "Natural Conv. Min",
    "Precio mínimo por Kilo de verde, oro, morteado convencional": "Verde Conv. Min",
    "Precio máximo por Kilo de fruto o cereza convencional": "Cereza Conv. Máx",
    "Precio máximo por Kilo de pergamino lavado convencional": "Perg. Lav. Conv. Máx",
    "Precio máximo por Kilo de natural convencional": "Natural Conv. Máx",
    "Precio máximo por Kilo de verde, oro, morteado convencional": "Verde Conv. Máx",
    "Precio mínimo por Kilo de pergamino lavado especial": "Perg. Lav. Esp. Min",
    "Precio mínimo por Kilo de pergamino honey especial": "Honey Esp. Min",
    "Precio mínimo por Kilo de pergamino semilavado especial": "Semilavado Esp. Min",
    "Precio mínimo por Kilo de natural especial": "Natural Esp. Min",
    "Precio mínimo por Kilo de café verde, oro, morteado especial": "Verde Esp. Min",
    "Precio máximo por Kilo de Pergamino lavado especial": "Perg. Lav. Esp. Máx",
    "Precio máximo por Kilo de pergamino honey especial": "Honey Esp. Máx",
    "Precio máximo por Kilo de pergamino semilavado especial": "Semilavado Esp. Máx",
    "Precio máximo por Kilo de natural especial": "Natural Esp. Máx",
    "Precio máximo por Kilo de café verde, oro o morteado especial": "Verde Esp. Máx",
}

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/45/A_small_cup_of_coffee.JPG/320px-A_small_cup_of_coffee.JPG",
    use_column_width=True,
)
st.sidebar.title("☕ Filtros")
estados_sel = st.sidebar.multiselect(
    "Estado",
    options=sorted(df["Estado"].unique()),
    default=sorted(df["Estado"].unique()),
)
tipo_sel = st.sidebar.multiselect(
    "Tipo de café",
    options=df["Tipo_cafe"].unique().tolist(),
    default=df["Tipo_cafe"].unique().tolist(),
)

df_f = df[df["Estado"].isin(estados_sel) & df["Tipo_cafe"].isin(tipo_sel)].copy()

# ── Header ───────────────────────────────────────────────────
st.title("☕ Estructura de Precios del Café Mexicano")
st.markdown(
    "**César Galindo · Robert H. Manson · Marisol Velázquez-Salazar**  \n"
    "Dashboard interactivo — 216 registros · 6 estados · 18 variables de precio"
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Productores", len(df_f))
k2.metric("Estados", df_f["Estado"].nunique())
conv_vals = df_f[CONV_MIN + CONV_MAX].stack().dropna()
esp_vals  = df_f[ESP_MIN  + ESP_MAX].stack().dropna()
k3.metric("Precio mediano Conv.", f"${conv_vals[conv_vals>0].median():.0f} MXN/kg" if len(conv_vals[conv_vals>0]) else "N/D")
k4.metric("Precio mediano Esp.",  f"${esp_vals[esp_vals>0].median():.0f} MXN/kg"  if len(esp_vals[esp_vals>0])  else "N/D")

st.divider()

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Distribución por Estado",
    "💰 Brechas de Precio",
    "🔗 Correlaciones",
    "🗂️ Clustering K-means",
    "📦 Boxplots Detallados",
])

# ════════════════════════════════════════════════
# TAB 1 – Distribución por estado
# ════════════════════════════════════════════════
with tab1:
    st.subheader("Productores por Estado y Tipo de Café")
    c1, c2 = st.columns(2)

    with c1:
        cnt = df_f.groupby(["Estado","Tipo_cafe"]).size().reset_index(name="n")
        fig = px.bar(
            cnt, x="Estado", y="n", color="Tipo_cafe",
            barmode="stack",
            color_discrete_map={"Arábica":"#1B6CA8","Robusta":"#D97706"},
            labels={"n":"Número de productores","Estado":"Estado","Tipo_cafe":"Tipo"},
            title="Productores por estado"
        )
        fig.update_layout(legend_title_text="Tipo")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        tot_e = df_f["Estado"].value_counts().reset_index()
        tot_e.columns = ["Estado","n"]
        fig2 = px.pie(
            tot_e, values="n", names="Estado",
            color="Estado",
            color_discrete_map=PALETTE,
            title="Proporción por estado",
            hole=0.35,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Precio Promedio por Estado (Convencional vs Especial)")
    rows = []
    for estado in df_f["Estado"].unique():
        sub = df_f[df_f["Estado"]==estado]
        c_med = sub[CONV_MIN+CONV_MAX].stack().replace(0,np.nan).dropna()
        e_med = sub[ESP_MIN+ESP_MAX].stack().replace(0,np.nan).dropna()
        rows.append({
            "Estado": estado,
            "Convencional": c_med.median() if len(c_med) else np.nan,
            "Especial":     e_med.median() if len(e_med) else np.nan,
        })
    avg_df = pd.DataFrame(rows).melt("Estado", var_name="Mercado", value_name="Precio Mediano")
    fig3 = px.bar(
        avg_df, x="Estado", y="Precio Mediano", color="Mercado",
        barmode="group",
        color_discrete_map={"Convencional":"#1B6CA8","Especial":"#D97706"},
        labels={"Precio Mediano":"MXN / kg"},
        title="Precio mediano por estado y tipo de mercado"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 2 – Brechas de precio
# ════════════════════════════════════════════════
with tab2:
    st.subheader("Brecha de Negociación (Precio Máximo − Mínimo)")
    st.markdown("La brecha refleja el margen de negociación por tipo de producto.")

    brecha_rows = []
    for col_min, col_max in zip(CONV_MIN, CONV_MAX):
        label = SHORT_LABELS[col_min].replace(" Min","")
        sub = df_f[[col_min, col_max, "Estado"]].dropna()
        sub = sub[(sub[col_min]>0) | (sub[col_max]>0)]
        if len(sub):
            sub["brecha"] = sub[col_max] - sub[col_min]
            for _, row in sub.iterrows():
                brecha_rows.append({"Producto": label, "Brecha": row["brecha"], "Estado": row["Estado"], "Mercado":"Convencional"})

    for col_min, col_max in zip(ESP_MIN, ESP_MAX):
        label = SHORT_LABELS[col_min].replace(" Min","")
        sub = df_f[[col_min, col_max, "Estado"]].dropna()
        sub = sub[(sub[col_min]>0) | (sub[col_max]>0)]
        if len(sub):
            sub["brecha"] = sub[col_max] - sub[col_min]
            for _, row in sub.iterrows():
                brecha_rows.append({"Producto": label, "Brecha": row["brecha"], "Estado": row["Estado"], "Mercado":"Especial"})

    brecha_df = pd.DataFrame(brecha_rows)

    mercado_opt = st.radio("Mercado", ["Convencional","Especial","Ambos"], horizontal=True)
    if mercado_opt != "Ambos":
        brecha_plot = brecha_df[brecha_df["Mercado"]==mercado_opt]
    else:
        brecha_plot = brecha_df

    fig4 = px.box(
        brecha_plot, x="Producto", y="Brecha", color="Estado",
        color_discrete_map=PALETTE,
        points="outliers",
        labels={"Brecha":"MXN / kg","Producto":"Tipo de producto"},
        title=f"Distribución de brechas — {mercado_opt}"
    )
    fig4.update_xaxes(tickangle=30)
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Brecha promedio por estado")
    avg_brecha = brecha_df.groupby(["Estado","Mercado"])["Brecha"].median().reset_index()
    fig5 = px.bar(
        avg_brecha, x="Estado", y="Brecha", color="Mercado",
        barmode="group",
        color_discrete_map={"Convencional":"#1B6CA8","Especial":"#D97706"},
        labels={"Brecha":"Brecha mediana (MXN/kg)"},
        title="Brecha mediana de negociación por estado"
    )
    st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 3 – Correlaciones de Spearman
# ════════════════════════════════════════════════
with tab3:
    st.subheader("Matriz de Correlación de Spearman")
    sel_cols = st.multiselect(
        "Variables a incluir",
        options=PRICE_COLS,
        default=CONV_MIN + CONV_MAX,
        format_func=lambda c: SHORT_LABELS.get(c, c),
    )

    if len(sel_cols) >= 2:
        sub = df_f[sel_cols].replace(0, np.nan).dropna(how="all")
        corr_mat, pval_mat = stats.spearmanr(sub, nan_policy="omit")
        if sub.shape[1] == 2:
            corr_mat = np.array([[1, corr_mat],[corr_mat, 1]])
        labels = [SHORT_LABELS.get(c, c) for c in sel_cols]
        fig6 = px.imshow(
            corr_mat,
            x=labels, y=labels,
            zmin=-1, zmax=1,
            color_continuous_scale="RdBu_r",
            text_auto=".2f",
            title="Correlación de Spearman entre variables de precio",
            aspect="auto",
        )
        fig6.update_layout(
            xaxis_tickangle=35,
            coloraxis_colorbar_title="ρ",
        )
        st.plotly_chart(fig6, use_container_width=True)

        # Scatter de las dos variables con mayor correlación
        if len(sel_cols) >= 2:
            st.subheader("Scatter: par seleccionado")
            ca, cb = st.columns(2)
            with ca:
                var_x = st.selectbox("Eje X", sel_cols, format_func=lambda c: SHORT_LABELS.get(c,c))
            with cb:
                var_y = st.selectbox("Eje Y", [c for c in sel_cols if c!=var_x], format_func=lambda c: SHORT_LABELS.get(c,c))
            scatter_df = df_f[[var_x, var_y, "Estado"]].replace(0,np.nan).dropna()
            fig7 = px.scatter(
                scatter_df, x=var_x, y=var_y, color="Estado",
                color_discrete_map=PALETTE,
                trendline="ols",
                labels={var_x: SHORT_LABELS.get(var_x,var_x), var_y: SHORT_LABELS.get(var_y,var_y)},
                title=f"ρ Spearman entre {SHORT_LABELS.get(var_x,var_x)} y {SHORT_LABELS.get(var_y,var_y)}"
            )
            st.plotly_chart(fig7, use_container_width=True)
    else:
        st.info("Selecciona al menos 2 variables.")

# ════════════════════════════════════════════════
# TAB 4 – Clustering K-means
# ════════════════════════════════════════════════
with tab4:
    st.subheader("Clustering K-means sobre variables de precio")

    feat_cols = [c for c in PRICE_COLS if df_f[c].notna().sum() > 10]
    k_val = st.slider("Número de clusters (k)", 2, 6, 3)

    sub_k = df_f[feat_cols + ["Estado"]].copy()
    sub_k[feat_cols] = sub_k[feat_cols].replace(0, np.nan)
    sub_k = sub_k.dropna(subset=feat_cols, how="all")
    sub_k[feat_cols] = sub_k[feat_cols].fillna(sub_k[feat_cols].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub_k[feat_cols])

    km = KMeans(n_clusters=k_val, random_state=42, n_init=10)
    sub_k["Cluster"] = km.fit_predict(X_scaled).astype(str)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    sub_k["PC1"] = coords[:,0]
    sub_k["PC2"] = coords[:,1]

    var_exp = pca.explained_variance_ratio_
    c1, c2 = st.columns([2,1])

    with c1:
        fig8 = px.scatter(
            sub_k, x="PC1", y="PC2", color="Cluster",
            symbol="Estado",
            title=f"PCA — K-means (k={k_val})",
            labels={
                "PC1": f"PC1 ({var_exp[0]*100:.1f}% var.)",
                "PC2": f"PC2 ({var_exp[1]*100:.1f}% var.)",
            },
            hover_data=["Estado"],
        )
        st.plotly_chart(fig8, use_container_width=True)

    with c2:
        st.markdown("**Composición de clusters**")
        comp = sub_k.groupby(["Cluster","Estado"]).size().reset_index(name="n")
        fig9 = px.bar(
            comp, x="Cluster", y="n", color="Estado",
            color_discrete_map=PALETTE, barmode="stack",
            labels={"n":"Productores"},
            title="Estados por cluster"
        )
        st.plotly_chart(fig9, use_container_width=True)

    st.subheader("Perfil de precios medianos por cluster")
    profile = sub_k.groupby("Cluster")[feat_cols].median()
    profile.columns = [SHORT_LABELS.get(c,c) for c in profile.columns]
    fig10 = px.imshow(
        profile.T,
        color_continuous_scale="YlOrRd",
        labels=dict(color="MXN/kg"),
        title="Heatmap: precio mediano por cluster y variable",
        aspect="auto",
        text_auto=".0f",
    )
    fig10.update_layout(xaxis_title="Cluster", yaxis_title="Variable")
    st.plotly_chart(fig10, use_container_width=True)

    # Elbow
    st.subheader("Método del codo (inercia)")
    inertias = []
    ks = range(2, 8)
    for ki in ks:
        km_i = KMeans(n_clusters=ki, random_state=42, n_init=10)
        km_i.fit(X_scaled)
        inertias.append({"k": ki, "Inercia": km_i.inertia_})
    inercia_df = pd.DataFrame(inertias)
    fig11 = px.line(
        inercia_df, x="k", y="Inercia", markers=True,
        title="Método del codo — selección de k",
        labels={"k":"Número de clusters","Inercia":"Inercia (SSE)"}
    )
    fig11.add_vline(x=k_val, line_dash="dash", line_color="red", annotation_text=f"k={k_val}")
    st.plotly_chart(fig11, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 5 – Boxplots detallados
# ════════════════════════════════════════════════
with tab5:
    st.subheader("Boxplots de Precios por Tipo de Producto")

    mercado_tab5 = st.radio("Mercado", ["Convencional","Especial"], horizontal=True, key="tab5_merc")
    cols_grp = (CONV_MIN + CONV_MAX) if mercado_tab5=="Convencional" else (ESP_MIN + ESP_MAX)

    long_rows = []
    for col in cols_grp:
        tmp = df_f[["Estado", col]].copy()
        tmp.columns = ["Estado","Precio"]
        tmp["Variable"] = SHORT_LABELS.get(col, col)
        tmp = tmp[tmp["Precio"].notna() & (tmp["Precio"]>0)]
        long_rows.append(tmp)
    long_df = pd.concat(long_rows, ignore_index=True)

    fig12 = px.box(
        long_df, x="Variable", y="Precio", color="Estado",
        color_discrete_map=PALETTE,
        points=False,
        labels={"Precio":"MXN / kg","Variable":"Tipo de producto"},
        title=f"Distribución de precios — Mercado {mercado_tab5}"
    )
    fig12.update_xaxes(tickangle=35)
    st.plotly_chart(fig12, use_container_width=True)

    st.subheader("Violin: distribución por estado")
    price_sel = st.selectbox(
        "Variable de precio",
        cols_grp,
        format_func=lambda c: SHORT_LABELS.get(c,c)
    )
    viol_df = df_f[["Estado", price_sel]].dropna()
    viol_df = viol_df[viol_df[price_sel]>0]
    fig13 = px.violin(
        viol_df, x="Estado", y=price_sel, color="Estado",
        box=True, points="outliers",
        color_discrete_map=PALETTE,
        labels={price_sel:"MXN / kg"},
        title=f"Distribución de: {SHORT_LABELS.get(price_sel, price_sel)}"
    )
    st.plotly_chart(fig13, use_container_width=True)

    # Tabla resumen
    st.subheader("Estadísticas descriptivas")
    summary = long_df.groupby("Variable")["Precio"].describe().round(1)
    st.dataframe(summary, use_container_width=True)

st.caption("Datos: Base_Cafe.csv — 216 productores · 6 estados · México")
