import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Earthquake Analytics Dashboard",
    layout="wide"
)

# ================= STYLE =================
st.markdown("""
<style>
/* Hapus padding/margin default Streamlit */
.css-18e3th9 {padding-top:0rem;}       /* container utama */
.block-container {padding-top:0rem;}   /* block content */
section[data-testid="stSidebar"] {padding-top:1rem;}  /* sidebar atas */
.stApp {background:#070d1a;color:#e5e7eb}
section[data-testid="stSidebar"] {
    background:#030712;
    padding-top:1.5rem
}
.sidebar-title {font-size:30px;font-weight:800;margin-bottom:1rem}
.card {
    background:linear-gradient(135deg,#0b1220,#020617);
    border-radius:14px;
    padding:30px;
    border:1px solid #1e293b
}
.card-title {font-size:12px;color:#94a3b8}
.card-value {font-size:26px;font-weight:700}
.delta-up {color:#10b981;font-size:12px}
.delta-down {color:#ef4444;font-size:12px}
.insight {
    font-size:13px;
    color:#c7d2fe;
    margin-top:6px
}
.footer {
    text-align:center;
    font-size:12px;
    color:#94a3b8;
    margin-top:2rem;
    padding:1rem;
    border-top:1px solid #1e293b
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv("katalog_gempa.csv")

    rename = {}
    for c in df.columns:
        cl = c.lower()
        if "lat" in cl: rename[c] = "lat"
        elif "lon" in cl: rename[c] = "lon"
        elif "depth" in cl or "kedalaman" in cl: rename[c] = "depth"
        elif "mag" in cl: rename[c] = "mag"
        elif "tgl" in cl or "date" in cl or "time" in cl: rename[c] = "time"
        elif "remark" in cl or "wilayah" in cl: rename[c] = "region"

    df = df.rename(columns=rename)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["year"] = df["time"].dt.year

    return df.dropna(subset=["mag","lat","lon","time"])

df = load_data()

# ================= HEADER =================
st.markdown("""
<div style="
    display:flex;
    align-items:center;
    background: linear-gradient(180deg,#020617,#030712);
    padding:30px 10px;
    border-radius:0 0 12px 12px;
    border-bottom:1px solid #1e293b;
    margin-bottom:0px;
">
    <div style="font-size:22px; font-weight:700;">Earthquake Analytics Dashboard</div>
</div>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    # CSS untuk menghapus padding atas sidebar
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] {
        padding-top: 0.5rem !important;  /* bisa diubah 0rem kalau mau nempel */
    }
    .sidebar-title {font-size:20px;font-weight:700;margin-bottom:1rem}
    </style>
    """, unsafe_allow_html=True)

    # Title / Icon Sidebar
    st.markdown('<div class="sidebar-title">🌍</div>', unsafe_allow_html=True)

    # Menu
    page = st.selectbox(
        "Menu",
        [
            "Overview",
            "Time Trend",
            "Distribution",
            "Relationship",
            "Spatial",
            "ML Analytics",
            "About"
        ],
        label_visibility="collapsed"
    )
    st.divider()

    st.divider()

    years = sorted(df["year"].unique())
    selected_years = st.multiselect(
        "Filter Year",
        years,
        default=years[-5:],
        max_selections=5
    )

dff = df[df["year"].isin(selected_years)]

# ================= HELPER =================
def metric_card(title, value, delta=None):
    delta_html = ""
    if delta is not None:
        cls = "delta-up" if delta >= 0 else "delta-down"
        arrow = "↑" if delta >= 0 else "↓"
        delta_html = f'<div class="{cls}">{arrow} {abs(delta):.1f}%</div>'

    st.markdown(f"""
    <div class="card">
        <div class="card-title">{title}</div>
        <div class="card-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def trend_direction(series):
    if len(series) < 2:
        return "stabil"
    return "meningkat" if series.iloc[-1] > series.iloc[0] else "menurun"

# =================================================
# ================= OVERVIEW ======================
# =================================================
if page == "Overview":

    prev_years = [y for y in years if y < min(selected_years)]
    prev_dff = df[df["year"].isin(prev_years[-len(selected_years):])] if prev_years else dff

    total = len(dff)
    prev_total = len(prev_dff)
    delta_total = (total - prev_total) / prev_total * 100 if prev_total else 0

    max_mag = dff.mag.max()
    delta_mag = ((max_mag - prev_dff.mag.max()) / prev_dff.mag.max() * 100) if prev_dff.mag.max() else 0

    avg_depth = dff.depth.mean()
    delta_depth = ((avg_depth - prev_dff.depth.mean()) / prev_dff.depth.mean() * 100) if prev_dff.depth.mean() else 0

    high_risk = (dff.mag >= 5).sum()
    delta_risk = ((high_risk - (prev_dff.mag >= 5).sum()) / (prev_dff.mag >= 5).sum() * 100) if (prev_dff.mag >= 5).sum() else 0

    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Total Events", f"{total:,}", delta_total)
    with c2: metric_card("Max Magnitude", f"{max_mag:.2f}", delta_mag)
    with c3: metric_card("Avg Depth (km)", f"{avg_depth:.1f}", delta_depth)
    with c4: metric_card("High Risk ≥5", high_risk, delta_risk)

    # ===== Charts =====
    col1,col2,col3 = st.columns(3)

    yearly = dff.groupby("year").size().reset_index(name="count")

    with col1:
        fig = px.bar(yearly, x="year", y="count", title="Earthquake Events per Year")
        st.plotly_chart(fig, use_container_width=True)

        max_year = yearly.loc[yearly["count"].idxmax()]
        st.markdown(
            f'<div class="insight">Aktivitas tertinggi terjadi pada tahun <b>{int(max_year.year)}</b> '
            f'dengan <b>{int(max_year["count"])}</b> kejadian.</div>',
            unsafe_allow_html=True
        )

    with col2:
        fig = px.histogram(dff, x="mag", nbins=25, title="Magnitude Distribution")
        st.plotly_chart(fig, use_container_width=True)

        dom_mag = dff.mag.round().mode()[0]
        st.markdown(
            f'<div class="insight">Magnitudo paling dominan berada di sekitar <b>M {dom_mag}</b>, '
            'menunjukkan mayoritas gempa berkekuatan ringan–menengah.</div>',
            unsafe_allow_html=True
        )

    with col3:
        fig = px.pie(
            dff.assign(Risk=np.where(dff.mag>=5,"High","Low")),
            names="Risk",
            title="Risk Composition"
        )
        st.plotly_chart(fig, use_container_width=True)

        risk_pct = (dff.mag>=5).mean()*100
        st.markdown(
            f'<div class="insight">Sebesar <b>{risk_pct:.1f}%</b> gempa termasuk kategori berisiko tinggi.</div>',
            unsafe_allow_html=True
        )

    st.subheader("📋 Data Preview")
    st.dataframe(dff.head(200), use_container_width=True)


# =================================================
# ================= TIME TREND ====================
# =================================================
elif page == "Time Trend":
      # ================= ROW 1 =================
    col1, col2, col3 = st.columns(3)

    # 1️⃣ Yearly Trend
    yearly = dff.groupby("year").size().reset_index(name="count")
    peak_year = yearly.loc[yearly["count"].idxmax()]
    trend = "meningkat" if yearly["count"].iloc[-1] > yearly["count"].iloc[0] else "menurun"

    with col1:
        fig = px.bar(yearly, x="year", y="count", title="Yearly Earthquake Trend")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f'<div class="insight">Aktivitas gempa tertinggi terjadi pada '
            f'<b>{int(peak_year.year)}</b> dengan <b>{int(peak_year["count"])}</b> kejadian. '
            f'Tren tahunan cenderung <b>{trend}</b>.</div>',
            unsafe_allow_html=True
        )

    # 2️⃣ Monthly Pattern
    monthly = dff.groupby(dff.time.dt.month).size()
    peak_month = monthly.idxmax()

    with col2:
        fig = px.line(
            monthly.reset_index(name="count"),
            x="time",
            y="count",
            title="Monthly Earthquake Pattern",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f'<div class="insight">Pola musiman menunjukkan aktivitas gempa '
            f'paling sering terjadi pada <b>bulan ke-{peak_month}</b>.</div>',
            unsafe_allow_html=True
        )

    # 3️⃣ Risk Composition
    risk_pct = (dff.mag >= 5).mean() * 100

    with col3:
        fig = px.pie(
            dff.assign(Risk=np.where(dff.mag >= 5, "High Risk", "Low Risk")),
            names="Risk",
            title="Risk Composition"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f'<div class="insight">Sekitar <b>{risk_pct:.1f}%</b> gempa '
            f'termasuk kategori berisiko tinggi.</div>',
            unsafe_allow_html=True
        )

    # ================= ROW 2 =================
    col4, col5, col6 = st.columns(3)

    # 4️⃣ Magnitude Distribution
    dom_mag = dff.mag.round().mode()[0]

    with col4:
        fig = px.histogram(dff, x="mag", nbins=25, title="Magnitude Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f'<div class="insight">Mayoritas gempa berada di sekitar '
            f'<b>M {dom_mag}</b>, menunjukkan dominasi gempa ringan–menengah.</div>',
            unsafe_allow_html=True
        )

    # 5️⃣ Depth Distribution
    median_depth = dff.depth.median()

    with col5:
        fig = px.box(dff, y="depth", title="Depth Distribution")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f'<div class="insight">Median kedalaman gempa berada di '
            f'<b>{median_depth:.1f} km</b>, menandakan dominasi gempa dangkal.</div>',
            unsafe_allow_html=True
        )

    # 6️⃣ Mag vs Depth
    corr = dff.mag.corr(dff.depth)
    strength = "lemah" if abs(corr) < 0.3 else "sedang" if abs(corr) < 0.6 else "kuat"

    with col6:
        fig = px.scatter(
            dff, x="depth", y="mag",
            title="Magnitude vs Depth",
            trendline="ols"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f'<div class="insight">Hubungan magnitudo dan kedalaman bersifat '
            f'<b>{strength}</b> (r = {corr:.2f}), sehingga kedalaman bukan '
            f'faktor utama penentu kekuatan gempa.</div>',
            unsafe_allow_html=True
        )

    # ================= ROW 3 =================
    col7, col8 = st.columns([2,1])

    # 7️⃣ Spatial Density
    top_region = dff.region.value_counts().idxmax()

    with col7:
        fig = px.density_mapbox(
            dff, lat="lat", lon="lon", z="mag",
            radius=10, mapbox_style="carto-darkmatter",
            zoom=3, title="Spatial Density of Earthquakes"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            f'<div class="insight">Konsentrasi gempa tertinggi teridentifikasi '
            f'di wilayah <b>{top_region}</b>, yang berpotensi sebagai zona seismik aktif.</div>',
            unsafe_allow_html=True
        )

    # 8️⃣ Top Regions
    top5 = dff.region.value_counts().head(5).reset_index()
    top5.columns = ["region", "count"]

    with col8:
        fig = px.bar(
            top5, x="count", y="region",
            orientation="h",
            title="Top 5 Earthquake Regions"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div class="insight">Lima wilayah teratas menyumbang proporsi '
            'kejadian gempa paling signifikan dibandingkan wilayah lain.</div>',
            unsafe_allow_html=True
        )

    st.subheader("📋 Data Preview")
    st.dataframe(dff.head(200), use_container_width=True)


# =================================================
# ================= DISTRIBUTION ==================
# =================================================
elif page == "Distribution":
    st.subheader("📦 Statistical Distribution")

    c1,c2 = st.columns(2)

    with c1:
        fig = px.box(dff, y="mag", title="Magnitude Spread")
        st.plotly_chart(fig, use_container_width=True)

        q1,q3 = dff.mag.quantile([0.25,0.75])
        st.markdown(
            f'<div class="insight">Sebagian besar magnitudo berada di rentang <b>{q1:.1f}–{q3:.1f}</b>.</div>',
            unsafe_allow_html=True
        )

    with c2:
        fig = px.violin(dff, y="depth", title="Depth Distribution")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="insight">Distribusi kedalaman menunjukkan variasi besar dengan beberapa outlier ekstrem.</div>',
            unsafe_allow_html=True
        )

    treemap = dff.groupby("region").size().reset_index(name="count")
    fig = px.treemap(treemap, path=["region"], values="count", title="Regional Hierarchy")
    st.plotly_chart(fig, use_container_width=True)

# =================================================
# ================= RELATIONSHIP ==================
# =================================================
elif page == "Relationship":
    st.subheader("🔗 Variable Relationship")

    fig = px.scatter(dff, x="depth", y="mag", color="year")
    st.plotly_chart(fig, use_container_width=True)

    corr = dff.mag.corr(dff.depth)
    strength = "lemah" if abs(corr)<0.3 else "sedang" if abs(corr)<0.6 else "kuat"

    st.markdown(
        f'<div class="insight">Korelasi magnitudo dan kedalaman bersifat <b>{strength}</b> '
        f'(r = {corr:.2f}).</div>',
        unsafe_allow_html=True
    )

# =================================================
# ================= SPATIAL =======================
# =================================================
elif page == "Spatial":
    st.subheader("🗺️ Geospatial Analytics")

    fig = px.density_mapbox(
        dff, lat="lat", lon="lon", z="mag",
        radius=10, mapbox_style="carto-darkmatter", zoom=3
    )
    st.plotly_chart(fig, use_container_width=True)

    top_region = dff.region.value_counts().idxmax()
    st.markdown(
        f'<div class="insight">Konsentrasi gempa tertinggi berada di wilayah <b>{top_region}</b>.</div>',
        unsafe_allow_html=True
    )

# =================================================
# ================= ML ============================
# =================================================
elif page == "ML Analytics":
    st.subheader("🧠 Machine Learning Insight")

    k = st.slider("Number of Clusters",2,6,3)
    coords = dff[["lat","lon"]].copy()
    km = KMeans(n_clusters=k, random_state=42)
    coords["cluster"] = km.fit_predict(coords)

    fig = px.scatter(coords, x="lon", y="lat", color="cluster")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        f'<div class="insight">Model KMeans mengelompokkan gempa menjadi <b>{k}</b> zona seismik utama '
        'berdasarkan pola geografis.</div>',
        unsafe_allow_html=True
    )
# ================= About =================
elif page == "About":
        st.markdown("""
        **Seismic Analytics Dashboard**  
        Version 1.0  

        Dashboard ini dirancang untuk menganalisis dan memvisualisasikan
        data gempa bumi berdasarkan waktu kejadian, magnitudo, kedalaman,
        lokasi geografis, serta tingkat risiko secara interaktif.

        **📊 Data Source**
        KAGGLE
        Katalog Gempa Bumi (BMKG / USGS)  
        Periode Data: 2008 – 2023  

        **🧠 Methodology**  
        • Descriptive & Exploratory Data Analysis  
        • Data Visualization (Plotly)  
        • Risk Classification (Magnitude-based)  
        • K-Means Clustering (Spatial Analysis)  

        **⚠️ Risk Level Definition**  
        • Low Risk  : Magnitude < 5.0  
        • High Risk : Magnitude ≥ 5.0  

        **🛠 Built With**  
        Python · Streamlit · Pandas · NumPy  
        Plotly · Scikit-learn  

        **🎓 Academic Purpose**  
        Dashboard ini dikembangkan sebagai media analisis data
        dan visualisasi untuk memenuhi projek Ujian Akhir Semester.
        """)
        
# ================= FOOTER =================
st.markdown(
    '<div class="footer">© 2025 Earthquake Analytics Dashboard — Data Visualization & Analytics</div>',
    unsafe_allow_html=True
)
