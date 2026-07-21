import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# =========================================================================
# 🌊 1. KONFIGURASI HALAMAN & CSS INSTRUMEN (ULTRA-SLIM RESPONSIVE)
# =========================================================================
st.set_page_config(page_title="Dashboard Pasut Hibrida Pasar Ikan", layout="wide", page_icon="waves")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

    /* --- FIX PEMBUNGKUS KALENDER FILTER SIDEBAR --- */
    [data-baseweb="popover"] {
        transform: scale(0.95) !important;
        transform-origin: top left !important;
    }
    [data-baseweb="popover"] > div {
        max-width: 260px !important;
    }

    /* Merapatkan kontainer utama ke batas paling atas layar secara aman */
    .block-container { 
        padding-top: 0.5rem !important; 
        padding-bottom: 0rem !important; 
        max-width: 95% !important; 
    }
    
    /* Menghilangkan gap vertikal bawaan Streamlit antar elemen */
    [data-testid="stVerticalBlock"] > div {
        gap: 0px !important;
    }

    .stApp { background-color: #ffffff; }
    
    /* Menyelaraskan teks judul utama di tengah */
    .header-text { 
        text-align: center; 
        width: 100%; 
        margin-top: 5px; 
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }

    /* ================= 🔥 FIX ABSOLUT: NORMAL HEIGHT TRANSPARENT HEADER 🔥 ================= */
    /* Mengembalikan tinggi normal header agar tombol tidak terpotong sistem overflow browser */
    [data-testid="stHeader"] {
        background-color: transparent !important;
        background: transparent !important;
        z-index: 99995 !important;
    }

    /* Mengubah tombol BUKA sidebar (hamburger ☰) menjadi widget navy solid yang stand-out */
    div[data-testid="collapsedControl"] {
        background-color: #0B3D4C !important; /* Navy instrumen pasut */
        border-radius: 8px !important;
        padding: 6px !important;
        box-shadow: 0 4px 12px rgba(11, 61, 76, 0.3) !important;
    }
    
    /* Menyelaraskan tombol TUTUP sidebar agar serupa */
    button[data-testid="stSidebarCollapseButton"] {
        background-color: #0B3D4C !important;
        border-radius: 8px !important;
    }

    /* Memastikan semua icon panah & hamburger berwarna putih kontras */
    div[data-testid="collapsedControl"] svg,
    button[data-testid="stSidebarCollapseButton"] svg {
        fill: #F8FAFC !important;
        color: #F8FAFC !important;
    }

    /* Sembunyikan hanya aksesori bawaan yang mengganggu (menu titik tiga, dll) */
    [data-testid="stToolbar"] {
        display: none !important;
    }

    /* GAYA METRIK KUSTOM (ULTRA SLIM READING PANEL) */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important; 
        border: 1px solid #e2e8f0 !important;
        border-left: 4px solid #0B3D4C !important; 
        padding: 4px 10px !important; 
        border-radius: 8px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
        min-height: 55px !important; 
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }
    
    div[data-testid="stMetricLabel"] { 
        color: #64748b !important; 
        font-weight: 600 !important; 
        font-size: 0.72rem !important; 
        margin-bottom: -4px !important; 
        white-space: nowrap !important;
    }

    [data-testid="stMetricValue"] { 
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 15px !important; 
        font-weight: 700 !important; 
        color: #0f172a !important; 
        white-space: nowrap !important;
    }

    div[data-testid="stMetricDelta"] { display: none !important; }
    div[data-testid="column"] { padding: 0 4px !important; }

    /* Summary Box Terintegrasi Melekat ke Header */
    .summary-box {
        background-color: #f1f5f9 !important; 
        padding: 6px 12px !important; 
        border-radius: 8px !important; 
        margin-top: 4px !important;
        margin-bottom: 8px !important; 
        border-left: 5px solid #0B3D4C !important; 
        text-align: center !important;
    }
    .summary-text { font-family: 'Inter', sans-serif; font-weight: 600; font-size: 0.82rem; color: #1e293b; }
    
    /* Responsive adjustment untuk mobile smartphone Android / iOS */
    @media (max-width: 767px) {
        .block-container { padding-top: 0.5rem !important; }
        .header-text h2 { font-size: 1.1rem !important; margin-top: 10px !important; }
        .summary-text { font-size: 0.72rem !important; }
        [data-testid="stMetricValue"] { font-size: 13px !important; }
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================================
# 📥 2. DATA PIPELINE (LOAD DATABASE MASTER)
# =========================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("HASIL_FINAL_TESIS_PASUT_HIBRIDA.csv", parse_dates=['Datetime'])
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Database utama gagal dimuat. Error: {e}")
    st.stop()

# =========================================================================
# ⚙️ 3. PANEL KONTROL SIDEBAR (PRESET ANATOMI STUDI KASUS)
# =========================================================================
st.sidebar.header("⚡ Kontrol Panel Analisis")

PRESETS = {
    "Studi Kasus 1: Periode Mei": {
        "start": "2026-05-14 00:00:00",
        "end": "2026-05-21 00:00:00",
        "desc": "Segmen awal bulan Mei."
    },
    "Studi Kasus 2: Periode Juni": {
        "start": "2026-06-12 00:00:00",
        "end": "2026-06-19 00:00:00",
        "desc": "Fase anomali residu meteorologis tinggi."
    },
    "Studi Kasus 3: Periode Juli": {
        "start": "2026-07-12 00:00:00",
        "end": "2026-07-19 00:00:00",
        "desc": "Batas data aktual observasi lapangan."
    },
    "🔮 MODE FORECASTING MASA DEPAN": {
        "start": "2026-07-21 19:00:00",
        "end": "2026-12-31 23:00:00",
        "desc": "Peramalan estafet bergulir tanpa data observasi riil."
    },
    "🎛️ Custom Rentang Waktu (Manual)": {
        "start": None,
        "end": None,
        "desc": "Bebas menentukan rentang analisis tanggal sendiri."
    }
}

pilihan_mode = st.sidebar.selectbox("Pilih Mode Analisis / Studi Kasus:", list(PRESETS.keys()), index=1)

if pilihan_mode == "🎛️ Custom Rentang Waktu (Manual)":
    min_date = df['Datetime'].min().date()
    max_date = df['Datetime'].max().date()
    start_date = st.sidebar.date_input("Tanggal Mulai", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("Tanggal Selesai", max_date, min_value=min_date, max_value=max_date)
    df_filtered = df[(df['Datetime'].dt.date >= start_date) & (df['Datetime'].dt.date <= end_date)].copy()
else:
    tgl_start = pd.to_datetime(PRESETS[pilihan_mode]['start'])
    tgl_end = pd.to_datetime(PRESETS[pilihan_mode]['end'])
    df_filtered = df[(df['Datetime'] >= tgl_start) & (df['Datetime'] <= tgl_end)].copy()

# =========================================================================
# 📊 4. KOMPUTASI KPI SCORECARDS & METADATA BAR
# =========================================================================
df_eval = df_filtered[df_filtered['TMA_Pasar_Ikan'].notna() & df_filtered['Prediksi_Hibrida_Final'].notna()]

# Main Header Title
st.markdown("""
    <div class="header-text">
        <h2 style="margin: 0; color: #0F172A; font-family: 'Space Grotesk', sans-serif; font-weight: bold; font-size: 1.55rem;">
            🌊 MONITORING PASUT HIBRIDA (UTIDE + LSTM) REAL-TIME
        </h2>
    </div>
""", unsafe_allow_html=True)

# Perhitungan nilai statistik validasi model
if len(df_eval) > 0:
    rmse_utide_curr = np.sqrt(np.mean((df_eval['TMA_Pasar_Ikan'] - df_eval['Prediksi_Harmonik_UTIDE']) ** 2))
    rmse_hib_curr = np.sqrt(np.mean((df_eval['TMA_Pasar_Ikan'] - df_eval['Prediksi_Hibrida_Final']) ** 2))
    peningkatan_curr = ((rmse_utide_curr - rmse_hib_curr) / rmse_utide_curr) * 100 if rmse_utide_curr > 0 else 0
    selisih_nominal = rmse_hib_curr - rmse_utide_curr
    
    # Render Informasi ke Panel Summary Box Melekat
    st.markdown(f"""
        <div class="summary-box">
            <span class="summary-text">
                📍 <b>Stasiun:</b> Pasar Ikan, Jakarta | 🛡️ <b>Fokus:</b> Koreksi Residu Non-Astronomis | 🔎 <b>Studi:</b> {PRESETS[pilihan_mode]['desc']}
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    
    m1.markdown(f"""
        <div data-testid="stMetric" style="border-left-color: #22c55e !important;">
            <label data-testid="stMetricLabel">📈 REDUKSI EROR (AKURASI)</label>
            <div data-testid="stMetricValue">
                {peningkatan_curr:.2f} % <span style="color: #22c55e; font-size: 0.72rem; font-weight: bold; font-family: 'Inter';">▲ OPTIMAL</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    m2.markdown(f"""
        <div data-testid="stMetric" style="border-left-color: #06B6D4 !important;">
            <label data-testid="stMetricLabel">📉 RMSE HARMONIK UTIDE MURNI</label>
            <div data-testid="stMetricValue">
                {rmse_utide_curr:.2f} cm <span style="color: #ef4444; font-size: 0.72rem; font-weight: bold; font-family: 'Inter';">+{abs(selisih_nominal):.1f} cm vs Hibrida</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    m3.markdown(f"""
        <div data-testid="stMetric" style="border-left-color: #4F46E5 !important;">
            <label data-testid="stMetricLabel">🏆 RMSE KOMPOSIT HIBRIDA (LSTM)</label>
            <div data-testid="stMetricValue">
                {rmse_hib_curr:.2f} cm <span style="color: #22c55e; font-size: 0.72rem; font-weight: bold; font-family: 'Inter';">-{abs(selisih_nominal):.1f} cm vs UTide</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
        <div class="summary-box" style="border-left-color: #d97706 !important;">
            <span class="summary-text" style="color: #b45309;">
                🔮 <b>MODE FORECASTING MASA DEPAN</b> | Grafik menampilkan kurva proyeksi. Metrik RMSE tidak dihitung karena data lapangan belum rilis.
            </span>
        </div>
    """, unsafe_allow_html=True)

# =========================================================================
# 📈 5. VISUALISASI GRAFIK TIMESERIES PLOTLY INTERAKTIF
# =========================================================================
st.markdown(f"<h3 style='margin:5px 0 3px 0; padding:0; font-size:19px; font-weight:600; color:#1E293B;'>📈 Grafik Analisis Perbandingan: {pilihan_mode}</h3>", unsafe_allow_html=True)

fig = go.Figure()

# 1. Data Observasi Lapangan: Solid Slate Grey (Garis utuh tanpa putus)
if df_filtered['TMA_Pasar_Ikan'].notna().sum() > 0:
    fig.add_trace(go.Scatter(
        x=df_filtered['Datetime'], y=df_filtered['TMA_Pasar_Ikan'],
        mode='lines', name='Observasi Stasiun (TMA Aktual)',
        line=dict(color='#64748B', width=2.5)
    ))

# 2. Prediksi UTide: Dotted (Titik rapat) Luminous Cyan Cerah
fig.add_trace(go.Scatter(
    x=df_filtered['Datetime'], y=df_filtered['Prediksi_Harmonik_UTIDE'],
    mode='lines', name='Prediksi UTide Murni (Astronomis)',
    line=dict(color='#06B6D4', width=2.0, dash='dot')
))

# 3. Prediksi Hibrida: Dashed (Putus-putus renggang) Premium Indigo Tebal
fig.add_trace(go.Scatter(
    x=df_filtered['Datetime'], y=df_filtered['Prediksi_Hibrida_Final'],
    mode='lines', name='Prediksi Hibrida (UTide + LSTM)',
    line=dict(color='#4F46E5', width=3.2, dash='dash')
))

# 4. Garis Batas Peringatan Rob Kompak
fig.add_hline(y=250, line_dash="dash", line_color="#DC2626", line_width=1.5)
fig.add_hline(y=230, line_dash="dash", line_color="#D97706", line_width=1.5)

# 5. Label Threshold Rob Presisi (Ditempel Tepat 1 Satuan Di Bawah Garis Batas)
fig.add_annotation(
    xref="paper", yref="y", x=0.005, y=249,
    text="<b>🚨 AWAS ROB (250 cm)</b>", showarrow=False,
    xanchor="left", yanchor="top", font=dict(color='#DC2626', size=11)
)

fig.add_annotation(
    xref="paper", yref="y", x=0.005, y=229,
    text="<b>⚠️ WASPADA ROB (230 cm)</b>", showarrow=False,
    xanchor="left", yanchor="top", font=dict(color='#D97706', size=11)
)

# --- KONFIGURASI LAYOUT RESPONSIF PLOTLY ---
fig.update_layout(
    height=410, 
    template="plotly_white", 
    margin=dict(l=10, r=10, t=25, b=10), 
    hovermode="x unified",
    hoverlabel=dict(bgcolor="white", font_size=11),
    xaxis=dict(tickfont=dict(size=10)),
    yaxis=dict(
        title=dict(text="Tinggi Air (cm)", font=dict(size=11)), 
        tickfont=dict(size=10)
    ),
    legend=dict(
        orientation="h", 
        yanchor="bottom", 
        y=1.02, 
        xanchor="right", 
        x=1, 
        font=dict(size=11)
    ),
    font=dict(family="Inter, sans-serif", color="#1E293B")
)

# Render Grafik Tanpa Modebar
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# =========================================================================
# 📋 6. INTEGRASI DATA TABULAR & DOWNLOAD CONTROLLER (Di Bawah Halaman)
# =========================================================================
st.divider()
st.markdown("<h4 style='margin:0 0 4px 0; padding:0; font-size:14px; font-weight:700; color:#1E293B;'>📋 Potongan Basis Data Numerik Terfilter</h4>", unsafe_allow_html=True)

kolom_tampilan = ['Datetime', 'TMA_Pasar_Ikan', 'Prediksi_Harmonik_UTIDE', 'Prediksi_Hibrida_Final']
st.dataframe(df_filtered[kolom_tampilan].reset_index(drop=True), use_container_width=True)

csv_data = df_filtered[kolom_tampilan].to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Unduh Data Potongan Kerja Ini (.CSV)",
    data=csv_data,
    file_name=f"DATA_INSPEKSI_PASUT_HIBRIDA.csv",
    mime="text/csv",
    use_container_width=True
)
