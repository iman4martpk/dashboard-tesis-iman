import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# =========================================================================
# 🌊 1. KONFIGURASI HALAMAN & CSS INSTRUMEN (ULTRA-SLIM RESPONSIVE)
# =========================================================================
st.set_page_config(
    page_title="Dashboard Pasut Hibrida Pasar Ikan",
    layout="wide",
    page_icon="🌊",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    [data-baseweb="popover"] {
        transform: scale(0.95) !important;
        transform-origin: top left !important;
    }
    [data-baseweb="popover"] > div {
        max-width: 260px !important;
    }
    .block-container { 
        padding-top: 3.2rem !important; 
        padding-bottom: 0rem !important; 
        max-width: 95% !important; 
    }
    [data-testid="stVerticalBlock"] > div {
        gap: 0px !important;
    }
    .stApp { background-color: #ffffff; }
    .header-text { 
        text-align: center; 
        width: 100%; 
        margin-top: 5px; 
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    html, body, [class*="css"] {
        font-family: Arial, Helvetica, sans-serif !important;
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
    }
    [data-testid="stHeader"] {
        background-color: transparent !important;
        background: transparent !important;
        z-index: 99995 !important;
    }
    div[data-testid="collapsedControl"] {
        background-color: #0B3D4C !important;
        border-radius: 8px !important;
        padding: 6px !important;
        box-shadow: 0 4px 12px rgba(11, 61, 76, 0.3) !important;
    }
    button[data-testid="stSidebarCollapseButton"] {
        background-color: #0B3D4C !important;
        border-radius: 8px !important;
    }
    div[data-testid="collapsedControl"] svg,
    button[data-testid="stSidebarCollapseButton"] svg {
        fill: #F8FAFC !important;
        color: #F8FAFC !important;
    }
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
        font-size: 0.68rem !important; 
        margin-bottom: -4px !important; 
        white-space: nowrap !important;
    }
    [data-testid="stMetricValue"] { 
        font-size: 14px !important; 
        font-weight: 700 !important; 
        color: #0f172a !important; 
        white-space: nowrap !important;
    }
    div[data-testid="stMetricDelta"] { display: none !important; }
    div[data-testid="column"] { padding: 0 4px !important; }
    .summary-box {
        background-color: #f1f5f9 !important; 
        padding: 6px 12px !important; 
        border-radius: 8px !important; 
        margin-top: 4px !important;
        margin-bottom: 8px !important; 
        border-left: 5px solid #0B3D4C !important; 
        text-align: center !important;
    }
    .summary-text { font-family: Arial, Helvetica, sans-serif !important; font-weight: 600; font-size: 0.82rem; color: #1e293b; }
    @media (max-width: 767px) {
        .block-container { padding-top: 3.4rem !important; }
        .header-text h2 { font-size: 1.1rem !important; margin-top: 10px !important; }
        .summary-text { font-size: 0.72rem !important; }
        [data-testid="stMetricValue"] { font-size: 12px !important; }
    }
    </style>
""", unsafe_allow_html=True)

# =========================================================================
# 📥 2. DATA PIPELINE (LOAD 2 FILE TERPISAH TANPA MERGE)
# =========================================================================
KOLOM_LSTM_MURNI = 'Prediksi_LSTM_Murni' 

@st.cache_data
def load_data():
    df_hibrida = pd.read_csv("HASIL_FINAL_TESIS_PASUT_HIBRIDA.csv", parse_dates=['Datetime'])
    df_lstm = pd.read_csv("HASIL_FINAL_TESIS_PASUT_LSTM_MURNI.csv", parse_dates=['Datetime'])
    return df_hibrida, df_lstm

try:
    df, df_lstm = load_data()
except Exception as e:
    st.error(f"❌ File CSV gagal dimuat. Pastikan nama file sudah benar. Error: {e}")
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
    mask = (df['Datetime'].dt.date >= start_date) & (df['Datetime'].dt.date <= end_date)
else:
    tgl_start = pd.to_datetime(PRESETS[pilihan_mode]['start'])
    tgl_end = pd.to_datetime(PRESETS[pilihan_mode]['end'])
    mask = (df['Datetime'] >= tgl_start) & (df['Datetime'] <= tgl_end)

df_filtered = df[mask].copy()
df_lstm_filtered = df_lstm[mask].copy()

# =========================================================================
# 📊 4. KOMPUTASI KPI SCORECARDS & METADATA BAR DENGAN LOGIKA DINAMIS
# =========================================================================
st.markdown("""
    <div class="header-text">
        <h2 style="margin: 0; color: #0F172A; font-family: Arial, Helvetica, sans-serif; font-weight: bold; font-size: 1.55rem;">
            🌊 MONITORING PASUT HIBRIDA (UTIDE + LSTM) REAL-TIME
        </h2>
    </div>
""", unsafe_allow_html=True)

if df_filtered['TMA_Pasar_Ikan'].notna().sum() > 0:
    valid_idx = df_filtered['TMA_Pasar_Ikan'].notna()
    
    # Menghitung metrik RMSE masing-masing metode
    rmse_utide_curr = np.sqrt(np.mean((df_filtered.loc[valid_idx, 'TMA_Pasar_Ikan'] - df_filtered.loc[valid_idx, 'Prediksi_Harmonik_UTIDE']) ** 2))
    rmse_lstm_curr = np.sqrt(np.mean((df_filtered.loc[valid_idx, 'TMA_Pasar_Ikan'] - df_lstm_filtered.loc[valid_idx, KOLOM_LSTM_MURNI]) ** 2))
    rmse_hib_curr = np.sqrt(np.mean((df_filtered.loc[valid_idx, 'TMA_Pasar_Ikan'] - df_filtered.loc[valid_idx, 'Prediksi_Hibrida_Final']) ** 2))
    
    peningkatan_curr = ((rmse_utide_curr - rmse_hib_curr) / rmse_utide_curr) * 100 if rmse_utide_curr > 0 else 0
    
    # ---------------- LOGIKA DINAMIS AKURASI TERTINGGI ----------------
    min_rmse = min(rmse_utide_curr, rmse_lstm_curr, rmse_hib_curr)
    badge_best = '<span style="color: #22c55e; font-size: 0.65rem; font-weight: bold;">🏆 AKURASI TERTINGGI</span>'
    
    if rmse_utide_curr == min_rmse:
        badge_utide = badge_best
    else:
        badge_utide = f'<span style="color: #ef4444; font-size: 0.65rem; font-weight: bold;">+{(rmse_utide_curr - min_rmse):.1f} cm vs Terbaik</span>'
        
    if rmse_lstm_curr == min_rmse:
        badge_lstm = badge_best
    else:
        badge_lstm = f'<span style="color: #ef4444; font-size: 0.65rem; font-weight: bold;">+{(rmse_lstm_curr - min_rmse):.1f} cm vs Terbaik</span>'
        
    if rmse_hib_curr == min_rmse:
        badge_hib = badge_best
    else:
        badge_hib = f'<span style="color: #ef4444; font-size: 0.65rem; font-weight: bold;">+{(rmse_hib_curr - min_rmse):.1f} cm vs Terbaik</span>'
    # ------------------------------------------------------------------

    st.markdown(f"""
        <div class="summary-box">
            <span class="summary-text">
                📍 <b>Stasiun:</b> Pasar Ikan, Jakarta | 🛡️ <b>Fokus:</b> Koreksi Residu Non-Astronomis | 🔎 <b>Studi:</b> {PRESETS[pilihan_mode]['desc']}
            </span>
        </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3, m4 = st.columns(4)
    
    m1.markdown(f"""
        <div data-testid="stMetric" style="border-left-color: #22c55e !important;">
            <label data-testid="stMetricLabel">📈 REDUKSI EROR (vs UTide)</label>
            <div data-testid="stMetricValue">
                {peningkatan_curr:.2f} % <span style="color: #22c55e; font-size: 0.68rem; font-weight: bold;">▲ OPTIMAL</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    m2.markdown(f"""
        <div data-testid="stMetric" style="border-left-color: #06B6D4 !important;">
            <label data-testid="stMetricLabel">📉 RMSE UTIDE MURNI</label>
            <div data-testid="stMetricValue">
                {rmse_utide_curr:.2f} cm {badge_utide}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    m3.markdown(f"""
        <div data-testid="stMetric" style="border-left-color: #F59E0B !important;">
            <label data-testid="stMetricLabel">📊 RMSE LSTM MURNI</label>
            <div data-testid="stMetricValue">
                {rmse_lstm_curr:.2f} cm {badge_lstm}
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    m4.markdown(f"""
        <div data-testid="stMetric" style="border-left-color: #4F46E5 !important;">
            <label data-testid="stMetricLabel">🏆 RMSE HIBRIDA</label>
            <div data-testid="stMetricValue">
                {rmse_hib_curr:.2f} cm {badge_hib}
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

if df_filtered['TMA_Pasar_Ikan'].notna().sum() > 0:
    fig.add_trace(go.Scatter(
        x=df_filtered['Datetime'], y=df_filtered['TMA_Pasar_Ikan'],
        mode='lines', name='Observasi Stasiun (TMA Aktual)',
        line=dict(color='#64748B', width=2.5)
    ))

fig.add_trace(go.Scatter(
    x=df_filtered['Datetime'], y=df_filtered['Prediksi_Harmonik_UTIDE'],
    mode='lines', name='Prediksi UTide Murni (Astronomis)',
    line=dict(color='#06B6D4', width=2.0, dash='dot')
))

fig.add_trace(go.Scatter(
    x=df_lstm_filtered['Datetime'], y=df_lstm_filtered[KOLOM_LSTM_MURNI],
    mode='lines', name='Prediksi LSTM Murni (Non-Astronomis)',
    line=dict(color='#F59E0B', width=2.0, dash='dashdot')
))

fig.add_trace(go.Scatter(
    x=df_filtered['Datetime'], y=df_filtered['Prediksi_Hibrida_Final'],
    mode='lines', name='Prediksi Hibrida (UTide + LSTM)',
    line=dict(color='#4F46E5', width=3.2, dash='dash')
))

fig.add_hline(y=250, line_dash="dash", line_color="#DC2626", line_width=1.5)
fig.add_hline(y=230, line_dash="dash", line_color="#D97706", line_width=1.5)

fig.add_annotation(
    xref="paper", yref="y", x=0.005, y=249,
    text="<b>🚨 AWAS ROB (250 cm)</b>", showarrow=False,
    xanchor="left", yanchor="top", font=dict(color='#DC2626', size=11, family="Arial")
)
fig.add_annotation(
    xref="paper", yref="y", x=0.005, y=229,
    text="<b>⚠️ WASPADA ROB (230 cm)</b>", showarrow=False,
    xanchor="left", yanchor="top", font=dict(color='#D97706', size=11, family="Arial")
)

fig.update_layout(
    height=410, 
    template="plotly_white", 
    margin=dict(l=10, r=10, t=25, b=10), 
    hovermode="x unified",
    hoverlabel=dict(bgcolor="white", font_size=11, font_family="Arial"),
    xaxis=dict(tickfont=dict(size=10, family="Arial")),
    yaxis=dict(
        title=dict(text="Tinggi Air (cm)", font=dict(size=11, family="Arial")), 
        tickfont=dict(size=10, family="Arial")
    ),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, family="Arial")
    ),
    font=dict(family="Arial, Helvetica, sans-serif", color="#1E293B")
)

st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# =========================================================================
# 📋 6. INTEGRASI DATA TABULAR & DOWNLOAD CONTROLLER
# =========================================================================
st.divider()
st.markdown("<h4 style='margin:0 0 4px 0; padding:0; font-size:14px; font-weight:700; color:#1E293B;'>📋 Potongan Basis Data Numerik Terfilter</h4>", unsafe_allow_html=True)

df_tampilan = pd.DataFrame({
    'Datetime': df_filtered['Datetime'],
    'TMA_Pasar_Ikan': df_filtered['TMA_Pasar_Ikan'],
    'Prediksi_Harmonik_UTIDE': df_filtered['Prediksi_Harmonik_UTIDE'],
    KOLOM_LSTM_MURNI: df_lstm_filtered[KOLOM_LSTM_MURNI].values,
    'Prediksi_Hibrida_Final': df_filtered['Prediksi_Hibrida_Final']
})

st.dataframe(df_tampilan.reset_index(drop=True), use_container_width=True)

csv_data = df_tampilan.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Unduh Data Potongan Kerja Ini (.CSV)",
    data=csv_data,
    file_name=f"DATA_INSPEKSI_PASUT_HIBRIDA.csv",
    mime="text/csv",
    use_container_width=True
)
