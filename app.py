import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================================================================
# 🌊 1. KONFIGURASI HALAMAN & THEME (RESPONSIVE MOBILE-FRIENDLY LAYOUT)
# =========================================================================
st.set_page_config(
    page_title="Dashboard Pasut Hibrida Pasar Ikan", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Suntik CSS kustom responsif: Menjaga kerapatan desktop & menyelamatkan tombol sidebar di HP
st.markdown("""
    <style>
        /* 💻 OPTIMASI UNTUK LAYAR DESKTOP / LAPTOP */
        @media (min-width: 768px) {
            [data-testid="stHeader"] {
                display: none !important;
            }
            .main .block-container {
                padding-top: 0.8rem !important; 
                padding-bottom: 0rem !important;
                padding-left: 2.5rem !important;
                padding-right: 2.5rem !important;
            }
        }
        
        /* 📱 OPTIMASI KHUSUS LAYAR HP (ANDROID / IOS) */
        @media (max-width: 767px) {
            /* Jangan sembunyikan header di HP agar tombol pemicu sidebar (☰) tetap bisa diakses */
            [data-testid="stHeader"] {
                background-color: transparent !important;
                color: #1E293B !important;
            }
            .main .block-container {
                padding-top: 3.5rem !important; /* Memberi ruang di atas agar tidak tertutup tombol sidebar mobile */
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            /* Mengecilkan font judul secara dinamis di HP agar pas satu layar */
            .responsive-title {
                font-size: 22px !important;
                line-height: 1.3 !important;
            }
            .responsive-subtitle {
                font-size: 12px !important;
                margin-top: 4px !important;
            }
        }
        
        /* Pengaturan global antar blok komponen */
        .stVerticalBlock {
            gap: 0.8rem !important;
        }
        div[data-testid="stMetric"] {
            padding: 5px 10px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Hero Header dengan Class Responsif CSS
st.markdown("""
    <div style='text-align: center; margin-bottom: 5px;'>
        <h1 class='responsive-title' style='margin: 0; padding: 0; font-size: 34px; font-weight: 800; color: #1E293B; letter-spacing: -0.5px;'>
            🌊 Dashboard Operasional Pasut Hibrida (UTide + LSTM)
        </h1>
        <p class='responsive-subtitle' style='margin: 6px 0 0 0; font-size: 14.5px; color: #64748B;'>
            <b>Stasiun Pemantauan:</b> Pasar Ikan, Jakarta &nbsp;|&nbsp; 
            <b>Fokus Riset:</b> Koreksi Residu Hidro-Oseanografi Non-Astronomis
        </p>
    </div>
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
    st.error(f"❌ Gagal memuat file database utama. Pastikan file 'HASIL_FINAL_TESIS_PASUT_HIBRIDA.csv' sudah di-upload ke GitHub. Error: {e}")
    st.stop()

# =========================================================================
# ⚙️ 3. PANEL KONTROL SIDEBAR (PRESET STUDI KASUS & CUSTOM FILTER)
# =========================================================================
st.sidebar.header("⚡ Kontrol Panel Analisis")

PRESETS = {
    "Studi Kasus 1: Periode Mei": {
        "start": "2026-05-14 00:00:00",
        "end": "2026-05-21 00:00:00",
        "desc": "Evaluasi performa model hibrida pada segmen awal bulan Mei."
    },
    "Studi Kasus 2: Periode Juni": {
        "start": "2026-06-12 00:00:00",
        "end": "2026-06-19 00:00:00",
        "desc": "Evaluasi pada fase anomali residu meteorologis tinggi."
    },
    "Studi Kasus 3: Periode Juli": {
        "start": "2026-07-12 00:00:00",
        "end": "2026-07-19 00:00:00",
        "desc": "Batas data historis aktual sebelum peramalan masa depan."
    },
    "🔮 MODE FORECASTING MASA DEPAN (Agustus - Desember 2026)": {
        "start": "2026-07-21 19:00:00",
        "end": "2026-12-31 23:00:00",
        "desc": "Sistem peramalan estafet bergulir tanpa data observasi lapangan."
    },
    "🎛️ Custom Rentang Waktu (Manual)": {
        "start": None,
        "end": None,
        "desc": "Bebas menentukan rentang tanggal analisis sendiri."
    }
}

pilihan_mode = st.sidebar.selectbox(
    "Pilih Mode Analisis / Studi Kasus:",
    list(PRESETS.keys()),
    index=1  # Default langsung menyorot ke Juni
)

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
# 📊 4. KOMPUTASI METRIK VALIDASI DINAMIS (KPI SCORECARDS)
# =========================================================================
st.markdown("<h3 style='margin:5px 0 3px 0; padding:0; font-size:19px; font-weight:600; color:#1E293B;'>📊 Performa Validasi Real-Time pada Rentang Terpilih</h3>", unsafe_allow_html=True)

df_eval = df_filtered[df_filtered['TMA_Pasar_Ikan'].notna() & df_filtered['Prediksi_Hibrida_Final'].notna()]

if len(df_eval) > 0:
    rmse_utide_curr = np.sqrt(np.mean((df_eval['TMA_Pasar_Ikan'] - df_eval['Prediksi_Harmonik_UTIDE']) ** 2))
    rmse_hib_curr = np.sqrt(np.mean((df_eval['TMA_Pasar_Ikan'] - df_eval['Prediksi_Hibrida_Final']) ** 2))
    
    if rmse_utide_curr > 0:
        peningkatan_curr = ((rmse_utide_curr - rmse_hib_curr) / rmse_utide_curr) * 100
    else:
        peningkatan_curr = 0
        
    selisih_nominal = rmse_hib_curr - rmse_utide_curr
    
    st.sidebar.info(f"ℹ️ **Deskripsi:**\n{PRESETS[pilihan_mode]['desc']}\n\n📈 **Akurasi Terdeteksi:** +{peningkatan_curr:.2f}%")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="📈 Reduksi Eror (Peningkatan Akurasi)", 
            value=f"{peningkatan_curr:.2f} %", 
            delta="LSTM Efektif Mengoreksi Residu",
            delta_color="normal"
        )
    with col2:
        st.metric(
            label="📉 RMSE Harmonik UTide Murni", 
            value=f"{rmse_utide_curr:.2f} cm", 
            delta=f"+{-selisih_nominal:.2f} cm vs Hibrida", 
            delta_color="normal"
        )
    with col3:
        st.metric(
            label="🏆 RMSE Komposit Hibrida (LSTM)", 
            value=f"{rmse_hib_curr:.2f} cm", 
            delta=f"{selisih_nominal:.2f} cm vs UTide", 
            delta_color="inverse"
        )
else:
    st.sidebar.info(f"ℹ️ **Deskripsi:**\n{PRESETS[pilihan_mode]['desc']}")
    st.warning("🔮 **Status:** Menampilkan Area Peramalan Masa Depan. Metrik akurasi tidak dihitung karena data observasi riil lapangan belum terjadi (Masa Depan).")

# =========================================================================
# 📈 5. GRAFIK INTERAKTIF PLOTLY (TIME-SERIES VISUALIZATION - NEW STYLING)
# =========================================================================
st.markdown(f"<h3 style='margin:5px 0 3px 0; padding:0; font-size:19px; font-weight:600; color:#1E293B;'>📈 Grafik Analisis Perbandingan: {pilihan_mode}</h3>", unsafe_allow_html=True)

fig = go.Figure()

# 1. Observasi Stasiun (TMA Aktual) - Solid Abu-Abu/Slate
if df_filtered['TMA_Pasar_Ikan'].notna().sum() > 0:
    fig.add_trace(go.Scatter(
        x=df_filtered['Datetime'], y=df_filtered['TMA_Pasar_Ikan'],
        mode='lines', name='Observasi Stasiun (TMA Aktual)',
        line=dict(color='#64748B', width=2.5)  # Solid Slate Gray
    ))

# 2. Prediksi UTide Murni (Astronomis) - Putus-Putus Rapat (Dot) Cyan Cerah
fig.add_trace(go.Scatter(
    x=df_filtered['Datetime'], y=df_filtered['Prediksi_Harmonik_UTIDE'],
    mode='lines', name='Prediksi UTide Murni (Astronomis)',
    line=dict(color='#06B6D4', width=2.0, dash='dot')  # Dotted Cyan
))

# 3. Prediksi Hibrida (UTide + LSTM) - Putus-Putus Regang (Dash) Indigo Premium Tebal
fig.add_trace(go.Scatter(
    x=df_filtered['Datetime'], y=df_filtered['Prediksi_Hibrida_Final'],
    mode='lines', name='Prediksi Hibrida (UTide + LSTM)',
    line=dict(color='#4F46E5', width=3.0, dash='dash')  # Dashed Indigo
))

# 4. Batas Waspada Rob (230 cm)
fig.add_trace(go.Scatter(
    x=[df_filtered['Datetime'].min(), df_filtered['Datetime'].max()],
    y=[230, 230],
    mode='lines',
    showlegend=False,
    line=dict(color='#F59E0B', width=1.5, dash='dash')
))

# 5. Batas Awas Rob (250 cm)
fig.add_trace(go.Scatter(
    x=[df_filtered['Datetime'].min(), df_filtered['Datetime'].max()],
    y=[250, 250],
    mode='lines',
    showlegend=False,
    line=dict(color='#DC2626', width=1.5, dash='dash')
))

# --- ANNOTATIONS: Didekatkan persis 1 cm di bawah masing-masing garis threshold ---
fig.add_annotation(
    xref="paper", yref="y",
    x=0.005, y=229,  # Mepet di bawah garis 230 cm
    text="<b>⚠️ Waspada Rob (230 cm)</b>",
    showarrow=False,
    xanchor="left",
    yanchor="top",
    font=dict(color='#F59E0B', size=11)
)

fig.add_annotation(
    xref="paper", yref="y",
    x=0.005, y=249,  # Mepet di bawah garis 250 cm
    text="<b>🚨 Awas Rob (250 cm)</b>",
    showarrow=False,
    xanchor="left",
    yanchor="top",
    font=dict(color='#DC2626', size=11)
)

fig.update_layout(
    height=400,  
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis_title="Tanggal & Waktu",
    yaxis_title="Tinggi Muka Air (cm)",
    hovermode="x unified",
    hoverlabel=dict(bgcolor="white", font_size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# 📋 6. INTEGRASI DATA TABULAR & FITUR EKSPOR DOKUMEN (Di luar frame utama)
# =========================================================================
st.markdown("---")
st.markdown("<h3 style='margin:0; padding:0; font-size:19px; font-weight:600; color:#1E293B;'>📋 Data Tabular Hasil Pemotongan</h3>", unsafe_allow_html=True)
st.markdown("Berikut adalah potongan baris data numerik yang merepresentasikan kurva grafik di atas:")

kolom_tampilan = ['Datetime', 'TMA_Pasar_Ikan', 'Prediksi_Harmonik_UTIDE', 'Prediksi_Hibrida_Final']
st.dataframe(df_filtered[kolom_tampilan].reset_index(drop=True), use_container_width=True)

csv_data = df_filtered[kolom_tampilan].to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Data Potongan Ini (.CSV)",
    data=csv_data,
    file_name=f"DATA_TESIS_STUDI_KASUS.csv",
    mime="text/csv",
)
