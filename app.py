import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================================================================
# 🌊 1. KONFIGURASI HALAMAN & THEME (COMPACT LAYOUT)
# =========================================================================
st.set_page_config(
    page_title="Dashboard Pasut Hibrida Pasar Ikan", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Suntik CSS kustom untuk merapatkan layout ke atas dan menghemat ruang vertikal
st.markdown("""
    <style>
        /* Mengurangi padding atas dan bawah pada kontainer utama */
        .main .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 0rem !important;
            padding-left: 2.5rem !important;
            padding-right: 2.5rem !important;
        }
        /* Memperkecil jarak default antar elemen Streamlit */
        .stVerticalBlock {
            gap: 0.6rem !important;
        }
        /* Mengurangi margin vertikal pada komponen metrik */
        div[data-testid="stMetric"] {
            padding: 5px 10px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Menggunakan HTML kustom agar Judul & Sub-judul tidak memakan banyak tempat
st.markdown("<h2 style='margin:0; padding:0; font-size:26px;'>🌊 Dashboard Operasional Pasut Hibrida (UTide + LSTM)</h2>", unsafe_allow_html=True)
st.markdown("<p style='margin: 0 0 10px 0; font-size:14px; color:#555;'><b>Stasiun Pemantauan:</b> Pasar Ikan, Jakarta | <b>Fokus Riset:</b> Koreksi Residu Hidro-Oseanografi Non-Astronomis</p>", unsafe_allow_html=True)

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
st.markdown("<h3 style='margin:5px 0 0 0; padding:0; font-size:18px;'>📊 Performa Validasi Real-Time pada Rentang Terpilih</h3>", unsafe_allow_html=True)

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
# 📈 5. GRAFIK INTERAKTIF PLOTLY (TIME-SERIES VISUALIZATION)
# =========================================================================
st.markdown(f"<h3 style='margin:10px 0 0 0; padding:0; font-size:18px;'>📈 Grafik Analisis Perbandingan: {pilihan_mode}</h3>", unsafe_allow_html=True)

fig = go.Figure()

if df_filtered['TMA_Pasar_Ikan'].notna().sum() > 0:
    fig.add_trace(go.Scatter(
        x=df_filtered['Datetime'], y=df_filtered['TMA_Pasar_Ikan'],
        mode='lines', name='Observasi Stasiun (TMA Aktual)',
        line=dict(color='black', width=2.5)
    ))

fig.add_trace(go.Scatter(
    x=df_filtered['Datetime'], y=df_filtered['Prediksi_Harmonik_UTIDE'],
    mode='lines', name='Prediksi UTide Murni (Astronomis)',
    line=dict(color='#1f77b4', width=1.8, dash='dot')
))

fig.add_trace(go.Scatter(
    x=df_filtered['Datetime'], y=df_filtered['Prediksi_Hibrida_Final'],
    mode='lines', name='Prediksi Hibrida (UTide + LSTM)',
    line=dict(color='#d62728', width=2.5)
))

fig.update_layout(
    height=400,  # Dikurangi dari 550 ke 400 agar muat dalam satu frame layar
    margin=dict(l=10, r=10, t=30, b=10), # Memperkecil margin internal grafik
    xaxis_title="Tanggal & Waktu",
    yaxis_title="Tinggi Muka Air (cm)",
    hovermode="x unified",
    hoverlabel=dict(bgcolor="white", font_size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# 📋 6. INTEGRASI DATA TABULAR & FITUR EKSPOR DOKUMEN (Di bagian bawah)
# =========================================================================
st.markdown("---")
st.markdown("<h3 style='margin:0; padding:0; font-size:18px;'>📋 Data Tabular Hasil Pemotongan</h3>", unsafe_allow_html=True)
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
