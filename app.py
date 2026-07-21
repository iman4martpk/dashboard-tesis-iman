import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================================================================
# 🌊 1. KONFIGURASI HALAMAN & THEME (PROFESSIONAL SLATE BLUE)
# =========================================================================
st.set_page_config(
    page_title="Dashboard Pasut Hibrida Pasar Ikan", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.title("🌊 Dashboard Operasional Pasut Hibrida (UTide + LSTM)")
st.markdown("**Stasiun Pemantauan:** Pasar Ikan, Jakarta | **Fokus Riset:** Koreksi Residu Hidro-Oseanografi Non-Astronomis")
st.markdown("---")

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
    "Studi Kasus 1: Awal Periode (Mei 2026)": {
        "start": "2026-05-01 00:00:00",
        "end": "2026-05-07 23:00:00",
        "desc": "Kondisi transisi awal seasonal pasut hidrologis murni."
    },
    "Studi Kasus 2: Tengah Periode (Juni 2026) [BEST PERFORMANCE]": {
        "start": "2026-06-01 00:00:00",
        "end": "2026-06-07 23:00:00",
        "desc": "Anomali residu meteorologis tinggi (Akurasi melonjak +26.08%)."
    },
    "Studi Kasus 3: Akhir Periode (Juli 2026)": {
        "start": "2026-07-14 19:00:00",
        "end": "2026-07-21 18:00:00",
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
    index=1  # Otomatis langsung menyorot Studi Kasus 2 (Juni) pas pertama dibuka
)

st.sidebar.info(f"ℹ️ **Deskripsi:**\n{PRESETS[pilihan_mode]['desc']}")

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
st.subheader("📊 Performa Validasi Real-Time pada Rentang Terpilih")

df_eval = df_filtered[df_filtered['TMA_Pasar_Ikan'].notna() & df_filtered['Prediksi_Hibrida_Final'].notna()]

if len(df_eval) > 0:
    # 1. Hitung RMSE Aktual saat ini
    rmse_utide_curr = np.sqrt(np.mean((df_eval['TMA_Pasar_Ikan'] - df_eval['Prediksi_Harmonik_UTIDE']) ** 2))
    rmse_hib_curr = np.sqrt(np.mean((df_eval['TMA_Pasar_Ikan'] - df_eval['Prediksi_Hibrida_Final']) ** 2))
    
    # 2. Hitung persentase reduksi eror
    if rmse_utide_curr > 0:
        peningkatan_curr = ((rmse_utide_curr - rmse_hib_curr) / rmse_utide_curr) * 100
    else:
        peningkatan_curr = 0
        
    # 3. Hitung selisih nominal eror antara UTide dan Hibrida (untuk memicu arah panah)
    selisih_nominal = rmse_hib_curr - rmse_utide_curr # Hasilnya pasti NEGATIF karena eror hibrida lebih kecil
    
   col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="📈 Reduksi Eror (Peningkatan Akurasi)", 
            value=f"{peningkatan_curr:.2f} %", 
            delta="LSTM Efektif Mengoreksi Residu",
            delta_color="normal" # Tetap HIJAU panah ATAS (Positif = Bagus)
        )
    with col2:
        st.metric(
            label="📉 RMSE Harmonik UTide Murni", 
            value=f"{rmse_utide_curr:.2f} cm", 
            delta=f"+{-selisih_nominal:.2f} cm vs Hibrida", 
            delta_color="normal" # 👈 UBAH KE "normal": Nilai positif akan berwarna MERAH (karena eror surplus)
        )
    with col3:
        st.metric(
            label="🏆 RMSE Komposit Hibrida (LSTM)", 
            value=f"{rmse_hib_curr:.2f} cm", 
            delta=f"{selisih_nominal:.2f} cm vs UTide", 
            delta_color="inverse" # 👈 UBAH KE "inverse": Nilai minus/turun dibalik jadi HIJAU (karena eror menyusut = bagus!)
        )
else:
    st.warning("🔮 **Status:** Menampilkan Area Peramalan Masa Depan. Metrik akurasi tidak dihitung karena data observasi riil lapangan belum terjadi (Masa Depan).")
# =========================================================================
# 📈 5. GRAFIK INTERAKTIF PLOTLY (TIME-SERIES VISUALIZATION)
# =========================================================================
st.subheader(f"📈 Grafik Analisis Perbandingan: {pilihan_mode}")

fig = go.Figure()

# 1. Garis Hitam: Observasi Riil Lapangan (Hanya muncul jika datanya tersedia)
if df_filtered['TMA_Pasar_Ikan'].notna().sum() > 0:
    fig.add_trace(go.Scatter(
        x=df_filtered['Datetime'], y=df_filtered['TMA_Pasar_Ikan'],
        mode='lines', name='Observasi Stasiun (TMA Aktual)',
        line=dict(color='black', width=2.5)
    ))

# 2. Garis Biru Putus-putus: UTide Murni
fig.add_trace(go.Scatter(
    x=df_filtered['Datetime'], y=df_filtered['Prediksi_Harmonik_UTIDE'],
    mode='lines', name='Prediksi UTide Murni (Astronomis)',
    line=dict(color='#1f77b4', width=1.8, dash='dot')
))

# 3. Garis Merah Tebal: Hibrida LSTM
fig.add_trace(go.Scatter(
    x=df_filtered['Datetime'], y=df_filtered['Prediksi_Hibrida_Final'],
    mode='lines', name='Prediksi Hibrida (UTide + LSTM)',
    line=dict(color='#d62728', width=2.5)
))

fig.update_layout(
    height=550,
    xaxis_title="Tanggal & Waktu",
    yaxis_title="Tinggi Muka Air (cm)",
    hovermode="x unified",
    hoverlabel=dict(bgcolor="white", font_size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# 📋 6. INTEGRASI DATA TABULAR & FITUR EKSPOR DOKUMEN
# =========================================================================
st.subheader("📋 Data Tabular Hasil Pemotongan")
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
