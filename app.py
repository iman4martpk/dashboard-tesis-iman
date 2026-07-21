import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. KONFIGURASI HALAMAN WEB
st.set_page_config(page_title="Dashboard Pasut Hibrida", layout="wide")

st.title("🌊 Dashboard Operasional Pasut Hibrida (UTide + LSTM)")
st.markdown("**Stasiun Pemantauan:** Pasar Ikan, Jakarta | **Target Riset:** Koreksi Residu Temporal")
st.markdown("---")

# 2. LOAD DATA (Menggunakan Cache Biar Ngebut)
@st.cache_data
def load_data():
    df = pd.read_csv("HASIL_FINAL_TESIS_PASUT_HIBRIDA.csv", parse_dates=['Datetime'])
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"❌ Gagal memuat file CSV. Pastikan file 'HASIL_FINAL_TESIS_PASUT_HIBRIDA.csv' ada di folder yang sama. Error: {e}")
    st.stop()

# 3. KOTAK METRIK UTAMA (KPI SCORECARD) - REKAPAN BAB 4 LU
st.subheader("📊 Rekapan Performa Model (Hasil Validasi)")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(label="Peningkatan Akurasi Maksimal (Juni)", value="26.08 %", delta="LSTM Superior")
with col2:
    st.metric(label="Rata-rata Reduksi Eror RMSE (Mei)", value="11.67 %", delta="Koreksi Positif")
with col3:
    st.metric(label="Batas Data Aktual Lapangan", value="21 Juli 2026", delta="18:00 WIB")

st.markdown("---")

# 4. SIDEBAR FILTER WAKTU & MODE TAMPILAN
st.sidebar.header("⚙️ Kontrol Panel")
min_date = df['Datetime'].min().date()
max_date = df['Datetime'].max().date()

start_date, end_date = st.sidebar.date_input(
    "Pilih Rentang Waktu Analisis",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

# Filter Mode: Tampilkan Semua atau Hanya Mode Forecasting Masa Depan
mode_tampilan = st.sidebar.radio(
    "Mode Tampilan Grafik:",
    ["Tampilkan Semua Data", "Hanya Peramalan Masa Depan (Agustus - Desember 2026)"]
)

# Filter dataframe sesuai tanggal pilihan user
mask = (df['Datetime'].dt.date >= start_date) & (df['Datetime'].dt.date <= end_date)
df_filtered = df.loc[mask].copy()

# Logika Mode Peramalan Masa Depan
if mode_tampilan == "Hanya Peramalan Masa Depan (Agustus - Desember 2026)":
    df_filtered = df_filtered[df_filtered['Datetime'] > '2026-07-21 18:00:00']

# 5. GRAFIK INTERAKTIF PLOTLY
st.subheader("📈 Visualisasi Perbandingan Model vs Observasi")

fig = go.Figure()

# Garis Observasi Asli (Hitam) - Hanya muncul jika datanya tidak NaN
if df_filtered['TMA_Pasar_Ikan'].notna().sum() > 0:
    fig.add_trace(go.Scatter(
        x=df_filtered['Datetime'], y=df_filtered['TMA_Pasar_Ikan'],
        mode='lines', name='Observasi Stasiun (TMA Riil)',
        line=dict(color='black', width=2)
    ))

# Garis UTide Murni (Biru Putus-putus)
fig.add_trace(go.Scatter(
    x=df_filtered['Datetime'], y=df_filtered['Prediksi_Harmonik_UTIDE'],
    mode='lines', name='UTide Murni (Astronomis)',
    line=dict(color='blue', width=1.5, dash='dot')
))

# Garis Prediksi Hibrida (Merah)
fig.add_trace(go.Scatter(
    x=df_filtered['Datetime'], y=df_filtered['Prediksi_Hibrida_Final'],
    mode='lines', name='Prediksi Hibrida (UTide + LSTM)',
    line=dict(color='red', width=2)
))

# Kustomisasi Layout Grafik (Label Y di-set ke cm sesuai data riil)
fig.update_layout(
    height=550,
    xaxis_title="Tanggal & Waktu",
    yaxis_title="Tinggi Muka Air (cm)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# 6. TABEL DATA MENTAH & TOMBOL DOWNLOAD
st.subheader("📋 Data Tabular & Integrasi Dokumen")
st.dataframe(df_filtered.reset_index(drop=True), use_container_width=True)

# Tombol unduh data langsung dari dashboard
csv_data = df_filtered.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Data yang Difilter (.CSV)",
    data=csv_data,
    file_name="EKSPOR_DATA_PASUT_ELKANA.csv",
    mime="text/csv",
)