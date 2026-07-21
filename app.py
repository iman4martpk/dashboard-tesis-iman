import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# =========================================================================
# 🌊 1. KONFIGURASI HALAMAN & THEME (INSTRUMEN PASUT — RESPONSIVE)
# =========================================================================
st.set_page_config(
    page_title="Dashboard Pasut Hibrida Pasar Ikan",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Suntik CSS kustom: font instrumen, header transparan tapi tombol sidebar SELALU kontras,
# dan hero band bergaya panel instrumen stasiun pasut.
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

        :root {
            --navy: #0B3D4C;
            --navy-2: #124E61;
            --teal: #0E7490;
            --foam: #F8FAFC;
            --ink: #1E293B;
            --slate: #64748B;
            --amber: #D97706;
            --alert-red: #DC2626;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            -webkit-font-smoothing: antialiased;
            text-rendering: optimizeLegibility;
        }

        /* ================= HEADER & TOMBOL SIDEBAR (SEMUA UKURAN LAYAR) ================= */
        /* Sembunyikan toolbar bawaan Streamlit (menu titik tiga, Deploy dsb) -- BUKAN tombol sidebar */
        [data-testid="stToolbar"] {
            visibility: hidden !important;
        }
        [data-testid="stHeader"] {
            background: transparent !important;
            height: 0rem !important;
        }

        /* Tombol buka/tutup sidebar dipaksa SELALU terlihat dengan latar solid kontras tinggi,
           supaya tidak lagi menyatu/hilang di layar HP */
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapseButton"] {
            display: flex !important;
            visibility: visible !important;
            opacity: 1 !important;
            background-color: var(--navy) !important;
            border-radius: 8px !important;
            box-shadow: 0 2px 10px rgba(11, 61, 76, 0.35) !important;
            padding: 4px !important;
            z-index: 999999 !important;
        }
        [data-testid="collapsedControl"] svg,
        [data-testid="stSidebarCollapseButton"] svg {
            fill: var(--foam) !important;
            color: var(--foam) !important;
        }

        /* ================= HERO / PANEL INSTRUMEN ================= */
        .hero-band {
            background-image:
                url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1200 120' preserveAspectRatio='none'%3E%3Cpath d='M0,50 C300,100 900,0 1200,60 L1200,120 L0,120 Z' fill='%23ffffff' fill-opacity='0.05'/%3E%3C/svg%3E"),
                linear-gradient(135deg, var(--navy) 0%, var(--navy-2) 100%);
            background-repeat: no-repeat, no-repeat;
            background-size: cover, cover;
            background-position: bottom, center;
            border-radius: 14px;
            padding: 22px 28px 26px 28px;
            margin-bottom: 14px;
            box-shadow: 0 6px 24px rgba(11, 61, 76, 0.18);
            text-align: center;
        }
        .hero-title {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            color: var(--foam);
            letter-spacing: -0.3px;
            margin: 0;
        }
        .hero-subtitle {
            font-family: 'Inter', sans-serif;
            font-weight: 400;
            color: #CBD5E1;
            margin: 6px 0 0 0;
        }

        /* 💻 DESKTOP / LAPTOP */
        @media (min-width: 768px) {
            .main .block-container {
                padding-top: 1.2rem !important;
                padding-bottom: 0rem !important;
                padding-left: 2.5rem !important;
                padding-right: 2.5rem !important;
            }
            .hero-title { font-size: 32px; }
            .hero-subtitle { font-size: 14.5px; }
        }

        /* 📱 HP (ANDROID / IOS) */
        @media (max-width: 767px) {
            .main .block-container {
                padding-top: 3.2rem !important; /* ruang aman di bawah tombol sidebar mengambang */
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            .hero-title { font-size: 21px; line-height: 1.35; }
            .hero-subtitle { font-size: 12px; margin-top: 5px; }
            .hero-band { padding: 16px 16px 20px 16px; border-radius: 12px; }
        }

        .stVerticalBlock { gap: 0.8rem !important; }

        /* ================= KARTU METRIK BERGAYA "GAUGE READOUT" ================= */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 4px solid var(--teal);
            border-radius: 10px;
            padding: 8px 14px !important;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        }
        div[data-testid="stMetricValue"] {
            font-family: 'IBM Plex Mono', monospace !important;
            color: var(--ink);
        }
        div[data-testid="stMetricLabel"] {
            font-family: 'Inter', sans-serif;
            color: var(--slate);
        }
        /* Bonus kosmetik: beda warna aksen tiap kartu KPI (navy / teal / amber).
           Ini bergantung struktur DOM st.columns Streamlit versi kamu -- kalau tidak
           berpengaruh di versimu, aman diabaikan, tidak merusak apa pun. */
        div[data-testid="column"]:nth-of-type(1) div[data-testid="stMetric"] { border-left-color: var(--navy); }
        div[data-testid="column"]:nth-of-type(2) div[data-testid="stMetric"] { border-left-color: var(--teal); }
        div[data-testid="column"]:nth-of-type(3) div[data-testid="stMetric"] { border-left-color: var(--amber); }
    </style>
""", unsafe_allow_html=True)

# Hero Header — panel instrumen navy dengan tekstur gelombang halus
st.markdown("""
    <div class="hero-band">
        <h1 class="hero-title">🌊 Dashboard Operasional Pasut Hibrida (UTide + LSTM)</h1>
        <p class="hero-subtitle">
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
# 📈 5. GRAFIK INTERAKTIF PLOTLY (TIME-SERIES VISUALIZATION — TEMA INSTRUMEN)
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

# 2. Prediksi UTide Murni (Astronomis) - Putus-Putus Rapat (Dot) Teal Laut
fig.add_trace(go.Scatter(
    x=df_filtered['Datetime'], y=df_filtered['Prediksi_Harmonik_UTIDE'],
    mode='lines', name='Prediksi UTide Murni (Astronomis)',
    line=dict(color='#0E7490', width=2.0, dash='dot')  # Dotted Teal
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
    line=dict(color='#D97706', width=1.5, dash='dash')
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
    x=0.005, y=229,
    text="<b>⚠️ Waspada Rob (230 cm)</b>",
    showarrow=False,
    xanchor="left",
    yanchor="top",
    font=dict(color='#D97706', size=11)
)

fig.add_annotation(
    xref="paper", yref="y",
    x=0.005, y=249,
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
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    font=dict(family="Inter, sans-serif", color="#1E293B")
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
