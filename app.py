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
        padding-top: 3.2rem !important; 
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

    /* ================= GLOBAL STANDARD FONT COUPLING ================= */
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
    # Load File Utama (Hibrida & UTide)
    df_hibrida = pd.read_csv("HASIL_FINAL_TESIS_PASUT_HIBRIDA.csv", parse_dates=['Datetime'])
    # Load File Pendukung (LSTM Murni) secara terpisah
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

# Logika Filter Tanggal (Diterapkan ke 2 dataframe sekaligus)
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

# Eksekusi filter secara independen
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
    
    # Hitung RMSE
    rmse_utide_curr = np.sqrt(np.mean((df_filtered.loc[valid_idx, 'TMA_Pasar_Ikan'] - df_filtered.loc[valid_idx, 'Prediksi_Harmonik_UTIDE']) ** 2))
    rmse_lstm_curr = np.sqrt(np.mean((df_filtered.loc[valid_idx, 'TMA_Pasar_Ikan'] - df_lstm_filtered.loc[valid_idx, KOLOM_LSTM_MURNI]) ** 2))
    rmse_hibOke bre, gw paham banget kegelisahan lu. Betul, di level tesis, akurasi istilah itu penting banget. Dosen penguji biasanya cukup *strict* soal definisi "Model". UTide itu pada dasarnya adalah *Harmonic Analysis* (analisis harmonik pasut), yang secara matematis memang sebuah model (model empiris/analitik), tapi sering dibedakan dengan "Model Prediksi" berbasis *Machine Learning* (kaya LSTM, SVR, dll).

Biar aman dan secara *scientific* lebih valid, mending kita hindari kata "Best Model" kalau lu membandingkan metode yang beda *nature*-nya.

## 1. Saran Istilah Scientific untuk Tesis

Berikut beberapa alternatif yang lebih *thesis-material* dan aman dipakai untuk membandingkan UTide dengan metode lain:

*   **Best Prediction Method / Metode Prediksi Terbaik:** Ini paling aman dan universal. Baik UTide maupun *Machine Learning* sama-sama berfungsi sebagai "Metode" untuk menghasilkan "Prediksi".
*   **Most Accurate Estimation / Estimasi Paling Akurat:** Sangat cocok kalau konteksnya membandingkan nilai selisih (error) terhadap data real-time.
*   **Optimal Forecasting Approach / Pendekatan Peramalan Optimal:** Terdengar sangat akademis. Cocok kalau tesis lu fokus pada sistem *forecasting* pasut ke depan.
*   **Superior Predictive Performance / Kinerja Prediksi Superior:** Biasanya dipakai di judul tabel evaluasi atau legenda grafik.

**Rekomendasi Gw:** Pakai **"Metode Prediksi Terbaik" (Best Prediction Method)**. Simpel, lugas, dan secara keilmuan mencakup analisis harmonik maupun *machine learning*.

---

## 2. Implementasi Kode Dinamis (Python)

Biar tulisannya bisa dinamis ngikutin nilai selisih (error) terkecil terhadap data real-time, kita harus bikin logikanya:
1. Hitung error (misal pakai *Root Mean Square Error* / RMSE atau *Mean Absolute Error* / MAE).
2. Cari nilai error terkecil.
3. Ambil nama metode yang menang, lalu masukin ke dalam teks/plot secara otomatis.

Ini contoh lengkap pakai Python (asumsi lu pakai Pandas dan Numpy, yang standar banget buat tesis data):

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. BIKIN DATA DUMMY (Anggap aja ini DataFrame lu)
# Kolom: Waktu, Data_Realtime, Prediksi_UTide, Prediksi_LSTM, Prediksi_SVR
data = {
    'Waktu': pd.date_range(start='2026-07-01', periods=5, freq='H'),
    'Realtime': [1.20, 1.45, 1.60, 1.55, 1.30],
    'UTide':    [1.22, 1.43, 1.58, 1.57, 1.28], # Selisih dikit
    'LSTM':     [1.15, 1.40, 1.65, 1.50, 1.35], # Selisih lumayan
    'SVR':      [1.25, 1.50, 1.55, 1.60, 1.25]  # Selisih lumayan
}
df = pd.DataFrame(data)

# 2. DEFINISIKAN METODE YANG MAU DIBANDINGKAN
# Key: Nama kolom di DataFrame, Value: Nama Keren buat di Tesis/Plot
metode_dict = {
    'UTide': 'Harmonic Analysis (UTide)',
    'LSTM': 'Deep Learning (LSTM)',
    'SVR': 'Support Vector Regression (SVR)'
}

# 3. HITUNG SELISIH TERBAIK (Pakai MAE atau RMSE)
# Kita pakai MAE (Mean Absolute Error) sebagai contoh selisih rata-rata
error_results = {}

for metode in metode_dict.keys():
    # Menghitung absolute error: |Realtime - Prediksi|
    abs_error = np.abs(df['Realtime'] - df[metode])
    mae = abs_error.mean()
    error_results[metode] = mae

# 4. CARI METODE DENGAN ERROR TERKECIL
best_method_key = min(error_results, key=error_results.get)
best_method_name = metode_dict[best_method_key]
best_error_value = error_results[best_method_key]

# 5. BIKIN TEKS DINAMISNYA
# Menggunakan istilah "Best Prediction Method"
dynamic_title = f"Evaluasi Pasang Surut: {best_method_name} sebagai Metode Prediksi Terbaik\n(MAE: {best_error_value:.3f} m)"

print("=== HASIL EVALUASI ===")
print(f"Metode Terbaik: {best_method_name}")
print(f"Teks Dinamis untuk Plot: \n{dynamic_title}\n")

# 6. CONTOH APLIKASI DI PLOT (Opsional, kalau lu mau nampilin grafiknya)
plt.figure(figsize=(10, 5))
plt.plot(df['Waktu'], df['Realtime'], label='Data Real-time (Bata)', color='black', linewidth=2, marker='o')

# Plot metode yang menang aja (atau bisa plot semua)
plt.plot(df['Waktu'], df[best_method_key], label=f'Prediksi: {best_method_name}', color='blue', linestyle='--')

plt.title(dynamic_title, fontsize=12, fontweight='bold')
plt.xlabel('Waktu')
plt.ylabel('Elevasi (m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show() # Uncomment ini buat nampilin plotnya di lokal lu
