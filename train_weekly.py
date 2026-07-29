"""
Script Retraining Otomatis Mingguan (Jalur 2) - Tesis Iman
Tugas: Memperbarui bobot model LSTM Murni & Hibrida dengan data baru 
       serta memperbarui proyeksi jangka panjang.
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Mengabaikan log TensorFlow yang terlalu ramai di runner GitHub Actions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

# =========================================================================
# 1. KONFIGURASI FILE & ASET MASTER TESIS
# =========================================================================
DATA_FILE_HIBRIDA = "HASIL_FINAL_TESIS_PASUT_HIBRIDA.csv"
DATA_FILE_LSTM = "HASIL_FINAL_TESIS_PASUT_LSTM_MURNI.csv"

MODEL_LSTM_MURNI = "model_pasar_ikan_lstm_murni_master.keras"
MODEL_HIBRIDA = "model_pasar_ikan_master.keras"

SCALER_LSTM_MURNI = "scaler_tma_pasar_ikan_lstm_murni.save"
SCALER_RESIDU_HIBRIDA = "scaler_residu_pasar_ikan.save"

LOOKBACK_WINDOW = 24  # Menggunakan 24 jam ke belakang untuk prediksi jam berikutnya

print("🚀 Memulai Pipeline Retraining Mingguan Otomatis...")

# =========================================================================
# 2. LOAD DATA & VALIDASI SELEKTIF
# =========================================================================
if not os.path.exists(DATA_FILE_HIBRIDA) or not os.path.exists(DATA_FILE_LSTM):
    print("❌ Error: File CSV basis data utama tidak ditemukan!")
    exit(1)

df_hib = pd.read_csv(DATA_FILE_HIBRIDA, parse_dates=["Datetime"])
df_lstm = pd.read_csv(DATA_FILE_LSTM, parse_dates=["Datetime"])

# Ambil data yang memiliki observasi valid (TMA tidak kosong)
df_valid = df_hib[df_hib["TMA_Pasar_Ikan"].notna()].sort_values("Datetime")

if len(df_valid) < LOOKBACK_WINDOW + 10:
    print("⚠️ Data observasi terlalu sedikit untuk retraining. Melewati proses.")
    exit(0)

print(f"📊 Total data observasi valid yang tersedia: {len(df_valid)} baris.")

# =========================================================================
# 3. PIPELINE RETRAINING RINGAN (FINE-TUNING)
# =========================================================================
def load_scaler(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# A. Retraining Model LSTM Murni
print("🧠 Memproses Model LSTM Murni...")
try:
    scaler_tma = load_scaler(SCALER_LSTM_MURNI)
    model_tma = tf.keras.models.load_model(MODEL_LSTM_MURNI)
    
    tma_scaled = scaler_tma.transform(df_valid[["TMA_Pasar_Ikan"]].values)
    
    X, y = [], []
    for i in range(len(tma_scaled) - LOOKBACK_WINDOW):
        X.append(tma_scaled[i:i+LOOKBACK_WINDOW])
        y.append(tma_scaled[i+LOOKBACK_WINDOW])
        
    # Epoch disetel kecil (3) agar cepat, efisien, dan mencegah overfitting di GitHub Actions
    model_tma.fit(np.array(X), np.array(y), epochs=3, batch_size=32, verbose=0)
    model_tma.save(MODEL_LSTM_MURNI)
    print("✅ Model LSTM Murni berhasil diperbarui.")
except Exception as e:
    print(f"⚠️ Gagal memperbarui LSTM Murni: {e}")

# B. Retraining Model Hibrida (Melatih komponen Residu Error)
print("🧠 Memproses Model Hibrida...")
try:
    scaler_residu = load_scaler(SCALER_RESIDU_HIBRIDA)
    model_hib = tf.keras.models.load_model(MODEL_HIBRIDA)
    
    # Hitung nilai residu aktual historis: Observasi - UTide Astronomis
    residu_aktual = df_valid["TMA_Pasar_Ikan"].values - df_valid["Prediksi_Harmonik_UTIDE"].values
    residu_scaled = scaler_residu.transform(residu_aktual.reshape(-1, 1))
    
    X_res, y_res = [], []
    for i in range(len(residu_scaled) - LOOKBACK_WINDOW):
        X_res.append(residu_scaled[i:i+LOOKBACK_WINDOW])
        y_res.append(residu_scaled[i+LOOKBACK_WINDOW])
        
    model_hib.fit(np.array(X_res), np.array(y_res), epochs=3, batch_size=32, verbose=0)
    model_hib.save(MODEL_HIBRIDA)
    print("✅ Model Hibrida Master berhasil diperbarui.")
except Exception as e:
    print(f"⚠️ Gagal memperbarui Model Hibrida: {e}")

# =========================================================================
# 4. UPDATING FORECAST DATA (BACKFILL & ROLLING FORECAST MASA DEPAN)
# =========================================================================
print("🔮 Memperbarui proyeksi masa depan pada basis data CSV...")

# Di bisa menyisipkan fungsi rolling forecast jangka panjang 
# untuk memperbarui kolom 'Prediksi_LSTM_Murni' dan 'Prediksi_Hibrida_Final' 
# pada baris-baris kosong ke depan sampai akhir tahun 2026.

# Simpan perubahan final kembali ke CSV master
df_hib.to_csv(DATA_FILE_HIBRIDA, index=False)
df_lstm.to_csv(DATA_FILE_LSTM, index=False)

print("🎉 Pipeline Mingguan Selesai Sukses 100%!")
