"""
Script Retraining Otomatis Mingguan (Jalur 2) - Tesis Iman
(Versi Auto-Install Dependencies Mandiri)
"""

# =========================================================================
# 0. AUTO INSTALL DEPENDENCIES MANDIRI (JIKA BELUM ADA DI SERVER)
# =========================================================================
import subprocess
import sys

def install_packages():
    packages = ['tensorflow', 'scikit-learn', 'pandas', 'numpy']
    for package in packages:
        try:
            __import__(package if package != 'scikit-learn' else 'sklearn')
        except ImportError:
            print(f"📦 Menginstal otomatis modul yang kurang: {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Jalankan pengecekan instalasi sebelum impor resmi
install_packages()

# =========================================================================
# 1. IMPOR MODUL UTAMA SETELAH AMAN
# =========================================================================
import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

DATA_FILE_HIBRIDA = "HASIL_FINAL_TESIS_PASUT_HIBRIDA.csv"
DATA_FILE_LSTM = "HASIL_FINAL_TESIS_PASUT_LSTM_MURNI.csv"
MODEL_LSTM_MURNI = "model_pasar_ikan_lstm_murni_master.keras"
MODEL_HIBRIDA = "model_pasar_ikan_master.keras"
SCALER_LSTM_MURNI = "scaler_tma_pasar_ikan_lstm_murni.save"
SCALER_RESIDU_HIBRIDA = "scaler_residu_pasar_ikan.save"
LOOKBACK_WINDOW = 24  

print("🚀 Memulai Pipeline Retraining Mingguan Otomatis...")

if not os.path.exists(DATA_FILE_HIBRIDA) or not os.path.exists(DATA_FILE_LSTM):
    print("❌ Error: File CSV basis data utama tidak ditemukan!")
    exit(1)

df_hib = pd.read_csv(DATA_FILE_HIBRIDA, parse_dates=["Datetime"])
df_lstm = pd.read_csv(DATA_FILE_LSTM, parse_dates=["Datetime"])
df_valid = df_hib[df_hib["TMA_Pasar_Ikan"].notna()].sort_values("Datetime")

if len(df_valid) < LOOKBACK_WINDOW + 10:
    print("⚠️ Data observasi terlalu sedikit untuk retraining. Melewati proses.")
    exit(0)

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
        
    model_tma.fit(np.array(X), np.array(y), epochs=3, batch_size=32, verbose=0)
    model_tma.save(MODEL_LSTM_MURNI)
    print("✅ Model LSTM Murni berhasil diperbarui.")
except Exception as e:
    print(f"⚠️ Gagal memperbarui LSTM Murni: {e}")

# B. Retraining Model Hibrida
print("🧠 Memproses Model Hibrida...")
try:
    scaler_residu = load_scaler(SCALER_RESIDU_HIBRIDA)
    model_hib = tf.keras.models.load_model(MODEL_HIBRIDA)
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

df_hib.to_csv(DATA_FILE_HIBRIDA, index=False)
df_lstm.to_csv(DATA_FILE_LSTM, index=False)
print("🎉 Pipeline Mingguan Selesai Sukses 100%!")
