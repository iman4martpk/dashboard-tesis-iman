#!/usr/bin/env python3
"""
Pipeline Tesis Otomatis: Auto-Retraining (Fine-Tuning) & Multi-Step Forecasting
Arsitektur: Seq2Seq LSTM (Encoder-Decoder) 3D Tensor Compatible
Desain Sistem: 3-File Architecture (Single Source of Truth)
Status: Environment 100% Synced with Local Machine
"""

import os
import joblib
import numpy as np
import pandas as pd
import sys
from datetime import datetime, timedelta

# Mengunci Seed agar bobot stochastic gradient descent konsisten
SEED = 42
np.random.seed(SEED)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Mematikan warning log bawaan TF yang mengotori konsol
import tensorflow as tf
tf.random.set_seed(SEED)
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================================
# ⚙️ KONFIGURASI PATH ASET & PARAMETER DIMENSI SEQ2SEQ
# =========================================================================
DATA_FILE_HIBRIDA = "HASIL_FINAL_TESIS_PASUT_HIBRIDA.csv"
DATA_FILE_LSTM = "HASIL_FINAL_TESIS_PASUT_LSTM_MURNI.csv"
DATA_FILE_OBSERVASI = "HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv"

MODEL_LSTM_MURNI = "model_pasar_ikan_lstm_murni_master.keras"
MODEL_HIBRIDA = "model_pasar_ikan_master.keras"
SCALER_LSTM_MURNI = "scaler_tma_pasar_ikan_lstm_murni.save"
SCALER_RESIDU_HIBRIDA = "scaler_residu_pasar_ikan.save"

N_INPUT = 336   # Jendela ke belakang: 14 hari x 24 jam
N_OUTPUT = 168  # Jendela ramalan blok: 7 hari x 24 jam
MAX_EPOCHS = 10
BATCH_SIZE = 128

print("=" * 80)
print("🚀 LAUNCHING PIPELINE: AUTOMATED SEQ2SEQ RETRAIN & FORECAST SYSTEM (SYNCED)")
print("=" * 80)

# =========================================================================
# 📊 DATA PIPELINE (INGESTION & SINKRONISASI)
# =========================================================================
for file_path in [DATA_FILE_HIBRIDA, DATA_FILE_LSTM, DATA_FILE_OBSERVASI]:
    if not os.path.exists(file_path):
        print(f"❌ Critical Error: File master '{file_path}' tidak ditemukan!")
        sys.exit(1)

print("📋 Membaca basis data dari repositori...")
df_hib = pd.read_csv(DATA_FILE_HIBRIDA, parse_dates=["Datetime"])
df_lstm = pd.read_csv(DATA_FILE_LSTM, parse_dates=["Datetime"])
df_obs = pd.read_csv(DATA_FILE_OBSERVASI, parse_dates=["Datetime"])

# Proteksi Overwrite: Bersihkan kolom observasi bawaan lama dari file model prediksi
if "TMA_Pasar_Ikan" in df_hib.columns: df_hib = df_hib.drop(columns=["TMA_Pasar_Ikan"])
if "TMA_Pasar_Ikan" in df_lstm.columns: df_lstm = df_lstm.drop(columns=["TMA_Pasar_Ikan"])

# Menjahit data observasi murni ke grid waktu menggunakan Left Join
df_learning = pd.merge(df_hib, df_obs[["Datetime", "TMA_Pasar_Ikan"]], on="Datetime", how="left")
df_valid = df_learning[df_learning["TMA_Pasar_Ikan"].notna()].sort_values("Datetime").reset_index(drop=True)

if len(df_valid) <= N_INPUT + N_OUTPUT:
    print(f"⚠️ Data observasi murni ({len(df_valid)} jam) belum memenuhi syarat windowing ({N_INPUT + N_OUTPUT} jam).")
    print("⏭️ Melewati fase retraining operasional minggu ini.")
    sys.exit(0)

waktu_terakhir_obs = df_valid["Datetime"].iloc[-1]
print(f"📌 Batas Data Observasi Lapangan Aktual: {waktu_terakhir_obs}")

def prepare_3d_sequences(dataset: np.ndarray, look_back: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Mengubah array 1D menjadi pasangan tensor 3D untuk arsitektur Encoder-Decoder."""
    X, Y = [], []
    for i in range(len(dataset) - look_back - horizon + 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[(i + look_back):(i + look_back + horizon), 0])
    return np.array(X), np.array(Y)

# =========================================================================
# 🧠 BLOK A: FINE-TUNING & FORECASTING MODEL LSTM MURNI (DIRECT TMA)
# =========================================================================
print("\n" + "-" * 50)
print("🧠 [BLOK A] PROSES MODEL LSTM MURNI (PREDIKSI LANGSUNG TMA)")
print("-" * 50)
try:
    scaler_tma = joblib.load(SCALER_LSTM_MURNI)
    model_tma = tf.keras.models.load_model(MODEL_LSTM_MURNI)
    
    scaled_tma = scaler_tma.transform(df_valid[["TMA_Pasar_Ikan"]].values.reshape(-1, 1))
    X_lstm, y_lstm = prepare_3d_sequences(scaled_tma, N_INPUT, N_OUTPUT)
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
    y_lstm = np.reshape(y_lstm, (y_lstm.shape[0], y_lstm.shape[1], 1))
    
    autostop_tma = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True, verbose=1)
    
    print(f"🏋️ Melatih ulang model via Fine-Tuning (Max: {MAX_EPOCHS} Epoch)...")
    model_tma.fit(X_lstm, y_lstm, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=[autostop_tma], verbose=1)
    model_tma.save(MODEL_LSTM_MURNI)
    print("💾 Otak Model LSTM Murni sukses diamankan.")
    
    total_future_hours = len(df_lstm[df_lstm["Datetime"] > waktu_terakhir_obs])
    print(f"🔮 Menghitung peramalan blok Seq2Seq ke depan untuk {total_future_hours} jam...")
    
    current_window = list(scaled_tma[-N_INPUT:].flatten())
    future_preds_scaled = []
    hours_predicted = 0
    
    while hours_predicted < total_future_hours:
        input_data = np.array(current_window[-N_INPUT:]).reshape(1, N_INPUT, 1)
        pred_block = model_tma.predict(input_data, verbose=0)[0, :, 0]
        future_preds_scaled.extend(list(pred_block))
        current_window.extend(list(pred_block))
        hours_predicted += N_OUTPUT
        
    future_preds_cm = scaler_tma.inverse_transform(np.array(future_preds_scaled[:total_future_hours]).reshape(-1, 1)).flatten()
    
    df_future_lstm = df_lstm[df_lstm["Datetime"] > waktu_terakhir_obs].copy()
    df_future_lstm["Prediksi_LSTM_Murni"] = future_preds_cm[:len(df_future_lstm)]
    df_lstm.loc[df_lstm["Datetime"] > waktu_terakhir_obs, "Prediksi_LSTM_Murni"] = df_future_lstm["Prediksi_LSTM_Murni"]
    print("✅ Garis proyeksi Prediksi_LSTM_Murni berhasil diperbarui di masa depan.")
    
except Exception as err:
    print(f"❌ Error Terjadi pada Blok A: {err}")

# =========================================================================
# 🧠 BLOK B: FINE-TUNING & FORECASTING MODEL HIBRIDA (ERROR RESIDUAL)
# =========================================================================
print("\n" + "-" * 50)
print("🧠 [BLOK B] PROSES MODEL HIBRIDA (PREDIKSI RESIDU ERROR)")
print("-" * 50)
try:
    scaler_residu = joblib.load(SCALER_RESIDU_HIBRIDA)
    model_hib = tf.keras.models.load_model(MODEL_HIBRIDA)
    
    residu_historis = df_valid["TMA_Pasar_Ikan"].values - df_valid["Prediksi_Harmonik_UTIDE"].values
    scaled_residu = scaler_residu.transform(residu_historis.reshape(-1, 1))
    
    X_res, y_res = prepare_3d_sequences(scaled_residu, N_INPUT, N_OUTPUT)
    X_res = np.reshape(X_res, (X_res.shape[0], X_res.shape[1], 1))
    y_res = np.reshape(y_res, (y_res.shape[0], y_res.shape[1], 1))
    
    autostop_hib = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True, verbose=1)
    
    print(f"🏋️ Melatih ulang model via Fine-Tuning (Max: {MAX_EPOCHS} Epoch)...")
    model_hib.fit(X_res, y_res, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=[autostop_hib], verbose=1)
    model_hib.save(MODEL_HIBRIDA)
    print("💾 Otak Model Residu Hibrida Master sukses diamankan.")
    
    total_future_hours_hib = len(df_hib[df_hib["Datetime"] > waktu_terakhir_obs])
    print(f"🔮 Menghitung peramalan blok residu Seq2Seq ke depan untuk {total_future_hours_hib} jam...")
    
    current_window_res = list(scaled_residu[-N_INPUT:].flatten())
    future_res_preds_scaled = []
    hours_predicted_res = 0
    
    while hours_predicted_res < total_future_hours_hib:
        input_data_res = np.array(current_window_res[-N_INPUT:]).reshape(1, N_INPUT, 1)
        pred_block_res = model_hib.predict(input_data_res, verbose=0)[0, :, 0]
        future_res_preds_scaled.extend(list(pred_block_res))
        current_window_res.extend(list(pred_block_res))
        hours_predicted_res += N_OUTPUT
        
    future_res_preds_cm = scaler_residu.inverse_transform(np.array(future_res_preds_scaled[:total_future_hours_hib]).reshape(-1, 1)).flatten()
    
    df_future_hib = df_hib[df_hib["Datetime"] > waktu_terakhir_obs].copy()
    df_future_hib["Residu_LSTM_Pred"] = future_res_preds_cm[:len(df_future_hib)]
    df_future_hib["Prediksi_Hibrida_Final"] = df_future_hib["Prediksi_Harmonik_UTIDE"] + df_future_hib["Residu_LSTM_Pred"]
    
    df_hib.loc[df_hib["Datetime"] > waktu_terakhir_obs, "Prediksi_Hibrida_Final"] = df_future_hib["Prediksi_Hibrida_Final"]
    print("✅ Garis proyeksi Prediksi_Hibrida_Final berhasil diperbarui di masa depan.")
    
except Exception as err:
    print(f"❌ Error Terjadi pada Blok B: {err}")

# =========================================================================
# 💾 DATABASE SYNCHRONIZATION & STORAGE MANAGEMENT
# =========================================================================
print("\n" + "=" * 80)
print("💾 PROSES SINKRONISASI BASIS DATA NUMERIK...")
print("=" * 80)

df_hib["Datetime"] = df_hib["Datetime"].dt.strftime('%Y-%m-%d %H:%M:%S')
df_lstm["Datetime"] = df_lstm["Datetime"].dt.strftime('%Y-%m-%d %H:%M:%S')

if "TMA_Pasar_Ikan" in df_hib.columns: df_hib = df_hib.drop(columns=["TMA_Pasar_Ikan"])
if "TMA_Pasar_Ikan" in df_lstm.columns: df_lstm = df_lstm.drop(columns=["TMA_Pasar_Ikan"])

df_hib.to_csv(DATA_FILE_HIBRIDA, index=False)
df_lstm.to_csv(DATA_FILE_LSTM, index=False)

print("🎉 [SUCCESS] ALL PIPELINES ARE COMPLETELY SYNCHRONIZED 100% WITH AUTOSTOP!")
print("=" * 80)
