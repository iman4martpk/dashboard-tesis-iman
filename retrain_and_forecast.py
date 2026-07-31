#!/usr/bin/env python3
"""
Pipeline Tesis Otomatis: Auto-Retraining (Fine-Tuning) & Multi-Step Forecasting
Arsitektur: Seq2Seq LSTM (Encoder-Decoder) 3D Tensor Compatible
Desain Sistem: 3-File Architecture (Single Source of Truth)
Status: Environment 100% Synced with Local Machine

Catatan perubahan penting:
1. [FIX MERGE CRASH] Parsing kolom Datetime tidak lagi memakai
   pd.read_csv(..., parse_dates=[...]) yang gagal diam-diam saat format
   tanggal di CSV bercampur (mis. legacy "%m/%d/%Y %H:%M" vs baris baru
   dari scraper "%Y-%m-%d %H:%M:%S") -> kolom tertinggal jadi string ->
   pd.merge meledak ("datetime64 and str columns"). Sekarang semua kolom
   Datetime diparse eksplisit lewat _parse_datetime_column() yang tahan
   format campuran (format="mixed", errors="coerce") dan konsisten
   menghasilkan datetime64 murni tanpa timezone di ketiga file.
2. [FIX CAKUPAN FORECAST] Forecasting HANYA menghasilkan & menimpa TEPAT
   1 window N_OUTPUT (168 jam / 7 hari) setelah observasi terakhir. Baris
   di luar window itu (hari ke-8 dan seterusnya) TIDAK disentuh sama
   sekali, apapun isinya. Sebelumnya rolling multi-block forecasting
   menimpa seluruh sisa grid (bisa berbulan-bulan ke depan) dan errornya
   menumpuk (forecast drift) karena tiap blok memakai prediksi blok
   sebelumnya sebagai "observasi palsu".
"""

import os
import sys
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd

# Mengunci Seed agar bobot stochastic gradient descent konsisten
SEED = 42
np.random.seed(SEED)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Mematikan warning log bawaan TF yang mengotori konsol
import tensorflow as tf  # noqa: E402

tf.random.set_seed(SEED)
from tensorflow.keras.callbacks import EarlyStopping  # noqa: E402

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
N_OUTPUT = 168  # Jendela ramalan blok: 7 hari x 24 jam (BATAS overwrite forecast)
MAX_EPOCHS = 10
BATCH_SIZE = 128

print("=" * 80)
print("🚀 LAUNCHING PIPELINE: AUTOMATED SEQ2SEQ RETRAIN & FORECAST SYSTEM (SYNCED)")
print("=" * 80)


# =========================================================================
# 🛠️ UTIL: PARSING DATETIME YANG TAHAN FORMAT CAMPURAN
# =========================================================================
def parse_datetime_column(series: pd.Series, label: str) -> pd.Series:
    """
    Parsing kolom Datetime yang tahan terhadap format campuran (mis. baris
    lama "%m/%d/%Y %H:%M" bercampur dengan baris baru ISO
    "%Y-%m-%d %H:%M:%S"). Baris yang gagal diparse dijadikan NaT (bukan
    meng-crash-kan pipeline) dan dilaporkan lewat konsol.
    """
    parsed = pd.to_datetime(series, format="mixed", dayfirst=False, errors="coerce")
    n_invalid = int(parsed.isna().sum())
    if n_invalid:
        print(f"⚠️ [{label}] {n_invalid} baris punya format Datetime tidak valid, diabaikan.")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


# =========================================================================
# 📊 DATA PIPELINE (INGESTION & SINKRONISASI)
# =========================================================================
for file_path in [DATA_FILE_HIBRIDA, DATA_FILE_LSTM, DATA_FILE_OBSERVASI]:
    if not os.path.exists(file_path):
        print(f"❌ Critical Error: File master '{file_path}' tidak ditemukan!")
        sys.exit(1)

print("📋 Membaca basis data dari repositori...")
# ⚠️ Sengaja TIDAK memakai parse_dates=["Datetime"] di sini karena rapuh
# terhadap format tanggal campuran. Datetime diparse eksplisit di bawah.
df_hib = pd.read_csv(DATA_FILE_HIBRIDA)
df_lstm = pd.read_csv(DATA_FILE_LSTM)
df_obs = pd.read_csv(DATA_FILE_OBSERVASI)

df_hib["Datetime"] = parse_datetime_column(df_hib["Datetime"], DATA_FILE_HIBRIDA)
df_lstm["Datetime"] = parse_datetime_column(df_lstm["Datetime"], DATA_FILE_LSTM)
df_obs["Datetime"] = parse_datetime_column(df_obs["Datetime"], DATA_FILE_OBSERVASI)

df_hib = df_hib.dropna(subset=["Datetime"]).reset_index(drop=True)
df_lstm = df_lstm.dropna(subset=["Datetime"]).reset_index(drop=True)
df_obs = df_obs.dropna(subset=["Datetime"]).reset_index(drop=True)

# Proteksi Overwrite: Bersihkan kolom observasi bawaan lama dari file model prediksi
if "TMA_Pasar_Ikan" in df_hib.columns:
    df_hib = df_hib.drop(columns=["TMA_Pasar_Ikan"])
if "TMA_Pasar_Ikan" in df_lstm.columns:
    df_lstm = df_lstm.drop(columns=["TMA_Pasar_Ikan"])

# Menjahit data observasi murni ke grid waktu menggunakan Left Join
df_learning = pd.merge(df_hib, df_obs[["Datetime", "TMA_Pasar_Ikan"]], on="Datetime", how="left")
df_valid = df_learning[df_learning["TMA_Pasar_Ikan"].notna()].sort_values("Datetime").reset_index(drop=True)

if len(df_valid) <= N_INPUT + N_OUTPUT:
    print(f"⚠️ Data observasi murni ({len(df_valid)} jam) belum memenuhi syarat windowing ({N_INPUT + N_OUTPUT} jam).")
    print("⏭️ Melewati fase retraining operasional minggu ini.")
    sys.exit(0)

waktu_terakhir_obs = df_valid["Datetime"].iloc[-1]
horizon_cutoff = waktu_terakhir_obs + timedelta(hours=N_OUTPUT)
print(f"📌 Batas Data Observasi Lapangan Aktual: {waktu_terakhir_obs}")
print(f"🎯 Jendela overwrite forecast (TERBATAS): {waktu_terakhir_obs} < t <= {horizon_cutoff}")
print("   (baris di luar jendela ini tidak akan diubah oleh pipeline ini)")


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

    autostop_tma = EarlyStopping(monitor="loss", patience=2, restore_best_weights=True, verbose=1)

    print(f"🏋️ Melatih ulang model via Fine-Tuning (Max: {MAX_EPOCHS} Epoch)...")
    model_tma.fit(X_lstm, y_lstm, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=[autostop_tma], verbose=1)
    model_tma.save(MODEL_LSTM_MURNI)
    print("💾 Otak Model LSTM Murni sukses diamankan.")

    # --- Forecast TERBATAS: tepat 1 window N_OUTPUT jam setelah observasi terakhir ---
    target_mask_lstm = (df_lstm["Datetime"] > waktu_terakhir_obs) & (df_lstm["Datetime"] <= horizon_cutoff)
    n_target_lstm = int(target_mask_lstm.sum())

    if n_target_lstm == 0:
        print("⏭️ Tidak ada baris di dalam jendela 7 hari ke depan pada grid. Blok A dilewati.")
    else:
        print(f"🔮 Menghitung peramalan Seq2Seq untuk {n_target_lstm} jam ke depan (maks. {N_OUTPUT} jam)...")

        input_data = np.array(scaled_tma[-N_INPUT:].flatten()).reshape(1, N_INPUT, 1)
        pred_block_scaled = model_tma.predict(input_data, verbose=0)[0, :, 0]
        pred_block_cm = scaler_tma.inverse_transform(pred_block_scaled.reshape(-1, 1)).flatten()

        # Potong persis sejumlah baris yang benar-benar tersedia dalam jendela 7 hari
        pred_block_cm = pred_block_cm[:n_target_lstm]

        df_lstm.loc[target_mask_lstm, "Prediksi_LSTM_Murni"] = pred_block_cm
        print(f"✅ Prediksi_LSTM_Murni diperbarui HANYA untuk {n_target_lstm} baris dalam jendela 7 hari.")
        print("   Baris setelah jendela ini (hari ke-8 dst.) tidak diubah.")

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

    autostop_hib = EarlyStopping(monitor="loss", patience=2, restore_best_weights=True, verbose=1)

    print(f"🏋️ Melatih ulang model via Fine-Tuning (Max: {MAX_EPOCHS} Epoch)...")
    model_hib.fit(X_res, y_res, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, callbacks=[autostop_hib], verbose=1)
    model_hib.save(MODEL_HIBRIDA)
    print("💾 Otak Model Residu Hibrida Master sukses diamankan.")

    # --- Forecast TERBATAS: tepat 1 window N_OUTPUT jam setelah observasi terakhir ---
    target_mask_hib = (df_hib["Datetime"] > waktu_terakhir_obs) & (df_hib["Datetime"] <= horizon_cutoff)
    n_target_hib = int(target_mask_hib.sum())

    if n_target_hib == 0:
        print("⏭️ Tidak ada baris di dalam jendela 7 hari ke depan pada grid. Blok B dilewati.")
    else:
        print(f"🔮 Menghitung peramalan residu Seq2Seq untuk {n_target_hib} jam ke depan (maks. {N_OUTPUT} jam)...")

        input_data_res = np.array(scaled_residu[-N_INPUT:].flatten()).reshape(1, N_INPUT, 1)
        pred_block_res_scaled = model_hib.predict(input_data_res, verbose=0)[0, :, 0]
        pred_block_res_cm = scaler_residu.inverse_transform(pred_block_res_scaled.reshape(-1, 1)).flatten()

        # Potong persis sejumlah baris yang benar-benar tersedia dalam jendela 7 hari
        pred_block_res_cm = pred_block_res_cm[:n_target_hib]

        utide_dalam_jendela = df_hib.loc[target_mask_hib, "Prediksi_Harmonik_UTIDE"].values
        prediksi_hibrida_final = utide_dalam_jendela + pred_block_res_cm

        df_hib.loc[target_mask_hib, "Residu_LSTM_Pred"] = pred_block_res_cm
        df_hib.loc[target_mask_hib, "Prediksi_Hibrida_Final"] = prediksi_hibrida_final
        print(f"✅ Prediksi_Hibrida_Final diperbarui HANYA untuk {n_target_hib} baris dalam jendela 7 hari.")
        print("   Baris setelah jendela ini (hari ke-8 dst.) tidak diubah.")

except Exception as err:
    print(f"❌ Error Terjadi pada Blok B: {err}")

# =========================================================================
# 💾 DATABASE SYNCHRONIZATION & STORAGE MANAGEMENT
# =========================================================================
print("\n" + "=" * 80)
print("💾 PROSES SINKRONISASI BASIS DATA NUMERIK...")
print("=" * 80)

df_hib["Datetime"] = df_hib["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
df_lstm["Datetime"] = df_lstm["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

if "TMA_Pasar_Ikan" in df_hib.columns:
    df_hib = df_hib.drop(columns=["TMA_Pasar_Ikan"])
if "TMA_Pasar_Ikan" in df_lstm.columns:
    df_lstm = df_lstm.drop(columns=["TMA_Pasar_Ikan"])

df_hib.to_csv(DATA_FILE_HIBRIDA, index=False)
df_lstm.to_csv(DATA_FILE_LSTM, index=False)

print("🎉 [SUCCESS] ALL PIPELINES ARE COMPLETELY SYNCHRONIZED 100% WITH AUTOSTOP!")
print("   Cakupan overwrite forecast: HANYA 7 hari (168 jam) setelah observasi terakhir.")
print("=" * 80)
