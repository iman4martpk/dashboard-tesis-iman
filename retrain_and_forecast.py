#!/usr/bin/env python3
"""
Pipeline mingguan: fine-tuning terkontrol dan peramalan 7 hari untuk model
LSTM Murni dan Hibrida (UTide + Seq2Seq LSTM residu), stasiun Pasar Ikan.

Arsitektur tiga berkas (observasi adalah satu-satunya sumber data aktual):
    HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv   dibaca saja, tidak pernah ditulis
    HASIL_FINAL_TESIS_PASUT_LSTM_MURNI.csv  diperbarui pada jendela 7 hari
    HASIL_FINAL_TESIS_PASUT_HIBRIDA.csv     diperbarui pada jendela 7 hari

Alur tiap eksekusi:
    1. Baca ketiga berkas, rapikan datetime dan deret observasi per jam.
    2. Verifikasi ramalan lama terhadap observasi baru, simpan ke log.
    3. Fine-tuning dengan holdout kronologis. Model baru dipakai hanya jika
       galat holdout-nya tidak lebih buruk daripada model sebelumnya.
    4. Ramal tepat N_OUTPUT jam setelah observasi terakhir. Baris di luar
       jendela itu tidak disentuh.

Catatan perubahan terhadap versi sebelumnya:
    - Deret observasi dibentuk ulang menjadi grid per jam. Versi lama membuang
      baris NaN lalu mengambil 336 baris terakhir, sehingga lookback bisa
      melompati gap waktu dan tidak sejajar dengan jam sebenarnya.
    - Window training hanya dari rentang kontinu tanpa NaN (sama dengan notebook).
    - Fine-tuning memakai validasi holdout dan learning rate kecil. Versi lama
      memantau training loss tanpa validasi dan selalu menimpa model master.
    - Lookback yang berlubang diisi dengan interpolasi gap pendek, lalu memori
      prediksi sebelumnya (perilaku yang sama dengan notebook rolling forecast).
    - Penempatan hasil ramalan berdasarkan timestamp, bukan irisan [:n].
    - Kolom Mode, Lead_Jam, Residu_Murni, dan Baseline ikut diperbarui.
    - Kegagalan satu blok tidak lagi tertelan: exit code 1 dan berkas tetap utuh.
    - Penulisan CSV dan model bersifat atomik, model sebelumnya dicadangkan.
"""

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau  # noqa: E402
from tensorflow.keras.optimizers import Adam  # noqa: E402

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
SEED = 42

FILE_HIBRIDA = "HASIL_FINAL_TESIS_PASUT_HIBRIDA.csv"
FILE_LSTM = "HASIL_FINAL_TESIS_PASUT_LSTM_MURNI.csv"
FILE_OBSERVASI = "HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv"
FILE_LOG_VERIFIKASI = "LOG_VERIFIKASI_MINGGUAN.csv"

MODEL_LSTM = "model_pasar_ikan_lstm_murni_master.keras"
MODEL_HIBRIDA = "model_pasar_ikan_master.keras"
SCALER_LSTM = "scaler_tma_pasar_ikan_lstm_murni.save"
SCALER_HIBRIDA = "scaler_residu_pasar_ikan.save"

COL_DATETIME = "Datetime"
COL_TMA = "TMA_Pasar_Ikan"
COL_UTIDE = "Prediksi_Harmonik_UTIDE"
COL_PRED_LSTM = "Prediksi_LSTM_Murni"
COL_PRED_HIB = "Prediksi_Hibrida_Final"
COL_RESIDU_PRED = "Residu_LSTM_Pred"
COL_RESIDU_OBS = "Residu_Murni"
COL_BASELINE = "Baseline_Naive_168h"

TRAIN_END = pd.Timestamp("2025-11-30 23:00:00")  # sama dengan notebook training
N_INPUT = 336            # lookback 14 hari
N_OUTPUT = 168           # horizon dan batas overwrite: 7 hari
MAX_GAP_INTERP = 3       # gap observasi <= 3 jam diinterpolasi

# Fine-tuning
FINETUNE_LR = 1e-5       # model master berhenti di LR 1e-6; 1e-5 cukup untuk adaptasi tanpa merusak bobot
MAX_EPOCHS = 10
PATIENCE = 3
BATCH_SIZE = 128
HOLDOUT_HARI = 28        # periode terakhir yang tidak ikut training, dipakai sebagai gerbang
TOLERANSI_VAL = 0.0      # model baru diterima bila val_mse <= val_mse_lama * (1 + toleransi)
STRIDE_LATIH = 1         # naikkan (2-6) bila runner terlalu lambat
MIN_WINDOW_LATIH = 1000

LABEL_MODE_LSTM = ("observasi", "campuran", "blind")
LABEL_MODE_HIBRIDA = ("Observasi", "Hibrid", "Rekursif")

logger = logging.getLogger("pipeline_mingguan")


# ---------------------------------------------------------------------------
# Utilitas berkas
# ---------------------------------------------------------------------------
def parse_datetime_column(series, label):
    """
    Parse kolom Datetime yang tahan format campuran (mis. "%m/%d/%Y %H:%M"
    bercampur "%Y-%m-%d %H:%M:%S"). Baris tidak valid menjadi NaT dan dilaporkan.
    """
    parsed = pd.to_datetime(series, format="mixed", dayfirst=False, errors="coerce")
    n_invalid = int(parsed.isna().sum())
    if n_invalid:
        logger.warning("[%s] %d baris punya format Datetime tidak valid, dibuang.", label, n_invalid)
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


def baca_csv(path, label):
    if not path.exists():
        raise FileNotFoundError(f"Berkas master tidak ditemukan: {path}")
    df = pd.read_csv(path)
    if COL_DATETIME not in df.columns:
        raise ValueError(f"[{label}] kolom '{COL_DATETIME}' tidak ada.")
    df[COL_DATETIME] = parse_datetime_column(df[COL_DATETIME], label)
    df = df.dropna(subset=[COL_DATETIME]).sort_values(COL_DATETIME)

    n_dup = int(df[COL_DATETIME].duplicated().sum())
    if n_dup:
        logger.warning("[%s] %d timestamp duplikat, nilai terakhir dipertahankan.", label, n_dup)
        df = df.drop_duplicates(COL_DATETIME, keep="last")
    return df.reset_index(drop=True)


def tulis_csv_atomik(df, path):
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def simpan_model_atomik(model, path):
    tmp = path.with_name(path.stem + ".tmp.keras")
    model.save(str(tmp))
    if path.exists():
        shutil.copy2(path, path.with_name(path.stem + ".prev.keras"))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Deret waktu
# ---------------------------------------------------------------------------
def fill_short_gaps(series, max_gap):
    """Interpolasi linear hanya untuk gap berurutan sepanjang <= max_gap."""
    is_na = series.isna()
    run_len = is_na.groupby((~is_na).cumsum()).transform("sum")
    interp = series.interpolate(method="linear", limit_area="inside")
    out = series.copy()
    mask = is_na & (run_len <= max_gap)
    out[mask] = interp[mask]
    return out


def bangun_seri_obs(df_obs):
    """Deret observasi per jam dari observasi pertama sampai observasi terakhir."""
    if COL_TMA not in df_obs.columns:
        raise ValueError(f"Kolom '{COL_TMA}' tidak ada di berkas observasi.")
    s = df_obs[[COL_DATETIME, COL_TMA]].copy()
    s[COL_TMA] = pd.to_numeric(s[COL_TMA], errors="coerce")
    s = s.dropna()

    bukan_jam = (s[COL_DATETIME].dt.minute != 0) | (s[COL_DATETIME].dt.second != 0)
    if bukan_jam.any():
        logger.warning("%d observasi tidak tepat pada jam penuh, diabaikan.", int(bukan_jam.sum()))
        s = s[~bukan_jam]
    if s.empty:
        raise ValueError("Tidak ada observasi valid.")

    s = s.set_index(COL_DATETIME)[COL_TMA]
    idx = pd.date_range(s.index.min(), s.index.max(), freq="h")
    return s.reindex(idx)


def seri_dari_df(df, kolom, idx):
    """Ambil kolom df sebagai deret per jam yang sejajar dengan idx (NaN bila tidak ada)."""
    if kolom not in df.columns:
        return pd.Series(np.nan, index=idx)
    return df.set_index(COL_DATETIME)[kolom].reindex(idx)


def posisi_jendela(df, t_last):
    """
    Baris df yang masuk jendela (t_last, t_last + N_OUTPUT jam], berdasarkan
    timestamp. Mengembalikan mask baris dan indeks lead (0..N_OUTPUT-1) tiap baris.
    """
    lead = (df[COL_DATETIME] - t_last) / pd.Timedelta(hours=1)
    mask = (lead >= 1) & (lead <= N_OUTPUT) & (lead == np.floor(lead))
    return mask.to_numpy(), lead[mask].astype(int).to_numpy() - 1


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------
class SimpanBobotTerbaik(Callback):
    """Menyimpan bobot di epoch dengan val_loss terendah (tidak bergantung versi Keras)."""

    def __init__(self):
        super().__init__()
        self.best = np.inf
        self.best_epoch = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        nilai = (logs or {}).get("val_loss")
        if nilai is not None and nilai < self.best:
            self.best = float(nilai)
            self.best_epoch = epoch + 1
            self.best_weights = self.model.get_weights()


def latih_ulang(model, scaler, seri_skala, path_model, label):
    """
    Fine-tuning dengan holdout kronologis pada HOLDOUT_HARI terakhir.

    Window train dan holdout tidak overlap. Model hasil fine-tuning disimpan
    hanya bila val MSE-nya tidak lebih buruk daripada model sebelum fine-tuning
    pada holdout yang sama; selain itu bobot lama dipulihkan.
    """
    span = N_INPUT + N_OUTPUT
    n = len(seri_skala)
    windows = sliding_window_view(seri_skala, span)
    valid = np.flatnonzero(~np.isnan(windows).any(axis=1))

    val_start = n - HOLDOUT_HARI * 24
    train_starts = valid[valid + span <= val_start][::STRIDE_LATIH]
    val_starts = valid[valid >= val_start]
    if len(train_starts) < MIN_WINDOW_LATIH or len(val_starts) == 0:
        logger.warning("[%s] Window tidak cukup (latih %d, holdout %d). Fine-tuning dilewati.",
                       label, len(train_starts), len(val_starts))
        return model

    w_tr, w_va = windows[train_starts], windows[val_starts]
    X_tr, y_tr = w_tr[:, :N_INPUT, None], w_tr[:, N_INPUT:, None]
    X_va, y_va = w_va[:, :N_INPUT, None], w_va[:, N_INPUT:, None]
    logger.info("[%s] Fine-tuning: %d window latih, %d window holdout (%d hari terakhir).",
                label, len(X_tr), len(X_va), HOLDOUT_HARI)

    rmse_per_mse = float(scaler.data_range_[0]) / 2  # MinMax (-1, 1): satuan asli = skala * range/2
    bobot_lama = model.get_weights()
    model.compile(optimizer=Adam(FINETUNE_LR), loss="mean_squared_error")
    val_lama = float(model.evaluate(X_va, y_va, batch_size=256, verbose=0))

    penyimpan = SimpanBobotTerbaik()
    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=False),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
            penyimpan,
        ],
        verbose=2,
    )

    val_baru = penyimpan.best
    logger.info("[%s] Holdout RMSE: sebelum %.4f | sesudah %.4f (epoch terbaik %d)",
                label, np.sqrt(val_lama) * rmse_per_mse, np.sqrt(val_baru) * rmse_per_mse,
                penyimpan.best_epoch)

    if penyimpan.best_weights is not None and val_baru <= val_lama * (1 + TOLERANSI_VAL):
        model.set_weights(penyimpan.best_weights)
        simpan_model_atomik(model, path_model)
        logger.info("[%s] Model hasil fine-tuning diterima dan disimpan (cadangan: .prev.keras).", label)
    else:
        model.set_weights(bobot_lama)
        logger.warning("[%s] Model hasil fine-tuning DITOLAK (holdout lebih buruk). "
                       "Model master tidak diubah.", label)
    return model


# ---------------------------------------------------------------------------
# Peramalan
# ---------------------------------------------------------------------------
def ramal_blok(model, scaler, seri_short, memori):
    """
    Ramal N_OUTPUT jam setelah indeks terakhir seri_short.

    Lookback: observasi (gap pendek sudah diinterpolasi), lalu memori ramalan
    sebelumnya untuk jam yang masih kosong. Bila tetap ada NaN, ramalan dibatalkan
    karena memasukkan data karangan ke model lebih berbahaya daripada tidak meramal.
    """
    if len(seri_short) < N_INPUT:
        raise ValueError(f"Deret hanya {len(seri_short)} jam, butuh minimal {N_INPUT}.")

    window = seri_short.iloc[-N_INPUT:]
    n_obs = int(window.notna().sum())
    if memori is not None:
        window = window.fillna(memori.reindex(window.index))
    n_kosong = int(window.isna().sum())
    if n_kosong:
        raise ValueError(f"Lookback masih berisi {n_kosong} jam kosong yang tidak bisa diisi "
                         f"(gap panjang tanpa memori ramalan).")

    x = scaler.transform(window.to_numpy().reshape(-1, 1)).astype("float32").reshape(1, N_INPUT, 1)
    keluaran = model(x, training=False).numpy()[0, :, 0]
    return scaler.inverse_transform(keluaran.reshape(-1, 1)).ravel(), n_obs


def label_mode(n_obs, label_set):
    if n_obs == N_INPUT:
        return label_set[0]
    if n_obs == 0:
        return label_set[2]
    return label_set[1]


def laporkan_di_luar_rentang(scaler, seri, label):
    nilai = seri.dropna().tail(HOLDOUT_HARI * 24).to_numpy().reshape(-1, 1)
    if len(nilai):
        skala = scaler.transform(nilai)
        persen = float(np.mean((skala < -1) | (skala > 1)) * 100)
        if persen > 0:
            logger.warning("[%s] %.2f%% data %d hari terakhir di luar rentang scaler.",
                           label, persen, HOLDOUT_HARI)


def blok_lstm_murni(base, df_lstm, seri_short, t_last, finetune):
    scaler = joblib.load(base / SCALER_LSTM)
    model = tf.keras.models.load_model(base / MODEL_LSTM)
    laporkan_di_luar_rentang(scaler, seri_short, "LSTM Murni")

    if finetune:
        try:
            skala = scaler.transform(seri_short.to_numpy().reshape(-1, 1)).ravel().astype("float32")
            model = latih_ulang(model, scaler, skala, base / MODEL_LSTM, "LSTM Murni")
        except Exception:
            logger.exception("[LSTM Murni] Fine-tuning gagal, memakai model master apa adanya.")
            model = tf.keras.models.load_model(base / MODEL_LSTM)

    memori = seri_dari_df(df_lstm, COL_PRED_LSTM, seri_short.index)
    pred, n_obs = ramal_blok(model, scaler, seri_short, memori)

    mask, lead_idx = posisi_jendela(df_lstm, t_last)
    if not mask.any():
        logger.warning("[LSTM Murni] Tidak ada baris grid pada jendela ramalan.")
        return
    label = label_mode(n_obs, LABEL_MODE_LSTM)
    df_lstm["Mode_Input"] = df_lstm.get("Mode_Input", pd.Series(index=df_lstm.index, dtype=object)).astype(object)
    df_lstm.loc[mask, COL_PRED_LSTM] = pred[lead_idx]
    df_lstm.loc[mask, "Mode_Input"] = label
    df_lstm.loc[mask, "Lead_Jam"] = lead_idx + 1

    if COL_BASELINE in df_lstm.columns:
        alas = seri_short.reindex(df_lstm[COL_DATETIME] - pd.Timedelta(hours=N_OUTPUT)).to_numpy()
        df_lstm[COL_BASELINE] = df_lstm[COL_BASELINE].fillna(pd.Series(alas, index=df_lstm.index))

    logger.info("[LSTM Murni] %d baris diperbarui | mode input: %s (%d/%d jam observasi).",
                int(mask.sum()), label, n_obs, N_INPUT)


def blok_hibrida(base, df_hib, seri_obs, t_last, finetune):
    idx = seri_obs.index
    utide = seri_dari_df(df_hib, COL_UTIDE, idx)
    residu = seri_obs - utide
    if residu.dropna().empty:
        raise ValueError("Residu tidak bisa dihitung: UTide tidak mencakup periode observasi.")
    residu_short = fill_short_gaps(residu, MAX_GAP_INTERP)

    scaler = joblib.load(base / SCALER_HIBRIDA)
    model = tf.keras.models.load_model(base / MODEL_HIBRIDA)
    laporkan_di_luar_rentang(scaler, residu_short, "Hibrida")

    if finetune:
        try:
            skala = scaler.transform(residu_short.to_numpy().reshape(-1, 1)).ravel().astype("float32")
            model = latih_ulang(model, scaler, skala, base / MODEL_HIBRIDA, "Hibrida")
        except Exception:
            logger.exception("[Hibrida] Fine-tuning gagal, memakai model master apa adanya.")
            model = tf.keras.models.load_model(base / MODEL_HIBRIDA)

    memori = seri_dari_df(df_hib, COL_RESIDU_PRED, idx)
    pred_res, n_obs = ramal_blok(model, scaler, residu_short, memori)

    # Residu_Murni untuk baris yang baru punya observasi (nilai lama tidak ditimpa)
    r_obs = df_hib[COL_DATETIME].map(residu.dropna())
    if COL_RESIDU_OBS in df_hib.columns:
        df_hib[COL_RESIDU_OBS] = df_hib[COL_RESIDU_OBS].fillna(r_obs)
    else:
        df_hib[COL_RESIDU_OBS] = r_obs

    mask, lead_idx = posisi_jendela(df_hib, t_last)
    if not mask.any():
        logger.warning("[Hibrida] Tidak ada baris grid pada jendela ramalan.")
        return
    utide_jendela = df_hib.loc[mask, COL_UTIDE].to_numpy(dtype="float64")
    if np.isnan(utide_jendela).any():
        logger.warning("[Hibrida] %d baris pada jendela tidak punya nilai UTide.",
                       int(np.isnan(utide_jendela).sum()))

    label = label_mode(n_obs, LABEL_MODE_HIBRIDA)
    df_hib["Mode_Prediksi"] = df_hib.get("Mode_Prediksi", pd.Series(index=df_hib.index, dtype=object)).astype(object)
    df_hib.loc[mask, COL_RESIDU_PRED] = pred_res[lead_idx]
    df_hib.loc[mask, COL_PRED_HIB] = utide_jendela + pred_res[lead_idx]
    df_hib.loc[mask, "Mode_Prediksi"] = label
    df_hib.loc[mask, "Lead_Jam"] = lead_idx + 1

    logger.info("[Hibrida] %d baris diperbarui | mode input: %s (%d/%d jam observasi).",
                int(mask.sum()), label, n_obs, N_INPUT)


# ---------------------------------------------------------------------------
# Verifikasi ramalan lama terhadap observasi baru
# ---------------------------------------------------------------------------
def hitung_metrik(y_true, y_pred):
    return {
        "N_Jam": int(len(y_true)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def verifikasi(seri_obs, seri_short, df_hib, df_lstm, t_last, path_log):
    """
    Bandingkan ramalan yang sudah tersimpan (dibuat sebelum observasi ini ada)
    dengan observasi aktual setelah TRAIN_END. Sampel dibuat identik untuk semua
    model agar perbandingannya adil. Hasil ditambahkan ke log mingguan.
    """
    idx = seri_obs.index
    tabel = pd.DataFrame({
        "Observasi": seri_obs,
        "LSTM_Murni": seri_dari_df(df_lstm, COL_PRED_LSTM, idx),
        "Hibrida": seri_dari_df(df_hib, COL_PRED_HIB, idx),
        "Harmonik": seri_dari_df(df_hib, COL_UTIDE, idx),
        "Naive_168h": seri_short.shift(N_OUTPUT),
    })
    tabel = tabel[tabel.index > TRAIN_END]
    kolom_model = [c for c in tabel.columns[1:] if tabel[c].notna().any()]
    tabel = tabel.dropna(subset=["Observasi"] + kolom_model)
    if tabel.empty or not kolom_model:
        logger.info("Verifikasi dilewati: belum ada ramalan yang beririsan dengan observasi.")
        return

    periode = {
        "kumulatif": tabel,
        "28_hari_terakhir": tabel[tabel.index > t_last - pd.Timedelta(days=28)],
    }
    waktu_run = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    baris = []
    for nama_p, sub in periode.items():
        if sub.empty:
            continue
        hasil = {m: hitung_metrik(sub["Observasi"], sub[m]) for m in kolom_model}
        for m, met in hasil.items():
            baris.append({"Waktu_Run": waktu_run, "Obs_Terakhir": t_last, "Periode": nama_p,
                          "Model": m, **met})
        ringkas = " | ".join(f"{m} {met['RMSE']:.4f}" for m, met in hasil.items())
        logger.info("Verifikasi %-17s (%d jam) RMSE: %s", nama_p, len(sub), ringkas)

        if {"Hibrida", "Harmonik"} <= hasil.keys() and hasil["Hibrida"]["RMSE"] > hasil["Harmonik"]["RMSE"]:
            logger.warning("Periode %s: hibrida tidak lebih baik daripada harmonik murni.", nama_p)
        if {"LSTM_Murni", "Naive_168h"} <= hasil.keys() and hasil["LSTM_Murni"]["RMSE"] > hasil["Naive_168h"]["RMSE"]:
            logger.warning("Periode %s: LSTM murni tidak lebih baik daripada baseline naive 168 jam.", nama_p)

    baru = pd.DataFrame(baris)
    if path_log.exists():
        lama = pd.read_csv(path_log)
        baru = pd.concat([lama, baru], ignore_index=True)
        baru["Obs_Terakhir"] = baru["Obs_Terakhir"].astype(str)
        baru = baru.drop_duplicates(["Obs_Terakhir", "Periode", "Model"], keep="last")
    tulis_csv_atomik(baru, path_log)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", type=Path, default=Path("."),
                        help="folder berisi seluruh berkas master (default: folder kerja)")
    parser.add_argument("--tanpa-finetune", action="store_true",
                        help="hanya meramal dengan model master, tanpa fine-tuning")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    t_mulai = time.time()
    base = args.base_dir
    tf.keras.utils.set_random_seed(SEED)

    # --- data ---
    try:
        df_hib = baca_csv(base / FILE_HIBRIDA, FILE_HIBRIDA)
        df_lstm = baca_csv(base / FILE_LSTM, FILE_LSTM)
        df_obs = baca_csv(base / FILE_OBSERVASI, FILE_OBSERVASI)
    except (FileNotFoundError, ValueError) as err:
        logger.error("%s", err)
        sys.exit(1)

    # observasi hanya boleh berasal dari berkas observasi
    df_hib = df_hib.drop(columns=[COL_TMA], errors="ignore")
    df_lstm = df_lstm.drop(columns=[COL_TMA], errors="ignore")

    seri_obs = bangun_seri_obs(df_obs)
    t_last = seri_obs.index[-1]
    if len(seri_obs) < N_INPUT:
        logger.warning("Observasi hanya %d jam (butuh %d). Pipeline dilewati.", len(seri_obs), N_INPUT)
        sys.exit(0)
    seri_short = fill_short_gaps(seri_obs, MAX_GAP_INTERP)

    logger.info("Observasi: %d jam grid, %d jam kosong, terakhir %s",
                len(seri_obs), int(seri_obs.isna().sum()), t_last)
    logger.info("Jendela overwrite: %s < t <= %s (di luar jendela tidak diubah)",
                t_last, t_last + pd.Timedelta(hours=N_OUTPUT))

    # --- verifikasi ramalan lama ---
    try:
        verifikasi(seri_obs, seri_short, df_hib, df_lstm, t_last, base / FILE_LOG_VERIFIKASI)
    except Exception:
        logger.exception("Verifikasi gagal (tidak menghentikan pipeline).")

    # --- blok model ---
    finetune = not args.tanpa_finetune
    gagal = []

    logger.info("-" * 60)
    try:
        blok_lstm_murni(base, df_lstm, seri_short, t_last, finetune)
    except Exception:
        logger.exception("[LSTM Murni] Blok gagal.")
        gagal.append("LSTM Murni")

    logger.info("-" * 60)
    try:
        blok_hibrida(base, df_hib, seri_obs, t_last, finetune)
    except Exception:
        logger.exception("[Hibrida] Blok gagal.")
        gagal.append("Hibrida")

    # --- simpan ---
    for df, nama in ((df_hib, FILE_HIBRIDA), (df_lstm, FILE_LSTM)):
        keluar = df.copy()
        keluar[COL_DATETIME] = keluar[COL_DATETIME].dt.strftime("%Y-%m-%d %H:%M:%S")
        tulis_csv_atomik(keluar, base / nama)

    logger.info("=" * 60)
    if gagal:
        logger.error("Selesai dengan kegagalan pada: %s", ", ".join(gagal))
        sys.exit(1)
    logger.info("Selesai dalam %.1f menit. Ramalan 7 hari setelah %s diperbarui.",
                (time.time() - t_mulai) / 60, t_last)


if __name__ == "__main__":
    main()
