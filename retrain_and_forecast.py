#!/usr/bin/env python3
"""
Pipeline mingguan peramalan pasang surut, stasiun Pasar Ikan:
model LSTM Murni dan model Hibrida (UTide + Seq2Seq LSTM residu).

Arsitektur tiga berkas (observasi adalah satu-satunya sumber data aktual):
    HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv   dibaca saja
    HASIL_FINAL_TESIS_PASUT_LSTM_MURNI.csv  diperbarui pada jendela 7 hari
    HASIL_FINAL_TESIS_PASUT_HIBRIDA.csv     diperbarui pada jendela 7 hari
Berkas pendukung:
    LOG_VERIFIKASI_MINGGUAN.csv             metrik ramalan lama vs observasi baru
    STATUS_PIPELINE.json                    riwayat run dan fine-tuning

Urutan kerja (ramalan selalu didahulukan agar tetap keluar bila fine-tuning lambat):
    1. Baca data, rapikan deret per jam, saring nilai mustahil.
    2. Verifikasi ramalan lama terhadap observasi baru.
    3. Ramal 7 hari setelah observasi terakhir dengan model master, simpan CSV.
    4. Putuskan fine-tuning per model (mode auto, lihat di bawah). Bila model baru
       diterima, ramalan diulang dengan model itu dan CSV ditulis ulang.

Mode fine-tuning (--finetune):
    auto   (default) jalan hanya bila ada pemicu: data baru cukup banyak sejak
           fine-tuning terakhir, atau akurasi 28 hari terakhir menurun.
    ya     paksa jalan sekarang.
    tidak  hanya meramal.
Model hasil fine-tuning dipakai hanya bila galat holdout-nya tidak lebih buruk
daripada model sebelumnya; selain itu bobot lama dipulihkan.

Opsi lain: --dry-run (tidak menulis apa pun), --base-dir.
Kode keluar: 0 sukses atau dilewati, 1 bila ada blok model yang gagal.
"""

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
FILE_STATUS = "STATUS_PIPELINE.json"

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

TRAIN_END = pd.Timestamp("2025-11-30 23:00:00")  # batas data latih notebook
N_INPUT = 336            # lookback 14 hari
N_OUTPUT = 168           # horizon dan batas overwrite: 7 hari
MAX_GAP_INTERP = 3       # gap observasi <= 3 jam diinterpolasi
BATAS_MARGIN_RANGE = 0.5  # nilai di luar [min - m*range, max + m*range] scaler dianggap mustahil

# Fine-tuning
FINETUNE_LR = 1e-5       # model master berhenti di LR 1e-6
MAX_EPOCHS = 10
PATIENCE = 3
BATCH_SIZE = 128
HOLDOUT_HARI = 28        # harus > 21 hari (satu window = 504 jam)
TOLERANSI_VAL = 0.0      # diterima bila val_mse <= val_mse_lama * (1 + toleransi)
STRIDE_LATIH = 1         # naikkan (2-6) bila runner terlalu lambat
MIN_WINDOW_LATIH = 1000
FT_BATAS_MENIT = 90      # batas waktu training, dipotong bila lewat

# Pemicu fine-tuning otomatis
FT_MIN_HARI_BARU = 56    # data baru sejak fine-tuning terakhir (atau TRAIN_END)
FT_COOLDOWN_HARI = 14    # jeda minimum antar percobaan (dihitung dari waktu data)
FT_RASIO_DRIFT = 1.25    # RMSE 28 hari / RMSE kumulatif
FT_MIN_JAM_DRIFT = 336   # sampel minimum agar deteksi drift dianggap sah

LABEL_MODE_LSTM = ("observasi", "campuran", "blind")
LABEL_MODE_HIBRIDA = ("Observasi", "Hibrid", "Rekursif")

logger = logging.getLogger("pipeline_mingguan")


# ---------------------------------------------------------------------------
# Spesifikasi model
# ---------------------------------------------------------------------------
@dataclass
class Spesifikasi:
    kunci: str
    nama: str
    kolom_metrik: str        # nama model pada tabel verifikasi
    pembanding: str          # baseline pada tabel verifikasi
    path_model: Path
    path_scaler: Path
    df: pd.DataFrame
    kolom_memori: str        # kolom ramalan lama yang dipakai sebagai memori lookback
    label_set: tuple
    bangun_seri: Callable
    tulis: Callable
    scaler: object = None
    model: object = None
    seri: pd.Series = None   # deret per jam (satuan model), gap pendek sudah terisi
    ramalan_ok: bool = False


# ---------------------------------------------------------------------------
# Berkas
# ---------------------------------------------------------------------------
def parse_datetime_column(series, label):
    """Parse Datetime tahan format campuran; baris tidak valid menjadi NaT."""
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


def tulis_json_atomik(data, path):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def simpan_model_atomik(model, path):
    tmp = path.with_name(path.stem + ".tmp.keras")
    model.save(str(tmp))
    if path.exists():
        shutil.copy2(path, path.with_name(path.stem + ".prev.keras"))
    os.replace(tmp, path)


def tulis_hasil_csv(df_hib, df_lstm, base):
    for df, nama in ((df_hib, FILE_HIBRIDA), (df_lstm, FILE_LSTM)):
        keluar = df.copy()
        keluar[COL_DATETIME] = keluar[COL_DATETIME].dt.strftime("%Y-%m-%d %H:%M:%S")
        tulis_csv_atomik(keluar, base / nama)


def baca_status(path):
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("finetune", {})
                return data
        except (OSError, ValueError):
            logger.warning("%s tidak terbaca, dianggap kosong.", path.name)
    return {"finetune": {}}


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
    """Deret observasi per jam dari observasi pertama sampai terakhir."""
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


def batas_wajar(scaler):
    """Rentang nilai yang masih wajar menurut scaler (dengan margin)."""
    margin = BATAS_MARGIN_RANGE * float(scaler.data_range_[0])
    return float(scaler.data_min_[0]) - margin, float(scaler.data_max_[0]) + margin


def saring_outlier(seri, scaler):
    """Nilai mustahil (mis. -999 dari scraper) dijadikan NaN agar tidak merusak lookback."""
    lo, hi = batas_wajar(scaler)
    buruk = (seri < lo) | (seri > hi)
    if buruk.any():
        logger.warning("%d observasi di luar rentang wajar [%.1f, %.1f] dibuang: %s",
                       int(buruk.sum()), lo, hi,
                       ", ".join(f"{t:%Y-%m-%d %H:%M}={v:.1f}" for t, v in seri[buruk].head(5).items()))
        seri = seri.mask(buruk)
    return seri


def seri_dari_df(df, kolom, idx):
    """Kolom df sebagai deret per jam sejajar dengan idx (NaN bila tidak ada)."""
    if kolom not in df.columns:
        return pd.Series(np.nan, index=idx)
    return df.set_index(COL_DATETIME)[kolom].reindex(idx)


def posisi_jendela(df, t_last):
    """Baris df pada jendela (t_last, t_last + N_OUTPUT jam] berdasarkan timestamp."""
    lead = (df[COL_DATETIME] - t_last) / pd.Timedelta(hours=1)
    mask = (lead >= 1) & (lead <= N_OUTPUT) & (lead == np.floor(lead))
    return mask.to_numpy(), lead[mask].astype(int).to_numpy() - 1


def kolom_objek(df, kolom):
    if kolom in df.columns:
        df[kolom] = df[kolom].astype(object)
    else:
        df[kolom] = pd.Series(index=df.index, dtype=object)


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


class BatasWaktu(Callback):
    """Menghentikan training bila melewati batas waktu (menjaga run CI tidak timeout)."""

    def __init__(self, menit):
        super().__init__()
        self.deadline = time.time() + menit * 60
        self.terpotong = False

    def on_train_batch_end(self, batch, logs=None):
        if time.time() > self.deadline:
            self.terpotong = True
            self.model.stop_training = True


def latih_ulang(spec, simpan):
    """
    Fine-tuning dengan holdout kronologis pada HOLDOUT_HARI terakhir.
    Window train dan holdout tidak overlap. Mengembalikan ringkasan hasil;
    spec.model berisi model terpilih (baru bila diterima, lama bila tidak).
    """
    model, scaler = spec.model, spec.scaler
    skala = scaler.transform(spec.seri.to_numpy().reshape(-1, 1)).ravel().astype("float32")

    span = N_INPUT + N_OUTPUT
    n = len(skala)
    windows = sliding_window_view(skala, span)
    valid = np.flatnonzero(~np.isnan(windows).any(axis=1))

    val_start = n - HOLDOUT_HARI * 24
    train_starts = valid[valid + span <= val_start][::STRIDE_LATIH]
    val_starts = valid[valid >= val_start]
    if len(train_starts) < MIN_WINDOW_LATIH or len(val_starts) == 0:
        logger.warning("[%s] Window tidak cukup (latih %d, holdout %d). Fine-tuning dilewati.",
                       spec.nama, len(train_starts), len(val_starts))
        return {"hasil": "dilewati"}

    w_tr, w_va = windows[train_starts], windows[val_starts]
    X_tr, y_tr = w_tr[:, :N_INPUT, None], w_tr[:, N_INPUT:, None]
    X_va, y_va = w_va[:, :N_INPUT, None], w_va[:, N_INPUT:, None]
    logger.info("[%s] Fine-tuning: %d window latih, %d window holdout (%d hari terakhir).",
                spec.nama, len(X_tr), len(X_va), HOLDOUT_HARI)

    fr = scaler.feature_range
    rmse_per_mse = float(scaler.data_range_[0]) / (fr[1] - fr[0])  # skala -> satuan asli
    bobot_lama = model.get_weights()
    model.compile(optimizer=Adam(FINETUNE_LR), loss="mean_squared_error")
    val_lama = float(model.evaluate(X_va, y_va, batch_size=256, verbose=0))

    penyimpan, batas = SimpanBobotTerbaik(), BatasWaktu(FT_BATAS_MENIT)
    model.fit(
        X_tr, y_tr,
        validation_data=(X_va, y_va),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            penyimpan,
            batas,
            EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=False),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        ],
        verbose=2,
    )
    if batas.terpotong:
        logger.warning("[%s] Training dipotong karena melewati %d menit.", spec.nama, FT_BATAS_MENIT)

    rmse_lama = float(np.sqrt(val_lama) * rmse_per_mse)
    ringkas = {"rmse_holdout_lama": rmse_lama, "terpotong": batas.terpotong}

    if penyimpan.best_weights is None:
        model.set_weights(bobot_lama)
        logger.warning("[%s] Tidak ada epoch yang selesai. Model master tidak diubah.", spec.nama)
        return {"hasil": "ditolak", **ringkas}

    rmse_baru = float(np.sqrt(penyimpan.best) * rmse_per_mse)
    ringkas.update(rmse_holdout_baru=rmse_baru, epoch_terbaik=penyimpan.best_epoch)
    logger.info("[%s] Holdout RMSE: sebelum %.4f | sesudah %.4f (epoch terbaik %d)",
                spec.nama, rmse_lama, rmse_baru, penyimpan.best_epoch)

    if penyimpan.best <= val_lama * (1 + TOLERANSI_VAL):
        model.set_weights(penyimpan.best_weights)
        if simpan:
            simpan_model_atomik(model, spec.path_model)
            logger.info("[%s] Model hasil fine-tuning diterima dan disimpan (cadangan: .prev.keras).",
                        spec.nama)
        else:
            logger.info("[%s] Model hasil fine-tuning diterima (dry-run, tidak disimpan).", spec.nama)
        return {"hasil": "diterima", **ringkas}

    model.set_weights(bobot_lama)
    logger.warning("[%s] Model hasil fine-tuning DITOLAK (holdout lebih buruk). Master tidak diubah.",
                   spec.nama)
    return {"hasil": "ditolak", **ringkas}


# ---------------------------------------------------------------------------
# Peramalan
# ---------------------------------------------------------------------------
def ramal_blok(model, scaler, seri, memori):
    """
    Ramal N_OUTPUT jam setelah indeks terakhir `seri`.

    Lookback: observasi (gap pendek sudah terisi), lalu memori ramalan lama untuk
    jam yang masih kosong. Bila tetap ada NaN ramalan dibatalkan.
    """
    if len(seri) < N_INPUT:
        raise ValueError(f"Deret hanya {len(seri)} jam, butuh minimal {N_INPUT}.")

    window = seri.iloc[-N_INPUT:]
    n_obs = int(window.notna().sum())
    window = window.fillna(memori.reindex(window.index))
    n_kosong = int(window.isna().sum())
    if n_kosong:
        raise ValueError(f"Lookback masih berisi {n_kosong} jam kosong yang tidak bisa diisi "
                         f"(gap panjang tanpa memori ramalan).")

    x = scaler.transform(window.to_numpy().reshape(-1, 1)).astype("float32").reshape(1, N_INPUT, 1)
    keluaran = model(x, training=False).numpy()[0, :, 0]
    return scaler.inverse_transform(keluaran.reshape(-1, 1)).ravel(), n_obs


def validasi_ramalan(pred, scaler, nama):
    """Ramalan yang tidak finite atau jauh di luar rentang latih tidak boleh ditulis."""
    if not np.isfinite(pred).all():
        raise ValueError(f"[{nama}] ramalan mengandung NaN/inf.")
    lo, hi = batas_wajar(scaler)
    if pred.min() < lo or pred.max() > hi:
        raise ValueError(f"[{nama}] ramalan di luar rentang wajar [{lo:.1f}, {hi:.1f}]: "
                         f"min {pred.min():.1f}, maks {pred.max():.1f}.")


def label_mode(n_obs, label_set):
    if n_obs == N_INPUT:
        return label_set[0]
    if n_obs == 0:
        return label_set[2]
    return label_set[1]


def seri_lstm(seri_obs, df):
    return fill_short_gaps(seri_obs, MAX_GAP_INTERP)


def seri_hibrida(seri_obs, df):
    residu = seri_obs - seri_dari_df(df, COL_UTIDE, seri_obs.index)
    if residu.dropna().empty:
        raise ValueError("Residu tidak bisa dihitung: UTide tidak mencakup periode observasi.")
    return fill_short_gaps(residu, MAX_GAP_INTERP)


def tulis_lstm(spec, t_last, pred, n_obs, seri_obs):
    df = spec.df
    mask, lead_idx = posisi_jendela(df, t_last)
    if not mask.any():
        logger.warning("[%s] Tidak ada baris grid pada jendela ramalan.", spec.nama)
        return 0, "-"
    label = label_mode(n_obs, spec.label_set)
    kolom_objek(df, "Mode_Input")
    df.loc[mask, COL_PRED_LSTM] = pred[lead_idx]
    df.loc[mask, "Mode_Input"] = label
    df.loc[mask, "Lead_Jam"] = lead_idx + 1

    if COL_BASELINE in df.columns:
        alas = spec.seri.reindex(df[COL_DATETIME] - pd.Timedelta(hours=N_OUTPUT)).to_numpy()
        df[COL_BASELINE] = df[COL_BASELINE].fillna(pd.Series(alas, index=df.index))
    return int(mask.sum()), label


def tulis_hibrida(spec, t_last, pred, n_obs, seri_obs):
    df = spec.df
    residu = seri_obs - seri_dari_df(df, COL_UTIDE, seri_obs.index)
    r_obs = df[COL_DATETIME].map(residu.dropna())
    if COL_RESIDU_OBS in df.columns:
        df[COL_RESIDU_OBS] = df[COL_RESIDU_OBS].fillna(r_obs)  # nilai lama tidak ditimpa
    else:
        df[COL_RESIDU_OBS] = r_obs

    mask, lead_idx = posisi_jendela(df, t_last)
    if not mask.any():
        logger.warning("[%s] Tidak ada baris grid pada jendela ramalan.", spec.nama)
        return 0, "-"
    utide = df.loc[mask, COL_UTIDE].to_numpy(dtype="float64")
    if np.isnan(utide).any():
        logger.warning("[%s] %d baris pada jendela tidak punya nilai UTide.", spec.nama,
                       int(np.isnan(utide).sum()))
    label = label_mode(n_obs, spec.label_set)
    kolom_objek(df, "Mode_Prediksi")
    df.loc[mask, COL_RESIDU_PRED] = pred[lead_idx]
    df.loc[mask, COL_PRED_HIB] = utide + pred[lead_idx]
    df.loc[mask, "Mode_Prediksi"] = label
    df.loc[mask, "Lead_Jam"] = lead_idx + 1
    return int(mask.sum()), label


def jalankan_ramalan(spec, t_last, seri_obs):
    spec.seri = spec.bangun_seri(seri_obs, spec.df)
    memori = seri_dari_df(spec.df, spec.kolom_memori, spec.seri.index)
    pred, n_obs = ramal_blok(spec.model, spec.scaler, spec.seri, memori)
    validasi_ramalan(pred, spec.scaler, spec.nama)
    n_baris, label = spec.tulis(spec, t_last, pred, n_obs, seri_obs)
    logger.info("[%s] %d baris diperbarui | mode input: %s (%d/%d jam observasi).",
                spec.nama, n_baris, label, n_obs, N_INPUT)
    spec.ramalan_ok = True


# ---------------------------------------------------------------------------
# Verifikasi
# ---------------------------------------------------------------------------
def hitung_metrik(y_true, y_pred):
    return {
        "N_Jam": int(len(y_true)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
        "R2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def hitung_verifikasi(seri_obs, seri_short, df_hib, df_lstm, t_last):
    """
    Bandingkan ramalan tersimpan (dibuat sebelum observasi ini ada) dengan observasi
    setelah TRAIN_END, pada sampel yang sama untuk semua model.
    Mengembalikan (hasil[periode][model] -> metrik, baris untuk log).
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
        return {}, []

    periode = {
        "kumulatif": tabel,
        "28_hari_terakhir": tabel[tabel.index > t_last - pd.Timedelta(days=28)],
    }
    waktu_run = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    hasil, baris = {}, []
    for nama_p, sub in periode.items():
        if sub.empty:
            continue
        hasil[nama_p] = {m: hitung_metrik(sub["Observasi"], sub[m]) for m in kolom_model}
        for m, met in hasil[nama_p].items():
            baris.append({"Waktu_Run": waktu_run, "Obs_Terakhir": t_last, "Periode": nama_p,
                          "Model": m, **met})
        ringkas = " | ".join(f"{m} {met['RMSE']:.4f}" for m, met in hasil[nama_p].items())
        logger.info("Verifikasi %-17s (%d jam) RMSE: %s", nama_p, len(sub), ringkas)
    return hasil, baris


def catat_verifikasi(baris, path_log):
    if not baris:
        return
    baru = pd.DataFrame(baris)
    if path_log.exists():
        lama = pd.read_csv(path_log)
        baru["Obs_Terakhir"] = baru["Obs_Terakhir"].astype(str)
        baru = pd.concat([lama, baru], ignore_index=True)
        baru = baru.drop_duplicates(["Obs_Terakhir", "Periode", "Model"], keep="last")
    tulis_csv_atomik(baru, path_log)


# ---------------------------------------------------------------------------
# Keputusan fine-tuning
# ---------------------------------------------------------------------------
def deteksi_drift(spec, hasil):
    m28 = hasil.get("28_hari_terakhir", {}).get(spec.kolom_metrik)
    mk = hasil.get("kumulatif", {}).get(spec.kolom_metrik)
    if not m28 or not mk or m28["N_Jam"] < FT_MIN_JAM_DRIFT:
        return None
    rasio = m28["RMSE"] / mk["RMSE"]
    if rasio > FT_RASIO_DRIFT:
        return f"RMSE 28 hari {rasio:.2f}x RMSE kumulatif"
    pembanding = hasil["28_hari_terakhir"].get(spec.pembanding)
    if pembanding and m28["RMSE"] > pembanding["RMSE"]:
        return (f"28 hari terakhir kalah dari {spec.pembanding} "
                f"({m28['RMSE']:.3f} > {pembanding['RMSE']:.3f})")
    return None


def putuskan_finetune(spec, mode, status, t_last, hasil):
    if mode == "tidak":
        return False, "dinonaktifkan manual"
    if mode == "ya":
        return True, "dipaksa manual"

    st = status.get("finetune", {}).get(spec.kunci, {})
    ref = pd.Timestamp(st["obs_saat_coba"]) if st.get("obs_saat_coba") else TRAIN_END
    hari_baru = (t_last - ref) / pd.Timedelta(days=1)

    if hari_baru < FT_COOLDOWN_HARI:
        return False, f"cooldown ({hari_baru:.0f} hari sejak percobaan terakhir, min {FT_COOLDOWN_HARI})"

    alasan = []
    if hari_baru >= FT_MIN_HARI_BARU:
        alasan.append(f"data baru {hari_baru:.0f} hari sejak {ref:%Y-%m-%d}")
    drift = deteksi_drift(spec, hasil)
    if drift:
        alasan.append(drift)
    if alasan:
        return True, "; ".join(alasan)
    return False, f"tidak ada pemicu (data baru {hari_baru:.0f} hari, tidak ada drift)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-dir", type=Path, default=Path("."),
                        help="folder berisi seluruh berkas master (default: folder kerja)")
    parser.add_argument("--finetune", choices=("auto", "ya", "tidak"), default="auto")
    parser.add_argument("--tanpa-finetune", action="store_true",
                        help="alias untuk --finetune tidak")
    parser.add_argument("--dry-run", action="store_true",
                        help="jalankan tanpa menulis CSV, model, log, atau status")
    args = parser.parse_args()
    mode = "tidak" if args.tanpa_finetune else args.finetune
    dry = args.dry_run

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                        datefmt="%H:%M:%S")
    t_mulai = time.time()
    base = args.base_dir
    tf.keras.utils.set_random_seed(SEED)
    if HOLDOUT_HARI * 24 <= N_INPUT + N_OUTPUT:
        logger.warning("HOLDOUT_HARI terlalu kecil: tidak ada window validasi, fine-tuning akan selalu dilewati.")
    if dry:
        logger.info("MODE DRY-RUN: tidak ada berkas yang ditulis.")

    # --- data ---
    try:
        df_hib = baca_csv(base / FILE_HIBRIDA, FILE_HIBRIDA)
        df_lstm = baca_csv(base / FILE_LSTM, FILE_LSTM)
        df_obs = baca_csv(base / FILE_OBSERVASI, FILE_OBSERVASI)
        seri_obs = bangun_seri_obs(df_obs)
    except (FileNotFoundError, ValueError) as err:
        logger.error("%s", err)
        sys.exit(1)
    df_hib = df_hib.drop(columns=[COL_TMA], errors="ignore")
    df_lstm = df_lstm.drop(columns=[COL_TMA], errors="ignore")

    status = baca_status(base / FILE_STATUS)
    gagal = []

    specs = [
        Spesifikasi("lstm", "LSTM Murni", "LSTM_Murni", "Naive_168h",
                    base / MODEL_LSTM, base / SCALER_LSTM, df_lstm, COL_PRED_LSTM,
                    LABEL_MODE_LSTM, seri_lstm, tulis_lstm),
        Spesifikasi("hibrida", "Hibrida", "Hibrida", "Harmonik",
                    base / MODEL_HIBRIDA, base / SCALER_HIBRIDA, df_hib, COL_RESIDU_PRED,
                    LABEL_MODE_HIBRIDA, seri_hibrida, tulis_hibrida),
    ]
    for spec in specs:
        try:
            spec.scaler = joblib.load(spec.path_scaler)
        except Exception:
            logger.exception("[%s] Scaler gagal dimuat.", spec.nama)
            gagal.append(spec.nama)

    if specs[0].scaler is not None:  # scaler TMA dipakai untuk menyaring nilai mustahil
        seri_obs = saring_outlier(seri_obs, specs[0].scaler)
    seri_obs = seri_obs.loc[:seri_obs.last_valid_index()]
    t_last = seri_obs.index[-1]

    if len(seri_obs) < N_INPUT:
        logger.warning("Observasi hanya %d jam (butuh %d). Pipeline dilewati.", len(seri_obs), N_INPUT)
        sys.exit(0)
    if mode == "auto" and status.get("obs_terakhir_sukses") == str(t_last):
        logger.info("Tidak ada observasi baru sejak run sukses terakhir (%s). Dilewati.", t_last)
        sys.exit(0)

    seri_short = fill_short_gaps(seri_obs, MAX_GAP_INTERP)
    logger.info("Observasi: %d jam grid, %d jam kosong, terakhir %s",
                len(seri_obs), int(seri_obs.isna().sum()), t_last)
    logger.info("Jendela overwrite: %s < t <= %s (di luar jendela tidak diubah)",
                t_last, t_last + pd.Timedelta(hours=N_OUTPUT))

    # --- verifikasi ramalan lama ---
    hasil_verif = {}
    try:
        hasil_verif, baris = hitung_verifikasi(seri_obs, seri_short, df_hib, df_lstm, t_last)
        if not dry:
            catat_verifikasi(baris, base / FILE_LOG_VERIFIKASI)
    except Exception:
        logger.exception("Verifikasi gagal (tidak menghentikan pipeline).")

    # --- ramalan dengan model master ---
    for spec in specs:
        if spec.nama in gagal:
            continue
        logger.info("-" * 60)
        try:
            spec.model = tf.keras.models.load_model(spec.path_model)
            jalankan_ramalan(spec, t_last, seri_obs)
        except Exception:
            logger.exception("[%s] Ramalan gagal.", spec.nama)
            gagal.append(spec.nama)
    if not dry:
        tulis_hasil_csv(df_hib, df_lstm, base)

    # --- fine-tuning bila diperlukan ---
    sekarang = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    for spec in specs:
        if not spec.ramalan_ok:
            continue
        perlu, alasan = putuskan_finetune(spec, mode, status, t_last, hasil_verif)
        logger.info("-" * 60)
        logger.info("[%s] Fine-tuning: %s (%s)", spec.nama, "YA" if perlu else "tidak", alasan)
        if not perlu:
            continue

        try:
            info = latih_ulang(spec, simpan=not dry)
            if info["hasil"] == "diterima":
                jalankan_ramalan(spec, t_last, seri_obs)  # ulang dengan model baru
                if not dry:
                    tulis_hasil_csv(df_hib, df_lstm, base)
        except Exception as err:
            logger.exception("[%s] Fine-tuning gagal, memakai model master.", spec.nama)
            info = {"hasil": "gagal", "error": str(err)[:200]}
            try:
                spec.model = tf.keras.models.load_model(spec.path_model)
            except Exception:
                pass
        status["finetune"][spec.kunci] = {"terakhir_coba": sekarang, "obs_saat_coba": str(t_last),
                                          "alasan": alasan, **info}

    # --- status ---
    status["terakhir_run"] = sekarang
    if not gagal:
        status["obs_terakhir_sukses"] = str(t_last)
    if not dry:
        tulis_json_atomik(status, base / FILE_STATUS)

    logger.info("=" * 60)
    if gagal:
        logger.error("Selesai dengan kegagalan pada: %s", ", ".join(dict.fromkeys(gagal)))
        sys.exit(1)
    logger.info("Selesai dalam %.1f menit. Ramalan 7 hari setelah %s diperbarui.",
                (time.time() - t_mulai) / 60, t_last)


if __name__ == "__main__":
    main()
