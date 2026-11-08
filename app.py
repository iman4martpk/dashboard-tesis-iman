"""
Dashboard Monitoring Pasut Hibrida (UTide + LSTM) - Stasiun Pasar Ikan, Jakarta.

Aplikasi Streamlit ini menampilkan perbandingan performa pendekatan
prediksi pasang surut air laut dengan data observasi independen secara REAL-TIME.

Catatan versi desain (v3, padat): seluruh logika data, metrik, filter, dan widget
sama dengan versi sebelumnya. Yang berubah hanya tampilan: tata letak lebih rapat,
informasi pendukung (periode aktif, zona siaga, metode, sumber data) dipindah ke
sidebar, dan informasi yang berulang di halaman utama dihapus.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# =========================================================================
# 1. KONSTANTA & KONFIGURASI GLOBAL
# =========================================================================

PAGE_TITLE = "Dashboard Pasut Hibrida Pasar Ikan"
PAGE_ICON = "🌊"

TZ_JKT = pytz.timezone("Asia/Jakarta")

def get_now_jkt() -> datetime:
    return datetime.now(TZ_JKT).replace(tzinfo=None)

# File Master Data
DATA_FILE_HIBRIDA = "HASIL_FINAL_TESIS_PASUT_HIBRIDA.csv"
DATA_FILE_LSTM = "HASIL_FINAL_TESIS_PASUT_LSTM_MURNI.csv"
DATA_FILE_OBSERVASI = "HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv"
DATA_FILE_15MIN = "HASIL_SCRAPING_15_MENITAN_DSDA.csv" # File baru

COL_DATETIME = "Datetime"
COL_OBSERVASI = "TMA_Pasar_Ikan"
COL_UTIDE = "Prediksi_Harmonik_UTIDE"
COL_LSTM = "Prediksi_LSTM_Murni"
COL_HIBRIDA = "Prediksi_Hibrida_Final"

# Palet warna: seri data memakai skema yang aman bagi pembaca buta warna
# (biru, hijau kebiruan, biru muda, magenta) dan dibedakan juga lewat gaya garis.
COLOR_PALETTE = {
    "primary": "#12263A",
    "success": "#0B6B4F",
    "danger": "#B42318",
    "observasi": "#12263A",
    "utide": "#56B4E9",
    "lstm": "#009E73",
    "hibrida": "#1F4E9C",
    "live_15m": "#B8337A",
    "aman": "#9CCBE3",
    "waspada": "#F2B35E",
    "awas": "#E5484D",
}

ALERT_ZONES = [
    (0, 200, COLOR_PALETTE["aman"], "Aman", 0.22),
    (201, 249, COLOR_PALETTE["waspada"], "Waspada rob", 0.30),
    (250, 500, COLOR_PALETTE["awas"], "Awas rob", 0.22),
]
ZONE_CLASSES = ["aman", "waspada", "awas"]

HARI_INI = get_now_jkt()
START_REALTIME = (HARI_INI - timedelta(days=2)).strftime("%Y-%m-%d 00:00:00")
END_REALTIME = (HARI_INI + timedelta(days=3)).strftime("%Y-%m-%d 23:00:00")

PRESETS = {
    "Ringkasan Real-Time (Jendela 5 Hari)": {
        "start": START_REALTIME,
        "end": END_REALTIME,
        "desc": "Jendela tampilan H-2 s.d. H+3.",
    },
    "Studi Kasus 1: Periode Desember 2025": {
        "start": "2025-12-02 00:00:00",
        "end": "2025-12-09 00:00:00",
        "desc": "Segmen Musim Baratan dan puncak musim hujan.",
    },
    "Studi Kasus 2: Periode Januari 2026": {
        "start": "2025-12-31 00:00:00",
        "end": "2026-01-06 00:00:00",
        "desc": "Segmen Musim Baratan dan puncak musim hujan.",
    },
    "Studi Kasus 3: Periode Juni 2026": {
        "start": "2026-06-12 00:00:00",
        "end": "2026-06-19 00:00:00",
        "desc": "Segmen Musim Timuran dan minim curah hujan.",
    },
    "Studi Kasus 4: Periode Juli 2026": {
        "start": "2026-07-10 00:00:00",
        "end": "2026-07-17 00:00:00",
        "desc": "Segmen Musim Timuran dan minim curah hujan.",
    },
    "Mode Forecasting Masa Depan": {
        "start": "2026-07-21 19:00:00",
        "end": "2026-12-31 23:00:00",
        "desc": "Peramalan estafet bergulir jangka panjang.",
    },
    "Custom Rentang Waktu (Manual)": {
        "start": None,
        "end": None,
        "desc": "Bebas menentukan rentang analisis tanggal sendiri.",
    },
}
CUSTOM_PRESET_KEY = "Custom Rentang Waktu (Manual)"
DEFAULT_PRESET_INDEX = 0


# =========================================================================
# 2. FUNGSI READ REAL-TIME POSKO BANJIR DSDA
# =========================================================================
def _parse_datetime_column(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="mixed", dayfirst=False, errors="coerce")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed

@st.cache_data(ttl=60)
def fetch_15min_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_FILE_15MIN)
        if df.empty: return pd.DataFrame()
        df[COL_DATETIME] = _parse_datetime_column(df[COL_DATETIME])
        return df.dropna(subset=[COL_DATETIME, COL_OBSERVASI]).sort_values(COL_DATETIME)
    except Exception:
        return pd.DataFrame()

def fetch_realtime_status() -> Optional[dict]:
    """Membaca nilai Teks Notifikasi dari Data 15-Menitan terbaru"""
    df_15m = fetch_15min_data()
    if df_15m.empty: return None
    latest_row = df_15m.iloc[-1]
    return {
        "jam": latest_row[COL_DATETIME].strftime("%H:%M"),
        "tma": float(latest_row[COL_OBSERVASI])
    }

# =========================================================================
# 3. STYLING & DATA PIPELINE SINKRONISASI
# =========================================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:opsz,wght@8..60,500;8..60,600&display=swap');

:root {
    --ink: #12263A;
    --muted: #5B6B7A;
    --rule: #D5DDE5;
    --paper: #F4F7FA;
    --card: #FFFFFF;
    --sea: #1F4E9C;
    --live: #B8337A;
    --good: #0B6B4F;
    --aman-bg: #E1F0F8;     --aman-fg: #0B5C80;
    --waspada-bg: #FCE8CC;  --waspada-fg: #8A4B05;
    --awas-bg: #FADADB;     --awas-fg: #9B1C1C;
    --sans: 'IBM Plex Sans', 'Segoe UI', Roboto, Arial, sans-serif;
    --serif: 'Source Serif 4', Georgia, 'Times New Roman', serif;
}

.stApp { background: var(--paper); color: var(--ink); font-family: var(--sans); }
header[data-testid="stHeader"] { background: transparent; height: 2.2rem; }
.block-container { max-width: 1280px; padding: 1.3rem 2rem 1.5rem; }
[data-testid="stMain"] [data-testid="stVerticalBlock"] { gap: 0.55rem; }

[data-testid="stMarkdownContainer"] *,
[data-testid="stWidgetLabel"] *,
[data-testid="stRadio"] label *,
[data-testid="stSelectbox"] *,
[data-testid="stDateInput"] *,
[data-testid="stCheckbox"] *,
[data-testid="stDownloadButton"] * { font-family: var(--sans) !important; }
[data-testid="stMarkdownContainer"] .serif { font-family: var(--serif) !important; }
[data-testid="stMarkdownContainer"] p { margin: 0; }
[data-testid="stWidgetLabel"] p { font-size: 0.85rem; font-weight: 500; color: var(--ink); }

/* ---------- Masthead (satu baris) ---------- */
.masthead { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;
            gap: 0.6rem 2rem; padding-bottom: 0.7rem; border-bottom: 2px solid var(--ink); }
.mast-title { font-size: 1.6rem; line-height: 1.2; font-weight: 600; letter-spacing: -0.01em; color: var(--ink); }
.mast-sub { margin-top: 0.2rem; font-size: 0.88rem; color: var(--muted); }
.mast-sub b { color: var(--ink); font-weight: 600; }

.live { display: flex; align-items: center; gap: 0.8rem; padding: 0.4rem 0.9rem;
        background: var(--card); border: 1px solid var(--rule); border-left: 3px solid var(--live); border-radius: 4px; }
.live.off { border-left-color: #9AA7B3; }
.live-val { font-size: 1.9rem; line-height: 1; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }
.live-val small { margin-left: 0.2rem; font-size: 0.85rem; font-weight: 500; color: var(--muted); font-family: var(--sans); }
.live-val-off { font-size: 1.05rem; color: var(--muted); }
.live-txt { font-size: 0.78rem; line-height: 1.35; color: var(--muted); }
.live-txt b { display: block; font-size: 0.82rem; color: var(--ink); font-weight: 600; }
.pulse { display: inline-block; width: 0.5rem; height: 0.5rem; margin-right: 0.35rem; border-radius: 50%;
         background: var(--live); animation: pulse 2.2s ease-in-out infinite; }
.pulse-off { background: #9AA7B3; animation: none; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

.pill { display: inline-block; padding: 0.08rem 0.55rem; border-radius: 999px; font-size: 0.74rem; font-weight: 600; }
.pill-aman { background: var(--aman-bg); color: var(--aman-fg); }
.pill-waspada { background: var(--waspada-bg); color: var(--waspada-fg); }
.pill-awas { background: var(--awas-bg); color: var(--awas-fg); }

/* ---------- Judul seksi ---------- */
.sec { display: flex; align-items: baseline; flex-wrap: wrap; gap: 0 0.7rem; margin: 0.9rem 0 0.1rem;
       padding-bottom: 0.3rem; border-bottom: 1px solid var(--rule); }
.sec-title { font-size: 1.15rem; line-height: 1.3; font-weight: 600; color: var(--ink); }
.sec-sub { font-size: 0.82rem; color: var(--muted); }

/* ---------- Ringkasan akurasi (satu baris, 4 kartu) ---------- */
.kpi-grid { display: grid; grid-template-columns: 1.3fr repeat(3, 1fr); gap: 0.6rem; }
.kpi { background: var(--card); border: 1px solid var(--rule); border-radius: 4px; padding: 0.65rem 0.9rem; }
.kpi-hero { border-top: 3px solid var(--sea); }
.kpi-label { display: flex; align-items: center; gap: 0.4rem; font-size: 0.78rem; font-weight: 500; color: var(--muted); }
.kpi-val { margin-top: 0.25rem; font-size: 1.55rem; line-height: 1.1; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }
.kpi-val small { margin-left: 0.15rem; font-size: 0.8rem; font-weight: 500; color: var(--muted); font-family: var(--sans); }
.kpi-hero .kpi-val { font-size: 2.3rem; color: var(--sea); }
.kpi-hero .kpi-val.kpi-empty { color: #9AA7B3; }
.kpi-note { margin-top: 0.25rem; font-size: 0.76rem; color: var(--muted); }
.kpi-note.best { font-weight: 600; color: var(--good); }
.kpi-bar { height: 0.3rem; margin-top: 0.4rem; background: #E8EDF2; border-radius: 3px; overflow: hidden; }
.kpi-bar > div { height: 100%; border-radius: 3px; }
.swatch { display: inline-block; width: 0.6rem; height: 0.6rem; border-radius: 2px; flex: none; }

/* ---------- Grafik ---------- */
[data-testid="stPlotlyChart"] { background: var(--card); border: 1px solid var(--rule); border-radius: 4px; padding: 0.3rem 0.4rem 0; }
[data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }
[data-testid="stElementContainer"]:has(.tbl-caption),
[data-testid="stElementContainer"]:has(.fig-caption) { overflow: visible; }
.fig-caption, .tbl-caption { display: block; max-width: 100ch; font-size: 0.8rem; line-height: 1.5; color: var(--muted); }
.tbl-caption { margin: 0.45rem 0 0.5rem; padding: 0.1rem 0 0.35rem; }
.fig-caption { margin: 0.1rem 0 0; padding-bottom: 0.3rem; }
.tbl-wrap { margin-top: 0.35rem; }
.fig-caption b, .tbl-caption b { color: var(--ink); }

/* ---------- Tabel ilmiah (tiga garis) ---------- */
.tbl-wrap { overflow-x: auto; }
.tbl { width: 100%; border-collapse: collapse; font-size: 0.86rem; font-variant-numeric: tabular-nums; background: transparent; }
.tbl th, .tbl td { border: none; }
.tbl thead th { padding: 0.4rem 0.7rem; text-align: right; font-weight: 600; color: var(--ink); white-space: nowrap;
                border-top: 2px solid var(--ink); border-bottom: 1px solid var(--ink); background: transparent; }
.tbl tbody td { padding: 0.38rem 0.7rem; text-align: right; border-bottom: 1px solid var(--rule); background: transparent; }
.tbl tbody tr:last-child td { border-bottom: 2px solid var(--ink); }
.tbl th:first-child, .tbl td:first-child { text-align: left; }
.tbl td.best { font-weight: 700; color: var(--good); background: #EAF6F1; }
.tbl td .swatch { margin-right: 0.5rem; }

.note { padding: 0.55rem 0.85rem; background: #EAF1F9; border: 1px solid #CFDDEC; border-left: 3px solid var(--sea);
        border-radius: 4px; font-size: 0.88rem; line-height: 1.5; }
.note b.lead { margin-right: 0.3rem; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: #EAEFF4; border-right: 1px solid var(--rule); }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding-top: 1rem; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.5rem; }
[data-testid="stSidebar"] hr { margin: 0.6rem 0; }
.side-title { font-size: 1.2rem; font-weight: 600; color: var(--ink); }
.side-h { margin: 0 0 0.3rem; font-size: 0.85rem; font-weight: 600; color: var(--ink); }
.side-txt { font-size: 0.8rem; line-height: 1.5; color: var(--muted); }
.side-txt b { color: var(--ink); font-weight: 600; }
.kv { display: grid; grid-template-columns: auto 1fr; gap: 0.15rem 0.8rem; font-size: 0.8rem; font-variant-numeric: tabular-nums; }
.kv span:nth-child(odd) { color: var(--muted); }
.kv span:nth-child(even) { text-align: right; color: var(--ink); font-weight: 500; }
.zone-row { display: flex; align-items: center; gap: 0.5rem; margin: 0.25rem 0; font-size: 0.82rem; }
.zone-swatch { width: 0.85rem; height: 0.85rem; border-radius: 2px; border: 1px solid rgba(18, 38, 58, 0.15); }
.sw-aman { background: var(--aman-bg); }
.sw-waspada { background: var(--waspada-bg); }
.sw-awas { background: var(--awas-bg); }
.zone-range { margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }
.meth { display: flex; gap: 0.5rem; margin: 0.3rem 0; font-size: 0.8rem; line-height: 1.4; color: var(--muted); }
.meth .swatch { margin-top: 0.3rem; }
.meth b { color: var(--ink); font-weight: 600; }

/* ---------- Tombol unduh ---------- */
[data-testid="stDownloadButton"] button { border: 1px solid var(--ink); background: #FFFFFF; color: var(--ink); border-radius: 4px; font-weight: 500; padding: 0.2rem 0.8rem; }
[data-testid="stDownloadButton"] button:hover { background: var(--ink); color: #FFFFFF; border-color: var(--ink); }

/* ---------- Layar kecil & preferensi gerak ---------- */
@media (max-width: 900px) {
    .block-container { padding: 1rem 1rem 1.5rem; }
    .mast-title { font-size: 1.35rem; }
    .kpi-grid { grid-template-columns: 1fr 1fr; }
    .kpi-hero { grid-column: 1 / -1; }
}
@media (prefers-reduced-motion: reduce) { .pulse { animation: none; } }
</style>
"""

def inject_custom_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def load_data() -> pd.DataFrame:
    df_master = pd.read_csv(DATA_FILE_HIBRIDA)
    df_lstm = pd.read_csv(DATA_FILE_LSTM)

    try:
        df_obs = pd.read_csv(DATA_FILE_OBSERVASI)
    except FileNotFoundError:
        df_obs = pd.DataFrame({COL_DATETIME: pd.Series(dtype="datetime64[ns]"), COL_OBSERVASI: pd.Series(dtype="float64")})

    df_master[COL_DATETIME] = _parse_datetime_column(df_master[COL_DATETIME])
    df_lstm[COL_DATETIME] = _parse_datetime_column(df_lstm[COL_DATETIME])
    df_obs[COL_DATETIME] = _parse_datetime_column(df_obs[COL_DATETIME])

    df_master = df_master.dropna(subset=[COL_DATETIME]).drop_duplicates(subset=[COL_DATETIME])
    df_lstm = df_lstm.dropna(subset=[COL_DATETIME]).drop_duplicates(subset=[COL_DATETIME])
    df_obs = df_obs.dropna(subset=[COL_DATETIME]).drop_duplicates(subset=[COL_DATETIME])

    if COL_OBSERVASI in df_master.columns: df_master = df_master.drop(columns=[COL_OBSERVASI])
    if COL_LSTM in df_master.columns: df_master = df_master.drop(columns=[COL_LSTM])

    df_master = pd.merge(df_master, df_lstm[[COL_DATETIME, COL_LSTM]], on=COL_DATETIME, how="left")
    df_master = pd.merge(df_master, df_obs[[COL_DATETIME, COL_OBSERVASI]], on=COL_DATETIME, how="left")

    return df_master

def filter_by_preset(df: pd.DataFrame, preset_name: str, custom_start=None, custom_end=None) -> pd.Series:
    if preset_name == CUSTOM_PRESET_KEY:
        return (df[COL_DATETIME].dt.date >= custom_start) & (df[COL_DATETIME].dt.date <= custom_end)
    tgl_start = pd.to_datetime(PRESETS[preset_name]["start"])
    tgl_end = pd.to_datetime(PRESETS[preset_name]["end"])
    return (df[COL_DATETIME] >= tgl_start) & (df[COL_DATETIME] <= tgl_end)

# =========================================================================
# 4. KOMPUTASI METRIK (KPI)
# =========================================================================
@dataclass
class MetodeMetrik:
    nama: str
    rmse: float
    warna: str
    is_terbaik: bool
    selisih_dari_terbaik: float

@dataclass
class KpiResult:
    reduksi_eror_persen: float
    utide: MetodeMetrik
    lstm: MetodeMetrik
    hibrida: MetodeMetrik

def compute_kpis(df_filtered: pd.DataFrame) -> Optional[KpiResult]:
    valid_idx = df_filtered[COL_OBSERVASI].notna()
    if valid_idx.sum() == 0: return None
    observasi = df_filtered.loc[valid_idx, COL_OBSERVASI]

    def rmse(prediksi: pd.Series) -> float:
        return float(np.sqrt(np.mean((observasi - prediksi) ** 2)))

    rmse_utide = rmse(df_filtered.loc[valid_idx, COL_UTIDE])
    rmse_lstm = rmse(df_filtered.loc[valid_idx, COL_LSTM])
    rmse_hibrida = rmse(df_filtered.loc[valid_idx, COL_HIBRIDA])

    reduksi_eror = ((rmse_utide - rmse_hibrida) / rmse_utide) * 100 if rmse_utide > 0 else 0.0
    min_rmse = min(rmse_utide, rmse_lstm, rmse_hibrida)

    def build_metrik(nama: str, rmse_val: float, warna: str) -> MetodeMetrik:
        is_terbaik = rmse_val == min_rmse
        return MetodeMetrik(nama=nama, rmse=rmse_val, warna=warna, is_terbaik=is_terbaik, selisih_dari_terbaik=0.0 if is_terbaik else rmse_val - min_rmse)

    return KpiResult(
        reduksi_eror_persen=reduksi_eror,
        utide=build_metrik("UTIDE", rmse_utide, COLOR_PALETTE["utide"]),
        lstm=build_metrik("LSTM", rmse_lstm, COLOR_PALETTE["lstm"]),
        hibrida=build_metrik("HIBRIDA", rmse_hibrida, COLOR_PALETTE["hibrida"]),
    )

# =========================================================================
# 5. UI RENDERING (PEMBANGUN HTML, GRAFIK & TABEL MENTAH)
# =========================================================================
def _zone_for_value(tma: float) -> tuple[str, str]:
    """Zona siaga untuk sebuah nilai TMA, memakai ambang yang sama dengan ALERT_ZONES."""
    nama, kelas = ALERT_ZONES[0][3], ZONE_CLASSES[0]
    for idx, (y0, _y1, _warna, label, _opacity) in enumerate(ALERT_ZONES):
        if tma >= y0:
            nama, kelas = label, ZONE_CLASSES[idx]
    return nama, kelas

def build_masthead_html(data_dsda: Optional[dict]) -> str:
    if data_dsda:
        zona, kelas = _zone_for_value(data_dsda["tma"])
        live = (
            '<div class="live">'
            f'<div class="serif live-val">{data_dsda["tma"]:g}<small>cm</small></div>'
            f'<div class="live-txt"><b><span class="pulse"></span>Observasi DSDA terbaru</b>'
            f'Pukul {data_dsda["jam"]} WIB &nbsp;<span class="pill pill-{kelas}">{zona}</span></div>'
            '</div>'
        )
    else:
        live = (
            '<div class="live off">'
            '<div class="serif live-val live-val-off">Tidak tersedia</div>'
            '<div class="live-txt"><b><span class="pulse pulse-off"></span>Observasi DSDA terbaru</b>'
            'Data live offline atau tertunda</div>'
            '</div>'
        )
    return (
        '<div class="masthead"><div class="mast-text">'
        '<div class="serif mast-title">Peramalan pasang surut hibrida UTide dan LSTM</div>'
        '<div class="mast-sub"><b>Stasiun Pasar Ikan, Jakarta.</b> Dashboard tesis oleh Iman, S.Si.</div>'
        f'</div>{live}</div>'
    )

def build_section_header_html(judul: str, sub: str = "") -> str:
    sub_html = f'<span class="sec-sub">{sub}</span>' if sub else ""
    return f'<div class="sec"><span class="serif sec-title">{judul}</span>{sub_html}</div>'

_NAMA_METODE = {"UTIDE": "UTide", "LSTM": "LSTM", "HIBRIDA": "Hibrida"}
_KET_METODE = {"UTIDE": "Astronomis (harmonik)", "LSTM": "Data-driven", "HIBRIDA": "Integrasi UTide dan LSTM"}

def build_kpi_html(kpi: Optional[KpiResult]) -> str:
    if kpi is None:
        return (
            '<div class="kpi-grid">'
            '<div class="kpi kpi-hero" style="grid-column:1/-1"><div class="kpi-label">Reduksi eror Hibrida terhadap UTide</div>'
            '<div class="serif kpi-val kpi-empty">&ndash;</div>'
            '<div class="kpi-note">Belum ada data observasi pada rentang waktu ini, sehingga RMSE belum dapat dihitung.</div></div>'
            '</div>'
        )

    if kpi.reduksi_eror_persen >= 0:
        catatan = "RMSE Hibrida lebih rendah daripada UTide."
    else:
        catatan = "RMSE Hibrida lebih tinggi daripada UTide."

    metode = [kpi.utide, kpi.lstm, kpi.hibrida]
    rmse_maks = max(m.rmse for m in metode) or 1.0
    kartu = ""
    for m in metode:
        lebar = max(m.rmse / rmse_maks * 100.0, 2.0)
        if m.is_terbaik:
            flag = '<div class="kpi-note best">Terendah</div>'
        else:
            flag = f'<div class="kpi-note">+{m.selisih_dari_terbaik:.1f} cm dari terendah</div>'
        kartu += (
            '<div class="kpi">'
            f'<div class="kpi-label"><span class="swatch" style="background:{m.warna}"></span>RMSE {_NAMA_METODE[m.nama]}</div>'
            f'<div class="serif kpi-val">{m.rmse:.2f}<small>cm</small></div>'
            f'<div class="kpi-bar"><div style="width:{lebar:.1f}%;background:{m.warna}"></div></div>{flag}'
            '</div>'
        )

    return (
        '<div class="kpi-grid">'
        '<div class="kpi kpi-hero"><div class="kpi-label">Reduksi eror Hibrida terhadap UTide</div>'
        f'<div class="serif kpi-val">{kpi.reduksi_eror_persen:.2f}<small>%</small></div>'
        f'<div class="kpi-note">{catatan}</div></div>'
        f'{kartu}'
        '</div>'
    )

def build_zone_legend_html() -> str:
    baris = ""
    for idx, (y0, y1, _warna, label, _opacity) in enumerate(ALERT_ZONES):
        if idx == 0:
            rentang = f"sampai {y1} cm"
        elif idx == len(ALERT_ZONES) - 1:
            rentang = f"{y0} cm ke atas"
        else:
            rentang = f"{y0} sampai {y1} cm"
        baris += (
            f'<div class="zone-row"><span class="zone-swatch sw-{ZONE_CLASSES[idx]}"></span>'
            f'<span>{label}</span><span class="zone-range">{rentang}</span></div>'
        )
    return f'<div class="side-h">Zona tinggi muka air</div>{baris}'

def build_method_legend_html() -> str:
    baris = ""
    for kode, warna in (("UTIDE", COLOR_PALETTE["utide"]), ("LSTM", COLOR_PALETTE["lstm"]), ("HIBRIDA", COLOR_PALETTE["hibrida"])):
        baris += (
            f'<div class="meth"><span class="swatch" style="background:{warna}"></span>'
            f'<span><b>{_NAMA_METODE[kode]}</b>: {_KET_METODE[kode]}</span></div>'
        )
    return f'<div class="side-h">Pendekatan peramalan</div>{baris}'

def build_active_period_html(pilihan_mode: str, df_filtered: pd.DataFrame) -> str:
    desc = html.escape(PRESETS[pilihan_mode]["desc"])
    if df_filtered.empty:
        awal = akhir = "-"
    else:
        awal = df_filtered[COL_DATETIME].min().strftime("%d %b %Y %H:%M")
        akhir = df_filtered[COL_DATETIME].max().strftime("%d %b %Y %H:%M")
    n_total = len(df_filtered)
    n_obs = int(df_filtered[COL_OBSERVASI].notna().sum())
    return (
        '<div class="side-h">Periode aktif</div>'
        f'<div class="side-txt" style="margin-bottom:0.4rem">{desc}</div>'
        '<div class="kv">'
        f'<span>Awal</span><span>{awal}</span>'
        f'<span>Akhir</span><span>{akhir}</span>'
        f'<span>Jam tersedia</span><span>{n_total}</span>'
        f'<span>Jam beramati</span><span>{n_obs}</span>'
        '</div>'
    )

def render_header(data_dsda: Optional[dict]) -> None:
    st.markdown(build_masthead_html(data_dsda), unsafe_allow_html=True)

def render_kpi_section(kpi: Optional[KpiResult]) -> None:
    st.markdown(
        build_section_header_html(
            "Ringkasan akurasi",
            "Selisih prediksi terhadap observasi per jam pada rentang yang dipilih. Batang lebih pendek berarti lebih akurat.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(build_kpi_html(kpi), unsafe_allow_html=True)

def _add_alert_zones(fig: go.Figure, dynamic_min: float, dynamic_max: float) -> None:
    for y0, y1, warna, label, opacity in ALERT_ZONES:
        y0_clip = max(y0, dynamic_min)
        y1_clip = min(y1, dynamic_max)
        if y0_clip >= y1_clip: continue
        fig.add_hrect(y0=y0_clip, y1=y1_clip, fillcolor=warna, opacity=opacity, line_width=0, layer="below")
        fig.add_annotation(xref="paper", yref="y", x=0.995, y=(y0_clip + y1_clip) / 2, text=label, showarrow=False, xanchor="right", yanchor="middle", font=dict(color="#5B6B7A", size=11))

def build_comparison_chart(df_filtered: pd.DataFrame, data_dsda: Optional[dict], show_15min: bool) -> go.Figure:
    all_mins = [df_filtered[COL_OBSERVASI].min(), df_filtered[COL_UTIDE].min(), df_filtered[COL_HIBRIDA].min(), df_filtered[COL_LSTM].min()]
    valid_mins = [v for v in all_mins if pd.notna(v)]
    current_min = min(valid_mins) if valid_mins else 100

    all_maxs = [df_filtered[COL_OBSERVASI].max(), df_filtered[COL_UTIDE].max(), df_filtered[COL_HIBRIDA].max(), df_filtered[COL_LSTM].max()]
    valid_maxs = [v for v in all_maxs if pd.notna(v)]
    current_max = max(valid_maxs) if valid_maxs else 280

    dynamic_min = current_min - 10
    dynamic_max = max(280, current_max + 10)

    fig = go.Figure()
    _add_alert_zones(fig, dynamic_min, dynamic_max)

    # 1. Plot Utama (Observasi Jam-jaman)
    if df_filtered[COL_OBSERVASI].notna().sum() > 0:
        fig.add_trace(go.Scatter(x=df_filtered[COL_DATETIME], y=df_filtered[COL_OBSERVASI], mode="lines", name="Observasi per jam", line=dict(color=COLOR_PALETTE["observasi"], width=2.4, shape="spline", smoothing=0.9), connectgaps=False))

    # 2. Plot Realtime 15-Menit (Jika dicentang user)
    if show_15min:
        df_15 = fetch_15min_data()
        if not df_15.empty:
            # Filter rentang waktu 15 menitan agar sama persis dengan master view
            min_dt, max_dt = df_filtered[COL_DATETIME].min(), df_filtered[COL_DATETIME].max()
            df_15_filt = df_15[(df_15[COL_DATETIME] >= min_dt) & (df_15[COL_DATETIME] <= max_dt)]
            if not df_15_filt.empty:
                fig.add_trace(go.Scatter(
                    x=df_15_filt[COL_DATETIME], y=df_15_filt[COL_OBSERVASI],
                    mode="lines+markers",
                    name="Live DSDA per 15 menit",
                    marker=dict(size=5, color=COLOR_PALETTE["live_15m"], line=dict(width=1, color="#FFFFFF")),
                    line=dict(color=COLOR_PALETTE["live_15m"], width=1.6, dash="solid", shape="spline"),
                    connectgaps=False
                ))

    # 3. Prediksi Model (dibedakan warna dan gaya garis)
    fig.add_trace(go.Scatter(x=df_filtered[COL_DATETIME], y=df_filtered[COL_UTIDE], mode="lines", name="Prediksi UTide", line=dict(color=COLOR_PALETTE["utide"], width=1.8, dash="dash", shape="spline", smoothing=0.9)))
    fig.add_trace(go.Scatter(x=df_filtered[COL_DATETIME], y=df_filtered[COL_LSTM], mode="lines", name="Prediksi LSTM", line=dict(color=COLOR_PALETTE["lstm"], width=2.0, dash="dot", shape="spline", smoothing=0.9)))
    fig.add_trace(go.Scatter(x=df_filtered[COL_DATETIME], y=df_filtered[COL_HIBRIDA], mode="lines", name="Prediksi Hibrida", line=dict(color=COLOR_PALETTE["hibrida"], width=2.8, shape="spline", smoothing=0.9)))

    # Garis Waktu Sekarang
    # CATATAN PERBAIKAN: fig.add_vline(..., annotation_text=...) memicu TypeError pada sumbu waktu
    # di plotly versi baru (error di sum(x)). Jadi garis dan label dibuat terpisah:
    # add_shape untuk garis vertikal, add_annotation untuk labelnya.
    if data_dsda and data_dsda["tma"] is not None:
        waktu_sekarang_jam = get_now_jkt()
        min_date = df_filtered[COL_DATETIME].min()
        max_date = df_filtered[COL_DATETIME].max()
        if pd.notna(min_date) and pd.notna(max_date) and min_date <= waktu_sekarang_jam <= max_date:
            jam_menit_str = waktu_sekarang_jam.strftime("%H:%M")
            fig.add_shape(
                type="line",
                x0=waktu_sekarang_jam, x1=waktu_sekarang_jam, xref="x",
                y0=0, y1=1, yref="paper",
                line=dict(width=1.4, dash="dot", color="#12263A"),
            )
            fig.add_annotation(
                x=waktu_sekarang_jam, y=1, xref="x", yref="paper",
                text=f" Sekarang, {jam_menit_str} WIB",
                showarrow=False, xanchor="left", yanchor="top",
                font=dict(color="#12263A", size=11),
                bgcolor="rgba(255,255,255,0.85)",
            )

    fonts = "IBM Plex Sans, Segoe UI, Arial, sans-serif"
    sumbu_font = dict(family=fonts, size=11, color="#5B6B7A")
    fig.update_layout(
        height=430,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=8, r=12, t=34, b=8),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="white", bordercolor="#D5DDE5", font=dict(size=12, family=fonts, color="#12263A")),
        font=dict(family=fonts, color="#12263A", size=12),
        xaxis=dict(
            tickfont=sumbu_font, showgrid=False, showline=True, linecolor="#9AA7B3", ticks="outside", tickcolor="#9AA7B3",
            tickformat="%d %b", hoverformat="%d %b %Y, %H:%M",
            tickformatstops=[dict(dtickrange=[None, 86400000], value="%d %b<br>%H:%M")],
        ),
        yaxis=dict(
            title=dict(text="Tinggi muka air (cm)", font=dict(size=12, color="#5B6B7A")),
            tickfont=sumbu_font, range=[dynamic_min, dynamic_max],
            gridcolor="#E8EDF2", gridwidth=1, zeroline=False, showline=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
    )
    return fig

def render_chart_section(pilihan_mode: str, df_filtered: pd.DataFrame, data_dsda: Optional[dict], show_15min: bool) -> None:
    st.markdown(
        build_section_header_html("Observasi dan prediksi", html.escape(pilihan_mode)),
        unsafe_allow_html=True,
    )
    fig = build_comparison_chart(df_filtered, data_dsda, show_15min)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(
        '<div class="fig-caption"><b>Gambar 1.</b> Tinggi muka air observasi dibandingkan dengan prediksi '
        'UTide, LSTM, dan Hibrida. Pita warna menandai zona siaga rob (rincian di panel samping).</div>',
        unsafe_allow_html=True,
    )

def render_data_table(df_filtered: pd.DataFrame) -> None:
    st.markdown(
        build_section_header_html("Data numerik", "Dasar perhitungan Gambar 1."),
        unsafe_allow_html=True,
    )
    df_tampilan = pd.DataFrame({
        COL_DATETIME: df_filtered[COL_DATETIME],
        COL_OBSERVASI: df_filtered[COL_OBSERVASI],
        COL_UTIDE: df_filtered[COL_UTIDE],
        COL_LSTM: df_filtered[COL_LSTM],
        COL_HIBRIDA: df_filtered[COL_HIBRIDA],
    })

    st.markdown('<div class="tbl-caption"><b>Tabel 2.</b> Data numerik mentah pada rentang waktu yang ditampilkan (cm).</div>', unsafe_allow_html=True)
    st.dataframe(df_tampilan.reset_index(drop=True), use_container_width=True, hide_index=True, height=300)
    csv_data = df_tampilan.to_csv(index=False).encode("utf-8")
    st.download_button(label="Unduh data tabular (.csv)", data=csv_data, file_name="DATA_INSPEKSI_PASUT.csv", mime="text/csv")

def render_sidebar_about() -> None:
    st.sidebar.markdown(
        '<div class="side-h">Sumber dan pembaruan</div>'
        '<div class="side-txt">Observasi: Posko Banjir DSDA DKI Jakarta.<br>'
        'Prediksi: harmonik UTide dan LSTM residual.<br>'
        'Halaman diperbarui otomatis setiap 15 menit.</div>',
        unsafe_allow_html=True,
    )


# =========================================================================
# 6. SEKSI EVALUASI TESIS (ANALISIS MENDALAM)
# =========================================================================
_KOLOM_METRIK = [
    # (nama kolom, format, arah terbaik)
    ("Akurasi (%) ↑", "{:.2f}%", "max"),
    ("RMSE (cm) ↓", "{:.3f}", "min"),
    ("MAE (cm) ↓", "{:.3f}", "min"),
    ("Koefisien Determinasi (R²) ↑", "{:.3f}", "max"),
]
_WARNA_BARIS_METRIK = [COLOR_PALETTE["utide"], COLOR_PALETTE["lstm"], COLOR_PALETTE["hibrida"]]

def build_metrics_table_html(df_metrics: pd.DataFrame) -> str:
    """Tabel metrik bergaya jurnal (tiga garis). Sel terbaik tiap kolom ditebalkan."""
    terbaik = {}
    for kolom, _fmt, arah in _KOLOM_METRIK:
        nilai = df_metrics[kolom]
        terbaik[kolom] = nilai.min() if arah == "min" else nilai.max()

    kepala = "<th>Pendekatan peramalan</th>" + "".join(f"<th>{k}</th>" for k, _f, _a in _KOLOM_METRIK)
    isi = ""
    for i, (_, baris) in enumerate(df_metrics.iterrows()):
        warna = _WARNA_BARIS_METRIK[i % len(_WARNA_BARIS_METRIK)]
        sel = f'<td><span class="swatch" style="background:{warna}"></span>{html.escape(str(baris["Pendekatan Peramalan"]))}</td>'
        for kolom, fmt, _arah in _KOLOM_METRIK:
            kelas = ' class="best"' if baris[kolom] == terbaik[kolom] else ""
            sel += f"<td{kelas}>{fmt.format(baris[kolom])}</td>"
        isi += f"<tr>{sel}</tr>"
    return f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{kepala}</tr></thead><tbody>{isi}</tbody></table></div>'

def build_interpretation_html(best_method: str, best_rmse: float, impr_utide: float) -> str:
    if "Hibrida" in best_method:
        isi = (
            "Pendekatan <b>Hibrida UTide+LSTM</b> memberi peramalan paling akurat (RMSE terendah "
            f"<b>{best_rmse:.2f} cm</b>). Komponen <i>data-driven</i> menangkap anomali non-astronomis sehingga "
            f"eror UTide murni turun <b>{impr_utide:.1f}%</b>."
        )
    else:
        isi = (
            f"Pada rentang ini, pendekatan <b>{html.escape(best_method)}</b> paling presisi "
            f"dengan RMSE <b>{best_rmse:.2f} cm</b>."
        )
    return f'<div class="note"><b class="lead">Interpretasi.</b>{isi}</div>'

def render_thesis_analysis(df_master: pd.DataFrame, df_filtered: pd.DataFrame) -> None:
    st.markdown(
        build_section_header_html(
            "Evaluasi kinerja peramalan",
            "Akurasi, RMSE, MAE, dan R² terhadap observasi per jam.",
        ),
        unsafe_allow_html=True,
    )

    scope = st.radio("Cakupan analisis data", ["Sesuai Tampilan Grafik", "Rekap Bulanan", "Rentang Kustom"], horizontal=True)
    eval_df = df_filtered.copy()

    if scope == "Rekap Bulanan":
        df_master_copy = df_master.copy()
        df_master_copy['MonthYear'] = df_master_copy[COL_DATETIME].dt.to_period('M')
        months = sorted(df_master_copy['MonthYear'].dropna().unique(), reverse=True)
        last_month_period = pd.Period(get_now_jkt(), 'M') - 1
        default_idx = months.index(last_month_period) if last_month_period in months else 0

        col_month, _ = st.columns([1, 2])
        with col_month: selected_month = st.selectbox("Bulan evaluasi", months, index=default_idx, format_func=lambda x: x.strftime('%B %Y'))
        mask_month = df_master_copy['MonthYear'] == selected_month
        eval_df = df_master_copy[mask_month].copy()

    elif scope == "Rentang Kustom":
        min_date, max_date = df_master[COL_DATETIME].min().date(), df_master[COL_DATETIME].max().date()
        col_date, _ = st.columns([1.5, 1.5])
        with col_date: custom_dates = st.date_input("Rentang tanggal", [min_date, max_date], min_value=min_date, max_value=max_date)

        if len(custom_dates) == 2:
            c_start, c_end = custom_dates
            mask_custom = (df_master[COL_DATETIME].dt.date >= c_start) & (df_master[COL_DATETIME].dt.date <= c_end)
            eval_df = df_master[mask_custom].copy()
        else:
            st.info("Silakan pilih tanggal akhir untuk memproses analisis.")
            return

    valid_idx = eval_df[COL_OBSERVASI].notna()
    if valid_idx.sum() < 2:
        st.warning("Data observasi tidak mencukupi untuk dihitung metrik akurasinya pada rentang waktu ini.")
        return

    obs = eval_df.loc[valid_idx, COL_OBSERVASI].values
    utide = eval_df.loc[valid_idx, COL_UTIDE].values
    lstm = eval_df.loc[valid_idx, COL_LSTM].values
    hibrida = eval_df.loc[valid_idx, COL_HIBRIDA].values

    def calc_metrics(pred):
        rmse = np.sqrt(np.mean((obs - pred) ** 2))
        mae = np.mean(np.abs(obs - pred))
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        safe_obs = np.where(obs == 0, 1e-6, obs)
        mape = np.mean(np.abs((obs - pred) / safe_obs)) * 100
        akurasi = max(0.0, 100.0 - mape)
        return rmse, mae, r2, akurasi

    rmse_u, mae_u, r2_u, acc_u = calc_metrics(utide)
    rmse_l, mae_l, r2_l, acc_l = calc_metrics(lstm)
    rmse_h, mae_h, r2_h, acc_h = calc_metrics(hibrida)

    df_metrics = pd.DataFrame({
        "Pendekatan Peramalan": ["Harmonik UTide (Astronomis Konvensional)", "LSTM Murni (Model Data-Driven)", "Hibrida UTide+LSTM (Integrasi Residual)"],
        "Akurasi (%) ↑": [acc_u, acc_l, acc_h],
        "RMSE (cm) ↓": [rmse_u, rmse_l, rmse_h],
        "MAE (cm) ↓": [mae_u, mae_l, mae_h],
        "Koefisien Determinasi (R²) ↑": [r2_u, r2_l, r2_h]
    })

    best_rmse = df_metrics["RMSE (cm) ↓"].min()
    best_method = df_metrics.loc[df_metrics["RMSE (cm) ↓"] == best_rmse, "Pendekatan Peramalan"].values[0]
    impr_utide = ((rmse_u - rmse_h) / rmse_u) * 100 if rmse_u else 0

    st.markdown(
        f'<div class="tbl-caption"><b>Tabel 1.</b> Metrik akurasi tiga pendekatan terhadap '
        f'<b>{len(obs)} jam</b> observasi aktual. Nilai terbaik per kolom ditebalkan.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(build_metrics_table_html(df_metrics), unsafe_allow_html=True)
    st.markdown(build_interpretation_html(best_method, best_rmse, impr_utide), unsafe_allow_html=True)


# =========================================================================
# 7. SIDEBAR CONTROLS & MAIN APP
# =========================================================================
def render_sidebar_controls(df: pd.DataFrame) -> tuple[str, Optional[object], Optional[object], bool]:
    st.sidebar.markdown('<div class="serif side-title">Panel analisis</div>', unsafe_allow_html=True)
    pilihan_mode = st.sidebar.selectbox("Mode analisis / studi kasus", list(PRESETS.keys()), index=DEFAULT_PRESET_INDEX)
    custom_start = custom_end = None
    if pilihan_mode == CUSTOM_PRESET_KEY:
        min_date = df[COL_DATETIME].min().date()
        max_date = df[COL_DATETIME].max().date()
        custom_start = st.sidebar.date_input("Tanggal mulai", min_date, min_value=min_date, max_value=max_date)
        custom_end = st.sidebar.date_input("Tanggal selesai", max_date, min_value=min_date, max_value=max_date)

    show_15min = st.sidebar.checkbox("Tampilkan garis 15 menit (live scrape)", value=True, help="Menampilkan titik potong observasi DSDA per 15 menit. Grafik akan ditumpuk dengan data jam-jaman.")

    return pilihan_mode, custom_start, custom_end, show_15min

def render_sidebar_info(pilihan_mode: str, df_filtered: pd.DataFrame) -> None:
    """Informasi pendukung (konteks, legenda, sumber) agar halaman utama tetap padat."""
    st.sidebar.divider()
    st.sidebar.markdown(build_active_period_html(pilihan_mode, df_filtered), unsafe_allow_html=True)
    st.sidebar.divider()
    st.sidebar.markdown(build_zone_legend_html(), unsafe_allow_html=True)
    st.sidebar.divider()
    st.sidebar.markdown(build_method_legend_html(), unsafe_allow_html=True)
    st.sidebar.divider()
    render_sidebar_about()

def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon=PAGE_ICON, initial_sidebar_state="expanded")

    # Fitur Auto-Refresh Streamlit: Refresh browser otomatis tiap 15 menit (15 * 60 * 1000 milidetik)
    st_autorefresh(interval=15 * 60 * 1000, key="data_refresh_15m")

    inject_custom_css()
    data_dsda = fetch_realtime_status()

    try:
        df_master = load_data()
    except Exception as e:
        st.error(f"File CSV gagal dimuat. Error: {e}")
        st.stop()
        return

    pilihan_mode, custom_start, custom_end, show_15min = render_sidebar_controls(df_master)
    mask = filter_by_preset(df_master, pilihan_mode, custom_start, custom_end)
    df_filtered = df_master[mask].copy()

    kpi = compute_kpis(df_filtered)
    render_sidebar_info(pilihan_mode, df_filtered)

    render_header(data_dsda)
    render_kpi_section(kpi)

    # 1. GRAFIK TERLEBIH DAHULU (Dengan parameter show_15min)
    render_chart_section(pilihan_mode, df_filtered, data_dsda, show_15min)

    # 2. ANALISIS KINERJA (EVALUASI TESIS)
    render_thesis_analysis(df_master, df_filtered)

    # 3. DATA MENTAH TABULAR (PALING BAWAH)
    render_data_table(df_filtered)


if __name__ == "__main__":
    main()
