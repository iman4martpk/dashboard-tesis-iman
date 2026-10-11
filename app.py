"""
Dashboard Monitoring Pasut Hibrida (UTide + LSTM) - Stasiun Pasar Ikan, Jakarta.

Aplikasi Streamlit ini menampilkan perbandingan performa pendekatan
prediksi pasang surut air laut dengan data observasi independen secara REAL-TIME.

* REVISI: Pembacaan data aktual kini tidak melakukan live-scrape web (untuk menghindari
          loading lama dan potensi blokir server). Data observasi aktual hanya 
          dibaca murni dari baris terakhir CSV lokal yang diperbarui otomatis 
          oleh background script (scrape_hourly iman.py).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pytz

# =========================================================================
# 1. KONSTANTA & KONFIGURASI GLOBAL
# =========================================================================

PAGE_TITLE = "Dashboard Pasut Hibrida Pasar Ikan"
PAGE_ICON = "🌊"

# Zona waktu konsisten untuk server Streamlit Cloud / GitHub
TZ_JKT = pytz.timezone("Asia/Jakarta")

def get_now_jkt() -> datetime:
    """Mengembalikan waktu saat ini di Jakarta tanpa timezone (naive) untuk mencocokkan CSV."""
    return datetime.now(TZ_JKT).replace(tzinfo=None)

# 3 File Master Data
DATA_FILE_HIBRIDA = "HASIL_FINAL_TESIS_PASUT_HIBRIDA.csv"
DATA_FILE_LSTM = "HASIL_FINAL_TESIS_PASUT_LSTM_MURNI.csv"
DATA_FILE_OBSERVASI = "HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv"

COL_DATETIME = "Datetime"
COL_OBSERVASI = "TMA_Pasar_Ikan"
COL_UTIDE = "Prediksi_Harmonik_UTIDE"
COL_LSTM = "Prediksi_LSTM_Murni"
COL_HIBRIDA = "Prediksi_Hibrida_Final"

# --- Palet warna baru (Lebih elegan & modern) -------------------------------
COLOR_PALETTE = {
    "primary": "#0B3D4C",
    "success": "#22c55e",
    "danger": "#ef4444",
    
    # GARIS GRAFIK
    "observasi": "#475569",                 # Slate-600
    "utide": "rgba(0, 194, 255, 0.45)",     # Cyan elektrik
    "lstm": "rgba(16, 185, 129, 0.45)",     # Hijau Emerald
    "hibrida": "rgba(37, 99, 235, 0.65)",   # Royal Blue
    
    # PITA GRADASI (Background)
    "aman": "#BAE6FD",     # Biru muda kalem
    "waspada": "#EA580C",  # Jingga tua
    "awas": "#FF0000",     # Merah
}

# --- Zona / pita siaga pada grafik (cm) ----------------------------------
ALERT_ZONES = [
    (0, 230, COLOR_PALETTE["aman"], "KONDISI AMAN", 0.25),
    (230, 250, COLOR_PALETTE["waspada"], "WASPADA ROB", 0.32),
    (250, 500, COLOR_PALETTE["awas"], "AWAS ROB", 0.25), 
]

# --- LOGIKA DINAMIS 2 HARI KE BELAKANG & 2 HARI KE DEPAN ---
HARI_INI = get_now_jkt()
START_REALTIME = (HARI_INI - timedelta(days=2)).strftime("%Y-%m-%d 00:00:00")
END_REALTIME = (HARI_INI + timedelta(days=2)).strftime("%Y-%m-%d 23:00:00")

PRESETS = {
    "📊 Ringkasan Real-Time (Jendela 4 Hari)": {
        "start": START_REALTIME,
        "end": END_REALTIME,
        "desc": "Kondisi operasional berjalan (2 hari lalu s.d 2 hari ke depan).",
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
    "🔮 MODE FORECASTING MASA DEPAN": {
        "start": "2026-07-21 19:00:00",
        "end": "2026-12-31 23:00:00",
        "desc": "Peramalan estafet bergulir jangka panjang.",
    },
    "🎛️ Custom Rentang Waktu (Manual)": {
        "start": None,
        "end": None,
        "desc": "Bebas menentukan rentang analisis tanggal sendiri.",
    },
}
CUSTOM_PRESET_KEY = "🎛️ Custom Rentang Waktu (Manual)"
DEFAULT_PRESET_INDEX = 0


# =========================================================================
# 2. FUNGSI PEMBACAAN DATA REAL-TIME LOKAL (TANPA HTTP REQUEST)
# =========================================================================

@st.cache_data(ttl=60)
def fetch_realtime_data() -> Optional[dict]:
    """
    Membaca nilai observasi TMA (Pasar Ikan) langsung dari baris terakhir
    file HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv yang dikelola oleh bot background.
    Tidak ada scraping web di UI untuk menjaga kecepatan render.
    """
    try:
        df = pd.read_csv(DATA_FILE_OBSERVASI)
        if df.empty:
            return None
            
        # Konversi waktu untuk pengurutan
        df[COL_DATETIME] = pd.to_datetime(df[COL_DATETIME], format="mixed", errors="coerce")
        df_valid = df.dropna(subset=[COL_DATETIME, COL_OBSERVASI]).sort_values(COL_DATETIME)
        
        if df_valid.empty:
            return None
            
        # Ambil baris data paling terbaru
        latest_row = df_valid.iloc[-1]
        waktu_terakhir = latest_row[COL_DATETIME]
        tma_terakhir = latest_row[COL_OBSERVASI]
        
        return {
            "jam": waktu_terakhir.strftime("%H:%M"),
            "tma": float(tma_terakhir)
        }
    except Exception as e:
        return None


# =========================================================================
# 3. STYLING & DATA PIPELINE SINKRONISASI
# =========================================================================
def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 3.2rem !important; padding-bottom: 2rem !important; max-width: 95% !important; }}
        .header-text {{ text-align: center; width: 100%; margin-top: 5px; margin-bottom: 0px !important; padding-bottom: 0px !important; }}
        html, body, [class*="css"] {{ font-family: Arial, Helvetica, sans-serif !important; }}
        div[data-testid="stMetric"] {{ background-color: #ffffff !important; border: 1px solid #e2e8f0 !important; border-left: 4px solid #e2e8f0 !important; padding: 4px 10px !important; border-radius: 8px !important; min-height: 55px !important; }}
        div[data-testid="stMetricLabel"] {{ color: #64748b !important; font-weight: 600 !important; font-size: 0.68rem !important; margin-bottom: -4px !important; }}
        [data-testid="stMetricValue"] {{ font-size: 14px !important; font-weight: 700 !important; color: #0f172a !important; }}
        .summary-box {{ background-color: #f1f5f9 !important; padding: 6px 12px !important; border-radius: 8px !important; margin-top: 4px !important; margin-bottom: 8px !important; border-left: 5px solid {COLOR_PALETTE['primary']} !important; text-align: center !important; }}
        .summary-text {{ font-family: Arial, Helvetica, sans-serif !important; font-weight: 600; font-size: 0.82rem; color: #1e293b; }}
        .eval-box {{ background-color: #f8fafc; padding: 15px; border-radius: 8px; border-left: 5px solid {COLOR_PALETTE['primary']}; margin-bottom: 15px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def _parse_datetime_column(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="mixed", dayfirst=False, errors="coerce")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed

def load_data() -> pd.DataFrame:
    """
    Semua dataframe (Hibrida, LSTM, Observasi) dilebur ke dalam
    satu grid waktu master (Left Merge) agar ukurannya dijamin sama.
    """
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

    if COL_OBSERVASI in df_master.columns:
        df_master = df_master.drop(columns=[COL_OBSERVASI])
    if COL_LSTM in df_master.columns:
        df_master = df_master.drop(columns=[COL_LSTM])

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
    if valid_idx.sum() == 0:
        return None

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
        utide=build_metrik("UTIDE", rmse_utide, "#00C2FF"),
        lstm=build_metrik("LSTM", rmse_lstm, "#10B981"),
        hibrida=build_metrik("HIBRIDA", rmse_hibrida, "#2563EB"),
    )


# =========================================================================
# 5. UI RENDERING (GRAFIK & TABEL MENTAH)
# =========================================================================
def render_header() -> None:
    st.markdown("""<div class="header-text"><h2 style="margin: 0; color: #0F172A; font-family: Arial, Helvetica, sans-serif; font-weight: bold; font-size: 1.55rem;">🌊 Dashboard Tesis: Peramalan Pasang Surut Hibrida Jakarta • Iman, S.Si.</h2></div>""", unsafe_allow_html=True)

def render_summary_box(pilihan_mode: str, data_dsda: Optional[dict]) -> None:
    info_realtime = f" | 🔴 <b>DSDA Real-time ({data_dsda['jam']}):</b> <span style='color:#DC2626;'>{data_dsda['tma']} cm</span>" if data_dsda else " | 🔴 <b>DSDA Real-time:</b> <span style='color:#64748B;'>Offline/Delay</span>"
    st.markdown(f"""<div class="summary-box"><span class="summary-text">📍 <b>Stasiun:</b> Pasar Ikan, Jakarta | 🛡️ <b>Mode:</b> {PRESETS[pilihan_mode]['desc']}{info_realtime}</span></div>""", unsafe_allow_html=True)

def _metrik_badge(metrik: MetodeMetrik) -> str:
    if metrik.is_terbaik: return '<span style="color: #22c55e; font-size: 0.65rem; font-weight: bold;">🏆 TERENDAH</span>'
    return f'<span style="color: #ef4444; font-size: 0.65rem; font-weight: bold;">+{metrik.selisih_dari_terbaik:.1f} cm vs Terbaik</span>'

def _render_metric_card(column, label: str, value_html: str, border_color: str) -> None:
    column.markdown(f'<div data-testid="stMetric" style="border-left-color: {border_color} !important;"><label data-testid="stMetricLabel">{label}</label><div data-testid="stMetricValue">{value_html}</div></div>', unsafe_allow_html=True)

def render_kpi_cards(kpi: KpiResult) -> None:
    col1, col2, col3, col4 = st.columns(4)
    _render_metric_card(col1, "📈 REDUKSI EROR (vs UTide)", f'{kpi.reduksi_eror_persen:.2f} % <span style="color: #22c55e; font-size: 0.68rem; font-weight: bold;">▲ OPTIMAL</span>', COLOR_PALETTE["success"])
    _render_metric_card(col2, "📉 RMSE UTIDE (Astronomis)", f"{kpi.utide.rmse:.2f} cm {_metrik_badge(kpi.utide)}", kpi.utide.warna)
    _render_metric_card(col3, "📊 RMSE LSTM (Data-Driven)", f"{kpi.lstm.rmse:.2f} cm {_metrik_badge(kpi.lstm)}", kpi.lstm.warna)
    _render_metric_card(col4, "🏆 RMSE HIBRIDA (Integrasi)", f"{kpi.hibrida.rmse:.2f} cm {_metrik_badge(kpi.hibrida)}", kpi.hibrida.warna)

def render_empty_kpi_cards() -> None:
    col1, col2, col3, col4 = st.columns(4)
    _render_metric_card(col1, "📈 REDUKSI EROR (vs UTide)", '<span style="color: #64748B;">Tidak ada data</span>', COLOR_PALETTE["observasi"])
    _render_metric_card(col2, "📉 RMSE UTIDE (Astronomis)", '<span style="color: #64748B;">Tidak ada data</span>', COLOR_PALETTE["observasi"])
    _render_metric_card(col3, "📊 RMSE LSTM (Data-Driven)", '<span style="color: #64748B;">Tidak ada data</span>', COLOR_PALETTE["observasi"])
    _render_metric_card(col4, "🏆 RMSE HIBRIDA (Integrasi)", '<span style="color: #64748B;">Tidak ada data</span>', COLOR_PALETTE["observasi"])

def _add_alert_zones(fig: go.Figure, dynamic_min: float, dynamic_max: float) -> None:
    for y0, y1, warna, label, opacity in ALERT_ZONES:
        y0_clip = max(y0, dynamic_min)
        y1_clip = min(y1, dynamic_max)
        if y0_clip >= y1_clip: continue
        fig.add_hrect(y0=y0_clip, y1=y1_clip, fillcolor=warna, opacity=opacity, line_width=0, layer="below")
        fig.add_annotation(xref="paper", yref="y", x=0.005, y=(y0_clip + y1_clip) / 2, text=f"<b>{label}</b>", showarrow=False, xanchor="left", yanchor="middle", font=dict(color="#1E293B", size=10, family="Arial"), bgcolor="rgba(255,255,255,0.55)")

def build_comparison_chart(df_filtered: pd.DataFrame, data_dsda: Optional[dict]) -> go.Figure:
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

    if df_filtered[COL_OBSERVASI].notna().sum() > 0:
        fig.add_trace(go.Scatter(x=df_filtered[COL_DATETIME], y=df_filtered[COL_OBSERVASI], mode="lines", name="Observasi Stasiun (TMA Aktual)", line=dict(color=COLOR_PALETTE["observasi"], width=2.5, shape="spline", smoothing=0.9), connectgaps=False))

    fig.add_trace(go.Scatter(x=df_filtered[COL_DATETIME], y=df_filtered[COL_UTIDE], mode="lines", name="Prediksi UTide (Pendekatan Astronomis)", line=dict(color=COLOR_PALETTE["utide"], width=1.5, shape="spline", smoothing=0.9)))
    fig.add_trace(go.Scatter(x=df_filtered[COL_DATETIME], y=df_filtered[COL_LSTM], mode="lines", name="Prediksi LSTM (Pendekatan Data-Driven)", line=dict(color=COLOR_
