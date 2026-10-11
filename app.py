"""
Dashboard Monitoring Pasut Hibrida (UTide + LSTM) - Stasiun Pasar Ikan, Jakarta.

Aplikasi Streamlit ini menampilkan perbandingan performa pendekatan
prediksi pasang surut air laut dengan data observasi independen secara REAL-TIME.
"""

from __future__ import annotations

import re
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

COLOR_PALETTE = {
    "primary": "#0B3D4C",
    "success": "#22c55e",
    "danger": "#ef4444",
    "observasi": "#475569",                 
    "utide": "rgba(0, 194, 255, 0.45)",     
    "lstm": "rgba(16, 185, 129, 0.45)",     
    "hibrida": "rgba(37, 99, 235, 0.65)",   
    "live_15m": "#E11D48", # Merah muda menyala untuk 15-min
    "aman": "#BAE6FD",     
    "waspada": "#EA580C",  
    "awas": "#FF0000",     
}

ALERT_ZONES = [
    (0, 230, COLOR_PALETTE["aman"], "KONDISI AMAN", 0.25),
    (230, 250, COLOR_PALETTE["waspada"], "WASPADA ROB", 0.32),
    (250, 500, COLOR_PALETTE["awas"], "AWAS ROB", 0.25), 
]

HARI_INI = get_now_jkt()
START_REALTIME = (HARI_INI - timedelta(days=3)).strftime("%Y-%m-%d 00:00:00")
END_REALTIME = (HARI_INI + timedelta(days=4)).strftime("%Y-%m-%d 23:00:00")

PRESETS = {
    "📊 Ringkasan Real-Time (Jendela 7 Hari)": {
        "start": START_REALTIME,
        "end": END_REALTIME,
        "desc": "Jendela pengamatan H-3 s.d. H+4.",
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
        .summary-text {{ font-family: Arial, Helvetica, sans-serif !important; font-weight: 600; font-size: 1rem; color: #1e293b; }}
        .eval-box {{ background-color: #f8fafc; padding: 15px; border-radius: 8px; border-left: 5px solid {COLOR_PALETTE['primary']}; margin-bottom: 15px; }}
        
        /* Animasi Berkedip (Blinking) untuk Indikator Live */
        @keyframes pulse-dot {{
            0% {{ opacity: 1; transform: scale(1); }}
            50% {{ opacity: 0.3; transform: scale(0.85); }}
            100% {{ opacity: 1; transform: scale(1); }}
        }}
        @keyframes flash-text {{
            0%, 100% {{ color: #DC2626; }}
            50% {{ color: #EF4444; text-shadow: 0 0 5px rgba(220,38,38,0.5); }}
        }}
        .live-dot {{ display: inline-block; animation: pulse-dot 1.5s infinite ease-in-out; }}
        .live-val {{ animation: flash-text 2s infinite; font-weight: 800; font-size: 1.1rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

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
    if data_dsda:
        info_realtime = f" | <span class='live-dot'>🔴</span> <b>DSDA Live (per 15 mnt) - Update {data_dsda['jam']}:</b> <span class='live-val'>{data_dsda['tma']} cm</span>"
    else:
        info_realtime = " | 🔴 <b>DSDA Live (per 15 mnt):</b> <span style='color:#64748B;'>Offline/Delay</span>"
        
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
    for col, judul in zip([col1, col2, col3, col4], ["📈 REDUKSI EROR (vs UTide)", "📉 RMSE UTIDE (Astronomis)", "📊 RMSE LSTM (Data-Driven)", "🏆 RMSE HIBRIDA (Integrasi)"]):
        _render_metric_card(col, judul, '<span style="color: #64748B;">Tidak ada data</span>', COLOR_PALETTE["observasi"])

def _add_alert_zones(fig: go.Figure, dynamic_min: float, dynamic_max: float) -> None:
    for y0, y1, warna, label, opacity in ALERT_ZONES:
        y0_clip = max(y0, dynamic_min)
        y1_clip = min(y1, dynamic_max)
        if y0_clip >= y1_clip: continue
        fig.add_hrect(y0=y0_clip, y1=y1_clip, fillcolor=warna, opacity=opacity, line_width=0, layer="below")
        fig.add_annotation(xref="paper", yref="y", x=0.005, y=(y0_clip + y1_clip) / 2, text=f"<b>{label}</b>", showarrow=False, xanchor="left", yanchor="middle", font=dict(color="#1E293B", size=10, family="Arial"), bgcolor="rgba(255,255,255,0.55)")

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
        fig.add_trace(go.Scatter(x=df_filtered[COL_DATETIME], y=df_filtered[COL_OBSERVASI], mode="lines", name="Observasi Evaluasi (Per Jam)", line=dict(color=COLOR_PALETTE["observasi"], width=2.5, shape="spline", smoothing=0.9), connectgaps=False))

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
                    name="Live Scrape DSDA (Per 15 Menit)", 
                    marker=dict(size=4, color=COLOR_PALETTE["live_15m"]),
                    line=dict(color=COLOR_PALETTE["live_15m"], width=2.0, dash="dot", shape="spline"), 
                    connectgaps=False
                ))

    # 3. Prediksi Model
    fig.add_trace(go.Scatter(x=df_filtered[COL_DATETIME], y=df_filtered[COL_UTIDE], mode="lines", name="Prediksi UTide", line=dict(color=COLOR_PALETTE["utide"], width=1.5, shape="spline", smoothing=0.9)))
    fig.add_trace(go.Scatter(x=df_filtered[COL_DATETIME], y=df_filtered[COL_LSTM], mode="lines", name="Prediksi LSTM", line=dict(color=COLOR_PALETTE["lstm"], width=1.5, shape="spline", smoothing=0.9)))
    fig.add_trace(go.Scatter(x=df_filtered[COL_DATETIME], y=df_filtered[COL_HIBRIDA], mode="lines", name="Prediksi Hibrida (Integrasi)", line=dict(color=COLOR_PALETTE["hibrida"], width=2.0, shape="spline", smoothing=0.9)))

    # Garis Waktu Sekarang
    if data_dsda and data_dsda["tma"] is not None:
        waktu_sekarang_jam = get_now_jkt()
        min_date = df_filtered[COL_DATETIME].min()
        max_date = df_filtered[COL_DATETIME].max()
        if pd.notna(min_date) and pd.notna(max_date) and min_date <= waktu_sekarang_jam <= max_date:
            jam_menit_str = waktu_sekarang_jam.strftime("%H:%M")
            fig.add_vline(x=waktu_sekarang_jam, line_width=1.5, line_dash="dot", line_color="#334155", annotation_text=f" Waktu Sekarang ({jam_menit_str} WIB)", annotation_position="top right")

    fig.update_layout(height=450, template="plotly_white", margin=dict(l=10, r=10, t=25, b=10), hovermode="x unified", hoverlabel=dict(bgcolor="white", font_size=11, font_family="Arial"), xaxis=dict(tickfont=dict(size=10, family="Arial")), yaxis=dict(title=dict(text="Tinggi Air (cm)", font=dict(size=11, family="Arial")), tickfont=dict(size=10, family="Arial"), range=[dynamic_min, dynamic_max]), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, family="Arial")), font=dict(family="Arial, Helvetica, sans-serif", color="#1E293B"))
    return fig

def render_data_table(df_filtered: pd.DataFrame) -> None:
    st.divider()
    st.markdown("<h4 style='margin:0 0 10px 0; padding:0; font-size:16px; font-weight:700; color:#1E293B;'>📋 Data Numerik Mentah (Tabular)</h4>", unsafe_allow_html=True)
    df_tampilan = pd.DataFrame({
        COL_DATETIME: df_filtered[COL_DATETIME],
        COL_OBSERVASI: df_filtered[COL_OBSERVASI],
        COL_UTIDE: df_filtered[COL_UTIDE],
        COL_LSTM: df_filtered[COL_LSTM],
        COL_HIBRIDA: df_filtered[COL_HIBRIDA],
    })
    
    st.dataframe(df_tampilan.reset_index(drop=True), use_container_width=True, hide_index=True)
    csv_data = df_tampilan.to_csv(index=False).encode("utf-8")
    st.download_button(label="📥 Unduh Data Tabular Ini (.CSV)", data=csv_data, file_name="DATA_INSPEKSI_PASUT.csv", mime="text/csv")


# =========================================================================
# 6. SEKSI EVALUASI TESIS (ANALISIS MENDALAM)
# =========================================================================
def render_thesis_analysis(df_master: pd.DataFrame, df_filtered: pd.DataFrame) -> None:
    st.divider()
    st.markdown("""<div class="eval-box"><h3 style='color:#1E293B; font-size: 18px; margin: 0px;'>🔬 Analisis Kinerja Peramalan (Evaluasi Tesis)</h3><p style='color:#64748b; font-size: 13px; margin: 5px 0 0 0;'>Modul perhitungan metrik akurasi (Akurasi, RMSE, MAE, Koefisien Determinasi) untuk membuktikan peningkatan performa hasil observasi berdasarkan rekaman per jam.</p></div>""", unsafe_allow_html=True)
    
    scope = st.radio("Pilih Cakupan Analisis Data:", ["Sesuai Tampilan Grafik", "Rekap Bulanan", "Rentang Kustom"], horizontal=True)
    eval_df = df_filtered.copy()
    
    if scope == "Rekap Bulanan":
        df_master_copy = df_master.copy()
        df_master_copy['MonthYear'] = df_master_copy[COL_DATETIME].dt.to_period('M')
        months = sorted(df_master_copy['MonthYear'].dropna().unique(), reverse=True)
        last_month_period = pd.Period(get_now_jkt(), 'M') - 1
        default_idx = months.index(last_month_period) if last_month_period in months else 0
        
        col_month, _ = st.columns([1, 2])
        with col_month: selected_month = st.selectbox("📅 Pilih Bulan Evaluasi:", months, index=default_idx, format_func=lambda x: x.strftime('%B %Y'))
        mask_month = df_master_copy['MonthYear'] == selected_month
        eval_df = df_master_copy[mask_month].copy()
        
    elif scope == "Rentang Kustom":
        min_date, max_date = df_master[COL_DATETIME].min().date(), df_master[COL_DATETIME].max().date()
        col_date, _ = st.columns([1.5, 1.5])
        with col_date: custom_dates = st.date_input("📅 Pilih Rentang Tanggal:", [min_date, max_date], min_value=min_date, max_value=max_date)
            
        if len(custom_dates) == 2:
            c_start, c_end = custom_dates
            mask_custom = (df_master[COL_DATETIME].dt.date >= c_start) & (df_master[COL_DATETIME].dt.date <= c_end)
            eval_df = df_master[mask_custom].copy()
        else:
            st.info("⚠️ Silakan pilih tanggal akhir untuk memproses analisis.")
            return
    
    valid_idx = eval_df[COL_OBSERVASI].notna()
    if valid_idx.sum() < 2:
        st.warning("⚠️ Data observasi tidak mencukupi untuk dihitung metrik akurasinya pada rentang waktu ini.")
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
    
    st.markdown(f"*(Data divalidasi berdasarkan sampel **{len(obs)} jam observasi aktual**)*")
    st.dataframe(df_metrics.style.highlight_min(subset=["RMSE (cm) ↓", "MAE (cm) ↓"], color='#bbf7d0', axis=0).highlight_max(subset=["Akurasi (%) ↑", "Koefisien Determinasi (R²) ↑"], color='#bbf7d0', axis=0).format({"Akurasi (%) ↑": "{:.2f}%", "RMSE (cm) ↓": "{:.3f}", "MAE (cm) ↓": "{:.3f}", "Koefisien Determinasi (R²) ↑": "{:.3f}"}), use_container_width=True, hide_index=True)
    
    if "Hibrida" in best_method:
        st.success(f"**Interpretasi Akademis:** Berdasarkan metrik evaluasi di atas, pendekatan **Hibrida UTide+LSTM (Integrasi Residual)** terbukti memberikan performa peramalan paling akurat dengan tingkat *Root Mean Square Error* (RMSE) terendah sebesar **{best_rmse:.2f} cm**. Penggunaan model pembelajaran mesin *data-driven* ini berhasil menangkap pola anomali non-astronomis dan mampu mereduksi tingkat kesalahan dari pendekatan astronomis murni (UTide) secara signifikan sebesar **{impr_utide:.1f}%**.", icon="✅")
    else:
        st.info(f"**Interpretasi Akademis:** Pada rentang observasi saat ini, pendekatan **{best_method}** menunjukkan kinerja paling presisi dibandingkan pendekatan lain dengan nilai hamburan eror (RMSE) sebesar **{best_rmse:.2f} cm**.", icon="ℹ️")


# =========================================================================
# 7. SIDEBAR CONTROLS & MAIN APP
# =========================================================================
def render_sidebar_controls(df: pd.DataFrame) -> tuple[str, Optional[object], Optional[object], bool]:
    st.sidebar.header("⚡ Kontrol Panel Analisis")
    pilihan_mode = st.sidebar.selectbox("Pilih Mode Analisis / Studi Kasus:", list(PRESETS.keys()), index=DEFAULT_PRESET_INDEX)
    custom_start = custom_end = None
    if pilihan_mode == CUSTOM_PRESET_KEY:
        min_date = df[COL_DATETIME].min().date()
        max_date = df[COL_DATETIME].max().date()
        custom_start = st.sidebar.date_input("Tanggal Mulai", min_date, min_value=min_date, max_value=max_date)
        custom_end = st.sidebar.date_input("Tanggal Selesai", max_date, min_value=min_date, max_value=max_date)
        
    st.sidebar.divider()
    st.sidebar.markdown("**Fitur Visualisasi Tambahan:**")
    show_15min = st.sidebar.checkbox("🔴 Tampilkan Garis 15-Menitan (Live Scrape)", value=True, help="Menampilkan titik potong observasi DSDA per 15 menit. Grafik akan ditumpuk dengan data jam-jaman.")
    
    return pilihan_mode, custom_start, custom_end, show_15min

def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon=PAGE_ICON, initial_sidebar_state="expanded")
    
    # Fitur Auto-Refresh Streamlit: Refresh browser otomatis tiap 15 menit (15 * 60 * 1000 milidetik)
    st_autorefresh(interval=15 * 60 * 1000, key="data_refresh_15m")
    
    inject_custom_css()
    data_dsda = fetch_realtime_status()

    try:
        df_master = load_data()
    except Exception as e:
        st.error(f"❌ File CSV gagal dimuat. Error: {e}")
        st.stop()
        return

    pilihan_mode, custom_start, custom_end, show_15min = render_sidebar_controls(df_master)
    mask = filter_by_preset(df_master, pilihan_mode, custom_start, custom_end)
    df_filtered = df_master[mask].copy()

    kpi = compute_kpis(df_filtered)

    render_header()
    render_summary_box(pilihan_mode, data_dsda)

    if kpi is not None:
        render_kpi_cards(kpi)
    else:
        render_empty_kpi_cards()
    
    st.markdown(f"""<div style="display: flex; align-items: baseline; margin: 8px 0 3px 0;"><h3 style="margin:0; padding:0; font-size:19px; font-weight:600; color:#1E293B;">📈 Grafik Analisis Perbandingan: {pilihan_mode}</h3></div>""", unsafe_allow_html=True)

    # 1. TAMPILKAN GRAFIK TERLEBIH DAHULU (Dengan parameter show_15min)
    fig = build_comparison_chart(df_filtered, data_dsda, show_15min)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    
    # 2. TAMPILKAN ANALISIS KINERJA (EVALUASI TESIS)
    render_thesis_analysis(df_master, df_filtered)
    
    # 3. TAMPILKAN DATA MENTAH TABULAR (PALING BAWAH)
    render_data_table(df_filtered)


if __name__ == "__main__":
    main()
