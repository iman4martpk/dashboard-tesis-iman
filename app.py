"""
Dashboard Monitoring Pasut Hibrida (UTide + LSTM) - Stasiun Pasar Ikan, Jakarta.

Aplikasi Streamlit ini menampilkan perbandingan performa tiga metode
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
import requests
import streamlit as st
from lxml import html

# =========================================================================
# 1. KONSTANTA & KONFIGURASI GLOBAL
# =========================================================================

PAGE_TITLE = "Dashboard Pasut Hibrida Pasar Ikan"
PAGE_ICON = "🌊"

# 3 File Master Data
DATA_FILE_HIBRIDA = "HASIL_FINAL_TESIS_PASUT_HIBRIDA.csv"
DATA_FILE_LSTM = "HASIL_FINAL_TESIS_PASUT_LSTM_MURNI.csv"
DATA_FILE_OBSERVASI = "HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv"

COL_DATETIME = "Datetime"
COL_OBSERVASI = "TMA_Pasar_Ikan"
COL_UTIDE = "Prediksi_Harmonik_UTIDE"
COL_LSTM = "Prediksi_LSTM_Murni"
COL_HIBRIDA = "Prediksi_Hibrida_Final"

# --- Palet warna baru (lebih eye-catching) -------------------------------
COLOR_PALETTE = {
    "primary": "#0B3D4C",
    "success": "#22c55e",
    "danger": "#ef4444",
    # Garis: observasi solid gelap, prediksi transparan (dipakai lewat rgba di bawah)
    "observasi": "#0F172A",                 # slate-900, 100% solid, garis "kebenaran lapangan"
    "utide": "rgba(0, 194, 255, 0.50)",     # cyan elektrik, transparan 50%
    "lstm": "rgba(255, 45, 149, 0.50)",     # magenta terang, transparan 50%
    "hibrida": "rgba(124, 58, 237, 0.60)",  # ungu vivid, transparan 60% (sedikit lebih tebal sbg andalan)
    
    # Pita gradasi level siaga (dipakai di background chart)
    "aman": "#BAE6FD",     # biru muda kalem (sky-200)
    "waspada": "#EA580C",  # jingga tua
    "awas": "#DC2626",     # merah
}

# --- Zona / pita siaga pada grafik (cm) ----------------------------------
Y_AXIS_MIN = 100
Y_AXIS_MAX = 280

ALERT_ZONES = [
    # (y0, y1, warna, label, opacity)
    (Y_AXIS_MIN, 230, COLOR_PALETTE["aman"], "KONDISI AMAN", 0.25),
    (230, 250, COLOR_PALETTE["waspada"], "WASPADA ROB", 0.32),
    (250, Y_AXIS_MAX, COLOR_PALETTE["awas"], "AWAS ROB", 0.30),
]

# --- LOGIKA DINAMIS 2 HARI KE BELAKANG & 2 HARI KE DEPAN ---
HARI_INI = datetime.now()
START_REALTIME = (HARI_INI - timedelta(days=2)).strftime("%Y-%m-%d 00:00:00")
END_REALTIME = (HARI_INI + timedelta(days=2)).strftime("%Y-%m-%d 23:00:00")

PRESETS = {
    "📊 Ringkasan Real-Time (Jendela 4 Hari)": {
        "start": START_REALTIME,
        "end": END_REALTIME,
        "desc": "Kondisi operasional berjalan (2 hari lalu s.d 2 hari ke depan).",
    },
    "Studi Kasus 1: Periode Mei": {
        "start": "2026-05-14 00:00:00",
        "end": "2026-05-21 00:00:00",
        "desc": "Segmen awal bulan Mei.",
    },
    "Studi Kasus 2: Periode Juni": {
        "start": "2026-06-12 00:00:00",
        "end": "2026-06-19 00:00:00",
        "desc": "Fase anomali residu meteorologis tinggi.",
    },
    "Studi Kasus 3: Periode Juli": {
        "start": "2026-07-12 00:00:00",
        "end": "2026-07-19 00:00:00",
        "desc": "Batas data aktual observasi lapangan Tesis.",
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
# 2. FUNGSI SCRAPING REAL-TIME BPBD (CACHE 10 MENIT TETAP AKTIF)
# =========================================================================
NAMA_POS_TARGET = "Pasar Ikan"
TMA_MIN, TMA_MAX = -300, 500


def _extract_pasar_ikan_reading(tree: html.HtmlElement) -> Optional[dict]:
    """
    Cari baris "Pasar Ikan" secara generik (nama pos & posisi kolom, bukan
    index baris/kolom tetap) supaya tidak gampang gagal saat markup situs
    BPBD sedikit berubah.
    """
    header_cells = tree.xpath("//table//thead//tr[last()]//*[self::th or self::td]")
    jam_list = []
    for cell in header_cells:
        text = " ".join(cell.itertext())
        match = re.search(r"\b\d{1,2}:\d{2}\b", text)
        if match:
            jam_list.append(match.group())
    if not jam_list:
        return None
    jam = jam_list[0]

    target_rows = tree.xpath(
        f"//table//tbody//tr[.//*[self::td or self::th]"
        f"[1][contains(normalize-space(.), '{NAMA_POS_TARGET}')]]"
    )
    if not target_rows:
        return None

    cells = target_rows[0].xpath(".//*[self::td or self::th]")
    if len(cells) < 2:
        return None

    value_text = " ".join(cells[1].itertext())
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value_text)
    if not match:
        return None
    angka_tma = float(match.group())

    if not (TMA_MIN <= angka_tma <= TMA_MAX):
        return None

    return {"jam": jam, "tma": angka_tma}


@st.cache_data(ttl=600)
def fetch_realtime_data() -> Optional[dict]:
    url = "https://bpbd.jakarta.go.id/waterlevel"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        tree = html.fromstring(response.content)
        return _extract_pasar_ikan_reading(tree)
    except Exception:
        return None


# =========================================================================
# 3. STYLING (CSS) & DATA PIPELINE
# =========================================================================
def inject_custom_css() -> None:
    st.markdown(
        f"""
        <style>
        [data-baseweb="popover"] {{ transform: scale(0.95) !important; transform-origin: top left !important; }}
        [data-baseweb="popover"] > div {{ max-width: 260px !important; }}
        .block-container {{ padding-top: 3.2rem !important; padding-bottom: 0rem !important; max-width: 95% !important; }}
        [data-testid="stVerticalBlock"] > div {{ gap: 0px !important; }}
        .stApp {{ background-color: #ffffff; }}
        .header-text {{ text-align: center; width: 100%; margin-top: 5px; margin-bottom: 0px !important; padding-bottom: 0px !important; }}
        html, body, [class*="css"] {{ font-family: Arial, Helvetica, sans-serif !important; -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; }}
        [data-testid="stHeader"] {{ background-color: transparent !important; background: transparent !important; z-index: 99995 !important; }}
        div[data-testid="collapsedControl"] {{ background-color: {COLOR_PALETTE['primary']} !important; border-radius: 8px !important; padding: 6px !important; box-shadow: 0 4px 12px rgba(11, 61, 76, 0.3) !important; }}
        button[data-testid="stSidebarCollapseButton"] {{ background-color: {COLOR_PALETTE['primary']} !important; border-radius: 8px !important; }}
        div[data-testid="collapsedControl"] svg, button[data-testid="stSidebarCollapseButton"] svg {{ fill: #F8FAFC !important; color: #F8FAFC !important; }}
        div[data-testid="stMetric"] {{ background-color: #ffffff !important; border: 1px solid #e2e8f0 !important; border-left: 4px solid {COLOR_PALETTE['primary']} !important; padding: 4px 10px !important; border-radius: 8px !important; box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important; min-height: 55px !important; display: flex !important; flex-direction: column !important; justify-content: center !important; }}
        div[data-testid="stMetricLabel"] {{ color: #64748b !important; font-weight: 600 !important; font-size: 0.68rem !important; margin-bottom: -4px !important; white-space: nowrap !important; }}
        [data-testid="stMetricValue"] {{ font-size: 14px !important; font-weight: 700 !important; color: #0f172a !important; white-space: nowrap !important; }}
        div[data-testid="stMetricDelta"] {{ display: none !important; }}
        div[data-testid="column"] {{ padding: 0 4px !important; }}
        .summary-box {{ background-color: #f1f5f9 !important; padding: 6px 12px !important; border-radius: 8px !important; margin-top: 4px !important; margin-bottom: 8px !important; border-left: 5px solid {COLOR_PALETTE['primary']} !important; text-align: center !important; }}
        .summary-text {{ font-family: Arial, Helvetica, sans-serif !important; font-weight: 600; font-size: 0.82rem; color: #1e293b; }}
        @media (max-width: 767px) {{ .block-container {{ padding-top: 3.4rem !important; }} .header-text h2 {{ font-size: 1.1rem !important; margin-top: 10px !important; }} .summary-text {{ font-size: 0.72rem !important; }} [data-testid="stMetricValue"] {{ font-size: 12px !important; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _parse_datetime_column(series: pd.Series) -> pd.Series:
    """
    Parsing Datetime yang tahan terhadap format campuran (mis. file lama
    berformat "%m/%d/%Y %H:%M" bercampur dengan baris baru dari scraper yang
    berformat ISO "%Y-%m-%d %H:%M:%S"). Baris yang gagal diparse dijadikan
    NaT (bukan meng-crash-kan seluruh aplikasi) lalu dibuang.
    """
    parsed = pd.to_datetime(series, format="mixed", dayfirst=False, errors="coerce")
    n_invalid = parsed.isna().sum()
    if n_invalid:
        st.warning(
            f"⚠️ {n_invalid} baris memiliki format tanggal yang tidak valid dan diabaikan."
        )
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed


# ⚠️ CACHE DIMATIKAN DI SINI AGAR STREAMLIT SELALU BACA CSV TERBARU DARI GITHUB ⚠️
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # 1. Baca data Prediksi (Grid Utuh)
    df_hibrida = pd.read_csv(DATA_FILE_HIBRIDA)
    df_lstm = pd.read_csv(DATA_FILE_LSTM)

    # 2. Baca data Observasi (Independent File)
    try:
        df_obs = pd.read_csv(DATA_FILE_OBSERVASI)
    except FileNotFoundError:
        df_obs = pd.DataFrame({COL_DATETIME: pd.Series(dtype="datetime64[ns]"), COL_OBSERVASI: pd.Series(dtype="float64")})

    # 🔥 Paksa semua kolom Datetime menjadi format datetime murni tanpa timezone,
    # tahan terhadap format string yang campur-campur (lihat _parse_datetime_column).
    df_hibrida[COL_DATETIME] = _parse_datetime_column(df_hibrida[COL_DATETIME])
    df_lstm[COL_DATETIME] = _parse_datetime_column(df_lstm[COL_DATETIME])
    df_obs[COL_DATETIME] = _parse_datetime_column(df_obs[COL_DATETIME])

    df_hibrida = df_hibrida.dropna(subset=[COL_DATETIME])
    df_lstm = df_lstm.dropna(subset=[COL_DATETIME])
    df_obs = df_obs.dropna(subset=[COL_DATETIME])

    # 3. Hapus kolom observasi bawaan di df_hibrida jika masih ada (agar tidak bentrok)
    if COL_OBSERVASI in df_hibrida.columns:
        df_hibrida = df_hibrida.drop(columns=[COL_OBSERVASI])

    # 4. GABUNGKAN (Merge) - Sekarang dijamin presisi karena tipe datanya sudah kembar!
    df_hibrida = pd.merge(df_hibrida, df_obs[[COL_DATETIME, COL_OBSERVASI]], on=COL_DATETIME, how="left")

    return df_hibrida, df_lstm


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


def compute_kpis(df_filtered: pd.DataFrame, df_lstm_filtered: pd.DataFrame) -> Optional[KpiResult]:
    valid_idx = df_filtered[COL_OBSERVASI].notna()
    if valid_idx.sum() == 0:
        return None

    observasi = df_filtered.loc[valid_idx, COL_OBSERVASI]

    def rmse(prediksi: pd.Series) -> float:
        return float(np.sqrt(np.mean((observasi - prediksi) ** 2)))

    rmse_utide = rmse(df_filtered.loc[valid_idx, COL_UTIDE])
    rmse_lstm = rmse(df_lstm_filtered.loc[valid_idx, COL_LSTM])
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
# 5. KOMPONEN TAMPILAN (UI RENDERING)
# =========================================================================
def render_header() -> None:
    st.markdown(
        """<div class="header-text"><h2 style="margin: 0; color: #0F172A; font-family: Arial, Helvetica, sans-serif; font-weight: bold; font-size: 1.55rem;">🌊 MONITORING PASUT HIBRIDA (UTIDE + LSTM) REAL-TIME</h2></div>""",
        unsafe_allow_html=True,
    )


def render_summary_box(pilihan_mode: str, data_dsda: Optional[dict]) -> None:
    if data_dsda:
        info_realtime = f" | 🔴 <b>DSDA Real-time ({data_dsda['jam']}):</b> <span style='color:#DC2626;'>{data_dsda['tma']} cm</span>"
    else:
        info_realtime = " | 🔴 <b>DSDA Real-time:</b> <span style='color:#64748B;'>Offline/Delay</span>"

    st.markdown(
        f"""
        <div class="summary-box">
            <span class="summary-text">
                📍 <b>Stasiun:</b> Pasar Ikan, Jakarta | 🛡️ <b>Mode:</b> {PRESETS[pilihan_mode]['desc']}{info_realtime}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _metrik_badge(metrik: MetodeMetrik) -> str:
    if metrik.is_terbaik:
        return '<span style="color: #22c55e; font-size: 0.65rem; font-weight: bold;">🏆 AKURASI TERTINGGI</span>'
    return f'<span style="color: #ef4444; font-size: 0.65rem; font-weight: bold;">+{metrik.selisih_dari_terbaik:.1f} cm vs Terbaik</span>'


def _render_metric_card(column, label: str, value_html: str, border_color: str) -> None:
    column.markdown(f'<div data-testid="stMetric" style="border-left-color: {border_color} !important;"><label data-testid="stMetricLabel">{label}</label><div data-testid="stMetricValue">{value_html}</div></div>', unsafe_allow_html=True)


def render_kpi_cards(kpi: KpiResult) -> None:
    col1, col2, col3, col4 = st.columns(4)
    _render_metric_card(col1, "📈 REDUKSI EROR (vs UTide)", f'{kpi.reduksi_eror_persen:.2f} % <span style="color: #22c55e; font-size: 0.68rem; font-weight: bold;">▲ OPTIMAL</span>', COLOR_PALETTE["success"])
    _render_metric_card(col2, "📉 RMSE UTIDE MURNI", f"{kpi.utide.rmse:.2f} cm {_metrik_badge(kpi.utide)}", "#00C2FF")
    _render_metric_card(col3, "📊 RMSE LSTM MURNI", f"{kpi.lstm.rmse:.2f} cm {_metrik_badge(kpi.lstm)}", "#FF2D95")
    _render_metric_card(col4, "🏆 RMSE HIBRIDA", f"{kpi.hibrida.rmse:.2f} cm {_metrik_badge(kpi.hibrida)}", "#7C3AED")


def render_empty_kpi_cards() -> None:
    col1, col2, col3, col4 = st.columns(4)
    _render_metric_card(col1, "📈 REDUKSI EROR (vs UTide)", '<span style="color: #64748B;">No Obs Data</span>', COLOR_PALETTE["observasi"])
    _render_metric_card(col2, "📉 RMSE UTIDE MURNI", '<span style="color: #64748B;">No Obs Data</span>', COLOR_PALETTE["observasi"])
    _render_metric_card(col3, "📊 RMSE LSTM MURNI", '<span style="color: #64748B;">No Obs Data</span>', COLOR_PALETTE["observasi"])
    _render_metric_card(col4, "🏆 RMSE HIBRIDA", '<span style="color: #64748B;">No Obs Data</span>', COLOR_PALETTE["observasi"])


def _add_alert_zones(fig: go.Figure) -> None:
    """Gambar pita gradasi level siaga sebagai background chart (bukan garis)."""
    for y0, y1, warna, label, opacity in ALERT_ZONES:
        y0_clip = max(y0, Y_AXIS_MIN)
        y1_clip = min(y1, Y_AXIS_MAX)
        if y0_clip >= y1_clip:
            continue
        fig.add_hrect(
            y0=y0_clip, y1=y1_clip,
            fillcolor=warna, opacity=opacity,
            line_width=0, layer="below",
        )
        fig.add_annotation(
            xref="paper", yref="y",
            x=0.005, y=(y0_clip + y1_clip) / 2,
            text=f"<b>{label}</b>",
            showarrow=False, xanchor="left", yanchor="middle",
            font=dict(color="#1E293B", size=10, family="Arial"),
            bgcolor="rgba(255,255,255,0.55)",
        )


def build_comparison_chart(df_filtered: pd.DataFrame, df_lstm_filtered: pd.DataFrame, data_dsda: Optional[dict]) -> go.Figure:
    fig = go.Figure()

    # 0. Pita gradasi level siaga di lapisan paling belakang
    _add_alert_zones(fig)

    # --- Garis prediksi digambar dulu (transparan), observasi digambar
    #     terakhir di atas (solid) supaya perpotongan antar garis tetap terlihat.

    # 1. Prediksi UTide Murni (transparan, smooth)
    fig.add_trace(go.Scatter(
        x=df_filtered[COL_DATETIME], y=df_filtered[COL_UTIDE],
        mode="lines", name="Prediksi UTide Murni (Astronomis)",
        line=dict(color=COLOR_PALETTE["utide"], width=2.4, shape="spline", smoothing=0.9),
    ))

    # 2. Prediksi LSTM Murni (transparan, smooth)
    fig.add_trace(go.Scatter(
        x=df_lstm_filtered[COL_DATETIME], y=df_lstm_filtered[COL_LSTM],
        mode="lines", name="Prediksi LSTM Murni (Non-Astronomis)",
        line=dict(color=COLOR_PALETTE["lstm"], width=2.4, shape="spline", smoothing=0.9),
    ))

    # 3. Prediksi Hibrida (transparan, smooth)
    fig.add_trace(go.Scatter(
        x=df_filtered[COL_DATETIME], y=df_filtered[COL_HIBRIDA],
        mode="lines", name="Prediksi Hibrida (UTide + LSTM)",
        line=dict(color=COLOR_PALETTE["hibrida"], width=3.0, shape="spline", smoothing=0.9),
    ))

    # 4. Observasi Historis - SOLID, di lapisan paling atas
    if df_filtered[COL_OBSERVASI].notna().sum() > 0:
        fig.add_trace(go.Scatter(
            x=df_filtered[COL_DATETIME], y=df_filtered[COL_OBSERVASI],
            mode="lines", name="Observasi Stasiun (TMA Aktual)",
            line=dict(color=COLOR_PALETTE["observasi"], width=2.6, shape="spline", smoothing=0.9),
            connectgaps=False,
        ))

    # 5. Garis vertikal waktu sekarang
    if data_dsda and data_dsda["tma"] is not None:
        waktu_sekarang_jam = datetime.now().replace(minute=0, second=0, microsecond=0)
        min_date = df_filtered[COL_DATETIME].min()
        max_date = df_filtered[COL_DATETIME].max()

        if pd.notna(min_date) and pd.notna(max_date) and min_date <= waktu_sekarang_jam <= max_date:
            fig.add_vline(
                x=waktu_sekarang_jam.timestamp() * 1000,
                line_width=1.5, line_dash="dot", line_color="#334155",
            )

    fig.update_layout(
        height=430, template="plotly_white", margin=dict(l=10, r=10, t=25, b=10), hovermode="x unified",
        hoverlabel=dict(bgcolor="white", font_size=11, font_family="Arial"), xaxis=dict(tickfont=dict(size=10, family="Arial")),
        yaxis=dict(
            title=dict(text="Tinggi Air (cm)", font=dict(size=11, family="Arial")),
            tickfont=dict(size=10, family="Arial"),
            range=[Y_AXIS_MIN, Y_AXIS_MAX],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, family="Arial")),
        font=dict(family="Arial, Helvetica, sans-serif", color="#1E293B"),
    )
    return fig


def render_data_table(df_filtered: pd.DataFrame, df_lstm_filtered: pd.DataFrame) -> None:
    st.divider()
    st.markdown("<h4 style='margin:0 0 4px 0; padding:0; font-size:14px; font-weight:700; color:#1E293B;'>📋 Potongan Basis Data Numerik Terfilter</h4>", unsafe_allow_html=True)
    df_tampilan = pd.DataFrame({
        COL_DATETIME: df_filtered[COL_DATETIME],
        COL_OBSERVASI: df_filtered[COL_OBSERVASI],
        COL_UTIDE: df_filtered[COL_UTIDE],
        COL_LSTM: df_lstm_filtered[COL_LSTM].values,
        COL_HIBRIDA: df_filtered[COL_HIBRIDA],
    })

    st.dataframe(df_tampilan.reset_index(drop=True), width="stretch")
    csv_data = df_tampilan.to_csv(index=False).encode("utf-8")
    st.download_button(label="📥 Unduh Data Potongan Kerja Ini (.CSV)", data=csv_data, file_name="DATA_INSPEKSI_PASUT_HIBRIDA.csv", mime="text/csv", width="stretch")


# =========================================================================
# 6. SIDEBAR CONTROLS & MAIN
# =========================================================================
def render_sidebar_controls(df: pd.DataFrame) -> tuple[str, Optional[object], Optional[object]]:
    st.sidebar.header("⚡ Kontrol Panel Analisis")
    pilihan_mode = st.sidebar.selectbox("Pilih Mode Analisis / Studi Kasus:", list(PRESETS.keys()), index=DEFAULT_PRESET_INDEX)
    custom_start = custom_end = None
    if pilihan_mode == CUSTOM_PRESET_KEY:
        min_date = df[COL_DATETIME].min().date()
        max_date = df[COL_DATETIME].max().date()
        custom_start = st.sidebar.date_input("Tanggal Mulai", min_date, min_value=min_date, max_value=max_date)
        custom_end = st.sidebar.date_input("Tanggal Selesai", max_date, min_value=min_date, max_value=max_date)
    return pilihan_mode, custom_start, custom_end


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide", page_icon=PAGE_ICON, initial_sidebar_state="expanded")
    inject_custom_css()

    data_dsda = fetch_realtime_data()

    try:
        df, df_lstm = load_data()
    except Exception as e:
        st.error(f"❌ File CSV gagal dimuat. Pastikan nama file sudah benar. Error: {e}")
        st.stop()
        return

    pilihan_mode, custom_start, custom_end = render_sidebar_controls(df)
    mask = filter_by_preset(df, pilihan_mode, custom_start, custom_end)
    df_filtered = df[mask].copy()
    df_lstm_filtered = df_lstm[mask].copy()

    kpi = compute_kpis(df_filtered, df_lstm_filtered)

    render_header()
    render_summary_box(pilihan_mode, data_dsda)

    if kpi is not None:
        render_kpi_cards(kpi)
    else:
        render_empty_kpi_cards()

    st.markdown(
        f"""
        <div style="display: flex; align-items: baseline; margin: 8px 0 3px 0;">
            <h3 style="margin:0; padding:0; font-size:19px; font-weight:600; color:#1E293B;">📈 Grafik Analisis Perbandingan: {pilihan_mode}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fig = build_comparison_chart(df_filtered, df_lstm_filtered, data_dsda)

    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    render_data_table(df_filtered, df_lstm_filtered)


if __name__ == "__main__":
    main()
