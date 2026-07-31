"""
Scraper BPBD -> Update File Khusus Observasi.
Versi Final: Murni APPEND (Tambah 1 baris ke paling bawah tanpa rewrite)
"""

import sys
import os
import re
from datetime import datetime, timedelta
import pandas as pd
import pytz
import requests
from lxml import html

# Konfigurasi Zona Waktu & Nama File
TZ = pytz.timezone("Asia/Jakarta")
FILE_OBSERVASI = "HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv"

XPATH_JAM = '/html/body/main/div/div/article/div/div[2]/div/div[1]/div[1]/div[2]/div/table/thead/tr[2]/td[2]/text()'
XPATH_TMA = '/html/body/main/div/div/article/div/div[2]/div/div[1]/div[1]/div[2]/div/table/tbody/tr[9]/td[2]/text()'

def fetch_realtime_data():
    """Scrape jam & nilai TMA Pasar Ikan terbaru dari halaman BPBD."""
    url = "https://bpbd.jakarta.go.id/waterlevel"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        tree = html.fromstring(response.content)

        jam_mentah = tree.xpath(XPATH_JAM)
        tma_mentah = tree.xpath(XPATH_TMA)

        if jam_mentah and tma_mentah:
            jam_str = re.search(r"\d{2}:\d{2}", jam_mentah[0]).group()
            angka_tma = float(re.search(r"[-+]?\d+", tma_mentah[0]).group())
            return jam_str, angka_tma
    except Exception as e:
        print(f"❌ Error scraping: {e}")
    return None, None

def resolve_target_datetime(now_naive: datetime, jam_str: str) -> pd.Timestamp:
    """Menggabungkan tanggal hari ini dengan jam hasil scrape."""
    target_time = datetime.strptime(jam_str, "%H:%M").time()
    candidate = datetime.combine(now_naive.date(), target_time)
    
    if candidate > now_naive:
        candidate -= timedelta(days=1)
        
    return pd.Timestamp(candidate)

def main() -> int:
    now_naive = datetime.now(TZ).replace(tzinfo=None)
    
    # 1. Scrape data dari web
    jam_str, angka_tma = fetch_realtime_data()
    if jam_str is None or angka_tma is None:
        print("🚨 Gagal mendapatkan data valid dari web BPBD.")
        return 1

    # 2. Tentukan timestamp dari data web tersebut
    target_ts = resolve_target_datetime(now_naive, jam_str)
    
    # 3. Cek apakah file sudah ada, dan apakah jam tersebut sudah tercatat
    file_exists = os.path.exists(FILE_OBSERVASI)
    if file_exists:
        # Cuma baca kolom Datetime biar sangat ringan di memori
        df_exist = pd.read_csv(FILE_OBSERVASI, usecols=["Datetime"])
        existing_dates = pd.to_datetime(df_exist["Datetime"]).dt.tz_localize(None)
        
        # Kalau jam tersebut sudah ada, langsung KELUAR tanpa nyentuh/nulis ulang CSV
        if target_ts in existing_dates.values:
            print(f"ℹ️ Data jam {target_ts} sudah ada di database. Skip append.")
            return 0

    # 4. Jika datanya benar-benar baru, siapkan HANYA 1 baris data tersebut
    new_row = pd.DataFrame([{
        "Datetime": target_ts.strftime("%Y-%m-%d %H:%M:%S"), 
        "TMA_Pasar_Ikan": angka_tma
    }])
    
    # 5. APPEND (Suntikkan di baris paling bawah) tanpa merusak/menulis ulang data di atasnya
    try:
        new_row.to_csv(FILE_OBSERVASI, mode='a', header=not file_exists, index=False)
        print(f"📝 Berhasil APPEND murni 1 baris baru: {target_ts} -> {angka_tma} cm")
    except Exception as e:
        print(f"❌ Gagal melakukan append ke file: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
