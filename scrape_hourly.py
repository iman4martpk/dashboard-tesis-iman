"""
Scraper BPBD -> Update File Khusus Observasi.
Versi Final: Single Source of Truth (SSOT)
"""

import sys
import re
from datetime import datetime, timedelta
import pandas as pd
import pytz
import requests
from lxml import html

# Konfigurasi Zona Waktu & Nama File
TZ = pytz.timezone("Asia/Jakarta")
FILE_OBSERVASI = "HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv"

# XPath sesuai struktur web BPBD Jakarta
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
            # Ekstrak jam (misal: "08:00")
            jam_str = re.search(r"\d{2}:\d{2}", jam_mentah[0]).group()
            # Ekstrak angka TMA (misal: "162")
            angka_tma = float(re.search(r"[-+]?\d+", tma_mentah[0]).group())
            return jam_str, angka_tma
    except Exception as e:
        print(f"❌ Error scraping: {e}")
    return None, None

def resolve_target_datetime(now_naive: datetime, jam_str: str) -> pd.Timestamp:
    """Menggabungkan tanggal hari ini dengan jam hasil scrape."""
    target_time = datetime.strptime(jam_str, "%H:%M").time()
    candidate = datetime.combine(now_naive.date(), target_time)
    
    # Kalau jam di web lebih besar dari jam sekarang (web telat update), mundurkan 1 hari
    if candidate > now_naive:
        candidate -= timedelta(days=1)
        
    return pd.Timestamp(candidate)

def main() -> int:
    # 1. Ambil waktu sekarang (WIB) tanpa timezone info agar cocok dengan Pandas
    now_naive = datetime.now(TZ).replace(tzinfo=None)
    
    # 2. Scrape data
    jam_str, angka_tma = fetch_realtime_data()
    if jam_str is None or angka_tma is None:
        print("🚨 Gagal mendapatkan data valid dari web BPBD.")
        return 1

    # 3. Tentukan target timestamp
    target_ts = resolve_target_datetime(now_naive, jam_str)
    
    # 4. Buka file observasi
    try:
        df = pd.read_csv(FILE_OBSERVASI)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
    except FileNotFoundError:
        # Buat dataframe kosong jika file belum ada/terhapus
        df = pd.DataFrame(columns=["Datetime", "TMA_Pasar_Ikan"])
        df["Datetime"] = pd.to_datetime(df["Datetime"])

    # 5. Cek apakah jam tersebut sudah ada di CSV
    mask = df["Datetime"] == target_ts

    if mask.sum() > 0:
        # Jika ada -> Update baris tersebut
        df.loc[mask, "TMA_Pasar_Ikan"] = angka_tma
        print(f"✅ Update baris lama: {target_ts} -> {angka_tma} cm")
    else:
        # Jika tidak ada -> Tambah baris baru di bawah
        new_row = pd.DataFrame([{"Datetime": target_ts, "TMA_Pasar_Ikan": angka_tma}])
        df = pd.concat([df, new_row], ignore_index=True)
        print(f"📝 Tambah baris baru: {target_ts} -> {angka_tma} cm")

    # 6. Rapikan urutan dan format waktu, lalu save
    df = df.sort_values("Datetime").reset_index(drop=True)
    df["Datetime"] = df["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        df.to_csv(FILE_OBSERVASI, index=False)
        print("💾 File observasi berhasil disimpan ke repositori.")
    except Exception as e:
        print(f"❌ Gagal menyimpan file: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
