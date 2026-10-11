#!/usr/bin/env python3
"""
Scraper Pasut -> Update File Observasi TMA Pasar Ikan (Untuk Dashboard Tesis).

Perilaku:
- Menjalankan Selenium (Headless) mengambil teknik Deep Scan dari scraper_khusus.
- Menyimpan hasil observasi ke HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv.
- Menggunakan File Lock agar aman jika dieksekusi berulang kali oleh Cron.
"""

from __future__ import annotations

import os
import time
import fcntl
import logging
import re
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
import pandas as pd
import pytz

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# --------------------------------------------------------------------------- #
# Konfigurasi
# --------------------------------------------------------------------------- #
TZ = pytz.timezone("Asia/Jakarta")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_OBSERVASI = os.path.join(BASE_DIR, "HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv")
LOCK_FILE = os.path.join(BASE_DIR, ".scraper_bpbd_observasi.lock")

URL_DSDA = "https://poskobanjir.dsdadki.web.id/"

# Range kewajaran nilai TMA Pasar Ikan dalam Centimeter (cm)
TMA_MIN_CM, TMA_MAX_CM = -300, 500

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("scraper_hibrida")

# --------------------------------------------------------------------------- #
# Logika Scraper (Adaptasi dari scraper_khusus)
# --------------------------------------------------------------------------- #
def run_selenium_scraper() -> float | None:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = None
    nilai_cm = None
    
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        log.info("Mengakses website DSDA (Deep Scan)...")
        driver.get(URL_DSDA)
        time.sleep(15)  # Tunggu Javascript me-render tabel
        
        all_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Pasar Ikan')]/ancestor::tr")
        
        for row in all_elements:
            row_text = row.get_attribute("innerText")
            nums = re.findall(r"(\d{3})", row_text)
            for n in nums:
                val_n = int(n)
                # Validasi nilai wajar dalam cm (TMA Pasar Ikan biasanya 120-300an cm)
                if 120 <= val_n <= TMA_MAX_CM: 
                    nilai_cm = float(val_n) # Tesis Iman menggunakan cm, bukan meter
                    break
            if nilai_cm is not None: 
                break

        if nilai_cm is None:
            log.info("Skenario 1 gagal, mencari via snippet text body...")
            all_text = driver.find_element(By.TAG_NAME, "body").get_attribute("innerText")
            pos = all_text.find("Pasar Ikan")
            if pos != -1:
                snippet = all_text[pos:pos+200]
                nums = re.findall(r"(\d{3})", snippet)
                for n in nums:
                    val_n = int(n)
                    if 120 <= val_n <= TMA_MAX_CM:
                        nilai_cm = float(val_n)
                        break

    except Exception as e:
        log.error(f"Scraper Driver Error: {e}")
    finally:
        if driver: 
            driver.quit()
            
    return nilai_cm

# --------------------------------------------------------------------------- #
# Penyimpanan (Append-only & Idempotent)
# --------------------------------------------------------------------------- #
def append_to_csv(waktu: datetime, nilai_cm: float):
    waktu_str = waktu.strftime("%Y-%m-%d %H:%M:%S")
    file_exists = os.path.exists(FILE_OBSERVASI)
    
    # Cek duplikasi agar cron yang jalan berulang tidak memasukkan data ganda pada jam/menit yang sama
    if file_exists:
        try:
            df_exist = pd.read_csv(FILE_OBSERVASI)
            # Normalisasi kolom waktu untuk komparasi (sampai batas menit)
            exist_times = pd.to_datetime(df_exist["Datetime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
            if waktu.strftime("%Y-%m-%d %H:%M") in exist_times.values:
                log.info(f"Data untuk {waktu.strftime('%Y-%m-%d %H:%M')} sudah ada. Skip append.")
                return
        except Exception as e:
            log.warning(f"Gagal membaca CSV lama: {e}")
            
    # Buat baris baru
    new_row = pd.DataFrame([{"Datetime": waktu_str, "TMA_Pasar_Ikan": nilai_cm}])
    new_row.to_csv(FILE_OBSERVASI, mode="a", header=not file_exists, index=False)
    log.info(f"✅ Berhasil APPEND 1 baris baru: {waktu_str} -> {nilai_cm} cm")

@contextmanager
def file_lock(lock_path: str):
    """Cegah dua proses cron menulis CSV secara bersamaan."""
    open(lock_path, 'a').close()
    with open(lock_path, "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield
        except BlockingIOError:
            log.warning("Proses cron lain sedang berjalan (lock aktif). Exit.")
            raise SystemExit(0)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)

# --------------------------------------------------------------------------- #
# Main Execution
# --------------------------------------------------------------------------- #
def main():
    sekarang = datetime.now(TZ).replace(tzinfo=None)
    
    # Pembulatan waktu ke jam terdekat atau per 15 menit (sesuaikan kebutuhan)
    # Jika ingin mencatat data tiap jam pas:
    menit_bulat = (sekarang.minute // 15) * 15
    waktu_fixed = sekarang.replace(minute=menit_bulat, second=0, microsecond=0)
    
    with file_lock(LOCK_FILE):
        log.info(f"Mulai scraping data TMA untuk {waktu_fixed.strftime('%Y-%m-%d %H:%M')}")
        nilai_tma_cm = run_selenium_scraper()
        
        if nilai_tma_cm is not None:
            append_to_csv(waktu_fixed, nilai_tma_cm)
        else:
            log.error("Gagal mendapatkan nilai yang valid dari DSDA.")
            sys.exit(1)

if __name__ == "__main__":
    main()
