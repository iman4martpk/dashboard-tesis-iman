#!/usr/bin/env python3
"""
Scraper Posko Banjir DSDA -> Simpan ke 2 File CSV Terpisah:
1. HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv (Per Jam Pas untuk Evaluasi ML)
2. HASIL_SCRAPING_15_MENITAN_DSDA.csv (Per 15 Menit untuk UI/Grafik Realtime)
"""

import os
import time
import re
import pandas as pd
from datetime import datetime
import pytz

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# --- KONFIGURASI FILE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_HOURLY = os.path.join(BASE_DIR, 'HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv')
FILE_15MIN = os.path.join(BASE_DIR, 'HASIL_SCRAPING_15_MENITAN_DSDA.csv')

LIMIT_SENSOR_ERROR_CM = 350.0  # Batas Atas (cm)
MIN_VALID_VALUE_CM = 50.0      # Batas Bawah (cm)


def append_to_csv(filename, waktu_fixed, nilai_cm, label_tipe):
    waktu_str = waktu_fixed.strftime('%Y-%m-%d %H:%M:%S')
    waktu_check = waktu_fixed.strftime('%Y-%m-%d %H:%M')
    
    new_data = pd.DataFrame({'Datetime': [waktu_str], 'TMA_Pasar_Ikan': [nilai_cm]})

    if not os.path.exists(filename):
        new_data.to_csv(filename, index=False)
        print(f"✅ CSV Baru [{label_tipe}]: {waktu_str} -> {nilai_cm} cm")
    else:
        try:
            old_data = pd.read_csv(filename)
            if 'Datetime' in old_data.columns and not old_data.empty:
                old_times = pd.to_datetime(old_data['Datetime'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
                if waktu_check in old_times.values:
                    print(f"⚠️ [{label_tipe}] Waktu {waktu_check} sudah ada. Skip nulis.")
                    return
            
            combined = pd.concat([old_data, new_data]).sort_values('Datetime')
            combined.to_csv(filename, index=False)
            print(f"✅ BERHASIL NULIS [{label_tipe}]: {waktu_str} -> {nilai_cm} cm")
        except Exception as e:
            print(f"Error simpan CSV [{label_tipe}]: {e}")


def process_and_save(waktu_sekarang, nilai_cm):
    if nilai_cm is None or nilai_cm > LIMIT_SENSOR_ERROR_CM or nilai_cm < MIN_VALID_VALUE_CM:
        print(f"DEBUG: Nilai {nilai_cm} cm diblokir (di luar range valid).")
        return

    # 1. SIMPAN KE FILE HOURLY (Dibulatkan ke Jam Pas :00:00)
    waktu_hourly = waktu_sekarang.replace(minute=0, second=0, microsecond=0)
    append_to_csv(FILE_HOURLY, waktu_hourly, nilai_cm, "HOURLY TESIS")

    # 2. SIMPAN KE FILE 15-MENITAN (Dibulatkan ke :00, :15, :30, :45)
    menit_15 = (waktu_sekarang.minute // 15) * 15
    waktu_15min = waktu_sekarang.replace(minute=menit_15, second=0, microsecond=0)
    append_to_csv(FILE_15MIN, waktu_15min, nilai_cm, "15-MENIT DSDA")


def run_scraper_dsda():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = None
    nilai_cm = None

    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get("https://poskobanjir.dsdadki.web.id/")
        time.sleep(15)

        all_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Pasar Ikan')]/ancestor::tr")
        for row in all_elements:
            row_text = row.get_attribute("innerText")
            nums = re.findall(r"\b(\d{2,3})\b", row_text)
            for n in nums:
                val_n = int(n)
                if MIN_VALID_VALUE_CM <= val_n <= LIMIT_SENSOR_ERROR_CM:
                    nilai_cm = float(val_n)
                    break
            if nilai_cm:
                break

        if not nilai_cm:
            all_text = driver.find_element(By.TAG_NAME, "body").get_attribute("innerText")
            pos = all_text.find("Pasar Ikan")
            if pos != -1:
                snippet = all_text[pos:pos + 200]
                nums = re.findall(r"\b(\d{2,3})\b", snippet)
                for n in nums:
                    val_n = int(n)
                    if MIN_VALID_VALUE_CM <= val_n <= LIMIT_SENSOR_ERROR_CM:
                        nilai_cm = float(val_n)
                        break

    except Exception as e:
        print(f"❌ Scraper Error: {e}")
    finally:
        if driver:
            driver.quit()

    return nilai_cm


if __name__ == "__main__":
    tz_jkt = pytz.timezone('Asia/Jakarta')
    sekarang = datetime.now(tz_jkt).replace(tzinfo=None)
    print(f"\nEksekusi: {sekarang.strftime('%Y-%m-%d %H:%M:%S')} WIB")

    nilai_dsda_cm = run_scraper_dsda()

    if nilai_dsda_cm is not None:
        process_and_save(sekarang, nilai_dsda_cm)
