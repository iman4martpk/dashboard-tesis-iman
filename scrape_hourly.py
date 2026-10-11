#!/usr/bin/env python3
"""
Scraper BPBD (Posko Banjir DSDA) -> Update HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv
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

# --- KONFIGURASI FILE & FILTER TESIS IMAN ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_OBSERVASI = os.path.join(BASE_DIR, 'HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv')

LIMIT_SENSOR_ERROR_CM = 350.0  # Batas Atas (cm)
MIN_VALID_VALUE_CM = 120.0     # Batas Bawah (cm)


def save_to_csv(filename, waktu, nilai_cm):
    # FILTER: Cek batas bawah dan batas atas
    if nilai_cm is None or nilai_cm > LIMIT_SENSOR_ERROR_CM or nilai_cm < MIN_VALID_VALUE_CM:
        print(f"DEBUG: Nilai {nilai_cm} cm diblokir (di luar range {MIN_VALID_VALUE_CM} - {LIMIT_SENSOR_ERROR_CM} cm)")
        return

    # Format string waktu untuk CSV Tesis Iman
    waktu_str = waktu.strftime('%Y-%m-%d %H:%M:%S')
    waktu_check = waktu.strftime('%Y-%m-%d %H:%M')
    
    new_data = pd.DataFrame({'Datetime': [waktu_str], 'TMA_Pasar_Ikan': [nilai_cm]})

    if not os.path.exists(filename):
        new_data.to_csv(filename, index=False)
        print(f"✅ CSV Baru Dibuat: {waktu_str} -> {nilai_cm} cm")
    else:
        try:
            old_data = pd.read_csv(filename)
            if 'Datetime' in old_data.columns and not old_data.empty:
                # Cek agar tidak duplikat di menit yang sama
                old_times = pd.to_datetime(old_data['Datetime'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M')
                if waktu_check in old_times.values:
                    print(f"⚠️ Data waktu {waktu_check} sudah ada di CSV. Skip nulis.")
                    return
            
            combined = pd.concat([old_data, new_data]).sort_values('Datetime')
            combined.to_csv(filename, index=False)
            print(f"✅ BERHASIL NULIS KE CSV: {waktu_str} -> {nilai_cm} cm")
        except Exception as e:
            print(f"Error simpan CSV: {e}")


def run_scraper_bpbd():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

    driver = None
    nilai_cm = None

    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        print("\n--- Scraping BPBD Pasar Ikan (Metode Deep Scan 3.0) ---")
        driver.get("https://poskobanjir.dsdadki.web.id/")
        time.sleep(15)

        print("Mencari baris Pasar Ikan...")
        all_elements = driver.find_elements(By.XPATH, "//*[contains(text(), 'Pasar Ikan')]/ancestor::tr")

        for row in all_elements:
            row_text = row.get_attribute("innerText")
            nums = re.findall(r"(\d{3})", row_text)
            for n in nums:
                val_n = int(n)
                # Filter angka CM (biasanya 120cm - 350cm)
                if 120 <= val_n <= 350:
                    nilai_cm = float(val_n)
                    print(f"✅ BPBD Berhasil (Row Text): {nilai_cm} cm")
                    break
            if nilai_cm:
                break

        if not nilai_cm:
            print("Skenario 1 gagal, mencari snippet...")
            all_text = driver.find_element(By.TAG_NAME, "body").get_attribute("innerText")
            pos = all_text.find("Pasar Ikan")
            if pos != -1:
                snippet = all_text[pos:pos + 200]
                nums = re.findall(r"(\d{3})", snippet)
                for n in nums:
                    val_n = int(n)
                    if 120 <= val_n <= 350:
                        nilai_cm = float(val_n)
                        print(f"✅ BPBD Berhasil (Snippet): {nilai_cm} cm")
                        break

    except Exception as e:
        print(f"❌ BPBD Gagal: {e}")
    finally:
        if driver:
            driver.quit()

    return nilai_cm


if __name__ == "__main__":
    tz_jkt = pytz.timezone('Asia/Jakarta')
    sekarang = datetime.now(tz_jkt).replace(tzinfo=None)
    print(f"\nEksekusi: {sekarang.strftime('%Y-%m-%d %H:%M:%S')} WIB")

    # Scrape data BPBD Pasar Ikan
    nilai_bpbd_cm = run_scraper_bpbd()

    # Simpan ke file CSV Observasi Tesis Iman
    if nilai_bpbd_cm is not None:
        save_to_csv(FILE_OBSERVASI, sekarang, nilai_bpbd_cm)
