#!/usr/bin/env python3
"""
Scraper Posko Banjir DSDA (Pompa Pasar Ikan 1) -> 2 CSV, append-only.

1. HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv   : per jam pas (untuk evaluasi ML)
2. HASIL_SCRAPING_15_MENITAN_DSDA.csv      : per 15 menit (untuk UI realtime)

Perbaikan utama:
- Baris dicari lewat NAMA pos, nilai diambil per SEL (bukan regex angka).
- Timestamp memakai waktu pembacaan dari web DSDA, bukan jam runner.
- Hourly hanya ditulis kalau pembacaan dekat jam pas (tidak salah label).
- CSV append-only (tidak ditulis ulang), dedupe per timestamp.
- Exit code != 0 kalau gagal / data DSDA basi, supaya job merah.
"""
import csv
import os
import re
import sys
from datetime import datetime, timedelta

import pytz

URL = "https://poskobanjir.dsdadki.web.id/"
POS_NAME = "Pompa Pasar Ikan 1"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_HOURLY = os.path.join(BASE_DIR, "HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv")
FILE_15MIN = os.path.join(BASE_DIR, "HASIL_SCRAPING_15_MENITAN_DSDA.csv")

MIN_VALID_CM = 50.0
MAX_VALID_CM = 350.0
HOURLY_TOLERANCE_MIN = 10   # pembacaan harus <= 10 menit dari jam pas
STALE_AFTER_MIN = 90        # data DSDA lebih tua dari ini -> run ditandai gagal

DATE_RE = re.compile(r"\d{2}/\d{2}/\d{4}")
TIME_RE = re.compile(r"\d{1,2}:\d{2}")
NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def parse_cells(texts, pos_name=POS_NAME):
    """texts = teks tiap sel satu baris. Return (nilai_cm, datetime_pembacaan) / None."""
    texts = [t.strip() for t in texts]
    idx = next((i for i, t in enumerate(texts) if pos_name.lower() in t.lower()), None)
    if idx is None:
        return None
    # nilai = sel angka PERTAMA setelah nama pos (nomor urut ada sebelum nama, jadi tidak ikut)
    nilai = next((float(t.replace(",", ".")) for t in texts[idx + 1:] if NUM_RE.fullmatch(t)), None)
    tgl = next((t for t in texts if DATE_RE.fullmatch(t)), None)
    jam = next((t for t in texts if TIME_RE.fullmatch(t)), None)
    if nilai is None or not tgl or not jam:
        return None
    return nilai, datetime.strptime(f"{tgl} {jam}", "%d/%m/%Y %H:%M")


def fetch_reading():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from webdriver_manager.chrome import ChromeDriverManager

    opts = Options()
    opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    driver = None
    try:
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=opts)
        driver.get(URL)

        def ambil(d):
            rows = d.find_elements(
                By.XPATH, f"//tr[not(.//tr) and contains(normalize-space(.), '{POS_NAME}')]"
            )
            for row in rows:
                cells = [c.get_attribute("innerText") or "" for c in row.find_elements(By.XPATH, "./td|./th")]
                hasil = parse_cells(cells)
                if hasil:
                    print(f"DEBUG baris: {[c.strip() for c in cells]}")
                    return hasil
            return False

        return WebDriverWait(driver, 45, poll_frequency=2).until(ambil)
    except Exception as e:
        print(f"❌ Scraper error: {type(e).__name__}: {e}")
        return None
    finally:
        if driver:
            driver.quit()


def existing_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="") as f:
        return {r["Datetime"][:16] for r in csv.DictReader(f) if r.get("Datetime")}


def append_row(path, ts, nilai, label):
    key = ts.strftime("%Y-%m-%d %H:%M")
    if key in existing_keys(path):
        print(f"⚠️ [{label}] {key} sudah ada, skip.")
        return False
    baru = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if baru:
            w.writerow(["Datetime", "TMA_Pasar_Ikan"])
        w.writerow([ts.strftime("%Y-%m-%d %H:%M:%S"), f"{nilai:.1f}"])
    print(f"✅ [{label}] {ts:%Y-%m-%d %H:%M:%S} -> {nilai:.1f} cm")
    return True


def nearest_hour(ts):
    dasar = ts.replace(minute=0, second=0, microsecond=0)
    jam = min((dasar, dasar + timedelta(hours=1)), key=lambda x: abs((ts - x).total_seconds()))
    return jam, abs((ts - jam).total_seconds()) / 60


def main():
    sekarang = datetime.now(pytz.timezone("Asia/Jakarta")).replace(tzinfo=None)
    print(f"Eksekusi: {sekarang:%Y-%m-%d %H:%M:%S} WIB")

    hasil = fetch_reading()
    if not hasil:
        print("❌ Pembacaan tidak ditemukan. Cek POS_NAME / struktur tabel DSDA.")
        return 1
    nilai, ts_data = hasil
    umur = (sekarang - ts_data).total_seconds() / 60
    print(f"Pembacaan DSDA: {ts_data:%Y-%m-%d %H:%M} = {nilai} cm (umur {umur:.0f} menit)")

    if not (MIN_VALID_CM <= nilai <= MAX_VALID_CM):
        print(f"❌ Nilai {nilai} cm di luar range valid {MIN_VALID_CM}-{MAX_VALID_CM}.")
        return 1

    ts15 = ts_data.replace(minute=(ts_data.minute // 15) * 15, second=0, microsecond=0)
    append_row(FILE_15MIN, ts15, nilai, "15-MENIT")

    jam, selisih = nearest_hour(ts_data)
    if selisih <= HOURLY_TOLERANCE_MIN:
        append_row(FILE_HOURLY, jam, nilai, "HOURLY")
    else:
        print(f"ℹ️ Pembacaan {selisih:.0f} menit dari jam pas, tidak ditulis ke hourly.")

    if umur > STALE_AFTER_MIN:
        print(f"❌ Data DSDA basi ({umur:.0f} menit). Sensor/website mungkin mati.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
