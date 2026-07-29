import pandas as pd
import requests
from lxml import html
import re
from datetime import datetime
import pytz

# Set zona waktu Jakarta (WIB)
tz = pytz.timezone('Asia/Jakarta')
now = datetime.now(tz)

# Target File CSV yang akan ditulis (Skenario 1)
DATA_FILES = [
    "HASIL_FINAL_TESIS_PASUT_HIBRIDA.csv",
    "HASIL_FINAL_TESIS_PASUT_LSTM_MURNI.csv"
]

def fetch_realtime_data():
    url = "https://bpbd.jakarta.go.id/waterlevel"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        tree = html.fromstring(response.content)
        
        # XPath sesuai website BPBD Jakarta
        xpath_jam = '/html/body/main/div/div/article/div/div[2]/div/div[1]/div[1]/div[2]/div/table/thead/tr[2]/td[2]/text()'
        xpath_tma = '/html/body/main/div/div/article/div/div[2]/div/div[1]/div[1]/div[2]/div/table/tbody/tr[9]/td[2]/text()'
        
        jam_mentah = tree.xpath(xpath_jam)
        tma_mentah = tree.xpath(xpath_tma)
        
        if jam_mentah and tma_mentah:
            # Ekstrak Jam (Misal dari "01:00 WIB" jadi "01:00")
            jam = jam_mentah[0].strip()
            match_jam = re.search(r'\d{2}:\d{2}', jam)
            jam_str = match_jam.group() if match_jam else "00:00"

            # Ekstrak TMA
            teks_tma = tma_mentah[0].strip()
            match_tma = re.search(r'[-+]?\d+', teks_tma)
            angka_tma = float(match_tma.group()) if match_tma else None
            
            return jam_str, angka_tma
    except Exception as e:
        print(f"❌ Error scraping: {e}")
    return None, None

def main():
    jam_str, angka_tma = fetch_realtime_data()

    if jam_str and angka_tma is not None:
        target_time = datetime.strptime(jam_str, "%H:%M").time()
        target_date = now.date()
        
        # Logika anti-bug: Jika script jalan jam 00:25, tapi data BPBD masih jam 23:00 (telat update),
        # maka data tersebut adalah milik hari kemarin.
        if now.hour == 0 and target_time.hour >= 22:
            target_date = target_date - pd.Timedelta(days=1)
            
        target_datetime_str = f"{target_date} {jam_str}:00"
        print(f"🎯 Target Row Datetime: {target_datetime_str} | Nilai Scrape: {angka_tma} cm")

        # Tulis ke dalam 2 File CSV Master
        for file in DATA_FILES:
            try:
                df = pd.read_csv(file)
                # Cari baris yang Datetime-nya cocok
                mask = df['Datetime'] == target_datetime_str
                
                if mask.sum() > 0:
                    df.loc[mask, 'TMA_Pasar_Ikan'] = angka_tma
                    df.to_csv(file, index=False)
                    print(f"✅ Sukses mengupdate {file} pada baris {target_datetime_str}")
                else:
                    print(f"⚠️ Peringatan: Datetime {target_datetime_str} tidak ditemukan di {file}")
            except Exception as e:
                print(f"❌ Error memproses file {file}: {e}")
    else:
        print("🚨 Gagal mendapatkan data valid dari web BPBD.")

if __name__ == "__main__":
    main()
