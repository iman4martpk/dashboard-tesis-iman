#!/usr/bin/env python3
"""
Scraper BPBD -> Update File Observasi TMA Pasar Ikan.

Perilaku:
- Mengambil data TMA (Tinggi Muka Air) terbaru untuk pos "Pasar Ikan - Laut"
  dari https://bpbd.jakarta.go.id/waterlevel
- Menambahkan (append) satu baris baru ke file CSV observasi, TANPA menulis
  ulang (rewrite) baris-baris yang sudah ada.
- Aman dijalankan berulang (idempotent): jika jam yang sama sudah tercatat,
  proses akan di-skip.

Perbaikan dari versi sebelumnya:
1. XPath tidak lagi bergantung pada index baris tetap (tr[9]) yang rapuh
   terhadap perubahan struktur tabel. Baris target dicari berdasarkan teks
   nama pos ("Pasar Ikan"), dan sel header/isi dicari secara generik
   (menerima <th> maupun <td>).
2. Regex nilai TMA kini mendukung angka desimal dan negatif.
3. Logging terstruktur (bukan print) sehingga kegagalan mudah didiagnosis.
4. HTTP request memakai retry/backoff dan timeout eksplisit.
5. Validasi rentang nilai TMA untuk menolak hasil scrape yang tidak masuk akal.
6. File locking sederhana (POSIX flock) agar aman dari eksekusi tumpang tindih
   (mis. dijadwalkan lewat cron).
"""

from __future__ import annotations

import fcntl
import logging
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import pytz
import requests
from lxml import html
from requests.adapters import HTTPAdapter, Retry

# --------------------------------------------------------------------------- #
# Konfigurasi
# --------------------------------------------------------------------------- #

TZ = pytz.timezone("Asia/Jakarta")

BASE_DIR = Path(__file__).resolve().parent
FILE_OBSERVASI = BASE_DIR / "HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv"
LOCK_FILE = BASE_DIR / ".scraper_bpbd_observasi.lock"

URL = "https://bpbd.jakarta.go.id/waterlevel"
NAMA_POS = "Pasar Ikan"  # substring pencarian nama pos target (bebas huruf besar/kecil)

# Rentang nilai wajar TMA Pasar Ikan (cm) - untuk validasi hasil scrape.
TMA_MIN, TMA_MAX = -300, 500

REQUEST_TIMEOUT = 10  # detik
MAX_RETRIES = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scraper_bpbd")


@dataclass(frozen=True)
class ObservasiRow:
    waktu: pd.Timestamp
    tma_cm: float


# --------------------------------------------------------------------------- #
# HTTP session dengan retry
# --------------------------------------------------------------------------- #

def build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=MAX_RETRIES,
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8",
        }
    )
    return session


# --------------------------------------------------------------------------- #
# Scraping
# --------------------------------------------------------------------------- #

def fetch_page(session: requests.Session) -> Optional[html.HtmlElement]:
    """Ambil dan parse halaman waterlevel BPBD. Return None jika gagal total."""
    try:
        response = session.get(URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as exc:
        log.error("Gagal mengambil halaman BPBD: %s", exc)
        return None
    return html.fromstring(response.content)


def extract_pasar_ikan_reading(tree: html.HtmlElement) -> Optional[ObservasiRow]:
    """
    Cari baris tabel untuk pos "Pasar Ikan" secara generik (tidak bergantung
    pada index tetap), lalu ambil kolom pertama (jam terbaru) beserta nilainya.

    Struktur tabel yang diasumsikan:
      thead: baris terakhir berisi label kolom jam, mis. "19:00", "18:00", ...
      tbody: setiap baris = 1 pos, kolom pertama = nama pos, kolom berikutnya
             = nilai TMA per jam (kolom pertama setelah nama = jam terbaru).
    """
    # Header jam: ambil semua sel (th ATAU td) di baris terakhir <thead>,
    # lalu saring yang match format jam (HH:MM).
    header_cells = tree.xpath(
        "//table//thead//tr[last()]//*[self::th or self::td]"
    )
    jam_list = []
    for cell in header_cells:
        text = " ".join(cell.itertext())
        match = re.search(r"\b\d{1,2}:\d{2}\b", text)
        if match:
            jam_list.append(match.group())

    if not jam_list:
        log.error("Tidak menemukan label jam di header tabel (struktur mungkin berubah).")
        return None

    jam_str = jam_list[0]  # kolom jam pertama = data terbaru

    # Cari baris body yang nama posnya mengandung NAMA_POS.
    target_rows = tree.xpath(
        f"//table//tbody//tr[.//*[self::td or self::th]"
        f"[1][contains(normalize-space(.), '{NAMA_POS}')]]"
    )
    if not target_rows:
        log.error("Baris pos '%s' tidak ditemukan di tabel.", NAMA_POS)
        return None

    row = target_rows[0]
    cells = row.xpath(".//*[self::td or self::th]")
    if len(cells) < 2:
        log.error("Baris pos '%s' ditemukan tapi tidak punya kolom data.", NAMA_POS)
        return None

    value_text = " ".join(cells[1].itertext())
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value_text)
    if not match:
        log.error("Tidak bisa mem-parse nilai TMA dari teks: %r", value_text)
        return None

    angka_tma = float(match.group())

    if not (TMA_MIN <= angka_tma <= TMA_MAX):
        log.error(
            "Nilai TMA %.2f cm di luar rentang wajar (%d..%d). Ditolak demi keamanan data.",
            angka_tma, TMA_MIN, TMA_MAX,
        )
        return None

    now_naive = datetime.now(TZ).replace(tzinfo=None)
    waktu = resolve_target_datetime(now_naive, jam_str)
    return ObservasiRow(waktu=waktu, tma_cm=angka_tma)


def resolve_target_datetime(now_naive: datetime, jam_str: str) -> pd.Timestamp:
    """Gabungkan tanggal hari ini dengan jam hasil scrape (mundur 1 hari jika perlu)."""
    target_time = datetime.strptime(jam_str, "%H:%M").time()
    candidate = datetime.combine(now_naive.date(), target_time)
    if candidate > now_naive:
        candidate -= timedelta(days=1)
    return pd.Timestamp(candidate)


# --------------------------------------------------------------------------- #
# Penyimpanan (append-only)
# --------------------------------------------------------------------------- #

def is_already_recorded(target_ts: pd.Timestamp) -> bool:
    """
    Cek apakah target_ts sudah tercatat di file observasi.

    File lama diketahui berisi format datetime yang tidak seragam (mis. baris
    lama "%m/%d/%Y %H:%M" bercampur dengan baris baru "%Y-%m-%d %H:%M:%S"),
    jadi parsing memakai format='mixed' + errors='coerce' agar tidak crash,
    dan baris yang gagal diparse dilaporkan sebagai peringatan (bukan fatal).
    """
    if not FILE_OBSERVASI.exists():
        return False

    df_exist = pd.read_csv(FILE_OBSERVASI, usecols=["Datetime"])
    parsed = pd.to_datetime(df_exist["Datetime"], format="mixed", dayfirst=False, errors="coerce")

    n_invalid = parsed.isna().sum()
    if n_invalid:
        log.warning(
            "%d baris di %s memiliki format Datetime yang tidak bisa diparse dan diabaikan "
            "saat pengecekan duplikat.",
            n_invalid, FILE_OBSERVASI.name,
        )

    existing_dates = parsed.dropna()
    if existing_dates.dt.tz is not None:
        existing_dates = existing_dates.dt.tz_localize(None)

    return target_ts in existing_dates.values


def append_row(row: ObservasiRow) -> None:
    file_exists = FILE_OBSERVASI.exists()
    new_row = pd.DataFrame(
        [{
            "Datetime": row.waktu.strftime("%Y-%m-%d %H:%M:%S"),
            "TMA_Pasar_Ikan": row.tma_cm,
        }]
    )
    new_row.to_csv(FILE_OBSERVASI, mode="a", header=not file_exists, index=False)


@contextmanager
def file_lock(lock_path: Path) -> Iterator[None]:
    """Cegah dua proses menulis CSV secara bersamaan (mis. cron overlap)."""
    lock_path.touch(exist_ok=True)
    with open(lock_path, "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.warning("Proses lain sedang berjalan (lock aktif). Keluar tanpa aksi.")
            raise SystemExit(0)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    with file_lock(LOCK_FILE):
        session = build_session()
        tree = fetch_page(session)
        if tree is None:
            return 1

        reading = extract_pasar_ikan_reading(tree)
        if reading is None:
            log.error("Gagal mendapatkan data valid dari web BPBD.")
            return 1

        if is_already_recorded(reading.waktu):
            log.info("Data jam %s sudah ada di database. Skip append.", reading.waktu)
            return 0

        try:
            append_row(reading)
        except OSError as exc:
            log.error("Gagal melakukan append ke file: %s", exc)
            return 1

        log.info(
            "Berhasil APPEND 1 baris baru: %s -> %.2f cm",
            reading.waktu, reading.tma_cm,
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
