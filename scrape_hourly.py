name: Jalur 1 - Scraper TMA 15 Menitan (Hourly)

on:
  repository_dispatch:
    types: [run_scrape_hourly]
  schedule:
    - cron: "7,22,37,52 * * * *"   # cadangan, hindari menit :00 yang padat
  workflow_dispatch:

concurrency:
  group: scrape-tma
  cancel-in-progress: false

permissions:
  contents: write

jobs:
  scrape:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - run: pip install selenium webdriver-manager pytz

      - name: Eksekusi Script Scraper TMA
        run: python scrape_hourly.py

      - name: Commit & Push hasil scrape
        if: ${{ !cancelled() }}
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          git add HASIL_FINAL_TESIS_PASUT_OBSERVASI.csv HASIL_SCRAPING_15_MENITAN_DSDA.csv
          if git diff --cached --quiet; then echo "Tidak ada data baru"; exit 0; fi
          git commit -m "Auto-scrape: update data observasi TMA Pasar Ikan"
          git pull --rebase origin main
          git push
