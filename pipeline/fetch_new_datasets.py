"""
Download candidate datasets from Kaggle using Bearer token + GCS redirect.
Saves to datasets/260516/ (today's date folder).
"""
import requests, zipfile, io, os
from pathlib import Path

TOKEN = "KGAT_fed75909a0bee69dfe451def47640e3f"
H = {"Authorization": f"Bearer {TOKEN}"}
BASE = "https://www.kaggle.com/api/v1"
DEST = Path("/Users/apple/Desktop/RealEstate_SocialMedia_ML/datasets/260516")

# Tab 2 — Social Media candidates
# Tab 3 — Real Estate Sale candidates
DATASETS = [
    # Tab 2
    "jsonk11/social-media-advertising-dataset",
    "atharvasoundankar/viral-social-media-trends-and-engagement-analysis",
    "nayakganesh007/google-ads-sales-dataset",
    # Tab 3
    "yasirumanujith/usa-real-estate-dataset",
    "alaasweed/southern-states-zillow-data",
    "visheshptel/housing-dataset-prices-and-sales-prediction",
    "kanchana1990/illinois-real-estate-sold-properties-data-2026",
    "feeldidaxie/king-county-house-sales-usa",
]

def download(ref):
    folder = DEST / ref.split("/")[1]
    if folder.exists() and any(folder.iterdir()):
        print(f"  [skip] {ref} already downloaded")
        return
    folder.mkdir(parents=True, exist_ok=True)

    print(f"  Downloading {ref}...")
    r = requests.get(f"{BASE}/datasets/download/{ref}",
                     headers=H, stream=True, allow_redirects=True)
    if r.status_code != 200:
        print(f"  [ERR] {ref}: {r.status_code}")
        return

    content = r.content
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            z.extractall(folder)
        files = list(folder.rglob("*.csv")) + list(folder.rglob("*.parquet"))
        mb = sum(f.stat().st_size for f in files) / 1e6
        print(f"  [OK ] {ref} → {len(files)} file(s), {mb:.1f}MB extracted")
    except zipfile.BadZipFile:
        # might be a raw CSV
        outfile = folder / "data.csv"
        outfile.write_bytes(content)
        print(f"  [OK ] {ref} → raw file {len(content)/1e6:.1f}MB")

for ref in DATASETS:
    try:
        download(ref)
    except Exception as e:
        print(f"  [ERR] {ref}: {e}")

print("\nAll done.")
