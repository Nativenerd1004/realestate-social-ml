"""
Step 01 — Kaggle Dataset Fetcher
Searches Kaggle REST API for datasets and downloads via kagglehub.
Uses the new KGAT_ token format via KAGGLE_API_TOKEN env var.
"""
import json
import os
import shutil
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASETS, KAGGLE_SEARCH_QUERIES, MAX_DATASETS_PER_QUERY


KAGGLE_API_BASE = "https://www.kaggle.com/api/v1"


def get_token() -> str:
    token = os.getenv("KAGGLE_API_TOKEN", "")
    if not token:
        access_token_file = Path.home() / ".kaggle" / "access_token"
        if access_token_file.exists():
            token = access_token_file.read_text().strip()
    if not token:
        print("[!] KAGGLE_API_TOKEN not set and ~/.kaggle/access_token not found")
        sys.exit(1)
    return token


def search_datasets(query: str, token: str, max_results: int = 5) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}"}
    params  = {"search": query, "filetype": "csv", "maxSize": 104857600, "pageSize": max_results}
    try:
        r = requests.get(f"{KAGGLE_API_BASE}/datasets/list", headers=headers,
                         params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
        print(f"    [!] Search returned {r.status_code}: {r.text[:200]}")
        return []
    except Exception as e:
        print(f"    [!] Search error: {e}")
        return []


def download_dataset(ref: str, dest: Path, token: str) -> list[Path]:
    import kagglehub
    os.environ["KAGGLE_API_TOKEN"] = token

    print(f"    [↓] Downloading: {ref}")
    try:
        path = kagglehub.dataset_download(ref)
        src = Path(path)
        dest.mkdir(parents=True, exist_ok=True)

        csv_files = []
        for f in src.rglob("*.csv"):
            target = dest / f.name
            shutil.copy2(f, target)
            csv_files.append(target)

        return csv_files
    except Exception as e:
        print(f"    [!] Download failed for {ref}: {e}")
        return []


def run():
    print("=" * 60)
    print("STEP 01 — Kaggle Dataset Fetcher")
    print("=" * 60)

    token = get_token()
    print("[✓] Kaggle token found")
    DATASETS.mkdir(parents=True, exist_ok=True)

    downloaded = []
    seen = set()

    for query in KAGGLE_SEARCH_QUERIES:
        print(f"\n[→] Searching: '{query}'")
        results = search_datasets(query, token, max_results=MAX_DATASETS_PER_QUERY + 3)
        count = 0

        for ds in results:
            ref = ds.get("ref", "")
            if not ref or ref in seen or count >= MAX_DATASETS_PER_QUERY:
                continue
            seen.add(ref)

            dest = DATASETS / ref.replace("/", "__")
            csv_files = download_dataset(ref, dest, token)

            if csv_files:
                downloaded.append({
                    "ref": ref,
                    "path": str(dest),
                    "files": [str(f) for f in csv_files],
                })
                print(f"    [✓] {len(csv_files)} CSV file(s) saved")
                count += 1

    # Fallback: download curated real estate datasets if search returns nothing
    if not downloaded:
        print("\n[!] Search returned no results — downloading curated fallback datasets")
        fallback_refs = [
            "shree1992/housedata",
            "amitabhajoy/bengaluru-house-data",
            "gregoiredc/airbnb-nyc",
        ]
        for ref in fallback_refs:
            dest = DATASETS / ref.replace("/", "__")
            csv_files = download_dataset(ref, dest, token)
            if csv_files:
                downloaded.append({"ref": ref, "path": str(dest),
                                   "files": [str(f) for f in csv_files]})

    manifest_path = DATASETS / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(downloaded, f, indent=2)

    print(f"\n[✓] Step 01 complete — {len(downloaded)} datasets downloaded")
    print(f"    Manifest: {manifest_path}")
    return downloaded


if __name__ == "__main__":
    run()
