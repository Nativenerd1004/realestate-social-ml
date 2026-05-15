"""
Step 02 — PostgreSQL Loader
Creates the database, reads all downloaded CSVs, and loads them as raw tables.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
import sqlalchemy as sa

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASETS, PG_DB, PG_HOST, PG_PASSWORD, PG_PORT, PG_URL, PG_USER


def get_engine(db=PG_DB):
    return sa.create_engine(f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{db}")


def create_database():
    # Connect to postgres default db to create our db if needed
    admin_url = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/postgres"
    engine = sa.create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        exists = conn.execute(
            sa.text("SELECT 1 FROM pg_database WHERE datname = :db"), {"db": PG_DB}
        ).fetchone()
        if not exists:
            conn.execute(sa.text(f'CREATE DATABASE "{PG_DB}"'))
            print(f"[✓] Database '{PG_DB}' created")
        else:
            print(f"[✓] Database '{PG_DB}' already exists")
    engine.dispose()


def safe_table_name(path: str) -> str:
    name = Path(path).stem
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
    return f"raw_{name[:50]}"


def load_csvs():
    manifest_path = DATASETS / "manifest.json"
    if not manifest_path.exists():
        print("[!] No manifest found — run Step 01 first")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    engine = get_engine()
    loaded_tables = []

    for ds in manifest:
        for csv_path in ds["files"]:
            table_name = safe_table_name(csv_path)
            print(f"\n[→] Loading: {Path(csv_path).name} → table '{table_name}'")
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                df.columns = [re.sub(r"[^a-zA-Z0-9_]", "_", c).lower() for c in df.columns]
                df.to_sql(table_name, engine, if_exists="replace", index=False, chunksize=5000)
                print(f"    [✓] {len(df):,} rows × {len(df.columns)} cols")
                loaded_tables.append({
                    "table": table_name,
                    "source": csv_path,
                    "rows": len(df),
                    "cols": list(df.columns),
                })
            except Exception as e:
                print(f"    [!] Failed: {e}")

    # Save table registry
    registry_path = DATASETS / "table_registry.json"
    with open(registry_path, "w") as f:
        json.dump(loaded_tables, f, indent=2)

    print(f"\n[✓] Step 02 complete — {len(loaded_tables)} tables loaded into PostgreSQL")
    return loaded_tables


def run():
    print("=" * 60)
    print("STEP 02 — PostgreSQL Loader")
    print("=" * 60)
    create_database()
    return load_csvs()


if __name__ == "__main__":
    run()
