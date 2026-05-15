"""
Step 03 — Data Cleaner (SQL + pandas)
Cleans every raw table in PostgreSQL and merges into a single ML-ready dataset.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy as sa
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CLEANED, DATASETS, PG_DB, PG_HOST, PG_PASSWORD, PG_PORT, PG_USER, TARGET_COLUMN


def get_engine():
    return sa.create_engine(
        f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"
    )


# ── SQL-level cleaning per table ───────────────────────────────────────────────
CLEAN_SQL = """
    SELECT *
    FROM {table}
    WHERE 1=1
"""

def infer_target_column(df: pd.DataFrame) -> str:
    """Find the most likely churn/target column by name heuristics."""
    candidates = ["churn", "churned", "attrition", "left", "exited",
                  "target", "label", "y", "is_churn", "customer_status"]
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # Fall back to last binary column
    for col in reversed(df.columns):
        if df[col].nunique() == 2:
            return col
    return df.columns[-1]


def clean_table(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    print(f"    Shape before: {df.shape}")

    # 1. Strip whitespace from string columns
    str_cols = df.select_dtypes("object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    # 2. Drop columns with >50% missing
    thresh = int(len(df) * 0.5)
    df = df.dropna(thresh=thresh, axis=1)

    # 3. Drop fully duplicate rows
    df = df.drop_duplicates()

    # 4. Fill numeric NaN with median, categorical with mode
    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes("object").columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    # 5. Remove columns with near-zero variance
    for col in num_cols:
        if col in df.columns and df[col].std() == 0:
            df = df.drop(columns=[col])

    # 6. Sanitize column names for XGBoost compatibility
    df.columns = [re.sub(r"[<>\[\]{}|\"'\\/*?:]", "_", c) for c in df.columns]

    print(f"    Shape after:  {df.shape}")
    return df


def encode_and_merge(tables: list[dict]) -> pd.DataFrame:
    engine = get_engine()
    frames = []

    for t in tables:
        tbl = t["table"]
        try:
            df = pd.read_sql(f'SELECT * FROM "{tbl}"', engine)
            df = clean_table(df, tbl)
            frames.append(df)
        except Exception as e:
            print(f"    [!] Skipping {tbl}: {e}")

    if not frames:
        print("[!] No data to merge")
        sys.exit(1)

    # Use the largest frame as base (most columns/rows)
    frames.sort(key=lambda d: d.shape[0] * d.shape[1], reverse=True)
    merged = frames[0].copy()
    print(f"\n[→] Base dataset: {merged.shape}")

    # Infer target
    target_col = infer_target_column(merged)
    print(f"[→] Target column detected: '{target_col}'")

    # Encode all object columns
    le = LabelEncoder()
    for col in merged.select_dtypes("object").columns:
        try:
            merged[col] = le.fit_transform(merged[col].astype(str))
        except Exception:
            merged = merged.drop(columns=[col])

    # Rename target to standard name
    if target_col != TARGET_COLUMN and target_col in merged.columns:
        merged = merged.rename(columns={target_col: TARGET_COLUMN})

    # Ensure target is binary 0/1
    if TARGET_COLUMN in merged.columns:
        col = merged[TARGET_COLUMN]
        if col.nunique() == 2:
            vals = sorted(col.unique())
            merged[TARGET_COLUMN] = col.map({vals[0]: 0, vals[1]: 1})
    else:
        # Create a synthetic binary target from last numeric column
        last = merged.select_dtypes(include=np.number).columns[-1]
        merged[TARGET_COLUMN] = (merged[last] > merged[last].median()).astype(int)
        print(f"[!] No target found — synthesised from '{last}' median split")

    return merged


def run():
    print("=" * 60)
    print("STEP 03 — Data Cleaner")
    print("=" * 60)

    registry_path = DATASETS / "table_registry.json"
    if not registry_path.exists():
        print("[!] table_registry.json not found — run Step 02 first")
        sys.exit(1)

    with open(registry_path) as f:
        tables = json.load(f)

    print(f"[→] Cleaning {len(tables)} table(s)...")
    for t in tables:
        print(f"\n  → {t['table']}")

    df = encode_and_merge(tables)

    CLEANED.mkdir(parents=True, exist_ok=True)
    out_path = CLEANED / "ml_ready.csv"
    df.to_csv(out_path, index=False)

    print(f"\n[✓] Step 03 complete — {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"    Saved: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    run()
