import pandas as pd
from pathlib import Path
import sys

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # 1) strip whitespace, lowercase all column names
    df.columns = [c.strip().lower() for c in df.columns]
    # 2) rename the two expected cols if they exist
    rename_map = {}
    if 'label' in df.columns:   rename_map['label'] = 'label'
    if 'text' in df.columns:    rename_map['text']  = 'text'
    df = df.rename(columns=rename_map)
    # 3) drop rows that are entirely empty
    return df.dropna(how='all')

def main():
    root = Path(__file__).parent / "datasets_texts"
    if not root.exists():
        print(f"ERROR: folder not found: {root}")
        sys.exit(1)

    csv_files = sorted(root.glob("*.csv"))
    if not csv_files:
        print("ERROR: No CSV files in datasets_texts")
        sys.exit(1)

    frames = []
    for path in csv_files:
        try:
            # read everything as str to avoid dtype surprises
            df = pd.read_csv(path, encoding='utf-8', dtype=str)
        except Exception as e:
            print(f"⚠️  Skipping {path.name}: read error ({e})")
            continue

        df = clean_df(df)
        if df.empty:
            print(f"⚠️  Skipping {path.name}: no data after cleaning")
            continue

        disease = path.stem.strip()
        df['disease'] = disease
        frames.append(df)
        print(f"✔ Loaded {path.name}: {len(df)} rows, cols={list(df.columns)}")

    if not frames:
        print("ERROR: No valid data to merge.")
        sys.exit(1)

    # 4) concat, sort, reorder
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.sort_values(by='disease').reset_index(drop=True)

    # put disease first, then label & text if present
    base_cols = ['disease', 'label', 'text']
    cols = [c for c in base_cols if c in merged.columns] + \
           [c for c in merged.columns if c not in base_cols]
    merged = merged[cols]

    # 5) write out
    out = Path(__file__).parent / "merged_datasets_texts.xlsx"
    merged.to_excel(out, index=False)
    print(f"\n✅ Merged {len(frames)} files → {out} ({len(merged)} total rows)")

if __name__ == "__main__":
    main()
