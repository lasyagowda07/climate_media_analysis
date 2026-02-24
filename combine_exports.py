import pandas as pd
from pathlib import Path

export_folder = Path("data_raw/mediacloud_exports")
files = list(export_folder.glob("*.csv"))

print(f"Found {len(files)} export files")

dfs = []
for file in files:
    print(f"Reading: {file.name}")
    df = pd.read_csv(file)
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

print("\nTotal rows combined:", len(combined))

output_path = Path("data_raw/combined/master_dataset/mediacloud_combined.csv")
combined.to_csv(output_path, index=False)

print(f"\nSaved combined dataset → {output_path}")