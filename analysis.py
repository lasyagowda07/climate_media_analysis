import pandas as pd

# Update this if your filename is different
FILE = "mc-onlinenews-mediacloud-20260208211645-content.csv"

df = pd.read_csv(FILE)

print("\n--- SHAPE (rows, columns) ---")
print(df.shape)

print("\n--- COLUMNS ---")
print(df.columns.tolist())

print("\n--- FIRST 5 ROWS (selected) ---")
cols_to_try = [c for c in ["title", "publish_date", "media_name", "url", "language", "text"] if c in df.columns]
print(df[cols_to_try].head() if cols_to_try else df.head())

print("\n--- MISSING VALUES (top 15) ---")
print(df.isna().sum().sort_values(ascending=False).head(15))

print("\n--- DUPLICATE URL COUNT (if url exists) ---")
if "url" in df.columns:
    print(df["url"].duplicated().sum())
else:
    print("No 'url' column found.")