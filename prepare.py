import pandas as pd

INPUT_FILE = "climate_headlines.csv"
OUTPUT_FILE = "climate_prepared.csv"

def main():
    df = pd.read_csv(INPUT_FILE)

    print("\n--- Loaded climate dataset ---")
    print("Rows, Cols:", df.shape)
    print("Columns:", list(df.columns))

    # Keep essential columns
    df = df[["publish_date", "media_name", "title", "url"]].copy()

    # Convert date
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df = df.dropna(subset=["publish_date", "title"])

    # Time features
    df["date"] = df["publish_date"].dt.date
    df["year"] = df["publish_date"].dt.year
    df["month"] = df["publish_date"].dt.month
    df["year_month"] = df["publish_date"].dt.to_period("M").astype(str)
    df["week"] = df["publish_date"].dt.to_period("W").astype(str)

    # Clean outlet names
    df["media_name"] = df["media_name"].str.lower().str.strip()

    # Clean titles
    df["title_clean"] = (
        df["title"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Remove duplicates
    df = df.drop_duplicates(subset=["media_name", "title_clean"])

    print("\n--- After cleaning ---")
    print("Rows:", len(df))
    print("Date range:", df["date"].min(), "to", df["date"].max())

    # Daily volume (important for your dataset)
    print("\n--- Articles per day (sample) ---")
    print(df["date"].value_counts().sort_index().head(10))

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved prepared dataset -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()