import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = "climate_with_sentiment.csv"
OUTPUT_FILE = "climate_with_embeddings.csv"

TEXT_COL = "title_clean"   
MODEL_NAME = "all-MiniLM-L6-v2"  

def main():
    df = pd.read_csv(INPUT_FILE)

    # Keep only rows we need
    df = df.dropna(subset=[TEXT_COL, "media_name", "date"]).copy()

    texts = df[TEXT_COL].astype(str).tolist()

    print("Rows:", len(df))
    print("Embedding model:", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    # Encode headlines
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True 
    )

    # Save embeddings as one column (list) so it stays in CSV
    df["embedding"] = [emb.tolist() for emb in embeddings]

    df.to_csv(OUTPUT_FILE, index=False)
    print("Saved ->", OUTPUT_FILE)

if __name__ == "__main__":
    main()