import os
import ast
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

INPUT_FILE = "poc_step2_anchor_filter/climate_filtered_by_anchor.csv"
OUT_DIR = "poc_step3_dedup_syndication"
MIN_OUTLETS_TO_CALL_SYNDICATED = 2   # if same title appears in >=2 outlets in same month -> drop
MIN_HEADLINES_PER_OUTLET_MONTH = 3

os.makedirs(OUT_DIR, exist_ok=True)

def parse_embedding(x):
    if isinstance(x, str):
        x = x.strip()

        # Case 1: Proper Python list format (with commas)
        if "," in x:
            try:
                return np.array(ast.literal_eval(x), dtype=float)
            except:
                pass

        # Case 2: Space-separated numpy-style string (your case)
        try:
            # Remove brackets and split by whitespace
            x = x.replace("[", "").replace("]", "")
            values = x.split()
            return np.array([float(v) for v in values], dtype=float)
        except:
            return None

    if isinstance(x, (list, tuple, np.ndarray)):
        return np.array(x, dtype=float)

    return None

def compute_centroids(df_month):
    grouped = df_month.groupby(["media_name", "year_month"])
    rows = []
    for (outlet, ym), g in grouped:
        if len(g) < MIN_HEADLINES_PER_OUTLET_MONTH:
            continue
        vecs = np.stack(g["embedding"].values)
        rows.append({"media_name": outlet, "year_month": ym, "centroid": vecs.mean(axis=0)})
    return pd.DataFrame(rows)

def month_score(centroids):
    X = np.stack(centroids["centroid"].values)
    D = cosine_distances(X)
    tri = np.triu_indices_from(D, k=1)
    return float(np.mean(D[tri]))

# -------- MAIN --------
df = pd.read_csv(INPUT_FILE)

required = {"year_month", "media_name", "embedding"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# choose a text key for dedup
# Prefer title_clean if present, else title
text_col = "title_clean" if "title_clean" in df.columns else "title"
if text_col not in df.columns:
    raise ValueError("Need either 'title_clean' or 'title' column for dedup.")

df["embedding"] = df["embedding"].apply(parse_embedding)
df = df[df["embedding"].notna()].copy()

# mark syndicated within each month: same headline appears across multiple outlets
counts = (
    df.groupby(["year_month", text_col])["media_name"]
      .nunique()
      .reset_index(name="num_outlets_sharing")
)

df = df.merge(counts, on=["year_month", text_col], how="left")

before_rows = len(df)
dedup = df[df["num_outlets_sharing"] < MIN_OUTLETS_TO_CALL_SYNDICATED].copy()
after_rows = len(dedup)

dedup.to_csv(os.path.join(OUT_DIR, "climate_anchor_filtered_dedup.csv"), index=False)

print(f"\nRows before dedup: {before_rows}")
print(f"Rows after dedup : {after_rows}")
print(f"Saved -> {OUT_DIR}/climate_anchor_filtered_dedup.csv")

# compare monthly polarization before vs after dedup
months = sorted(df["year_month"].unique().tolist())
rows = []

for ym in months:
    a = df[df["year_month"] == ym].copy()
    b = dedup[dedup["year_month"] == ym].copy()

    ca = compute_centroids(a)
    cb = compute_centroids(b)

    score_a = month_score(ca) if len(ca) >= 3 else np.nan
    score_b = month_score(cb) if len(cb) >= 3 else np.nan

    rows.append({
        "year_month": ym,
        "score_before_dedup": score_a,
        "outlets_before_dedup": len(ca),
        "rows_before_dedup": len(a),
        "score_after_dedup": score_b,
        "outlets_after_dedup": len(cb),
        "rows_after_dedup": len(b)
    })

compare = pd.DataFrame(rows).sort_values("year_month")
compare.to_csv(os.path.join(OUT_DIR, "monthly_scores_before_vs_after_dedup.csv"), index=False)

print(f"\nSaved -> {OUT_DIR}/monthly_scores_before_vs_after_dedup.csv")