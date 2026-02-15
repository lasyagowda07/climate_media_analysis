import os
import ast
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sentence_transformers import SentenceTransformer

INPUT_FILE = "climate_with_embeddings.csv"
OUT_DIR = "poc_step2_anchor_filter"
MODEL_NAME = "all-MiniLM-L6-v2"  # MUST match what you used for embeddings
ANCHOR_TEXT = "climate change global warming emissions carbon dioxide fossil fuels renewable energy net zero policy"
SIM_THRESHOLD = 0.30  # tweak after you inspect distribution
MIN_HEADLINES_PER_OUTLET_MONTH = 3
TOP_K = 10
SAMPLE_HEADLINES_PER_OUTLET = 2

os.makedirs(OUT_DIR, exist_ok=True)

def parse_embedding(x):
    if isinstance(x, str):
        return np.array(ast.literal_eval(x), dtype=float)
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

def month_score_from_centroids(centroids):
    X = np.stack(centroids["centroid"].values)
    D = cosine_distances(X)
    tri = np.triu_indices_from(D, k=1)
    return float(np.mean(D[tri])), len(centroids)

def top_pairs(centroids, k=10):
    outlets = centroids["media_name"].values
    X = np.stack(centroids["centroid"].values)
    D = cosine_distances(X)
    np.fill_diagonal(D, np.nan)
    tri = np.triu_indices_from(D, k=1)

    pairs = [(outlets[i], outlets[j], float(D[i, j])) for i, j in zip(tri[0], tri[1])]
    pairs_sorted = sorted(pairs, key=lambda t: t[2])
    return pairs_sorted[:k], pairs_sorted[-k:][::-1]

def sample_titles(df_month, outlet, n=2):
    return df_month[df_month["media_name"] == outlet]["title"].dropna().head(n).tolist()

# -------- MAIN --------
df = pd.read_csv(INPUT_FILE)
required = {"year_month", "media_name", "title", "embedding"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["embedding"] = df["embedding"].apply(parse_embedding)
df = df[df["embedding"].notna()].copy()

# anchor embedding
model = SentenceTransformer(MODEL_NAME)
anchor_vec = model.encode([ANCHOR_TEXT], convert_to_numpy=True)[0].reshape(1, -1)

# cosine similarity headline->anchor
X = np.stack(df["embedding"].values)
sims = cosine_similarity(X, anchor_vec).reshape(-1)
df["anchor_similarity"] = sims

# save distribution quick view
desc = df["anchor_similarity"].describe()
desc.to_csv(os.path.join(OUT_DIR, "anchor_similarity_describe.csv"))
print("\nAnchor similarity describe saved.")

# filter
df_f = df[df["anchor_similarity"] >= SIM_THRESHOLD].copy()
df_f.to_csv(os.path.join(OUT_DIR, "climate_filtered_by_anchor.csv"), index=False)
print(f"\nFiltered rows: {len(df_f)} / {len(df)} (threshold={SIM_THRESHOLD})")
print(f"Saved -> {OUT_DIR}/climate_filtered_by_anchor.csv")

# rerun monthly scores BEFORE vs AFTER
months = sorted(df["year_month"].unique().tolist())
rows = []
appendix = ["# Step 2 Appendix: Anchor Filter Impact\n",
            f"- Model: {MODEL_NAME}\n- Anchor: {ANCHOR_TEXT}\n- Threshold: {SIM_THRESHOLD}\n"]

for ym in months:
    b = df[df["year_month"] == ym].copy()
    a = df_f[df_f["year_month"] == ym].copy()

    # before
    cb = compute_centroids(b)
    if len(cb) >= 3:
        score_b, outlets_b = month_score_from_centroids(cb)
    else:
        score_b, outlets_b = np.nan, len(cb)

    # after
    ca = compute_centroids(a)
    if len(ca) >= 3:
        score_a, outlets_a = month_score_from_centroids(ca)
    else:
        score_a, outlets_a = np.nan, len(ca)

    rows.append({
        "year_month": ym,
        "score_before": score_b,
        "outlets_before": outlets_b,
        "rows_before": len(b),
        "score_after": score_a,
        "outlets_after": outlets_a,
        "rows_after": len(a)
    })

    # also save pair extremes after-filter for each month (qual evidence)
    if len(ca) >= 3:
        closest, farthest = top_pairs(ca, k=TOP_K)
        pd.DataFrame(closest, columns=["outlet_1","outlet_2","cosine_distance"]).to_csv(
            os.path.join(OUT_DIR, f"{ym}_closest_pairs_AFTER.csv"), index=False
        )
        pd.DataFrame(farthest, columns=["outlet_1","outlet_2","cosine_distance"]).to_csv(
            os.path.join(OUT_DIR, f"{ym}_farthest_pairs_AFTER.csv"), index=False
        )

        # add 2 example pairs to appendix (short)
        appendix.append(f"\n## {ym}\n")
        appendix.append("### Example Closest Pair (After Filter)\n")
        o1, o2, d = closest[0]
        appendix.append(f"- {o1} ↔ {o2} (distance={d:.6f})")
        for outlet in [o1, o2]:
            for t in sample_titles(a, outlet, n=SAMPLE_HEADLINES_PER_OUTLET):
                appendix.append(f"  - {outlet}: {t}")

        appendix.append("\n### Example Most-Distant Pair (After Filter)\n")
        o1, o2, d = farthest[0]
        appendix.append(f"- {o1} ↔ {o2} (distance={d:.6f})")
        for outlet in [o1, o2]:
            for t in sample_titles(a, outlet, n=SAMPLE_HEADLINES_PER_OUTLET):
                appendix.append(f"  - {outlet}: {t}")

compare = pd.DataFrame(rows).sort_values("year_month")
compare.to_csv(os.path.join(OUT_DIR, "monthly_scores_before_vs_after_anchor.csv"), index=False)

with open(os.path.join(OUT_DIR, "APPENDIX_anchor_filter_examples.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(appendix))

print("\nSaved:")
print(f"- {OUT_DIR}/monthly_scores_before_vs_after_anchor.csv")
print(f"- {OUT_DIR}/APPENDIX_anchor_filter_examples.md")