import os
import ast
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

INPUT_FILE = "climate_with_embeddings.csv"
OUT_DIR = "poc_step1_pairs_all_months"
TOP_K = 10
SAMPLE_HEADLINES_PER_OUTLET = 3
MIN_HEADLINES_PER_OUTLET_MONTH = 3  # skip outlets with tiny counts (noise)

os.makedirs(OUT_DIR, exist_ok=True)

def parse_embedding(x):
    # safer than eval; handles stringified lists
    if isinstance(x, str):
        return np.array(ast.literal_eval(x), dtype=float)
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.array(x, dtype=float)
    return None

def compute_centroids(df_month: pd.DataFrame) -> pd.DataFrame:
    # group by outlet-month, average embeddings
    grouped = df_month.groupby(["media_name", "year_month"])
    rows = []
    for (outlet, ym), g in grouped:
        if len(g) < MIN_HEADLINES_PER_OUTLET_MONTH:
            continue
        vecs = np.stack(g["embedding"].values)
        centroid = vecs.mean(axis=0)
        rows.append({"media_name": outlet, "year_month": ym, "centroid": centroid})
    return pd.DataFrame(rows)

def top_pairs_for_month(centroids_month: pd.DataFrame, k=10):
    outlets = centroids_month["media_name"].values
    X = np.stack(centroids_month["centroid"].values)
    D = cosine_distances(X)

    # ignore diagonal
    np.fill_diagonal(D, np.nan)

    # get upper triangle indices to avoid duplicates
    tri = np.triu_indices_from(D, k=1)
    pairs = []
    for i, j in zip(tri[0], tri[1]):
        pairs.append((outlets[i], outlets[j], float(D[i, j])))

    pairs_sorted = sorted(pairs, key=lambda t: t[2])
    closest = pairs_sorted[:k]
    farthest = pairs_sorted[-k:][::-1]
    return closest, farthest

def sample_headlines(df_month: pd.DataFrame, outlet: str, n=3):
    # use title_clean if you want consistent; but for human appendix use original title
    s = df_month[df_month["media_name"] == outlet]["title"].dropna().head(n).tolist()
    return s

# ---------- MAIN ----------
df = pd.read_csv(INPUT_FILE)
required = {"year_month", "media_name", "title", "embedding"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns in {INPUT_FILE}: {missing}")

df["embedding"] = df["embedding"].apply(parse_embedding)
df = df[df["embedding"].notna()].copy()

months = sorted(df["year_month"].dropna().unique().tolist())
print(f"Found months: {months}")

summary_rows = []
appendix_lines = ["# POC Appendix: Closest vs Most-Distant Outlet Pairs (All Months)\n"]

for ym in months:
    df_month = df[df["year_month"] == ym].copy()
    centroids = compute_centroids(df_month)

    if len(centroids) < 3:
        print(f"[SKIP] {ym}: not enough outlets after filtering (need >=3).")
        continue

    closest, farthest = top_pairs_for_month(centroids, k=TOP_K)

    # save pair tables
    closest_df = pd.DataFrame(closest, columns=["outlet_1", "outlet_2", "cosine_distance"])
    farthest_df = pd.DataFrame(farthest, columns=["outlet_1", "outlet_2", "cosine_distance"])

    closest_path = os.path.join(OUT_DIR, f"{ym}_top{TOP_K}_closest_pairs.csv")
    farthest_path = os.path.join(OUT_DIR, f"{ym}_top{TOP_K}_farthest_pairs.csv")
    closest_df.to_csv(closest_path, index=False)
    farthest_df.to_csv(farthest_path, index=False)

    # appendix text
    appendix_lines.append(f"\n## Month: {ym}\n")
    appendix_lines.append(f"Outlets used (after min-count filter): **{len(centroids)}**\n")

    appendix_lines.append("### Top 10 Closest Pairs + Example Headlines\n")
    for o1, o2, d in closest:
        appendix_lines.append(f"- **{o1} ↔ {o2}** (distance={d:.6f})")
        for outlet in [o1, o2]:
            heads = sample_headlines(df_month, outlet, n=SAMPLE_HEADLINES_PER_OUTLET)
            for h in heads:
                appendix_lines.append(f"  - {outlet}: {h}")
        appendix_lines.append("")

    appendix_lines.append("### Top 10 Most-Distant Pairs + Example Headlines\n")
    for o1, o2, d in farthest:
        appendix_lines.append(f"- **{o1} ↔ {o2}** (distance={d:.6f})")
        for outlet in [o1, o2]:
            heads = sample_headlines(df_month, outlet, n=SAMPLE_HEADLINES_PER_OUTLET)
            for h in heads:
                appendix_lines.append(f"  - {outlet}: {h}")
        appendix_lines.append("")

    # summary row (use avg pairwise for month)
    X = np.stack(centroids["centroid"].values)
    D = cosine_distances(X)
    tri = np.triu_indices_from(D, k=1)
    month_score = float(np.mean(D[tri]))
    summary_rows.append({"year_month": ym, "polarization_score": month_score, "num_outlets": len(centroids)})

    print(f"[DONE] {ym}: saved pair tables + appendix entries.")

# save summary + appendix
summary_df = pd.DataFrame(summary_rows).sort_values("year_month")
summary_df.to_csv(os.path.join(OUT_DIR, "monthly_polarization_scores_step1.csv"), index=False)

with open(os.path.join(OUT_DIR, "APPENDIX_pairs_examples.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(appendix_lines))

print("\nSaved:")
print(f"- {OUT_DIR}/monthly_polarization_scores_step1.csv")
print(f"- {OUT_DIR}/APPENDIX_pairs_examples.md")
print(f"- {OUT_DIR}/*_closest_pairs.csv and *_farthest_pairs.csv")