import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

INPUT_FILE = "poc_step3_dedup_syndication/climate_anchor_filtered_dedup.csv"
MONTH = "2025-12"

TOP_K_PAIRS = 10          # top closest + farthest outlet pairs
HEADLINE_EXAMPLES = 3     # how many headline matches to print per outlet pair
MAX_HEADLINES_PER_OUTLET = 250  # safety cap to avoid huge pairwise matrices

def parse_embedding(x):
    x = str(x).strip().replace("\n", " ")
    x = x.replace("[", "").replace("]", "")
    return np.fromstring(x, sep=" ")

def get_headline_matches(df_month, outlet_a, outlet_b, k=3):
    """
    Returns k most similar headline pairs between outlet_a and outlet_b.
    """
    A = df_month[df_month["media_name"] == outlet_a].copy()
    B = df_month[df_month["media_name"] == outlet_b].copy()

    # Safety cap (prevents huge N x M)
    A = A.head(MAX_HEADLINES_PER_OUTLET)
    B = B.head(MAX_HEADLINES_PER_OUTLET)

    if len(A) == 0 or len(B) == 0:
        return []

    XA = np.vstack(A["embedding"].values)
    XB = np.vstack(B["embedding"].values)

    # cosine distance between each A headline and each B headline
    D = cosine_distances(XA, XB)

    # find k smallest distances
    flat_idx = np.argsort(D, axis=None)[:k]
    pairs = []
    for idx in flat_idx:
        i, j = np.unravel_index(idx, D.shape)
        pairs.append({
            "distance": float(D[i, j]),
            "headline_a": A.iloc[i]["title"],
            "headline_b": B.iloc[j]["title"],
            "url_a": A.iloc[i].get("url", ""),
            "url_b": B.iloc[j].get("url", "")
        })
    return pairs


# -----------------------------
# Load and prep
# -----------------------------
df = pd.read_csv(INPUT_FILE)

df["embedding"] = df["embedding"].apply(parse_embedding)

month_df = df[df["year_month"] == MONTH].copy()

print(f"\nMonth selected: {MONTH}")
print("Rows:", len(month_df))
print("Unique outlets:", month_df["media_name"].nunique())

# -----------------------------
# Compute outlet centroids
# -----------------------------
grouped = month_df.groupby("media_name")

centroids = []
outlets = []

for outlet, group in grouped:
    vectors = np.stack(group["embedding"].values)
    centroid = np.mean(vectors, axis=0)
    centroids.append(centroid)
    outlets.append(outlet)

X = np.vstack(centroids)

# Pairwise distance between outlet centroids
dist_matrix = cosine_distances(X)
np.fill_diagonal(dist_matrix, np.nan)

pairs = []
for i in range(len(outlets)):
    for j in range(i + 1, len(outlets)):
        pairs.append((outlets[i], outlets[j], dist_matrix[i, j]))

pairs_df = pd.DataFrame(pairs, columns=["outlet_1", "outlet_2", "distance"]).sort_values("distance")

print("\n===== Top 10 Closest Outlet Pairs =====")
print(pairs_df.head(TOP_K_PAIRS))

print("\n===== Top 10 Most Distant Outlet Pairs =====")
print(pairs_df.tail(TOP_K_PAIRS))


# -----------------------------
# Print headline examples for closest pairs
# -----------------------------
print("\n\n==============================")
print("  EXAMPLES FOR CLOSEST PAIRS  ")
print("==============================\n")

for _, row in pairs_df.head(TOP_K_PAIRS).iterrows():
    o1, o2, d = row["outlet_1"], row["outlet_2"], row["distance"]
    print(f"\nPair: {o1}  <->  {o2} | centroid_distance={d:.4f}")

    matches = get_headline_matches(month_df, o1, o2, k=HEADLINE_EXAMPLES)

    if not matches:
        print("No headline pairs found.")
        continue

    for m in matches:
        print(f"  - headline_distance={m['distance']:.4f}")
        print(f"    {o1}: {m['headline_a']}")
        print(f"    {o2}: {m['headline_b']}")
        print()

# -----------------------------
# Print headline examples for farthest pairs
# -----------------------------
print("\n\n==============================")
print("  EXAMPLES FOR MOST DISTANT   ")
print("==============================\n")

for _, row in pairs_df.tail(TOP_K_PAIRS).iterrows():
    o1, o2, d = row["outlet_1"], row["outlet_2"], row["distance"]
    print(f"\nPair: {o1}  <->  {o2} | centroid_distance={d:.4f}")

    matches = get_headline_matches(month_df, o1, o2, k=HEADLINE_EXAMPLES)

    if not matches:
        print("No headline pairs found.")
        continue

    # For far pairs, we still print the "closest headline matches" just to show
    # even their nearest content is not that close.
    for m in matches:
        print(f"  - headline_distance={m['distance']:.4f}")
        print(f"    {o1}: {m['headline_a']}")
        print(f"    {o2}: {m['headline_b']}")
        print()