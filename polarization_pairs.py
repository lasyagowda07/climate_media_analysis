import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import ast

# ---------- Load centroids ----------
centroids = pd.read_csv("centroids.csv")
centroids["centroid"] = centroids["centroid"].apply(ast.literal_eval).apply(np.array)

# ---------- Load headlines (for examples) ----------
data = pd.read_csv("climate_with_sentiment.csv")

month = "2025-12"
month_data = centroids[centroids["year_month"] == month].copy()

print(f"\nMonth selected: {month}")
print("Outlets:", len(month_data))

outlets = month_data["media_name"].values
X = np.vstack(month_data["centroid"].values)

dist = cosine_distances(X)
np.fill_diagonal(dist, np.nan)

# ---------- Build pair list (upper triangle only) ----------
pairs = []
n = len(outlets)

for i in range(n):
    for j in range(i + 1, n):
        pairs.append((outlets[i], outlets[j], float(dist[i, j])))

pairs_df = pd.DataFrame(pairs, columns=["outlet_1", "outlet_2", "cosine_distance"])

# Sort
closest = pairs_df.sort_values("cosine_distance", ascending=True).head(10)
farthest = pairs_df.sort_values("cosine_distance", ascending=False).head(10)

print("\n--- Top 10 Closest Outlet Pairs ---")
print(closest.to_string(index=False))

print("\n--- Top 10 Most Distant Outlet Pairs ---")
print(farthest.to_string(index=False))

# ---------- Helper to print sample headlines ----------
def print_headlines_for_pair(o1, o2, k=3):
    print(f"\nPair: {o1}  <->  {o2}")
    for outlet in [o1, o2]:
        print(f"\nOutlet: {outlet}")
        sample = data[(data["media_name"] == outlet) & (data["year_month"] == month)]["title"].dropna().head(k)
        for t in sample:
            print("-", t)

print("\n\n===== Examples for Closest Pairs =====")
for _, row in closest.iterrows():
    print_headlines_for_pair(row["outlet_1"], row["outlet_2"], k=2)

print("\n\n===== Examples for Most Distant Pairs =====")
for _, row in farthest.iterrows():
    print_headlines_for_pair(row["outlet_1"], row["outlet_2"], k=2)

# Save the pair list for this month
pairs_df.to_csv(f"{month}_pairwise_distances.csv", index=False)
print(f"\nSaved -> {month}_pairwise_distances.csv")