import pandas as pd
import numpy as np

# Load dataset with embeddings
df = pd.read_csv("climate_with_embeddings.csv")

print("\n--- Loaded Dataset ---")
print("Rows:", len(df))
print("Columns:", df.columns)

# Convert embedding column from string to numpy array
df["embedding"] = df["embedding"].apply(eval)
df["embedding"] = df["embedding"].apply(np.array)

print("\nEmbedding sample shape:")
print(df["embedding"].iloc[0].shape)

# Group by outlet and month
grouped = df.groupby(["media_name", "year_month"])

centroids = []

for (outlet, month), group in grouped:
    vectors = np.stack(group["embedding"].values)
    centroid = np.mean(vectors, axis=0)

    centroids.append({
        "media_name": outlet,
        "year_month": month,
        "centroid": centroid
    })

centroids_df = pd.DataFrame(centroids)

print("\n--- Centroids Created ---")
print("Rows:", len(centroids_df))

from sklearn.metrics.pairwise import cosine_distances

polarization_scores = []

# Group centroids by month
months = centroids_df["year_month"].unique()

for month in months:
    month_data = centroids_df[centroids_df["year_month"] == month]

    vectors = np.stack(month_data["centroid"].values)

    # Compute pairwise cosine distances
    distances = cosine_distances(vectors)

    # We only want upper triangle (no duplicates, no diagonal)
    triu_indices = np.triu_indices_from(distances, k=1)
    pairwise_values = distances[triu_indices]

    avg_distance = np.mean(pairwise_values)

    polarization_scores.append({
        "year_month": month,
        "polarization_score": avg_distance,
        "num_outlets": len(month_data)
    })

polarization_df = pd.DataFrame(polarization_scores)

print("\n--- Monthly Polarization Scores ---")
print(polarization_df)

polarization_df.to_csv("monthly_polarization_scores.csv", index=False)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(8,5))
# plt.plot(polarization_df["year_month"], polarization_df["polarization_score"], marker="o")
# plt.title("Monthly Linguistic Polarization in Climate Coverage")
# plt.xlabel("Month")
# plt.ylabel("Average Cosine Distance")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# Save centroids to CSV
centroids_df_save = centroids_df.copy()

# Convert numpy arrays to list so CSV can store them
centroids_df_save["centroid"] = centroids_df_save["centroid"].apply(lambda x: x.tolist())

centroids_df_save.to_csv("centroids.csv", index=False)

print("\nSaved -> centroids.csv")