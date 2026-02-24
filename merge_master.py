import pandas as pd

# load datasets
media = pd.read_csv("data_raw/combined/master_dataset/mediacloud_combined.csv")
nlp = pd.read_csv("climate_with_embeddings.csv")

# convert dates
media['publish_date'] = pd.to_datetime(media['publish_date'])
nlp['publish_date'] = pd.to_datetime(nlp['publish_date'])

# merge on date
merged = media.merge(nlp, on="publish_date", how="inner")

# save
merged.to_csv("FINAL_MASTER_DATASET.csv", index=False)

print("Final dataset created")
print("Rows:", len(merged))


import pandas as pd
df = pd.read_csv("FINAL_MASTER_DATASET.csv")

print(df.shape)
print(df.columns)
print(df.head())
print(df.isna().sum().sort_values(ascending=False).head(10))

print(df['sentiment_label'].value_counts())
print(df['year_month'].value_counts().sort_index().tail(12))
print(df['media_name_x'].nunique())


#POLARIZATION INSIGHTS
print(df['sentiment_label'].value_counts())


df.groupby('year_month')['sentiment'].mean()


df.groupby('media_name_x')['sentiment'].mean().sort_values().head(10)


df.groupby('media_name_x')['sentiment'].mean().sort_values().tail(10)


df.groupby('year_month')['sentiment'].mean()


pd.read_csv("monthly_polarization_scores.csv")


df.groupby('year_month')['sentiment_label'].value_counts(normalize=True)


pd.read_csv("monthly_polarization_scores.csv")