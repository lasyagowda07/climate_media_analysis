import pandas as pd
import re

FILE = "mc-onlinenews-mediacloud-20260208211645-content.csv"

df = pd.read_csv(FILE)

# Keep only English articles (important for NLP later)
df = df[df["language"] == "en"]

# Climate-related keywords
CLIMATE_KEYWORDS = [
    "climate change",
    "global warming",
    "carbon",
    "emissions",
    "greenhouse",
    "renewable",
    "fossil fuel",
    "net zero",
    "energy transition",
    "climate policy",
    "paris agreement",
    "cop",
    "carbon tax",
    "clean energy"
]

# Build regex pattern
pattern = "|".join(CLIMATE_KEYWORDS)

# Filter based on headline (title)
climate_df = df[
    df["title"].str.lower().str.contains(pattern, regex=True, na=False)
]

print("Total articles:", len(df))
print("Climate-related articles:", len(climate_df))

print("\nSample climate headlines:\n")
print(climate_df["title"].head(10))

# Save filtered dataset
climate_df.to_csv("climate_headlines.csv", index=False)