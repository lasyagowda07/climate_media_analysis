import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download("vader_lexicon")

INPUT_FILE = "climate_prepared.csv"
OUTPUT_FILE = "climate_with_sentiment.csv"

def main():
    df = pd.read_csv(INPUT_FILE)

    print("\n--- Loaded dataset ---")
    print("Rows:", len(df))

    sia = SentimentIntensityAnalyzer()

    # Sentiment scores
    df["sentiment"] = df["title_clean"].apply(
        lambda x: sia.polarity_scores(x)["compound"]
    )

    # Label sentiment (for interpretation)
    def label_sentiment(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    df["sentiment_label"] = df["sentiment"].apply(label_sentiment)

    print("\n--- Sentiment distribution ---")
    print(df["sentiment_label"].value_counts())

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()