import json
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt


# Load JSON
file_path = "data/saps.json"
with open(file_path, 'r') as file:
    data = json.load(file)

# Parse metadata for 'RobinHoodPennyStocks'
metadata = data["RobinHoodPennyStocks"]["md"]
post_data_meta = metadata["postData"]
inter_keys = metadata["inter"]["keys"]
intra_keys = metadata["intra"]["keys"]

# Parse raw data for 'RobinHoodPennyStocks'
raw_posts = data["RobinHoodPennyStocks"]["raw"]["postData"]

# Convert raw posts to DataFrame
posts_df = pd.DataFrame(raw_posts, columns=["ticker", "title", "text", "flair", "timestamp"])

# Add financial data placeholders
posts_df["inter_fin_data"] = posts_df.index.map(lambda x: inter_keys[x] if x < len(inter_keys) else None)
posts_df["intra_fin_data"] = posts_df.index.map(lambda x: intra_keys[x] if x < len(intra_keys) else None)

# Initialize Hugging Face Sentiment Analysis Pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    if pd.isnull(text) or not isinstance(text, str) or len(text.strip()) == 0:
        return None, None
    try:
        result = sentiment_analyzer(text[:512])[0]
        return result['label'], result['score']
    except Exception as e:
        return None, None

posts_df[["sentiment", "sentiment_score"]] = posts_df["text"].apply(
    lambda text: pd.Series(analyze_sentiment(text))
)

# Save to CSV for further analysis
posts_df.to_csv("processed_posts_with_sentiment.csv", index=False)
print("Processed posts saved with sentiment analysis.")


df = pd.read_csv("processed_posts_with_sentiment.csv")

# Visualize sentiment distribution
df["sentiment"].value_counts().plot(kind="bar", title="Sentiment Distribution")
plt.show()

