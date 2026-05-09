import os
import pandas as pd
import joblib

from src.sms_model import train_sms_model

# =========================
# CREATE MODELS FOLDER
# =========================

os.makedirs("models", exist_ok=True)

# =========================
# SMS DATASET
# =========================

DATA_PATH = "data/sms_spam_dataset.csv"

sms_df = pd.read_csv(DATA_PATH)

sms_df["label"] = sms_df["label"].astype(str).str.lower().str.strip()

# =========================
# TRAIN MODEL
# =========================

result = train_sms_model(
    sms_df,
    model_name="BernoulliNB"
)

# =========================
# PRINT METRICS
# =========================

print("SMS Spam Detection Results")

for k, v in result["metrics"].items():
    print(k, ":", v)

# =========================
# SAVE MODEL
# =========================

joblib.dump(
    result["pipeline"],
    "models/sms_spam_pipeline.joblib"
)

print("Saved model to models/sms_spam_pipeline.joblib")


# =========================
# EMAIL VISUALIZATION
# =========================

from sklearn.model_selection import train_test_split

from visualization import (
    add_token_features,
    scatter_tokenization,
    heatmap_tokenization,
    word_frequency_graph,
    metrics_vs_threshold
)

from src.preprocessing import final_tokens
from src.email_graph_model import GraphSpamDetector

email_df = pd.read_csv("data/email_spam_dataset.csv")

email_df["label"] = email_df["label"].astype(str).str.lower().str.strip()

email_df = add_token_features(email_df)

scatter_tokenization(email_df)

heatmap_tokenization(email_df)

word_frequency_graph(
    email_df,
    "spam",
    "Resultant Graph for Frequency of Words in Spam Corpus"
)

word_frequency_graph(
    email_df,
    "ham",
    "Resultant Graph for Frequency of Words in Ham Corpus"
)

train_df, test_df = train_test_split(
    email_df,
    test_size=0.2,
    random_state=42,
    stratify=email_df["label"]
)

metrics_vs_threshold(
    train_df,
    test_df,
    final_tokens,
    GraphSpamDetector
)

print("All visualizations completed successfully.")