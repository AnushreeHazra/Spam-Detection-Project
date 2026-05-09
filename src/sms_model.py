"""SMS spam classification using TF-IDF and classical ML models."""

import re
from typing import Dict
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def clean_sms(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " urltoken ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_sms_pipeline(model_name: str = "BernoulliNB") -> Pipeline:
    if model_name == "MultinomialNB":
        clf = MultinomialNB()
    elif model_name == "LogisticRegression":
        clf = LogisticRegression(max_iter=1000)
    else:
        clf = BernoulliNB()
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_sms, max_features=5000, ngram_range=(1, 2))),
        ("model", clf),
    ])


def train_sms_model(df: pd.DataFrame, model_name: str = "BernoulliNB") -> Dict:
    message_col = "message" if "message" in df.columns else df.columns[0]
    df = df.copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    pipe = build_sms_pipeline(model_name)
    pipe.fit(train_df[message_col], train_df["label"])
    preds = pipe.predict(test_df[message_col])
    cm = confusion_matrix(test_df["label"], preds, labels=["ham", "spam"])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0
    metrics = {
        "accuracy": accuracy_score(test_df["label"], preds),
        "precision": precision_score(test_df["label"], preds, pos_label="spam", zero_division=0),
        "recall": recall_score(test_df["label"], preds, pos_label="spam", zero_division=0),
        "f1_score": f1_score(test_df["label"], preds, pos_label="spam", zero_division=0),
        "specificity": specificity,
        "confusion_matrix": cm.tolist(),
    }
    return {"pipeline": pipe, "metrics": metrics, "train_df": train_df, "test_df": test_df}
