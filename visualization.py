import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def count_words(text):
    return len(str(text).split())


def count_sentences(text):
    return len(re.split(r'[.!?]+', str(text)))


def add_token_features(df):
    df["subject"] = df["subject"].astype(str)
    df["body"] = df["body"].astype(str)

    df["num_characters_subject"] = df["subject"].apply(len)
    df["num_characters_body"] = df["body"].apply(len)

    df["num_words_subject"] = df["subject"].apply(count_words)
    df["num_words_body"] = df["body"].apply(count_words)

    df["num_sentences_subject"] = df["subject"].apply(count_sentences)
    df["num_sentences_body"] = df["body"].apply(count_sentences)

    return df


def scatter_tokenization(df):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=df,
        x="num_words_subject",
        y="num_words_body",
        hue="label"
    )
    plt.title("Scatter Plot of Different Tokenizations")
    plt.xlabel("Number of Words in Subject")
    plt.ylabel("Number of Words in Body")
    plt.show()


def heatmap_tokenization(df):
    cols = [
        "num_characters_subject",
        "num_characters_body",
        "num_words_subject",
        "num_words_body",
        "num_sentences_subject",
        "num_sentences_body"
    ]

    plt.figure(figsize=(8, 5))
    sns.heatmap(df[cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Heatmap of Different Tokenizations")
    plt.show()


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def word_frequency_graph(df, label_name, title):
    text = " ".join(df[df["label"] == label_name]["body"].astype(str))
    text = clean_text(text)

    words = text.split()

    stop_words = {
        "the", "is", "and", "to", "of", "in", "for", "you",
        "your", "a", "an", "on", "this", "that", "with", "be"
    }

    words = [word for word in words if word not in stop_words and len(word) > 2]

    word_counts = Counter(words).most_common(20)

    words = [x[0] for x in word_counts]
    counts = [x[1] for x in word_counts]

    plt.figure(figsize=(10, 5))
    sns.barplot(x=counts, y=words)
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.show()

def metrics_vs_threshold(train_df, test_df, final_tokens, GraphSpamDetector):
    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for threshold in thresholds:
        model = GraphSpamDetector(
            threshold,
            5,
            2,
            0.80
        )

        model.fit(train_df, final_tokens)

        y_true = []
        y_pred = []

        for _, row in test_df.iterrows():
            tokens = final_tokens(row)
            pred = model.predict(tokens)

            y_true.append(row["label"])
            y_pred.append(pred)

        accuracy_list.append(accuracy_score(y_true, y_pred))
        precision_list.append(precision_score(y_true, y_pred, pos_label="spam", zero_division=0))
        recall_list.append(recall_score(y_true, y_pred, pos_label="spam", zero_division=0))
        f1_list.append(f1_score(y_true, y_pred, pos_label="spam", zero_division=0))

    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, accuracy_list, marker="o", label="Accuracy")
    plt.plot(thresholds, precision_list, marker="o", label="Precision")
    plt.plot(thresholds, recall_list, marker="o", label="Recall")
    plt.plot(thresholds, f1_list, marker="o", label="F1 Score")

    plt.title("Metrics vs Similarity Threshold")
    plt.xlabel("Similarity Threshold")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()
