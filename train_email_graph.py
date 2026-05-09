import pandas as pd
from sklearn.model_selection import train_test_split
from src.preprocessing import final_email_tokens
from src.email_graph_model import GraphSpamDetector

DATA_PATH = "data/email_spam_dataset.csv"

df = pd.read_csv(DATA_PATH)
df["label"] = df["label"].astype(str).str.lower().str.strip()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

model = GraphSpamDetector(similarity_threshold=0.2, top_k=5, min_df=2, max_df_ratio=0.8)
model.fit(train_df, final_email_tokens)

metrics = model.evaluate(test_df, final_email_tokens)
print("Graph-Based Email Spam Detection Results")
for k, v in metrics.items():
    print(k, ":", v)
