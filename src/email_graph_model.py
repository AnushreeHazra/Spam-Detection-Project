"""Graph-based email spam classifier using token similarity and neighbourhood voting."""

import math
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Set, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class GraphSpamDetector:
    def __init__(self, similarity_threshold: float = 0.2, top_k: int = 5, min_df: int = 2, max_df_ratio: float = 0.8):
        self.threshold = similarity_threshold
        self.top_k = top_k
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio
        self.nodes: List[Dict] = []
        self.token_index: Dict[str, Set[int]] = defaultdict(set)
        self.allowed_tokens: Set[str] = set()

    @staticmethod
    def cosine_similarity(a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a.intersection(b)) / (math.sqrt(len(a)) * math.sqrt(len(b)) + 1e-9)

    def _build_allowed_tokens(self, all_tokens: List[Set[str]]) -> Set[str]:
        df_counter = Counter()
        total_docs = len(all_tokens)
        for tokens in all_tokens:
            for token in tokens:
                df_counter[token] += 1
        max_df = max(1, int(self.max_df_ratio * total_docs))
        return {tok for tok, count in df_counter.items() if self.min_df <= count <= max_df}

    def fit(self, df: pd.DataFrame, token_function: Callable) -> None:
        """Store email nodes and create inverted token index."""
        self.nodes = []
        self.token_index = defaultdict(set)

        raw_tokens = [token_function(row.to_dict()) for _, row in df.iterrows()]
        self.allowed_tokens = self._build_allowed_tokens(raw_tokens)

        for node_id, (_, row) in enumerate(df.iterrows()):
            tokens = raw_tokens[node_id].intersection(self.allowed_tokens)
            label = str(row["label"]).lower().strip()
            self.nodes.append({"tokens": tokens, "label": label, "row_index": node_id})
            for token in tokens:
                self.token_index[token].add(node_id)

    def _candidate_ids(self, tokens: Set[str]) -> Set[int]:
        candidates = set()
        for token in tokens:
            candidates.update(self.token_index.get(token, set()))
        return candidates

    def similar_neighbours(self, tokens: Set[str]) -> List[Tuple[int, float, str]]:
        tokens = tokens.intersection(self.allowed_tokens) if self.allowed_tokens else tokens
        candidates = self._candidate_ids(tokens)
        sims = []
        for idx in candidates:
            sim = self.cosine_similarity(tokens, self.nodes[idx]["tokens"])
            if sim >= self.threshold:
                sims.append((idx, sim, self.nodes[idx]["label"]))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[: self.top_k]

    def predict(self, tokens: Set[str]) -> str:
        neighbours = self.similar_neighbours(tokens)
        if not neighbours:
            return "ham"
        spam_votes = sum(1 for _, _, label in neighbours if label == "spam")
        return "spam" if spam_votes / len(neighbours) > 0.5 else "ham"

    def predict_with_details(self, tokens: Set[str]) -> Dict:
        neighbours = self.similar_neighbours(tokens)
        if not neighbours:
            return {"prediction": "ham", "spam_score": 0.0, "neighbours": []}
        spam_votes = sum(1 for _, _, label in neighbours if label == "spam")
        score = spam_votes / len(neighbours)
        return {"prediction": "spam" if score > 0.5 else "ham", "spam_score": score, "neighbours": neighbours}

    def evaluate(self, test_df: pd.DataFrame, token_function: Callable) -> Dict:
        actual, predicted = [], []
        for _, row in test_df.iterrows():
            actual.append(str(row["label"]).lower().strip())
            predicted.append(self.predict(token_function(row.to_dict())))

        labels = ["ham", "spam"]
        cm = confusion_matrix(actual, predicted, labels=labels)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) else 0
        return {
            "accuracy": accuracy_score(actual, predicted),
            "precision": precision_score(actual, predicted, pos_label="spam", zero_division=0),
            "recall": recall_score(actual, predicted, pos_label="spam", zero_division=0),
            "f1_score": f1_score(actual, predicted, pos_label="spam", zero_division=0),
            "specificity": specificity,
            "confusion_matrix": cm.tolist(),
        }
