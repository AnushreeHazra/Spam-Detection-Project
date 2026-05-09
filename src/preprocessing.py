"""Text and metadata preprocessing utilities for SMS and email spam detection."""

import re
from datetime import datetime
from typing import Any, Dict, Iterable, Set
from bs4 import BeautifulSoup

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    STOP_WORDS = set(stopwords.words("english"))
    LEMMATIZER = WordNetLemmatizer()
except Exception:
    STOP_WORDS = {
        "the", "is", "and", "or", "a", "an", "to", "for", "in", "on", "of",
        "this", "that", "you", "your", "we", "our", "are", "be", "with", "as"
    }
    LEMMATIZER = None


def clean_text(text: Any) -> str:
    """Clean raw text using lowercasing, HTML stripping, URL/email removal and whitespace normalization."""
    text = "" if text is None else str(text)
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+", " emailtoken ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_text(text: Any) -> Set[str]:
    """Tokenize, remove stopwords and lemmatize text."""
    cleaned = clean_text(text)
    tokens = []
    for tok in cleaned.split():
        if tok in STOP_WORDS or len(tok) <= 1:
            continue
        if LEMMATIZER is not None:
            tok = LEMMATIZER.lemmatize(tok)
        tokens.append(tok)
    return set(tokens)


def count_urls(text: Any) -> int:
    text = "" if text is None else str(text)
    return len(re.findall(r"http\S+|www\S+", text.lower()))


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def metadata_tokens(row: Dict[str, Any]) -> Set[str]:
    """Extract metadata-derived tokens from an email row."""
    tokens = set()

    sender = str(row.get("from_email", "") or row.get("from", "") or "")
    domain = sender.split("@")[-1].lower() if "@" in sender else "unknown"
    tokens.add(f"dom_{domain}")

    subject = str(row.get("subject", "") or "")
    subj_len = len(subject)
    if subj_len <= 20:
        tokens.add("subj_len_short")
    elif subj_len <= 50:
        tokens.add("subj_len_medium")
    else:
        tokens.add("subj_len_long")

    to_count = _to_int(row.get("to_count", 1), 1)
    if to_count <= 1:
        tokens.add("to_cnt_1")
    elif to_count <= 3:
        tokens.add("to_cnt_2_3")
    else:
        tokens.add("to_cnt_many")

    body = str(row.get("body", "") or row.get("message", "") or "")
    url_count = count_urls(body)
    if url_count == 0:
        tokens.add("url_cnt_0")
    elif url_count == 1:
        tokens.add("url_cnt_1")
    else:
        tokens.add("url_cnt_many")

    has_attachment = _to_int(row.get("has_attachment", 0), 0)
    tokens.add(f"attach_{1 if has_attachment else 0}")

    date_value = row.get("date", None)
    if date_value:
        try:
            dt = datetime.fromisoformat(str(date_value).replace("Z", ""))
            tokens.add(f"hour_{dt.hour}")
            tokens.add(f"weekday_{dt.weekday()}")
        except Exception:
            pass

    return tokens


def final_email_tokens(row: Dict[str, Any]) -> Set[str]:
    """Create final unified email token set: subject + body + metadata tokens."""
    subject = row.get("subject", "")
    body = row.get("body", "") or row.get("message", "")
    tokens = tokenize_text(subject).union(tokenize_text(body))
    tokens.update(metadata_tokens(row))
    return tokens


# Alias used in notebook/tutorial code
final_tokens = final_email_tokens
