# Spam-Detection-Project
# Unified SMS and Email Spam Detection

This is a GitHub-ready Streamlit project for **SMS and Email Spam Detection**.

The project contains two modules:

1. **SMS Spam Detection**  
   Uses NLP preprocessing, TF-IDF vectorization, and classical machine learning models such as Bernoulli Naive Bayes, Multinomial Naive Bayes, and Logistic Regression.

2. **Graph-Based Email Spam Detection**  
   Uses NLP preprocessing, metadata-driven feature extraction, cosine similarity, token indexing, top-k neighbours, graph pruning concepts, and neighbourhood majority voting.

---

## Dataset Details

### Email Dataset

The email module uses `data/email_spam_dataset.csv`.

Source-wise distribution used in the project:

- SpamAssassin: 9,352 emails
  - Ham: 6,954
  - Spam: 2,398
- Enron: 7,402 emails
- Gmail: 1,000 emails

Total email dataset: **17,754 emails**

Final combined dataset target distribution:

- Ham emails: 11,401
- Spam emails: 5,353



### SMS Dataset

The SMS module uses `data/sms_spam_dataset.csv`.

- Total SMS records: 10,000
- Ham SMS: 7,500
- Spam SMS: 2,500

---

## Project Workflow

### SMS Spam Detection Workflow

1. Dataset collection
2. Text cleaning
3. Lowercasing
4. Special character removal
5. TF-IDF vectorization
6. Train-test split
7. Model training
8. Prediction
9. Evaluation using accuracy, precision, recall, F1-score, specificity
10. Streamlit deployment

### Email Graph-Based Spam Detection Workflow

1. Email dataset collection
2. Text preprocessing
3. Metadata extraction
4. Token merging and final token representation
5. Train-test split
6. Message storage and token indexing
7. Similarity graph construction
8. Cosine similarity calculation
9. Similarity thresholding
10. Top-K neighbour retention
11. Neighbourhood majority voting
12. Spam/Ham prediction
13. Performance evaluation

---

## Email Graph-Based Model Explanation

Each email is treated as a node in a graph. The model extracts tokens from:

- Email subject
- Email body
- Sender domain
- Subject length
- Recipient count
- URL count
- Attachment presence
- Time features if available

Emails sharing similar token sets are treated as candidate neighbours. Cosine similarity is calculated between the new email and candidate emails. If similarity is above the threshold, the email becomes a neighbour. The model keeps only the top-k most similar neighbours.

For prediction:

- If more than 50% of neighbours are spam, the email is predicted as spam.
- Otherwise, it is predicted as ham.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run Training Scripts

### Train SMS model

```bash
python train_sms.py
```

### Train Email Graph model

```bash
python train_email_graph.py
```

---

## Run Streamlit App

```bash
streamlit run app.py
```

---

## Folder Structure

```text
unified_spam_detection_project/
│
├── app.py
├── train_sms.py
├── train_email_graph.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── email_spam_dataset_17754.csv
│   └── sms_spam_dataset_10000.csv
│
├── models/
│
└── src/
    ├── __init__.py
    ├── preprocessing.py
    ├── sms_model.py
    └── email_graph_model.py
```

---

## Explanation

This project implements a unified spam detection system for SMS and email. For SMS spam detection, I used NLP preprocessing and TF-IDF vectorization with classical machine learning models, mainly Naive Bayes. For email spam detection, I used a graph-based approach where each email is represented as a node. Textual tokens and metadata tokens are merged to create a feature representation. Cosine similarity is used to find similar email nodes, and neighbourhood majority voting is used to classify a new email as spam or ham. This approach helps capture relational similarity among emails and can adapt to evolving spam patterns.

# Spam-Detection-Project
