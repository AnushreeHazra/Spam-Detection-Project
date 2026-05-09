import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import streamlit as st
from sklearn.model_selection import train_test_split

from src.sms_model import train_sms_model
from src.preprocessing import final_email_tokens
from src.email_graph_model import GraphSpamDetector

st.set_page_config(
    page_title="SpamShield",
    page_icon="🛡️",
    layout="wide"
)
st.markdown("""
<style>

/* MAIN APP */
.stApp {
    background: linear-gradient(135deg, #07111f 0%, #0b1026 50%, #120c2b 100%);
    color: white !important;
}

/* GLOBAL TEXT */
html, body, [class*="css"] {
    color: white !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#090f22,#111733);
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] * {
    color: white !important;
}

/* TABS */
button[data-baseweb="tab"] {
    color: white !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

/* TITLES */
.big-title {
    font-size: 42px;
    font-weight: 800;
    color: white !important;
}

/* GRADIENT TEXT */
.gradient-text {
    background: linear-gradient(90deg,#7c3aed,#38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* CARDS */
.card {
    background: rgba(255,255,255,0.08);
    padding: 24px;
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}

/* METRIC CONTAINER */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}

/* METRIC LABEL */
[data-testid="metric-container"] label {
    color: #e2e8f0 !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}

/* METRIC VALUE */
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 34px !important;
    font-weight: 900 !important;
}

/* LABELS */
label, p, span, div {
    color: white !important;
}

/* INPUT LABELS */
[data-testid="stWidgetLabel"] label {
    color: white !important;
    font-size: 17px !important;
    font-weight: 700 !important;
}

/* TEXT INPUT */
.stTextInput input {
    background-color: #0f172a !important;
    color: white !important;
    border: 2px solid #d1d5db !important;
    border-radius: 14px !important;
}

/* TEXT AREA */
.stTextArea textarea {
    background-color: #0f172a !important;
    color: white !important;
    border: 2px solid #d1d5db !important;
    border-radius: 14px !important;
}

/* NUMBER INPUT */
.stNumberInput input {
    background-color: #0f172a !important;
    color: white !important;
    border: 2px solid #d1d5db !important;
    border-radius: 14px !important;
}

/* SLIDER TEXT */
.stSlider label {
    color: white !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

.stSlider span {
    color: white !important;
}

/* SELECT BOX */
.stSelectbox div {
    color: white !important;
    background-color: #0f172a !important;
}

/* BUTTON */
.stButton>button {
    background: linear-gradient(90deg,#6d28d9,#9333ea);
    color: white !important;
    border-radius: 14px;
    border: none;
    height: 50px;
    font-size: 17px;
    font-weight: 700;
}

/* TABLE */
table {
    color: white !important;
}

/* PLACEHOLDER */
input::placeholder,
textarea::placeholder {
    color: #cbd5e1 !important;
}

</style>
""", unsafe_allow_html=True)


st.title("🛡️ Unified SMS and Email Spam Detection")
st.write(
    "A Streamlit project with SMS spam detection using TF-IDF + Naive Bayes "
    "and email spam detection using a graph-based similarity classifier."
)

EMAIL_DATA = "data/email_spam_dataset.csv"
SMS_DATA = "data/sms_spam_dataset.csv"


@st.cache_data
def load_email_data():
    df = pd.read_csv(EMAIL_DATA)

    df["label"] = df["label"].astype(str).str.lower().str.strip()

    if "to_count" not in df.columns:
        df["to_count"] = 1

    if "has_attachment" not in df.columns:
        df["has_attachment"] = 0

    return df


@st.cache_data
def load_sms_data():
    df = pd.read_csv(SMS_DATA)
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    return df


@st.cache_resource
def train_sms_cached(model_name):
    return train_sms_model(load_sms_data(), model_name=model_name)


@st.cache_resource
def train_email_graph_cached(max_rows, threshold, top_k, min_df, max_df_ratio):
    df = load_email_data().sample(
        min(max_rows, len(load_email_data())),
        random_state=42
    )

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    model = GraphSpamDetector(threshold, top_k, min_df, max_df_ratio)
    model.fit(train_df, final_email_tokens)
    metrics = model.evaluate(test_df, final_email_tokens)

    return model, metrics, train_df, test_df


def clean_for_wordcloud(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def show_wordcloud(df, label_name, title):
    df = df.copy()
    df["label"] = df["label"].astype(str).str.lower().str.strip()

    if "body" not in df.columns:
        st.error("Column 'body' not found in email dataset.")
        st.write("Available columns:", df.columns.tolist())
        return

    filtered = df[df["label"] == label_name]

    if filtered.empty:
        st.warning(f"No rows found for label: {label_name}")
        return

    text = " ".join(filtered["body"].fillna("").astype(str))
    text = clean_for_wordcloud(text)

    if len(text.strip()) < 5:
        st.warning(f"No usable text available for {label_name}")
        return

    wc = WordCloud(
        width=900,
        height=450,
        background_color="white",
        max_words=100,
        collocations=False
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)

    st.pyplot(fig)


email_df = load_email_data()
sms_df = load_sms_data()

email_spam = int((email_df["label"] == "spam").sum())
email_ham = int((email_df["label"] == "ham").sum())
sms_spam = int((sms_df["label"] == "spam").sum())
sms_ham = int((sms_df["label"] == "ham").sum())

st.markdown("""
<div class="big-title">
Welcome to <span class="gradient-text">SpamShield</span>
</div>
<p style="color:#b8c0d9;font-size:18px;">
Detect and analyze spam in SMS and Emails using Machine Learning and Graph-Based Classification
</p>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="card">
        <div class="metric-label">📧 Total Emails</div>
        <div class="metric-value">{len(email_df):,}</div>
        <p>All email records</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <div class="metric-label">🚨 Email Spam</div>
        <div class="metric-value spam">{email_spam:,}</div>
        <p>Spam email records</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <div class="metric-label">💬 Total SMS</div>
        <div class="metric-value">{len(sms_df):,}</div>
        <p>All SMS records</p>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="card">
        <div class="metric-label">⚠️ SMS Spam</div>
        <div class="metric-value spam">{sms_spam:,}</div>
        <p>Spam SMS records</p>
    </div>
    """, unsafe_allow_html=True)
with st.sidebar:
    st.markdown("""
    <h1 style="color:white;">🛡️ Spam<span style="color:#7c3aed;">Shield</span></h1>
    <p style="color:#b8c0d9;">Smart Spam Detection</p>
    <hr>
    """, unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Dataset Overview",
        "📱 SMS Detection",
        "📧 Email Graph Detection",
        "ℹ️ Project Workflow"
    ]
)


with tab1:
    st.subheader("Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Email Records", len(email_df))
    c2.metric("SMS Records", len(sms_df))
    c3.metric("Email Spam", int((email_df["label"] == "spam").sum()))
    c4.metric("SMS Spam", int((sms_df["label"] == "spam").sum()))

    st.write("Email source distribution")
    if "source" in email_df.columns:
        st.dataframe(
            email_df["source"]
            .value_counts()
            .reset_index()
            .rename(columns={"source": "Source", "count": "Records"})
        )

    st.write("Email label distribution")
    st.dataframe(email_df["label"].value_counts().reset_index())

    st.write("SMS label distribution")
    st.dataframe(sms_df["label"].value_counts().reset_index())

    st.subheader("Email Word Cloud Visualization")

    col1, col2 = st.columns(2)

    with col1:
        show_wordcloud(email_df, "spam", "Spam Email Word Cloud")

    with col2:
        show_wordcloud(email_df, "ham", "Ham Email Word Cloud")


with tab2:
    st.subheader("SMS Spam Detection")

    model_name = st.selectbox(
        "Choose SMS Model",
        ["BernoulliNB", "MultinomialNB", "LogisticRegression"]
    )

    sms_result = train_sms_cached(model_name)
    metrics = sms_result["metrics"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    m2.metric("Precision", f"{metrics['precision']:.3f}")
    m3.metric("Recall", f"{metrics['recall']:.3f}")
    m4.metric("F1", f"{metrics['f1_score']:.3f}")
    m5.metric("Specificity", f"{metrics['specificity']:.3f}")

    sms_text = st.text_area(
        "Enter SMS Message",
        "Congratulations! You won a free cash prize. Click now to claim."
    )

    if st.button("Predict SMS"):
        pred = sms_result["pipeline"].predict([sms_text])[0]

        if pred == "spam":
            st.error("Prediction: SPAM")
        else:
            st.success("Prediction: HAM")

with tab3:
    st.subheader("Graph-Based Email Spam Detection")
    st.write(
        "This model converts subject, body and metadata into tokens, "
        "finds similar email nodes using cosine similarity, and predicts "
        "using majority voting."
    )

    with st.sidebar:
        st.header("Email Graph Hyperparameters")

        max_rows = st.slider(
            "Training rows for Streamlit demo",
            1000,
            min(17754, len(email_df)),
            min(5000, len(email_df)),
            step=500
        )

        threshold = st.slider(
            "Similarity Threshold",
            0.05,
            0.80,
            0.20,
            step=0.05
        )

        top_k = st.slider("Top-K Neighbours", 1, 20, 5)
        min_df = st.slider("min_df", 1, 10, 2)

        max_df_ratio = st.slider(
            "max_df_ratio",
            0.50,
            1.00,
            0.80,
            step=0.05
        )

    model, graph_metrics, train_df, test_df = train_email_graph_cached(
        max_rows,
        threshold,
        top_k,
        min_df,
        max_df_ratio
    )

    g1, g2, g3, g4, g5 = st.columns(5)
    g1.metric("Accuracy", f"{graph_metrics['accuracy']:.3f}")
    g2.metric("Precision", f"{graph_metrics['precision']:.3f}")
    g3.metric("Recall", f"{graph_metrics['recall']:.3f}")
    g4.metric("F1", f"{graph_metrics['f1_score']:.3f}")
    g5.metric("Specificity", f"{graph_metrics['specificity']:.3f}")

    subject = st.text_input("Email Subject", "Urgent account verification")

    body = st.text_area(
        "Email Body",
        "Your account will be blocked. Verify your details immediately by clicking this link."
    )

    from_email = st.text_input("Sender Email", "security@fakebank.net")

    to_count = st.number_input(
        "Recipient Count",
        min_value=1,
        value=1
    )

    has_attachment = st.selectbox("Has Attachment?", [0, 1])

    if st.button("Predict Email"):
        row = {
            "subject": subject,
            "body": body,
            "from_email": from_email,
            "to_count": to_count,
            "has_attachment": has_attachment
        }

        details = model.predict_with_details(final_email_tokens(row))

        if details["prediction"] == "spam":
            st.error(
                f"Prediction: SPAM | Spam Score: {details['spam_score']:.2f}"
            )
        else:
            st.success(
                f"Prediction: HAM | Spam Score: {details['spam_score']:.2f}"
            )

        st.write("Top similar neighbours:")
        st.dataframe(
            pd.DataFrame(
                details["neighbours"],
                columns=["node_id", "similarity", "label"]
            )
        )


with tab4:
    st.subheader("Project Workflow")

    st.markdown("""
    **SMS Module**: Dataset → Cleaning → TF-IDF Vectorization → Train-Test Split → Naive Bayes / ML Model → Prediction → Evaluation.

    **Email Module**: Dataset Collection → Text Preprocessing → Metadata Extraction → Token Merging → Train-Test Split → Message Storage & Token Indexing → Similarity Graph Construction → Thresholding → Top-K Neighbour Selection → Majority Voting → Spam/Ham Prediction → Evaluation.

    **Email Graph Features**:
    - Subject + body textual tokens
    - Sender domain token
    - Subject length bin
    - Recipient count bin
    - URL count token
    - Attachment indicator
    - Time-based tokens if date is available
    """)