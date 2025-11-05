import os, warnings

# Hide TF INFO/WARNING backend logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 2 hides INFO and WARNING; keeps errors visible [web:150]

# Suppress only this deprecation emitted by tf_keras losses
warnings.filterwarnings(
    "ignore",
    message=r".*tf\.losses\.sparse_softmax_cross_entropy.*deprecated.*"
)  # precise regex for that warning line [web:152]

import re
import string
import unidecode
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources once
nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# -------------------------
# Preprocessing
# -------------------------
def preprocess(text):
    text = re.sub(r"http\S+", "", text.lower())
    text = unidecode.unidecode(text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    return [w for w in tokens if w not in stop_words]

# -------------------------
# TF-IDF buzzword scoring
# -------------------------
def tfidf_score(text, buzzwords):
    vectorizer = TfidfVectorizer(vocabulary=buzzwords, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform([text])
    return tfidf_matrix.sum()

# -------------------------
# Main Clickbait Detector
# -------------------------
def detect_clickbait(title):
    score = 0
    mislead_score = 0
    reasons = []
    mislead_reasons = []

    title_clean = title.strip()
    normalized_title = unidecode.unidecode(title_clean.lower())

    tokens = preprocess(title_clean)
    pos_tags = pos_tag(tokens)

    # 🔥 Buzzwords & Phrases
    buzzwords = [
        "you won't believe", "shocking", "amazing", "unbelievable", "secret",
        "revealed", "top", "worst", "epic", "must see", "crazy", "miracle",
        "everything changes", "never seen before", "change your life", "watch this",
        "before you die", "blow your mind", "can't believe", "literally", "gone viral",
        "will make you cry"
    ]
    if any(phrase in normalized_title for phrase in buzzwords):
        score += 40
        reasons.append("Buzzwords or sensational phrases")

    if tfidf_score(normalized_title, buzzwords) > 1.5:
        score += 20
        reasons.append("TF-IDF score indicates clickbait phrasing")

    # POS Patterns
    pos_counts = {"JJ": 0, "RB": 0, "UH": 0}
    for _, tag in pos_tags:
        if tag in pos_counts:
            pos_counts[tag] += 1

    if pos_counts["JJ"] >= 3:
        score += 10
        reasons.append("Too many adjectives")

    if pos_counts["UH"] >= 1:
        score += 10
        reasons.append("Interjection/exclamatory structure")

    # Formatting Heuristics
    if any(w.isupper() and len(w) > 3 for w in title_clean.split()):
        score += 10
        reasons.append("ALL CAPS words used")

    if title.count("!") + title.count("?") >= 2:
        score += 10
        reasons.append("Excessive punctuation (!/?)")

    if len(title.split()) < 5:
        score += 10
        reasons.append("Very short/vague title")
        mislead_score += 5
        mislead_reasons.append("Too vague to understand intent")

    # Clickbait Pattern Matching
    clickbait_starts = [
        "you won’t believe", "this is why", "this is what", "is this the",
        "why you should", "watch how", "never do this", "before and after"
    ]
    if any(normalized_title.startswith(p) for p in clickbait_starts):
        score += 10
        reasons.append("Typical clickbait phrasing")

    if re.search(r'\b(top\s*\d+|\d+\s+(ways|things|facts|secrets))', normalized_title):
        score += 10
        reasons.append("Listicle format")

    # 🚨 Misleading Patterns
    misleading_phrases = [
        "cure for cancer", "billionaire overnight", "earn $1000 a day", "guaranteed results",
        "miracle", "every doctor hates", "proof that", "never get sick", "stop aging",
        "ai will destroy", "ai took over", "government hiding", "earth is flat"
    ]
    if any(phrase in normalized_title for phrase in misleading_phrases):
        mislead_score += 40
        mislead_reasons.append("False/exaggerated health/finance/science claim")

    if any(phrase in normalized_title for phrase in ["save the world", "change humanity", "live forever"]):
        mislead_score += 20
        mislead_reasons.append("Overpromising extreme result")

    # Cap Scores
    clickbait_score = min(score, 100)
    misleading_score = min(mislead_score, 100)

    return clickbait_score, reasons, misleading_score, mislead_reasons
