import os, warnings

# Hide TF INFO/WARNING backend logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 2 hides INFO and WARNING; keeps errors visible [web:150]

# Suppress only this deprecation emitted by tf_keras losses
warnings.filterwarnings(
    "ignore",
    message=r".*tf\.losses\.sparse_softmax_cross_entropy.*deprecated.*"
)  # precise regex for that warning line [web:152]

import wikipedia
from difflib import SequenceMatcher
import re

def clean_text(text):
    return re.sub(r'\W+', ' ', text).lower()

def similarity(a, b):
    return SequenceMatcher(None, clean_text(a), clean_text(b)).ratio()

def search_wikipedia(query, threshold=0.65):
    try:
        search_results = wikipedia.search(query)
        if not search_results:
            return None, None, 0.0  # summary, title, score

        best_match_title = None
        best_score = 0.0
        best_summary = ""

        for title in search_results:
            try:
                summary = wikipedia.summary(title, sentences=3)
                # Compare BOTH title similarity and summary similarity
                title_score = similarity(query, title)
                summary_score = similarity(query, summary)
                score = max(title_score, summary_score)

                if score > best_score:
                    best_score = score
                    best_match_title = title
                    best_summary = summary
            except Exception:
                continue

        if best_score >= threshold:
            return best_summary, best_match_title, best_score
        else:
            return None, None, best_score
    except Exception as e:
        print(f"❌ Wikipedia lookup failed: {e}")
        return None, None, 0.0

if __name__ == "__main__":
    q = input("🔍 Enter a claim: ")
    summary, title, score = search_wikipedia(q)
    if summary:
        print(f"✅ Matched Article: {title}\n📄 Summary:\n{summary}\n🔢 Similarity: {round(score * 100, 2)}%")
    else:
        print("❌ No reliable Wikipedia match found.")
