import os, warnings

# Hide TF INFO/WARNING backend logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 2 hides INFO and WARNING; keeps errors visible [web:150]

# Suppress only this deprecation emitted by tf_keras losses
warnings.filterwarnings(
    "ignore",
    message=r".*tf\.losses\.sparse_softmax_cross_entropy.*deprecated.*"
)  # precise regex for that warning line [web:152]

import torch
from transformers import pipeline

_music_classifier = None

def get_music_classifier():
    global _music_classifier
    if _music_classifier is not None:
        return _music_classifier
    try:
        device = 0 if torch.cuda.is_available() else -1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _music_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
    except Exception as e:
        print(f"⚠️ Music classifier GPU load failed ({e}), falling back to CPU...")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _music_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    return _music_classifier


def is_music_transcript(transcript):
    """
    Uses hybrid logic + zero-shot learning to accurately classify music transcripts.
    Prevents misclassifying educational videos as music videos.
    """

    # Normalize and clean transcript
    transcript = transcript.lower()
    lines = [line.strip() for line in transcript.splitlines() if line.strip()]

    if not lines:
        return False

    # --- 1. Heuristic check: short & repeated lines ---
    short_lines = [line for line in lines if len(line.split()) <= 5]
    unique_lines = set(lines)
    repetition_ratio = 1 - (len(unique_lines) / len(lines))

    if len(short_lines) >= 10 and repetition_ratio > 0.45:
        return True  # Strong lyrics pattern

    # --- 2. AI check: Zero-shot classification ---
    classifier = get_music_classifier()
    result = classifier(
        text_sample,
        text_sample,
        candidate_labels=[
            "music lyrics",
            "song",
            "rap",
            "educational video",
            "documentary",
            "podcast",
            "news",
            "speech",
            "storytelling"
        ],
        multi_label=True
    )

    labels = dict(zip(result['labels'], result['scores']))

    # --- 3. Final decision logic ---
    if labels.get("music lyrics", 0) > 0.7 or labels.get("song", 0) > 0.7:
        return True

    if labels.get("educational video", 0) > 0.6 or labels.get("documentary", 0) > 0.6:
        return False

    # Fallback: assume not music if no strong match
    return False
