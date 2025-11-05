# --- Quiet mode switches: BEFORE heavy imports ---
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide TF INFO/WARNING [web:150]
warnings.filterwarnings("ignore", message=r".*tf\.losses\.sparse_softmax_cross_entropy.*deprecated.*")  # tf-keras dep warn [web:152]
warnings.filterwarnings("ignore", message=r".*grouped_entities.*deprecated.*")  # Transformers token-classification dep [web:19]

from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
import re
import torch
from concurrent.futures import ThreadPoolExecutor
import logging

# Ensure NLTK punkt
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# -----------------------------
# Config
# -----------------------------
DEVICE_PIPE = 0 if torch.cuda.is_available() else -1
PRIMARY_MODEL = "facebook/bart-large-cnn"         # strong summarizer [web:197]
REFINER_MODEL = "facebook/bart-large-cnn"         # stylistic/length refiner [web:197]
LONG_MODEL = "allenai/led-base-16384"             # long-context fallback [web:200]
MAX_CHUNK_WORDS = 480                              # chunk target
LONG_THRESHOLD_WORDS = 2400                        # switch to LED if transcript is very long
BEAMS = 4                                          # beam size for determinism/quality
LENGTH_PENALTY = 1.0                               # conservative
NO_REPEAT_NGRAM_SIZE = 3                           # reduce redundancy

# -----------------------------
# Initialize pipelines
# -----------------------------
def _make_summarizer(model_name):
    return pipeline(
        "summarization",
        model=model_name,
        device=DEVICE_PIPE
    )  # [web:197]

try:
    summarizer = _make_summarizer(PRIMARY_MODEL)
    refiner = _make_summarizer(REFINER_MODEL)
    # Lazy LED init to avoid VRAM hit if not needed
    led_summarizer = None
except Exception as e:
    logging.critical(f"❌ Failed to load summarization pipelines: {e}")
    raise SystemExit("Failed to load summarizer/refiner.")  # [web:197]

# -----------------------------
# Helpers
# -----------------------------
def clean_transcript(text: str) -> str:
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    return text

def split_into_chunks(text: str, max_words: int = MAX_CHUNK_WORDS):
    sentences = sent_tokenize(text)
    chunks, cur, wc = [], [], 0
    for s in sentences:
        sw = len(s.split())
        if sw > max_words:
            logging.warning("Skipping very long sentence during chunking.")
            continue
        if wc + sw > max_words:
            if cur:
                chunks.append(" ".join(cur))
            cur, wc = [s], sw
        else:
            cur.append(s); wc += sw
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def remove_repetitive_sentences(text: str) -> str:
    seen, out = set(), []
    for s in sent_tokenize(text):
        norm = re.sub(r"\W+", "", s.lower())
        if norm not in seen:
            seen.add(norm)
            out.append(s)
    return " ".join(out)

def remove_weak_subjective_sentences(text: str) -> str:
    vague = [
        "we are all one","universe experiencing itself","you could say","meaning of life","the meaning of existence",
        "the great unknown","infinite wisdom","cosmic balance","oneness","eternal truth","divine truth","soul purpose",
        "inner truth","inner peace","the essence of being","collective consciousness","universal truth","we are the universe",
        "existential mystery","unfathomable mystery","pure consciousness","higher self","ultimate reality","cosmic energy",
        "vibration of the cosmos","spiritual awakening","alignment with the universe","raised vibrations","quantum energy",
        "energy shift","frequency alignment","divine purpose","soul contract","chakras in harmony","third eye opening",
        "ascension process","from a certain point of view","some say","it’s all connected","it depends on your perspective",
        "maybe the universe wants","a matter of interpretation","our limited understanding","cannot truly know","nobody really knows",
        "wibbly wobbly","timey wimey","threads of reality","timeless void","the great dance","journey of the soul","echoes of time",
        "unseen forces","a cosmic journey","symphony of creation","a divine orchestration","ripple of infinity","you are stardust",
        "we are stardust","we are made of stars","the universe chose you","everything happens for a reason","you are the observer",
        "just an illusion","a simulation maybe","nothing is real","reality is subjective","the simulation theory",
        "a glitch in the matrix","this changes everything","the truth they don’t want you to know","open your mind"
    ]
    kept = [s for s in sent_tokenize(text) if not any(p in s.lower() for p in vague)]
    return " ".join(kept)

def grammar_polish(text: str) -> str:
    text = re.sub(r"(?<!\w)([A-Z])\.\s+([A-Z])", r"\1. \2", text)
    text = re.sub(r"\b([A-Z])\.(\s)", r"\1. ", text)
    text = re.sub(r"\.\s*\.", ".", text)
    text = re.sub(r"\s+([.,;!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# -----------------------------
# Core summarization
# -----------------------------
def _run_summarizer(pipe, text: str, target_ratio=0.6, max_cap=230, beams=BEAMS):
    wc = len(text.split())
    if wc < 40:
        return text.strip()
    max_len = min(max_cap, max(70, int(wc * target_ratio)))
    min_len = max(50, int(max_len * 0.55))
    out = pipe(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        num_beams=beams,
        length_penalty=LENGTH_PENALTY,
        no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
        early_stopping=True,
    )  # deterministic, low repetition [web:197]
    return out[0]["summary_text"].strip()

def summarize_chunk(text: str) -> str:
    try:
        return _run_summarizer(summarizer, text)
    except Exception as e:
        logging.error(f"⚠️ Summarization error: {e}")
        return text

def refine_summary(text: str) -> str:
    try:
        wc = len(text.split())
        if wc < 60:
            return grammar_polish(text)
        # second pass: slightly tighter range
        wc = len(text.split())
        max_len = min(220, wc)
        min_len = max(80, int(max_len * 0.5))
        out = refiner(
            text,
            max_length=max_len,
            min_length=min_len,
            do_sample=False,
            num_beams=BEAMS,
            length_penalty=LENGTH_PENALTY,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            early_stopping=True,
        )
        refined = grammar_polish(out[0]["summary_text"].strip())
        return refined
    except Exception as e:
        logging.error(f"⚠️ Refinement error: {e}")
        return grammar_polish(text)

def summarize_long(text: str) -> str:
    # For very long documents, use LED to reduce chunk seams [web:200]
    global led_summarizer
    try:
        if led_summarizer is None:
            led_summarizer = pipeline("summarization", model=LONG_MODEL, device=DEVICE_PIPE)  # lazy init [web:200]
        return _run_summarizer(led_summarizer, text, target_ratio=0.5, max_cap=400)
    except Exception as e:
        logging.warning(f"LED fallback failed ({e}); using chunked summarization.")
        # fallback to chunked path
        chunks = split_into_chunks(text)
        with ThreadPoolExecutor(max_workers=4) as ex:
            pieces = list(ex.map(summarize_chunk, chunks))
        return " ".join(pieces)

# -----------------------------
# Public API
# -----------------------------
def summarize_transcript(transcript: str, is_music: bool = False) -> str:
    if is_music:
        return "🎵 Music video — summarization skipped."
    cleaned = clean_transcript(transcript)
    if not cleaned:
        return "⚠️ Empty or invalid transcript."

    # Route long inputs through LED first
    total_words = len(cleaned.split())
    if total_words >= LONG_THRESHOLD_WORDS:
        base = summarize_long(cleaned)  # long-context path [web:200]
    else:
        chunks = split_into_chunks(cleaned)
        if not chunks:
            return "⚠️ Could not process transcript into valid chunks."
        with ThreadPoolExecutor(max_workers=4) as executor:
            pieces = list(executor.map(summarize_chunk, chunks))
        base = " ".join(pieces)

    # Post-process and refine
    deduped = remove_repetitive_sentences(base)
    filtered = remove_weak_subjective_sentences(deduped)
    polished = grammar_polish(filtered)
    final = refine_summary(polished) if len(polished.split()) > 180 else polished
    out = final.strip()
    return out if out else "⚠️ Could not generate a valid summary."
