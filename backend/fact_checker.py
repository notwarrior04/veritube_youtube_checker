import os, warnings

# Hide TF INFO/WARNING backend logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 2 hides INFO and WARNING; keeps errors visible [web:150]

# Suppress only this deprecation emitted by tf_keras losses
warnings.filterwarnings(
    "ignore",
    message=r".*tf\.losses\.sparse_softmax_cross_entropy.*deprecated.*"
)  # precise regex for that warning line [web:152]

import json
import re
import spacy
import requests
import wikipedia
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from claim_rewriter import rewrite_claim

# -------------------------------
# Load NLP and Models
# -------------------------------
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")

# -------------------------------
# Text Helpers
# -------------------------------

def normalize(text):
    return re.sub(r'\W+', ' ', text).lower().strip()

def semantic_similarity(a, b):
    emb1 = embedder.encode(a, convert_to_tensor=True)
    emb2 = embedder.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2)[0][0])

def entailment_check(claim, context):
    result = nli_model(f"{claim} </s> {context}", truncation=True)[0]
    label, score = result['label'], result['score']
    if label == "ENTAILMENT" and score > 0.8:
        return "✅ TRUE", f"Entailment match from NLI (score: {round(score * 100)}%)"
    elif label == "CONTRADICTION" and score > 0.8:
        return "❌ FALSE", f"Contradiction found in NLI (score: {round(score * 100)}%)"
    return "⚠️ UNCERTAIN", "No strong entailment or contradiction found"

def extract_years(text):
    return [ent.text for ent in nlp(text).ents if ent.label_ == "DATE"]

# -------------------------------
# Claim Extraction
# -------------------------------

def extract_claims(text):
    doc = nlp(text)
    claims = []
    for sent in doc.sents:
        s = sent.text.strip()
        if 5 <= len(s.split()) <= 30 and s[-1] not in "?!" and any(
            word.lemma_ in ["be", "have", "cause", "lead", "show", "find", "report", "state"]
            for word in sent
        ):
            claims.append(s)
    return claims

# -------------------------------
# Load Local Claim DB
# -------------------------------

def load_claim_database(path="claim_database.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("claims", [])
    except Exception as e:
        print("")
        return []

# -------------------------------
# Local DB Checker
# -------------------------------

def check_local_database(claim, claim_db):
    for entry in claim_db:
        known = entry["claim"]
        score = semantic_similarity(claim, known)
        if score >= 0.8:
            return {
                "verdict": "✅ TRUE" if entry["verdict"].lower() == "true" else "❌ FALSE",
                "reason": f"Matched with local database (score: {round(score * 100)}%) - Source: {entry['source']}",
                "confidence": f"{round(score * 100)}%"
            }
    return None

# -------------------------------
# Enhanced Wikipedia Checker
# -------------------------------

def wikipedia_check(claim):
    try:
        search_results = wikipedia.search(claim)
        if not search_results:
            return None
        for result in search_results[:3]:
            try:
                page = wikipedia.page(result, auto_suggest=False)
                content = page.content[:3000]
                sim = semantic_similarity(claim, content)
                if sim >= 0.7:
                    return {
                        "verdict": "✅ TRUE",
                        "reason": f"Matched from Wikipedia content (similarity: {round(sim * 100)}%)",
                        "confidence": f"{round(sim * 100)}%"
                    }
                else:
                    verdict, reason = entailment_check(claim, content)
                    if verdict != "⚠️ UNCERTAIN":
                        return {
                            "verdict": verdict,
                            "reason": f"{reason} — Wikipedia content used",
                            "confidence": f"{round(sim * 100)}%"
                        }
            except Exception:
                continue
        return None
    except Exception:
        return None

# -------------------------------
# Serper.dev Web Search Checker
# -------------------------------

def check_with_serper(claim):
    api_key = "3297044fa25f72720ad3f6aa81bf1d4dcee76d4c"
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    data = {"q": claim}

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        results = response.json().get("organic", [])

        best_score = 0
        best_snippet = ""
        for result in results:
            snippet = result.get("snippet", "")
            if not snippet:
                continue

            sim = semantic_similarity(claim, snippet)
            if sim >= 0.7:
                return {
                    "verdict": "✅ TRUE",
                    "reason": f"Matched from Serper.dev (similarity: {round(sim * 100)}%) — \"{snippet[:120]}...\"",
                    "confidence": f"{round(sim * 100)}%"
                }
            elif sim >= 0.5:
                verdict, reason = entailment_check(claim, snippet)
                if verdict != "⚠️ UNCERTAIN":
                    return {
                        "verdict": verdict,
                        "reason": f"{reason} — Serper: \"{snippet[:120]}...\"",
                        "confidence": f"{round(sim * 100)}%"
                    }
            if sim > best_score:
                best_score = sim
                best_snippet = snippet

        if best_score >= 0.4:
            verdict, reason = entailment_check(claim, best_snippet)
            if verdict != "⚠️ UNCERTAIN":
                return {
                    "verdict": verdict,
                    "reason": f"{reason} — Serper.dev (low sim: {round(best_score * 100)}%)",
                    "confidence": f"{round(best_score * 100)}%"
                }

        return {
            "verdict": "⚠️ UNCERTAIN",
            "reason": f"No strong match found. Best evidence snippet: \"{best_snippet[:120]}...\"",
            "confidence": f"{round(best_score * 100)}%"
        }

    except Exception as e:
        return {
            "verdict": "⚠️ UNCERTAIN",
            "reason": f"Serper.dev search failed: {str(e)}",
            "confidence": "0%"
        }

# -------------------------------
# Subjectivity Filter
# -------------------------------

def is_subjective_or_philosophical(text):
    philosophical_keywords = [
        "meaning of life", "why we exist", "reason for existence", "consciousness", "awareness",
        "what is reality", "what is time", "what is space", "what are we", "who are we", 
        "what am i", "why are we here", "reality itself", "existence", "subjective", 
        "objective truth", "absolute truth", "relative truth",
        "soul", "spirit", "spiritual", "karma", "fate", "destiny", "cosmic plan",
        "divine", "energy field", "vibrations", "higher plane", "transcend", "awakening",
        "enlightenment", "infinite", "eternal", "beyond science", "alternate reality",
        "parallel universe", "multiverse", "unseen dimensions", "illusion", "maya",
        "dead stars", "made of stars", "stardust", "we are the universe", "universe's way",
        "we are one", "interconnected", "oneness", "cosmic purpose", "part of the universe",
        "connected to everything", "everything is connected",
        "i believe", "i think", "i feel", "some say", "you could say", "one might say", 
        "perhaps", "maybe", "could be", "it is said", "it feels like", "it's as if",
        "in my view", "philosophically speaking", "from a spiritual perspective"
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in philosophical_keywords)

# -------------------------------
# Music Transcript Detector
# -------------------------------

def is_music_transcript(text, metadata=None):
    if metadata:
        category = metadata.get('categories', [])
        channel = metadata.get('channel', '').lower()
        if any("music" in cat.lower() for cat in category):
            return True
        if any(keyword in channel for keyword in ["vevo", "official audio", "music"]):
            return True
    lines = [line.strip() for line in text.lower().splitlines() if line.strip()]
    if not lines or len(lines) < 5:
        return False
    short_lines = [line for line in lines if len(line.split()) <= 5]
    repetitive_short_lines = sum(short_lines.count(line) for line in set(short_lines) if short_lines.count(line) > 1)
    total_lines = len(lines)
    unique_lines = len(set(lines))
    repetition_ratio = 1 - (unique_lines / total_lines)
    endings = [line.split()[-1] for line in lines if len(line.split()) >= 2]
    rhyme_like = sum(endings.count(word) > 1 for word in set(endings))
    music_keywords = ['chorus', 'verse', 'repeat', 'hook', 'vocals', 'lyrics', 'sing']
    keyword_hits = sum(1 for kw in music_keywords if kw in text.lower())
    if repetition_ratio > 0.45 and repetitive_short_lines >= 5:
        return True
    if rhyme_like >= 5:
        return True
    if keyword_hits >= 2:
        return True
    return False

# -------------------------------
# Main Fact Check Orchestrator
# -------------------------------

def run_fact_check(transcript_text):
    text = transcript_text.strip()

    if is_music_transcript(text):
        print("🎵 This appears to be a music video — factual verification is not applicable.")
        return [{
            "claim": "Lyrics / Music content",
            "verdict": "🎵 MUSIC VIDEO",
            "reason": "Lyrics detected — factual verification is not applicable.",
            "confidence": "N/A"
        }]

    claims = extract_claims(text)
    claim_db = load_claim_database()
    results = []

    for claim in claims:
        rewritten_claim = rewrite_claim(claim)
        print(f"🔁 Rewritten: \"{claim}\" → \"{rewritten_claim}\"")

        if is_subjective_or_philosophical(rewritten_claim):
            results.append({
                "claim": claim,
                "verdict": "🌀 SUBJECTIVE",
                "reason": "Philosophical/subjective — not verifiable by factual search",
                "confidence": "N/A"
            })
            continue

        local_result = check_local_database(rewritten_claim, claim_db)
        if local_result:
            results.append({"claim": claim, **local_result})
            continue

        web_result = check_with_serper(rewritten_claim)
        if web_result["verdict"] != "⚠️ UNCERTAIN":
            results.append({"claim": claim, **web_result})
            continue

        wiki_result = wikipedia_check(rewritten_claim)
        if wiki_result:
            results.append({"claim": claim, **wiki_result})
        else:
            results.append({"claim": claim, **web_result})

    return results
