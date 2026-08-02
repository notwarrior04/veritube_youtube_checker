import os
import sys
import json
import re
import google.generativeai as genai

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def get_gemini_api_key():
    return os.environ.get("GEMINI_API_KEY", "").strip()

def is_gemini_available():
    return bool(get_gemini_api_key())

def configure_gemini():
    key = get_gemini_api_key()
    if key:
        genai.configure(api_key=key)

# Initial configuration if key exists
configure_gemini()

def _get_model(preferred_model="gemini-2.5-flash"):
    configure_gemini()
    generation_config = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
    }
    candidates = [preferred_model, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-flash-latest", "gemini-2.5-pro", "gemini-pro"]
    for name in candidates:
        try:
            m = genai.GenerativeModel(model_name=name, generation_config=generation_config)
            return m
        except Exception:
            continue
    return genai.GenerativeModel(model_name="gemini-2.5-flash", generation_config=generation_config)

def gemini_extract_and_rewrite_claims(transcript_text):
    """
    Uses Gemini to extract verifiable factual claims, normalize them into search-ready queries,
    and categorize subjective/philosophical statements.
    """
    if not is_gemini_available():
        return None

    prompt = f"""
You are an expert fact-checking AI assistant. Analyze the following transcript text.

Task:
1. Extract distinct, objective, verifiable factual claims made in the transcript.
2. For each claim, provide a normalized/rewritten version that makes the claim clear, self-contained, and optimized for search engine querying.
3. Identify any subjective, philosophical, or opinion-based statements and label them as "SUBJECTIVE".

Output format: Return ONLY a valid JSON object with the following schema:
{{
  "claims": [
    {{
      "original_claim": "exact text snippet from transcript",
      "rewritten_claim": "normalized search-ready factual claim assertion",
      "is_subjective": false
    }},
    {{
      "original_claim": "philosophy statement",
      "rewritten_claim": "philosophy statement",
      "is_subjective": true
    }}
  ]
}}

Transcript Text:
\"\"\"
{transcript_text[:12000]}
\"\"\"
"""
    try:
        model = _get_model()
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Clean JSON markdown fencing if present
        text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
        text = text.strip()

        data = json.loads(text)
        return data.get("claims", [])
    except Exception as e:
        print(f"⚠️ Gemini Claim Extraction Error: {e}")
        return None

def gemini_verify_claim_entailment(claim, evidence_snippets):
    """
    Uses Gemini NLI reasoning to evaluate if evidence snippets entail, contradict, or leave a claim uncertain.
    """
    if not is_gemini_available() or not evidence_snippets:
        return None

    snippets_str = "\n".join([f"- Snippet {i+1}: {s}" for i, s in enumerate(evidence_snippets)])

    prompt = f"""
You are a formal NLI (Natural Language Inference) verifier.

Claim: "{claim}"

Evidence Snippets:
{snippets_str}

Task:
Determine whether the evidence ENTAILS (supports), CONTRADICTS (refutes), or is NEUTRAL/UNCERTAIN regarding the claim.

Output format: Return ONLY a valid JSON object:
{{
  "verdict": "✅ TRUE" | "❌ FALSE" | "⚠️ UNCERTAIN",
  "reason": "Detailed concise explanation citing the evidence snippet",
  "confidence": "95%"
}}
"""
    try:
        model = _get_model()
        response = model.generate_content(prompt)
        text = response.text.strip()
        text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
        data = json.loads(text.strip())
        return data
    except Exception as e:
        print(f"⚠️ Gemini Verification Error: {e}")
        return None

def clean_plain_text_summary(text):
    if not text:
        return ""
    # Manually strip any symbols/hashes/asterisks before headings like Summary, Overview, Key Takeaways
    text = re.sub(r"^[^\w\s]*(Summary|Overview|Key Takeaways)[^\w\s]*", r"\1", text, flags=re.IGNORECASE | re.MULTILINE)
    # Strip any remaining #, *, _, `, ~, : symbols
    text = re.sub(r"[#*_`~]+", "", text)
    # Convert leading dashes/asterisks on bullet points to clean bullet char •
    text = re.sub(r"^\s*[-•]\s*", "• ", text, flags=re.MULTILINE)
    # Clean up multi-newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def gemini_summarize_transcript(transcript_text, is_music=False):
    """
    Generates a structured, high-quality summary in simple plain text without extra symbols.
    """
    if not is_gemini_available():
        return None

    if is_music:
        return "🎵 Music video lyrics detected — summarization skipped."

    prompt = f"""
Summarize the following video transcript concisely and accurately.

Requirements:
- Do NOT use markdown symbols like asterisks (** or *), hashes (#), or backticks.
- Format the output in simple, clean plain text with two sections:
  OVERVIEW
  [2-3 sentence overview paragraph]

  KEY TAKEAWAYS
  • [Bullet point 1]
  • [Bullet point 2]
  • [Bullet point 3]

Transcript:
\"\"\"
{transcript_text[:15000]}
\"\"\"
"""
    try:
        model = _get_model()
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        return clean_plain_text_summary(raw_text)
    except Exception as e:
        print(f"⚠️ Gemini Summarization Error: {e}")
        return None

def gemini_detect_clickbait_discrepancy(video_title, transcript_sample=""):
    """
    Analyzes title-transcript discrepancy and sensationalism using Gemini.
    """
    if not is_gemini_available():
        return None

    prompt = f"""
Analyze the video title for clickbait tactics, sensationalism, and misleading claims.

Video Title: "{video_title}"
Transcript Sample: \"\"\"{transcript_sample[:3000]}\"\"\"

Task:
1. Assign a Clickbait Score (0 to 100).
2. Assign a Misleading Score (0 to 100).
3. Provide reasoning for both scores.

Output format: Return ONLY a valid JSON object:
{{
  "cb_score": 40,
  "cb_reasons": ["Sensational wording used", "Exaggerated punctuation"],
  "ml_score": 10,
  "ml_reasons": ["Minor exaggeration but main premise is supported in transcript"]
}}
"""
    try:
        model = _get_model()
        response = model.generate_content(prompt)
        text = response.text.strip()
        text = re.sub(r"^```json\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^```\s*", "", text, flags=re.MULTILINE)
        data = json.loads(text.strip())
        return data
    except Exception as e:
        print(f"⚠️ Gemini Clickbait Detection Error: {e}")
        return None
