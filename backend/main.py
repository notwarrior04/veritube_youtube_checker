import os, warnings

# Hide TensorFlow INFO/WARNING logs (and oneDNN notes) without affecting execution
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0=all, 1=hide INFO, 2=hide INFO+WARNING, 3=hide all [web:150]

# Optional: disable oneDNN optimizations (removes its note; may reduce CPU perf)
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # only if you want to turn off the feature itself [web:145]

# Hide only this specific TF/Keras deprecation about sparse_softmax_cross_entropy
warnings.filterwarnings(
    "ignore",
    message=r".*tf\.losses\.sparse_softmax_cross_entropy.*deprecated.*"
)  # narrow filter for that deprecation [web:152]

# Hide only the Transformers token-classification deprecation (if used indirectly)
warnings.filterwarnings("ignore", message=r".*grouped_entities.*deprecated.*")  # task-specific [web:91]

import importlib
import clickbait_detector
import re
import yt_dlp
from urllib.parse import urlparse, parse_qs

# If your clickbait detector is hot-reloaded during dev
importlib.reload(clickbait_detector)
from clickbait_detector import detect_clickbait

# These modules should already be made quiet internally (yt-dlp opts, transformers filters, etc.)
from youtube_audio import download_audio
from transcriber import transcribe_audio
from fact_checker import run_fact_check, is_music_transcript
from summarizer import summarize_transcript

def extract_title(url):
    try:
        # Silence SABR/nsig advisories from yt-dlp while extracting metadata
        ydl_opts = {"quiet": True, "no_warnings": True}  # keeps real exceptions but hides advisories [web:92]
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get('title', ''), info
    except Exception as e:
        print("❌ Failed to extract title:", e)
        return "", {}

def basic_sentence_splitter(text):
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

if __name__ == "__main__":
    print("🎥 VeriTube - YouTube Verifier")

    url = input("\n🔗 Enter YouTube video URL: ").strip()

    # Step 1: Extract title (quiet)
    title, metadata = extract_title(url)
    print(f"\n📌 Video Title: {title}")

    # Step 2: Download audio (ensure your youtube_audio.download_audio uses quiet/no_warnings/logger)
    audio_path = download_audio(url)
    if not audio_path:
        print("❌ Aborting due to audio download failure.")
        raise SystemExit(1)

    # Step 3: Transcribe audio — pass video title to allow early music detection
    print("\n📝 Transcribing...")
    language, transcript = transcribe_audio(audio_path, video_title=title)

    # Step 4: Music video check using transcript (fallback)
    if is_music_transcript(transcript, metadata=metadata):
        print("🎵 This appears to be a music video — skipping clickbait analysis and factual verification.")

        print("\n📚 Fact Check Results:")
        print("   📝 Claim: Lyrics / Music content")
        print("   🔍 Verdict: 🎵 MUSIC VIDEO")
        print("   💡 Reason: Lyrics detected — factual verification is not applicable.")

        print("\n🎼 Full Lyrics (from transcript):\n")
        for line in basic_sentence_splitter(transcript):
            print(line)
        raise SystemExit(0)

    # Step 5: Clickbait Detection
    print("🔎 Running clickbait & misleading title analysis...")
    cb_score, cb_reason, ml_score, ml_reason = detect_clickbait(title)
    print(f"\n🚨 Clickbait Score: {cb_score}%")
    print(f"🧠 Reason(s): {cb_reason or '✅ Neutral/informative'}")
    print(f"\n⚠️  Misleading Score: {ml_score}%")
    print(f"🔍 Reason(s): {ml_reason or '✅ Seems honest'}")

    # Step 6: Fact-checking
    print("\n✅ Running Fact Check...")
    fact_results = run_fact_check(transcript)

    print("\n📚 Fact Check Results:")
    for idx, res in enumerate(fact_results, 1):
        print(f"\n{idx}. 📝 Claim: {res['claim']}")
        print(f"   🔍 Verdict: {res['verdict']}")
        print(f"   💡 Reason: {res['reason']}")

    # Step 7: Summarization
    print("\n📖 Generating Summary...")
    summary = summarize_transcript(transcript, is_music=False)
    print("\n📝 Summary:\n", summary)
