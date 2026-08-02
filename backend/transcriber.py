import os, warnings

# Hide TF INFO/WARNING backend logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 2 hides INFO and WARNING; keeps errors visible [web:150]

# Suppress only this deprecation emitted by tf_keras losses
warnings.filterwarnings(
    "ignore",
    message=r".*tf\.losses\.sparse_softmax_cross_entropy.*deprecated.*"
)  # precise regex for that warning line [web:152]

import os
import re
import warnings
import torch
import whisper
import requests
from bs4 import BeautifulSoup
from deepmultilingualpunctuation import PunctuationModel

# Silence ONLY the grouped_entities deprecation coming from TokenClassificationPipeline
warnings.filterwarnings(
    "ignore",
    message=r".*grouped_entities.*deprecated.*"  # narrow match so other warnings still show
)

SERPER_API_KEY = "3297044fa25f72720ad3f6aa81bf1d4dcee76d4c"

# Select device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU = DEVICE.startswith("cuda")

def is_music_video(title):
    music_keywords = [
        "official video", "official music video", "lyrics", "audio",
        "remastered", "song", "music", "cover", "instrumental",
        "remix", "live", "performance", "karaoke", "single", "track"
    ]
    lowered_title = title.lower()

    for kw in music_keywords:
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, lowered_title):
            return True

    music_title_patterns = [
        r".+\sby\s.+",
        r"🎵|🎶|♫|♬",
    ]
    for pattern in music_title_patterns:
        if re.search(pattern, lowered_title):
            return True

    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": title, "gl": "us", "hl": "en"}
        )
        if response.status_code == 200:
            results = response.json()
            for item in results.get("organic", []):
                snippet = item.get("snippet", "").lower()
                title_snippet = item.get("title", "").lower()

                snippet_keywords = ["lyrics", "album", "track", "released", "singer", "music", "song"]
                title_keywords = ["lyrics", "song", "track", "official audio"]

                for kw in snippet_keywords:
                    if re.search(r"\b" + re.escape(kw) + r"\b", snippet):
                        return True

                for kw in title_keywords:
                    if re.search(r"\b" + re.escape(kw) + r"\b", title_snippet):
                        return True
    except Exception as e:
        print(f"⚠️ Error in music detection: {e}")

    return False

def clean_lyrics_text(raw_text):
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

    junk_keywords = [
        "Translations", "Contributors", "Read More", "About", "Lyrics", "Embed",
        "Genius", "Album", "Track", "Produced by", "Written by", "Release Date",
        "Türkçe", "Português", "Deutsch", "Español", "Français", "Русский", "Українська",
        "©", "All Rights Reserved"
    ]
    for keyword in junk_keywords:
        text = re.sub(rf"(?im)^{re.escape(keyword)}.*?$", "", text)

    text = re.sub(r"(?is)^.*?(?=\n\[|(?<!\w)\n\w.*\n)", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\n +", "\n", text)

    section_match = re.search(r"(?im)^\[?(verse|intro|chorus|hook|bridge|outro)[^\]]*\]?", text)
    if section_match:
        text = text[section_match.start():]
    else:
        fallback_match = re.search(r"(?m)^(?:[A-Za-z0-9']{2,} ){3,}[^\n]+", text)
        if fallback_match:
            text = text[fallback_match.start():]

    return text.strip()

def fetch_lyrics_from_serper(song_title, api_key):
    query = f"{song_title} lyrics site:genius.com"
    headers = {"X-API-KEY": api_key}
    url = "https://google.serper.dev/search"
    payload = {"q": query}

    try:
        print("🔍 Searching for lyrics...")
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()

        for result in data.get("organic", []):
            link = result.get("link", "")
            if "genius.com" in link:
                print(f"🎯 Found Genius page: {link}")
                lyrics_page = requests.get(link)
                soup = BeautifulSoup(lyrics_page.text, "html.parser")
                lyrics_divs = soup.find_all("div", attrs={"data-lyrics-container": "true"})
                raw_lyrics = "\n".join(div.get_text(separator="\n").strip() for div in lyrics_divs)
                clean_lyrics = clean_lyrics_text(raw_lyrics)
                return clean_lyrics if clean_lyrics else None

        print("❌ Genius lyrics not found.")

    except Exception as e:
        print(f"⚠️ Error while fetching lyrics: {e}")

    return None

def transcribe_audio(audio_path="audio.mp3", video_title=""):
    is_music = is_music_video(video_title)

    if is_music:
        print("🔍 Music video detected — searching for lyrics...")
        lyrics = fetch_lyrics_from_serper(video_title, SERPER_API_KEY)
        if lyrics:
            print("🎵 Lyrics fetched successfully!")
            with open("transcription.txt", "w", encoding="utf-8") as f:
                f.write(lyrics)
            return "en", lyrics
        else:
            print("❌ Lyrics not found. ⚠️ Falling back to Whisper transcription...")
    else:
        print("")

_whisper_model = None

def get_whisper_model(model_name="medium"):
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model, DEVICE, GPU

    target_device = DEVICE
    is_gpu = GPU

    if is_gpu:
        try:
            print(f"🌀 Loading Whisper model [{model_name}] on {target_device}...")
            torch.cuda.empty_cache()
            _whisper_model = whisper.load_model(model_name, device=target_device)
            return _whisper_model, target_device, is_gpu
        except Exception as e:
            print(f"⚠️ Failed to load Whisper [{model_name}] on GPU: {e}")
            torch.cuda.empty_cache()
            if model_name != "small":
                try:
                    print("🔄 Retrying with Whisper model [small] on GPU...")
                    _whisper_model = whisper.load_model("small", device=target_device)
                    return _whisper_model, target_device, is_gpu
                except Exception as e2:
                    print(f"⚠️ Failed to load Whisper [small] on GPU: {e2}")
                    torch.cuda.empty_cache()

            print("🔄 Falling back to CPU for Whisper transcription...")
            target_device = "cpu"
            is_gpu = False

    try:
        _whisper_model = whisper.load_model("small", device=target_device)
    except Exception:
        _whisper_model = whisper.load_model("base", device=target_device)

    return _whisper_model, target_device, is_gpu

def transcribe_audio(audio_path="audio.mp3", video_title=""):
    is_music = is_music_video(video_title)

    if is_music:
        print("🔍 Music video detected — searching for lyrics...")
        lyrics = fetch_lyrics_from_serper(video_title, SERPER_API_KEY)
        if lyrics:
            print("🎵 Lyrics fetched successfully!")
            with open("transcription.txt", "w", encoding="utf-8") as f:
                f.write(lyrics)
            return "en", lyrics
        else:
            print("❌ Lyrics not found. ⚠️ Falling back to Whisper transcription...")
    else:
        print("")

    # Load Whisper lazily with memory fallback
    model, use_device, is_gpu = get_whisper_model("medium")

    # Transcribe; fp16 only on GPU
    print(f"🎧 Transcribing audio on {use_device}...")
    result = model.transcribe(
        audio_path,
        task="transcribe",
        language=None,
        fp16=is_gpu  # True if GPU, False on CPU
    )
    text = result["text"]
    lang = result["language"]
    print(f"📢 Detected Language: {lang}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Punctuation model to GPU if available
    print("⏳ Restoring punctuation...")
    try:
        punct_model = PunctuationModel()  # internally uses a TokenClassification pipeline [web:138][web:19]
        if is_gpu and hasattr(punct_model, "model"):
            punct_model.model.to(torch.device("cuda"))
        punctuated = punct_model.restore_punctuation(text)
    except Exception as e:
        print(f"ℹ️ Punctuation model processing notice: {e}. Using raw text.")
        punctuated = text

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("✅ Transcription done.")
    with open("transcription.txt", "w", encoding="utf-8") as f:
        f.write(punctuated)

    return lang, punctuated

