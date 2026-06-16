import sys
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import os, uuid, json, pathlib
from urllib.parse import urlparse, parse_qs
from flask import Flask, render_template, request, redirect, url_for, flash, send_file

# Your modules
from youtube_audio import download_audio
from transcriber import transcribe_audio
from fact_checker import run_fact_check, is_music_transcript
from summarizer import summarize_transcript
from clickbait_detector import detect_clickbait
import yt_dlp

app = Flask(__name__, instance_relative_config=True)
app.config['SECRET_KEY'] = 'dev'  # for flash messages [web:85]
app.config['JOBS_DIR'] = os.path.join(app.instance_path, 'jobs')
pathlib.Path(app.config['JOBS_DIR']).mkdir(parents=True, exist_ok=True)

def extract_title_and_meta(url: str):
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": {"node": {}}
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return info.get('title', ''), info  # title + full metadata [web:3]

def is_valid_youtube_url(u: str) -> tuple[bool, str]:
    try:
        u = u.strip()
        if not u:
            return False, u
        if not u.startswith(("http://", "https://")):
            u = "https://" + u
        p = urlparse(u)
        host = (p.netloc or "").lower()
        allowed = {
            "youtube.com", "www.youtube.com", "m.youtube.com",
            "music.youtube.com", "youtu.be"
        }
        if host not in allowed and not host.endswith(".youtube.com"):
            return False, u
        # Permit video forms
        if host == "youtu.be":
            return (bool(p.path.strip("/")), u)
        if p.path.startswith("/watch"):
            q = parse_qs(p.query)
            return ("v" in q and len(q["v"][0]) > 5, u)
        if p.path.startswith("/shorts/") or p.path.startswith("/live/"):
            return True, u
        return False, u
    except Exception:
        return False, u

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # URL form [web:85]

@app.route("/analyze", methods=["POST"])
def analyze():
    url_in = request.form.get("url", "")
    ok, url = is_valid_youtube_url(url_in)
    if not ok:
        flash("Please provide a valid YouTube URL.")
        return redirect(url_for("index"))  # simple UX loop [web:85]

    job_id = uuid.uuid4().hex
    job_dir = os.path.join(app.config['JOBS_DIR'], job_id)
    os.makedirs(job_dir, exist_ok=True)

    # 1) Title + metadata
    try:
        title, meta = extract_title_and_meta(url)
    except Exception as e:
        flash(f"Failed to read video metadata: {e}")
        return redirect(url_for("index"))  # graceful error [web:3]

    # 2) Audio
    audio_path = download_audio(url)  # expects path or None
    if not audio_path or not os.path.exists(audio_path):
        flash("Audio download failed.")
        return redirect(url_for("index"))  # keep UI tidy [web:3]
    audio_target = os.path.join(job_dir, os.path.basename(audio_path))
    if os.path.abspath(audio_path) != os.path.abspath(audio_target):
        try:
            import shutil
            shutil.copy2(audio_path, audio_target)
        except Exception:
            audio_target = audio_path  # fallback if copy not needed

    # 3) Transcription (music-aware)
    language, transcript = transcribe_audio(audio_target, video_title=title)

    # 4) Music transcript fallback path
    if is_music_transcript(transcript, metadata=meta):
        summary = "Lyrics detected — summarization skipped."
        result = {
            "title": title,
            "language": language,
            "transcript": transcript,
            "music_video": True,
            "cb_score": 0, "cb_reason": "N/A",
            "ml_score": 0, "ml_reason": "N/A",
            "fact_results": [{
                "claim": "Lyrics / Music content",
                "verdict": "🎵 MUSIC VIDEO",
                "reason": "Lyrics detected — factual verification is not applicable."
            }],
            "summary": summary,
            "meta": {"id": meta.get("id"), "uploader": meta.get("uploader"), "duration": meta.get("duration")}
        }
    else:
        # 5) Clickbait/Misleading
        cb_score, cb_reason, ml_score, ml_reason = detect_clickbait(title)

        # 6) Fact-checks
        fact_results = run_fact_check(transcript)

        # 7) Summary
        summary = summarize_transcript(transcript, is_music=False)

        result = {
            "title": title,
            "language": language,
            "transcript": transcript,
            "music_video": False,
            "cb_score": int(cb_score),
            "cb_reason": "; ".join(cb_reason) if isinstance(cb_reason, list) else str(cb_reason),
            "ml_score": int(ml_score),
            "ml_reason": "; ".join(ml_reason) if isinstance(ml_reason, list) else str(ml_reason),
            "fact_results": fact_results,
            "summary": summary,
            "meta": {"id": meta.get("id"), "uploader": meta.get("uploader"), "duration": meta.get("duration")}
        }

    # Persist result and audio pointer
    with open(os.path.join(job_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(os.path.join(job_dir, "audio.json"), "w", encoding="utf-8") as f:
        json.dump({"audio_path": audio_target}, f)

    return redirect(url_for("result", job_id=job_id))  # UX redirect [web:85]

@app.route("/result/<job_id>", methods=["GET"])
def result(job_id):
    job_dir = os.path.join(app.config['JOBS_DIR'], job_id)
    rfile = os.path.join(job_dir, "result.json")
    if not os.path.exists(rfile):
        flash("Result not found. Please try again.")
        return redirect(url_for("index"))
    with open(rfile, "r", encoding="utf-8") as f:
        result = json.load(f)
    return render_template("result.html", result=result, job_id=job_id)  # dashboard [web:85]

@app.route("/download/<job_id>", methods=["GET"])
def download_audio_route(job_id):
    job_dir = os.path.join(app.config['JOBS_DIR'], job_id)
    afile = os.path.join(job_dir, "audio.json")
    if not os.path.exists(afile):
        flash("Audio not available.")
        return redirect(url_for("result", job_id=job_id))
    with open(afile, "r", encoding="utf-8") as f:
        info = json.load(f)
    path = info.get("audio_path")
    if not path or not os.path.exists(path):
        flash("Audio file missing.")
        return redirect(url_for("result", job_id=job_id))
    return send_file(path, as_attachment=True)  # server-side download [web:120]

if __name__ == "__main__":
    pathlib.Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    app.run(debug=True)  # HTTP dev server; use a proper WSGI for prod [web:3]
