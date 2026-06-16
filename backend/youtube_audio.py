import os, warnings

# Hide TF INFO/WARNING backend logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 2 hides INFO and WARNING; keeps errors visible [web:150]

# Suppress only this deprecation emitted by tf_keras losses
warnings.filterwarnings(
    "ignore",
    message=r".*tf\.losses\.sparse_softmax_cross_entropy.*deprecated.*"
)  # precise regex for that warning line [web:152]

import yt_dlp
import os

def download_audio(url, output_path="audio.mp3"):
    base, _ = os.path.splitext(output_path)
    outtmpl = base + ".%(ext)s"

    class SilentLogger:
        def debug(self, msg): pass
        def warning(self, msg): pass
        def error(self, msg): pass

    ydl_opts = {
        # Prefer audio-only; yt-dlp will auto-fallback (HLS/DASH) under SABR
        "format": "ba/bestaudio",
        "js_runtimes": {"node": {}},
        # Output naming
        "outtmpl": outtmpl,
        # Ensure ffmpeg extraction to MP3
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "noplaylist": True,
        # Reduce console noise
        "no_warnings": True,     # hide advisory warnings (nsig/SABR notices)
        "quiet": True,           # minimal output
        "logger": SilentLogger(),# fully silence yt-dlp logger
        # Improve retry robustness when formats/segments hiccup
        "retries": 10,
        "fragment_retries": 10,
        "ignoreerrors": "only_download",  # continue on minor format errors
    }

    try:
        # Tip: keep yt-dlp updated elsewhere (pip install -U yt-dlp or yt-dlp -U)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Prefer the exact requested path; postprocessor writes .mp3
        if os.path.exists(output_path):
            print(f"✅ Audio saved as {output_path}")
            return output_path

        # Fallback: check next to template
        candidate = base + ".mp3"
        if os.path.exists(candidate):
            if candidate != output_path:
                os.replace(candidate, output_path)
            print(f"✅ Audio saved as {output_path}")
            return output_path

        print("❌ Failed to find the downloaded audio.")
        return None
    except Exception as e:
        print("❌ Error:", e)
        return None
