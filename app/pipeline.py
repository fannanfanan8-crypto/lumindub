"""خط معالجة الدبلجة: استخراج الصوت -> نسخ -> ترجمة بلهجة -> توليد صوت -> دمج."""
import asyncio
import json
import os
import subprocess
from pathlib import Path

import edge_tts
import httpx
from pydub import AudioSegment

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# اللغات واللهجات المدعومة + أصوات Edge TTS المجانية
LANGUAGES = {
    "darija": {
        "label": "الدارجة المغربية",
        "voices": {"male": "ar-MA-JamalNeural", "female": "ar-MA-MounaNeural"},
        "style": (
            "Moroccan Arabic (Darija) written in Arabic script, natural spoken street/daily "
            "register as heard in Casablanca or Rabat (e.g. using words like: بغيت، دابا، "
            "كيفاش، شنو، واخا، بزاف, ديال). Do NOT use Modern Standard Arabic."
        ),
    },
    "egyptian": {
        "label": "العامية المصرية",
        "voices": {"male": "ar-EG-ShakirNeural", "female": "ar-EG-SalmaNeural"},
        "style": (
            "Egyptian Arabic (Masri) written in Arabic script, natural spoken Cairene register "
            "(e.g. إزاي، دلوقتي، عايز، كده، أوي). Do NOT use Modern Standard Arabic."
        ),
    },
    "gulf": {
        "label": "اللهجة الخليجية",
        "voices": {"male": "ar-AE-HamdanNeural", "female": "ar-AE-FatimaNeural"},
        "style": (
            "Gulf Arabic (Khaleeji) written in Arabic script, natural spoken register "
            "(e.g. شلونك، وايد، زين، أبي، هالشي). Do NOT use Modern Standard Arabic."
        ),
    },
    "fr": {
        "label": "Français",
        "voices": {"male": "fr-FR-HenriNeural", "female": "fr-FR-DeniseNeural"},
        "style": "Natural, fluent spoken French suited to video narration.",
    },
    "en": {
        "label": "English",
        "voices": {"male": "en-US-GuyNeural", "female": "en-US-JennyNeural"},
        "style": "Natural, fluent spoken English suited to video narration.",
    },
}


def run(cmd: list[str]) -> str:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {res.stderr[-500:]}")
    return res.stdout


def video_duration(path: Path) -> float:
    out = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", str(path),
    ])
    return float(out.strip())


def extract_audio(video: Path, out: Path) -> None:
    run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1", "-ar", "16000",
         "-b:a", "48k", str(out)])


def transcribe(audio: Path) -> tuple[str, list[dict]]:
    """Whisper عبر Groq (طبقة مجانية). يرجع (لغة المصدر، مقاطع مع الأزمنة)."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY غير مضبوط")
    with open(audio, "rb") as f:
        r = httpx.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            files={"file": (audio.name, f, "audio/mpeg")},
            data={"model": "whisper-large-v3-turbo", "response_format": "verbose_json"},
            timeout=180,
        )
    r.raise_for_status()
    data = r.json()
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in data.get("segments", []) if s["text"].strip()
    ]
    if not segments:
        raise RuntimeError("لم يتم اكتشاف كلام في الفيديو")
    return data.get("language", "unknown"), segments


def translate(segments: list[dict], lang_key: str, source_lang: str) -> list[str]:
    """ترجمة مع تكيف ثقافي بلهجة الهدف عبر Gemini (طبقة مجانية)."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY غير مضبوط")
    cfg = LANGUAGES[lang_key]
    items = [
        {"id": i, "text": s["text"], "seconds": round(s["end"] - s["start"], 1)}
        for i, s in enumerate(segments)
    ]
    prompt = (
        "You are a professional dubbing adapter.\n"
        f"Source language code: {source_lang}.\n"
        f"Target: {cfg['style']}\n\n"
        "Rules:\n"
        "- Translate by MEANING, not word by word. Adapt idioms, humor, and cultural references "
        "so they feel native to the target audience.\n"
        "- Keep the speaker's tone and emotion.\n"
        "- Each line must be speakable within its 'seconds' budget: keep it concise, "
        "roughly the same spoken length as the original.\n"
        "- Keep brand names, numbers and technical terms accurate.\n"
        "- Return ONLY a JSON array of strings, same length and order as the input, "
        "no explanations.\n\n"
        f"Input:\n{json.dumps(items, ensure_ascii=False)}"
    )
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}"
        ":generateContent"
    )
    headers = {"x-goog-api-key": GEMINI_API_KEY}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.4, "responseMimeType": "application/json"},
    }
    last_err = None
    for _ in range(3):
        try:
            r = httpx.post(url, json=body, headers=headers, timeout=120)
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            out = json.loads(text)
            if isinstance(out, list) and len(out) == len(segments):
                return [str(x) for x in out]
            last_err = "طول الترجمة لا يطابق عدد المقاطع"
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code} من Gemini (النموذج: {GEMINI_MODEL})"
        except Exception as e:  # noqa: BLE001
            last_err = type(e).__name__
    raise RuntimeError(f"فشلت الترجمة: {last_err}")


async def _tts_all(texts: list[str], voice: str, folder: Path) -> list[Path]:
    paths = []
    for i, t in enumerate(texts):
        p = folder / f"seg_{i}.mp3"
        await edge_tts.Communicate(t, voice).save(str(p))
        paths.append(p)
    return paths


def _speed_up(src: Path, factor: float) -> Path:
    dst = src.with_name(src.stem + "_fast.mp3")
    run(["ffmpeg", "-y", "-i", str(src), "-filter:a", f"atempo={factor:.3f}", str(dst)])
    return dst


def build_dub_track(segments, texts, voice, folder: Path, total_sec: float) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    files = asyncio.run(_tts_all(texts, voice, folder))
    track = AudioSegment.silent(duration=int(total_sec * 1000) + 500)
    for seg, f in zip(segments, files):
        audio = AudioSegment.from_file(f)
        slot_ms = max(int((seg["end"] - seg["start"]) * 1000), 300)
        if len(audio) > slot_ms:
            factor = min(len(audio) / slot_ms, 1.5)  # لا نسرّع أكثر من 1.5x
            audio = AudioSegment.from_file(_speed_up(f, factor))
        track = track.overlay(audio, position=int(seg["start"] * 1000))
    out = folder / "dub.mp3"
    track.export(out, format="mp3")
    return out


def mux(video: Path, dub: Path, out: Path) -> None:
    run([
        "ffmpeg", "-y", "-i", str(video), "-i", str(dub),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-shortest", str(out),
    ])
