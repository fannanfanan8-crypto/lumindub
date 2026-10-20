"""خط معالجة الدبلجة: استخراج الصوت -> نسخ -> ترجمة بلهجة -> صقل -> توليد صوت -> دمج."""
import asyncio
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import edge_tts
import httpx
from pydub import AudioSegment
from pydub.effects import normalize

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

# اللغات واللهجات المدعومة + أصوات Edge TTS المجانية.
# examples: أمثلة تُظهر للنموذج الفرق بين اللهجة والفصحى (en -> target).
LANGUAGES = {
    "darija": {
        "label": "الدارجة المغربية",
        "arabic": True,
        "rate": "-4%",
        "voices": {"male": "ar-MA-JamalNeural", "female": "ar-MA-MounaNeural"},
        "style": (
            "Moroccan Arabic (Darija) in Arabic script, as spoken in Casablanca/Rabat "
            "in everyday conversation"
        ),
        "guide": (
            "Use real Darija vocabulary and grammar: بغيت، دابا، كيفاش، شنو، علاش، فين، واش، "
            "بزاف، شوية، ديال، كاين، ماكاينش، واخا، يالاه، ساعة، هاد، هادشي، مزيان، خايب، "
            "كنقول، كتقول، غادي نمشي. Verbs take ك/كا (present) and غادي (future). "
            "Never use MSA forms such as: أريد، الآن، كيف حالك، ماذا، لماذا، جداً، سوف، هذا. "
            "French loanwords common in Moroccan speech are fine when natural."
        ),
        "examples": [
            ("I want to show you something.", "بغيت نوريك شي حاجة."),
            ("Let's get started.", "يالاه نبداو."),
            ("It's very expensive.", "غالي بزاف."),
            ("What do you think?", "شنو كتشوف؟"),
            ("We will see how it works.", "غادي نشوفو كيفاش خدام."),
        ],
    },
    "egyptian": {
        "label": "العامية المصرية",
        "arabic": True,
        "rate": "-4%",
        "voices": {"male": "ar-EG-ShakirNeural", "female": "ar-EG-SalmaNeural"},
        "style": "Egyptian Arabic (Masri) in Arabic script, natural spoken Cairene",
        "guide": (
            "Use real Cairene vocabulary and grammar: إزاي، دلوقتي، عايز/عايزة، كده، أوي، "
            "إيه، ليه، فين، مش، بتاع، حاجة، خلاص، يلا، هنعمل، بنعمل، لسه، برضو، كمان. "
            "Present uses ب (بنعمل), future uses ه/هـ (هنعمل). "
            "Never use MSA forms such as: أريد، الآن، كيف، ماذا، لماذا، جداً، سوف، هذا."
        ),
        "examples": [
            ("I want to show you something.", "عايز أوريك حاجة."),
            ("Let's get started.", "يلا نبدأ."),
            ("It's very expensive.", "ده غالي أوي."),
            ("What do you think?", "رأيك إيه؟"),
            ("We will see how it works.", "هنشوف بيشتغل إزاي."),
        ],
    },
    "gulf": {
        "label": "اللهجة الخليجية",
        "arabic": True,
        "rate": "-4%",
        "voices": {"male": "ar-AE-HamdanNeural", "female": "ar-AE-FatimaNeural"},
        "style": "Gulf Arabic (Khaleeji, Emirati/Saudi-neutral) in Arabic script, natural spoken",
        "guide": (
            "Use real Khaleeji vocabulary and grammar: شلونك، شخبارك، وايد، زين، أبي/أبغى، "
            "هالشي، ليش، وين، إيش، شرايك، يالله، حق، مب، خلاص، توّه، الحين، هب. "
            "Never use MSA forms such as: أريد، الآن، كيف، ماذا، لماذا، جداً، سوف، هذا."
        ),
        "examples": [
            ("I want to show you something.", "أبي أوريك شي."),
            ("Let's get started.", "يالله نبدأ."),
            ("It's very expensive.", "وايد غالي."),
            ("What do you think?", "شرايك؟"),
            ("We will see how it works.", "بنشوف شلون يشتغل."),
        ],
    },
    "fr": {
        "label": "Français",
        "arabic": False,
        "rate": "+0%",
        "voices": {"male": "fr-FR-HenriNeural", "female": "fr-FR-DeniseNeural"},
        "style": "Natural, fluent spoken French suited to video narration",
        "guide": (
            "Use natural spoken French with contractions and everyday phrasing, "
            "not stiff written French. Use 'tu' or 'vous' consistently, matching the source tone."
        ),
        "examples": [],
    },
    "en": {
        "label": "English",
        "arabic": False,
        "rate": "+0%",
        "voices": {"male": "en-US-GuyNeural", "female": "en-US-JennyNeural"},
        "style": "Natural, fluent spoken American English suited to video narration",
        "guide": "Use natural spoken English with contractions, not stiff written English.",
        "examples": [],
    },
}


class Line(str):
    """نص مترجم للعرض، ويحمل في .tts صيغة مهيأة للنطق."""

    tts: str

    def __new__(cls, text: str, tts: str | None = None):
        obj = super().__new__(cls, text)
        obj.tts = (tts or text).strip() or text
        return obj


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


# ---------------------------------------------------------------- Gemini

def _gemini_json(prompt: str, temperature: float = 0.4):
    """استدعاء Gemini وإرجاع JSON، مع إعادة المحاولة عند تجاوز الحد المجاني."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY غير مضبوط")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}"
        ":generateContent"
    )
    headers = {"x-goog-api-key": GEMINI_API_KEY}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    last_err = "unknown"
    for attempt in range(4):
        try:
            r = httpx.post(url, json=body, headers=headers, timeout=120)
            if r.status_code == 429:
                last_err = "تجاوز حد الاستخدام المجاني لـ Gemini"
                time.sleep(6 * (attempt + 1))
                continue
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code} من Gemini (النموذج: {GEMINI_MODEL})"
            if e.response.status_code in (400, 401, 403, 404):
                break
        except Exception as e:  # noqa: BLE001
            last_err = type(e).__name__
    raise RuntimeError(f"فشل الاتصال بـ Gemini: {last_err}")


def _examples_block(cfg: dict) -> str:
    if not cfg["examples"]:
        return ""
    lines = "\n".join(f'  "{en}"  ->  "{tgt}"' for en, tgt in cfg["examples"])
    return f"Examples of the exact register wanted:\n{lines}\n"


def _translate_pass(segments: list[dict], cfg: dict, source_lang: str) -> list[str]:
    items = [
        {"id": i, "text": s["text"], "seconds": round(s["end"] - s["start"], 1)}
        for i, s in enumerate(segments)
    ]
    prompt = (
        f"You are a senior dubbing adapter and a native speaker of: {cfg['style']}.\n"
        f"Source language code: {source_lang}. The lines below are consecutive parts of ONE "
        "video, so keep terminology, names and the speaker's voice consistent.\n\n"
        f"{cfg['guide']}\n\n"
        f"{_examples_block(cfg)}\n"
        "Rules:\n"
        "- Translate by MEANING, never word by word. Rewrite idioms, humor and cultural "
        "references so they feel native to this audience.\n"
        "- Sound like a real person talking to a friend, not like a written text or a news anchor.\n"
        "- Keep the speaker's tone, energy and emotion.\n"
        "- Each line must be speakable within its 'seconds' budget (about 12-15 characters "
        "per second for Arabic, 14-17 for French/English). Prefer shorter.\n"
        "- Keep brand names, numbers and technical terms accurate.\n"
        "- Return ONLY a JSON array of strings, same length and order as the input.\n\n"
        f"Input:\n{json.dumps(items, ensure_ascii=False)}"
    )
    for _ in range(3):
        out = _gemini_json(prompt, temperature=0.5)
        if isinstance(out, list) and len(out) == len(segments):
            return [str(x).strip() for x in out]
    raise RuntimeError("فشلت الترجمة: طول الترجمة لا يطابق عدد المقاطع")


def _refine_pass(lines: list[str], cfg: dict) -> list[tuple[str, str]]:
    """صقل لهجي + تهيئة النطق. يرجع [(نص العرض، نص النطق)]."""
    if cfg["arabic"]:
        task = (
            "1. 'text': review each line as a strict native-dialect editor. Replace any "
            "Modern Standard Arabic words or stiff phrasing with the natural spoken dialect. "
            "Keep the meaning and the length. If the line is already natural, keep it as is.\n"
            "2. 'tts': a speech-ready version of 'text' for an Arabic text-to-speech engine: "
            "write every number, percentage, symbol and abbreviation as the words a speaker of "
            "this dialect would SAY; transliterate Latin-script names and brands into Arabic "
            "letters as they are pronounced; add short vowel marks (tashkeel) ONLY on words "
            "that would be ambiguous or mispronounced without them; add a comma where a natural "
            "short pause belongs. Do not change the wording otherwise."
        )
    else:
        task = (
            "1. 'text': review each line and make it sound like natural speech. Keep meaning "
            "and length. If already natural, keep it as is.\n"
            "2. 'tts': a speech-ready version of 'text': write numbers, symbols and "
            "abbreviations as spoken words; keep everything else identical."
        )
    prompt = (
        f"Target language/dialect: {cfg['style']}.\n{cfg['guide']}\n\n"
        f"{_examples_block(cfg)}\n"
        f"For every input line produce an object with two fields:\n{task}\n\n"
        "Return ONLY a JSON array of objects {\"text\": str, \"tts\": str}, same length and "
        "order as the input.\n\n"
        f"Input:\n{json.dumps(lines, ensure_ascii=False)}"
    )
    out = _gemini_json(prompt, temperature=0.2)
    if (
        not isinstance(out, list) or len(out) != len(lines)
        or not all(isinstance(o, dict) and o.get("text") for o in out)
    ):
        raise ValueError("bad refine output")
    return [(str(o["text"]).strip(), str(o.get("tts") or o["text"]).strip()) for o in out]


def translate(segments: list[dict], lang_key: str, source_lang: str) -> list[str]:
    """ترجمة بالمعنى بلهجة الهدف ثم صقل لهجي وتهيئة للنطق. يرجع Line (نص + .tts)."""
    cfg = LANGUAGES[lang_key]
    draft = _translate_pass(segments, cfg, source_lang)
    try:
        refined = _refine_pass(draft, cfg)
        return [Line(t, s) for t, s in refined]
    except Exception:  # noqa: BLE001  - الصقل تحسين اختياري، لا نُفشل المهمة بسببه
        return [Line(t) for t in draft]


def translate_many(segments: list[dict], langs: list[str], source_lang: str) -> dict:
    """يترجم كل اللغات بالتوازي (عمل شبكة فقط). يرجع {لغة: [Line]}."""
    with ThreadPoolExecutor(max_workers=min(3, len(langs))) as ex:
        futs = {k: ex.submit(translate, segments, k, source_lang) for k in langs}
        return {k: f.result() for k, f in futs.items()}


def _shorten(lines: list[str], cfg: dict, max_chars: list[int]) -> list[str]:
    items = [{"id": i, "text": t, "max_chars": m} for i, (t, m) in enumerate(zip(lines, max_chars))]
    prompt = (
        f"Target language/dialect: {cfg['style']}.\n"
        "Each line below is too long to be spoken in its time slot. Rewrite each one shorter, "
        "at most 'max_chars' characters, keeping the key meaning, the same dialect and a natural "
        "spoken tone. Return ONLY a JSON array of strings, same length and order.\n\n"
        f"Input:\n{json.dumps(items, ensure_ascii=False)}"
    )
    out = _gemini_json(prompt, temperature=0.3)
    if not isinstance(out, list) or len(out) != len(lines):
        raise ValueError("bad shorten output")
    return [str(x).strip() or l for x, l in zip(out, lines)]


# ------------------------------------------------------------------- TTS

def _lang_from_voice(voice: str) -> str:
    for key, cfg in LANGUAGES.items():
        if voice in cfg["voices"].values():
            return key
    return "en"


async def _tts_one(text: str, voice: str, rate: str, path: Path) -> None:
    last: Exception | None = None
    for attempt in range(3):
        try:
            await edge_tts.Communicate(text, voice, rate=rate).save(str(path))
            return
        except Exception as e:  # noqa: BLE001
            last = e
            await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"فشل توليد الصوت: {type(last).__name__}")


async def _tts_many(jobs: list[tuple[str, Path]], voice: str, rate: str) -> None:
    sem = asyncio.Semaphore(4)  # 4 طلبات متوازية: أسرع دون إغراق الخدمة

    async def one(text: str, path: Path) -> None:
        async with sem:
            await _tts_one(text, voice, rate, path)

    await asyncio.gather(*(one(t, p) for t, p in jobs))


SR = 24000  # معدل عيّنات منخفض أحادي: أخف 4 مرات في الذاكرة والمعالجة


def _load(path: Path) -> AudioSegment:
    return AudioSegment.from_file(path).set_frame_rate(SR).set_channels(1)


def _speed_up(src: Path, factor: float) -> Path:
    dst = src.with_name(src.stem + "_fast.mp3")
    run(["ffmpeg", "-y", "-i", str(src), "-filter:a", f"atempo={factor:.3f}", str(dst)])
    return dst


def build_dub_track(segments, texts, voice, folder: Path, total_sec: float) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    lang_key = _lang_from_voice(voice)
    cfg = LANGUAGES[lang_key]
    rate = cfg["rate"]
    n = len(segments)
    spoken = [getattr(t, "tts", None) or str(t) for t in texts]

    # الوقت المتاح لكل مقطع: حتى بداية المقطع التالي (لا حتى نهاية مقطعه فقط)
    limits = []
    for i, seg in enumerate(segments):
        nxt = segments[i + 1]["start"] if i + 1 < n else total_sec
        limits.append(max(int((nxt - seg["start"]) * 1000), 400))

    files = [folder / f"seg_{i}.mp3" for i in range(n)]
    asyncio.run(_tts_many(list(zip(spoken, files)), voice, rate))
    audios = [_load(f) for f in files]

    # مقاطع أطول بكثير من وقتها: نطلب نصاً أقصر بدل تسريع مزعج
    long_idx = [
        i for i in range(n)
        if len(audios[i]) > limits[i] * 1.25 and len(spoken[i]) > 15
    ]
    if long_idx:
        try:
            targets = [int(len(spoken[i]) * limits[i] / len(audios[i]) * 0.92) for i in long_idx]
            shorter = _shorten([spoken[i] for i in long_idx], cfg, targets)
            asyncio.run(_tts_many([(shorter[k], files[i]) for k, i in enumerate(long_idx)], voice, rate))
            for i in long_idx:
                audios[i] = _load(files[i])
        except Exception:  # noqa: BLE001
            pass  # نكمل بالتسريع فقط

    track = AudioSegment.silent(duration=int(total_sec * 1000) + 500, frame_rate=SR).set_channels(1)
    for i, seg in enumerate(segments):
        audio = audios[i]
        if len(audio) > limits[i]:
            factor = min(len(audio) / limits[i], 1.35)  # حد أقصى للتسريع حفاظاً على الوضوح
            audio = _load(_speed_up(files[i], factor))
        audio = audio.fade_in(15).fade_out(40)
        track = track.overlay(audio, position=int(seg["start"] * 1000))
    track = normalize(track, headroom=1.5)
    out = folder / "dub.mp3"
    track.export(out, format="mp3")
    return out


def mux(video: Path, dub: Path, out: Path) -> None:
    run([
        "ffmpeg", "-y", "-i", str(video), "-i", str(dub),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
        "-shortest", str(out),
    ])
