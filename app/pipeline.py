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
SYNC_API_KEY = os.getenv("SYNC_API_KEY", "")
SYNC_MODEL = os.getenv("SYNC_MODEL", "lipsync-2")
SYNC_BASE = "https://api.sync.so/v2"
LIPSYNC_ENABLED = bool(SYNC_API_KEY)
LIPSYNC_MAX_SECONDS = int(os.getenv("LIPSYNC_MAX_SECONDS", "20"))  # حد الخطة المجانية
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-flash-lite-latest")

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


# أصوات Edge TTS المتاحة لكل لغة: (المعرّف، الاسم المعروض، الجنس)
VOICE_OPTIONS = {
    "darija": [
        ("ar-MA-JamalNeural", "جمال (رجل)", "male"),
        ("ar-MA-MounaNeural", "منى (امرأة)", "female"),
    ],
    "egyptian": [
        ("ar-EG-ShakirNeural", "شاكر (رجل)", "male"),
        ("ar-EG-SalmaNeural", "سلمى (امرأة)", "female"),
    ],
    "gulf": [
        ("ar-AE-HamdanNeural", "حمدان - إماراتي (رجل)", "male"),
        ("ar-AE-FatimaNeural", "فاطمة - إماراتية (امرأة)", "female"),
        ("ar-SA-HamedNeural", "حامد - سعودي (رجل)", "male"),
        ("ar-SA-ZariyahNeural", "زارية - سعودية (امرأة)", "female"),
    ],
    "fr": [
        ("fr-FR-HenriNeural", "Henri (homme)", "male"),
        ("fr-FR-DeniseNeural", "Denise (femme)", "female"),
        ("fr-FR-RemyMultilingualNeural", "Rémy (homme)", "male"),
        ("fr-FR-VivienneMultilingualNeural", "Vivienne (femme)", "female"),
    ],
    "en": [
        ("en-US-GuyNeural", "Guy (man)", "male"),
        ("en-US-JennyNeural", "Jenny (woman)", "female"),
        ("en-US-DavisNeural", "Davis (man)", "male"),
        ("en-US-AriaNeural", "Aria (woman)", "female"),
        ("en-GB-RyanNeural", "Ryan - UK (man)", "male"),
        ("en-GB-SoniaNeural", "Sonia - UK (woman)", "female"),
    ],
}


def allowed_voices(lang_key: str) -> set[str]:
    return {v[0] for v in VOICE_OPTIONS[lang_key]}


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

def _gemini_json(prompt: str, temperature: float = 0.4, retries: int = 4):
    """استدعاء Gemini وإرجاع JSON.
    عند الازدحام (503/500) أو حد الدقيقة (429) نعيد المحاولة ونبدّل بين النموذج الأساسي
    والبديل. وإن استُنفدت الحصة اليومية لنموذج ننتقل للآخر."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY غير مضبوط")
    models = [GEMINI_MODEL] + [m for m in (GEMINI_FALLBACK_MODEL,) if m and m != GEMINI_MODEL]
    exhausted: set[str] = set()
    headers = {"x-goog-api-key": GEMINI_API_KEY}
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    last_err = "unknown"
    for attempt in range(retries):
        alive = [m for m in models if m not in exhausted]
        if not alive:
            break
        model = alive[attempt % len(alive)]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        try:
            r = httpx.post(url, json=body, headers=headers, timeout=120)
            if r.status_code == 429:
                if "perday" in r.text.lower().replace(" ", "").replace("_", ""):
                    exhausted.add(model)
                    last_err = "انتهت حصة Gemini اليومية المجانية، جرّب غداً"
                else:
                    last_err = "تجاوز حد الدقيقة المجاني لـ Gemini"
                    if attempt < retries - 1:
                        time.sleep(10 * (attempt + 1))
                continue
            if r.status_code in (500, 502, 503, 504):
                last_err = f"Gemini مزدحم مؤقتاً (HTTP {r.status_code})"
                if attempt < retries - 1:
                    time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except httpx.HTTPStatusError as e:
            last_err = f"HTTP {e.response.status_code} من Gemini (النموذج: {model})"
            if e.response.status_code in (400, 401, 403, 404):
                exhausted.add(model)
        except Exception as e:  # noqa: BLE001
            last_err = type(e).__name__
    raise RuntimeError(f"فشل الاتصال بـ Gemini: {last_err}")


def _examples_block(cfg: dict) -> str:
    if not cfg["examples"]:
        return ""
    lines = "\n".join(f'  "{en}"  ->  "{tgt}"' for en, tgt in cfg["examples"])
    return f"Examples of the exact register wanted:\n{lines}\n"


def _glossary_block(glossary: str) -> str:
    g = "\n".join(l.strip() for l in (glossary or "").splitlines() if l.strip())[:1500]
    if not g:
        return ""
    return (
        "Glossary written by the video owner. ALWAYS use exactly these renderings "
        "(left = source term, right = required output; if only one term is given, "
        "keep it untranslated):\n" + g + "\n\n"
    )


def _translate_pass(segments: list[dict], cfg: dict, source_lang: str, glossary: str = "") -> list[str]:
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
        f"{_glossary_block(glossary)}"
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
    out = _gemini_json(prompt, temperature=0.2, retries=1)
    if (
        not isinstance(out, list) or len(out) != len(lines)
        or not all(isinstance(o, dict) and o.get("text") for o in out)
    ):
        raise ValueError("bad refine output")
    return [(str(o["text"]).strip(), str(o.get("tts") or o["text"]).strip()) for o in out]


def translate(segments: list[dict], lang_key: str, source_lang: str, glossary: str = "") -> list[str]:
    """ترجمة بالمعنى بلهجة الهدف ثم صقل لهجي وتهيئة للنطق. يرجع Line (نص + .tts)."""
    cfg = LANGUAGES[lang_key]
    draft = _translate_pass(segments, cfg, source_lang, glossary)
    if not cfg["arabic"]:
        return [Line(t) for t in draft]  # الفرنسية/الإنجليزية: طلب واحد يوفر الحصة
    try:
        refined = _refine_pass(draft, cfg)
        return [Line(t, s) for t, s in refined]
    except Exception:  # noqa: BLE001  - الصقل تحسين اختياري، لا نُفشل المهمة بسببه
        return [Line(t) for t in draft]


def translate_many(segments: list[dict], langs: list[str], source_lang: str, glossary: str = "") -> dict:
    """يترجم كل اللغات بالتوازي (عمل شبكة فقط). يرجع {لغة: [Line]}."""
    with ThreadPoolExecutor(max_workers=min(2, len(langs))) as ex:
        futs = {k: ex.submit(translate, segments, k, source_lang, glossary) for k in langs}
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
    out = _gemini_json(prompt, temperature=0.3, retries=1)
    if not isinstance(out, list) or len(out) != len(lines):
        raise ValueError("bad shorten output")
    return [str(x).strip() or l for x, l in zip(out, lines)]


# ------------------------------------------------------------------- TTS

def _lang_from_voice(voice: str) -> str:
    for key in VOICE_OPTIONS:
        if voice in allowed_voices(key):
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
    return AudioSegment.from_file(path).set_frame_rate(SR).set_channels(1).set_sample_width(2)


def _speed_up(src: Path, factor: float) -> Path:
    dst = src.with_name(src.stem + "_fast.mp3")
    run(["ffmpeg", "-y", "-i", str(src), "-filter:a", f"atempo={factor:.3f}", str(dst)])
    return dst


def build_dub_track(segments, texts, voice, folder: Path, total_sec: float, rate: str | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    lang_key = _lang_from_voice(voice)
    cfg = LANGUAGES[lang_key]
    rate = rate or cfg["rate"]
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

    # تجميع المقاطع بتسلسل واحد (O(n)) بدل overlay الذي ينسخ المسار كاملاً مع كل جملة
    chunks: list[bytes] = []
    cursor = 0
    for i, seg in enumerate(segments):
        audio = audios[i]
        if len(audio) > limits[i]:
            factor = min(len(audio) / limits[i], 1.35)  # حد أقصى للتسريع حفاظاً على الوضوح
            audio = _load(_speed_up(files[i], factor))
        audio = audio.fade_in(15).fade_out(40)
        pos = max(int(seg["start"] * 1000), cursor)  # لا نسمح بتداخل الجمل
        if pos > cursor:
            chunks.append(AudioSegment.silent(duration=pos - cursor, frame_rate=SR).raw_data)
        chunks.append(audio.raw_data)
        cursor = pos + len(audio)
    end = int(total_sec * 1000) + 500
    if cursor < end:
        chunks.append(AudioSegment.silent(duration=end - cursor, frame_rate=SR).raw_data)
    track = AudioSegment(data=b"".join(chunks), sample_width=2, frame_rate=SR, channels=1)
    track = normalize(track, headroom=1.5)
    out = folder / "dub.mp3"
    track.export(out, format="mp3")
    return out


def _srt_time(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    sec_, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{sec_:02d},{ms:03d}"


def make_srt(segments: list[dict], texts: list[str], path: Path) -> None:
    blocks = []
    for i, (seg, t) in enumerate(zip(segments, texts), 1):
        blocks.append(f"{i}\n{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}\n{str(t).strip()}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def preview_tts(text: str, voice: str, rate: str, path: Path) -> None:
    asyncio.run(_tts_one(text, voice, rate, path))


def lipsync(video_url: str, audio_url: str, raw_out: Path, timeout_s: int = 600) -> None:
    """مطابقة الشفاه عبر Sync.so: ينتج فيديو بحركة فم تطابق الصوت المدبلج."""
    if not SYNC_API_KEY:
        raise RuntimeError("SYNC_API_KEY غير مضبوط")
    headers = {"x-api-key": SYNC_API_KEY, "Content-Type": "application/json"}
    media = [{"type": "video", "url": video_url}, {"type": "audio", "url": audio_url}]
    bodies = [
        {"model": SYNC_MODEL, "input": media, "options": {"sync_mode": "cut_off"}},
        {"model": SYNC_MODEL, "input": media},
        {"model": SYNC_MODEL, "video_url": video_url, "audio_url": audio_url},
    ]
    gen, last = None, ""
    for body in bodies:  # نجرّب الصيغ المحتملة للطلب بالترتيب
        r = httpx.post(f"{SYNC_BASE}/generate", json=body, headers=headers, timeout=60)
        if r.status_code in (401, 403):
            raise RuntimeError("مفتاح Sync.so غير صالح")
        if r.status_code in (402, 429):
            raise RuntimeError("انتهت حصة Sync.so المجانية أو تجاوزت الحد المسموح")
        if r.status_code in (400, 422):
            last = r.text[:160]
            continue
        r.raise_for_status()
        gen = r.json()
        break
    if not gen or not gen.get("id"):
        raise RuntimeError(f"رفضت خدمة Sync.so الطلب: {last or 'استجابة غير متوقعة'}")

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(8)
        r = httpx.get(f"{SYNC_BASE}/generate/{gen['id']}", headers=headers, timeout=60)
        r.raise_for_status()
        d = r.json()
        status = str(d.get("status", "")).lower()
        if status == "completed":
            url = d.get("outputUrl") or d.get("output_url") or d.get("video_url")
            if not url:
                raise RuntimeError("اكتملت المطابقة لكن بلا رابط ناتج")
            with httpx.stream("GET", url, timeout=180, follow_redirects=True) as resp:
                resp.raise_for_status()
                with open(raw_out, "wb") as f:
                    for chunk in resp.iter_bytes():
                        f.write(chunk)
            return
        if status in ("failed", "rejected", "canceled", "cancelled"):
            raise RuntimeError(f"فشلت مطابقة الشفاه: {d.get('error') or status}")
    raise RuntimeError("انتهت مهلة مطابقة الشفاه")


def mux(video: Path, dub: Path, out: Path) -> None:
    run([
        "ffmpeg", "-y", "-i", str(video), "-i", str(dub),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "96k",
        "-movflags", "+faststart", "-shortest", str(out),
    ])
