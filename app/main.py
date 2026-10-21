"""LuminDub - منصة دبلجة فيديو بالذكاء الاصطناعي (نسخة مجانية).

المسار: رفع ← تحليل (نسخ + ترجمة) ← مراجعة النص (اختيارية) ← دبلجة وتصدير.
"""
import datetime
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import pipeline as pl

BASE = Path(__file__).resolve().parent.parent
DATA = Path("/tmp/lumindub")
DATA.mkdir(parents=True, exist_ok=True)

MAX_MB = 50
MAX_SECONDS = 180
JOB_TTL = 3600           # حذف الملفات بعد ساعة (قرص Render المجاني مؤقت)
DAILY_LIMIT = 6          # عدد الدبلجات اليومية لكل زائر (لحماية الحصة المجانية)
MAX_PREVIEWS = 30        # معاينات الصوت لكل مهمة
MAX_LINE_CHARS = 400

app = FastAPI(title="LuminDub")
JOBS: dict[str, dict] = {}
JOB_LOCK = threading.Semaphore(1)  # مهمة ثقيلة واحدة في كل مرة: المعالج المجاني ضعيف
USAGE: dict[str, tuple[str, int]] = {}


def cleanup() -> None:
    now = time.time()
    for jid in list(JOBS):
        if now - JOBS[jid]["created"] > JOB_TTL:
            shutil.rmtree(DATA / jid, ignore_errors=True)
            JOBS.pop(jid, None)


def client_id(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() or (request.client.host if request.client else "?"))


def check_daily_limit(cid: str) -> None:
    today = datetime.date.today().isoformat()
    day, n = USAGE.get(cid, (today, 0))
    if day != today:
        n = 0
    if n >= DAILY_LIMIT:
        raise HTTPException(429, f"وصلت الحد اليومي ({DAILY_LIMIT} دبلجات). عُد غداً")
    USAGE[cid] = (today, n + 1)


def get_job(job_id: str) -> dict:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "المهمة غير موجودة أو انتهت صلاحيتها")
    return job


# ----------------------------------------------------------- المرحلة 1: التحليل

def analyze(job_id: str, review: bool) -> None:
    job = JOBS[job_id]
    video: Path = job["_video"]
    folder = DATA / job_id
    job.update(step="في قائمة الانتظار", progress=1)
    with JOB_LOCK:
        t0 = last = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal last
            now = time.perf_counter()
            print(f"[job {job_id}] {name}: {now - last:.1f}s (total {now - t0:.1f}s)", flush=True)
            last = now

        try:
            job.update(status="running", step="استخراج الصوت", progress=5)
            total = pl.video_duration(video)
            if total > MAX_SECONDS:
                raise RuntimeError(f"الحد الأقصى للمدة {MAX_SECONDS} ثانية في النسخة المجانية")
            audio = folder / "audio.mp3"
            pl.extract_audio(video, audio)
            lap("extract_audio")

            job.update(step="التعرف على الكلام", progress=15)
            src_lang, segments = pl.transcribe(audio)
            lap("transcribe")

            job.update(step="ترجمة وتكييف كل اللغات معاً", progress=25)
            all_texts = pl.translate_many(segments, job["langs"], src_lang, job["glossary"])
            lap("translate_all")

            job.update(
                source_language=src_lang, _segments=segments, _total=total,
                _lines=all_texts, created=time.time(),
            )
            if review:
                job["draft"] = {
                    "segments": [
                        {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"]}
                        for s in segments
                    ],
                    "texts": {k: [{"text": str(t), "tts": t.tts} for t in v] for k, v in all_texts.items()},
                }
                job.update(status="review", step="راجع النص ثم ابدأ الدبلجة", progress=100)
        except Exception as e:  # noqa: BLE001
            job.update(status="error", step="فشل", error=str(e))
            return
    if not review:
        render(job_id, {k: [str(t) for t in v] for k, v in job["_lines"].items()}, {}, None)


# ----------------------------------------------------------- المرحلة 2: الدبلجة

def render(job_id: str, texts: dict, voices: dict, rate: str | None) -> None:
    job = JOBS[job_id]
    folder = DATA / job_id
    video: Path = job["_video"]
    segments, total = job["_segments"], job["_total"]
    gender = job["gender"]
    langs = job["langs"]
    job.update(status="running", step="في قائمة الانتظار", progress=1, error=None, outputs={})
    with JOB_LOCK:
        t0 = last = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal last
            now = time.perf_counter()
            print(f"[job {job_id}] {name}: {now - last:.1f}s (total {now - t0:.1f}s)", flush=True)
            last = now

        try:
            n = len(langs)
            for i, key in enumerate(langs):
                label = pl.LANGUAGES[key]["label"]
                base = 5 + int(90 * i / n)
                draft_lines = job["_lines"][key]
                lines = []
                for j, new in enumerate(texts[key]):
                    old = draft_lines[j]
                    # إن لم يعدّل المستخدم الجملة نحتفظ بنسخة النطق المهيأة
                    lines.append(pl.Line(new, old.tts) if new.strip() == str(old).strip() else pl.Line(new))

                job.update(step=f"توليد الصوت: {label}", progress=base)
                voice = voices.get(key) or pl.LANGUAGES[key]["voices"][gender]
                dub = pl.build_dub_track(segments, lines, voice, folder / key, total, rate)
                lap(f"tts+mix[{key}]")

                job.update(step=f"دمج الفيديو: {label}", progress=base + int(60 / n))
                out = folder / f"{key}.mp4"
                pl.mux(video, dub, out)
                pl.make_srt(segments, lines, folder / f"{key}.srt")
                lap(f"mux[{key}]")
                job["outputs"][key] = {
                    "label": label,
                    "url": f"/files/{job_id}/{key}.mp4",
                    "srt": f"/files/{job_id}/{key}.srt",
                    "text": " ".join(lines),
                }
            job.update(status="done", step="اكتمل", progress=100, created=time.time())
        except Exception as e:  # noqa: BLE001
            job.update(status="error", step="فشل", error=str(e))


# --------------------------------------------------------------- نقاط النهاية

@app.post("/api/jobs")
async def create_job(
    request: Request,
    file: UploadFile = File(...),
    langs: str = Form(...),
    gender: str = Form("male"),
    consent: bool = Form(False),
    review: bool = Form(True),
    glossary: str = Form(""),
):
    cleanup()
    if not consent:
        raise HTTPException(400, "يجب الإقرار بأنك تملك حقوق الفيديو والصوت")
    keys = [k for k in dict.fromkeys(langs.split(",")) if k in pl.LANGUAGES]
    if not keys:
        raise HTTPException(400, "اختر لغة واحدة على الأقل")
    if gender not in ("male", "female"):
        gender = "male"
    check_daily_limit(client_id(request))

    job_id = uuid.uuid4().hex[:12]
    folder = DATA / job_id
    folder.mkdir(parents=True)
    video = folder / "input.mp4"
    size = 0
    with open(video, "wb") as f:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_MB * 1024 * 1024:
                f.close()
                shutil.rmtree(folder, ignore_errors=True)
                raise HTTPException(413, f"الحد الأقصى للملف {MAX_MB}MB")
            f.write(chunk)

    JOBS[job_id] = {
        "created": time.time(), "status": "queued", "step": "في الانتظار",
        "progress": 0, "outputs": {}, "error": None, "source_language": None,
        "langs": keys, "gender": gender, "glossary": glossary[:1500],
        "_video": video, "previews": 0,
    }
    threading.Thread(target=analyze, args=(job_id, review), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)
    hidden = {"created", "glossary", "previews"}
    return {k: v for k, v in job.items() if not k.startswith("_") and k not in hidden}


@app.post("/api/jobs/{job_id}/preview")
def preview(job_id: str, payload: dict = Body(...)):
    """معاينة صوت جملة واحدة قبل الدبلجة الكاملة."""
    job = get_job(job_id)
    lang = payload.get("lang")
    text = str(payload.get("text", "")).strip()[:MAX_LINE_CHARS]
    if lang not in job["langs"] or not text:
        raise HTTPException(400, "طلب غير صالح")
    voice = payload.get("voice") or pl.LANGUAGES[lang]["voices"][job["gender"]]
    if voice not in pl.allowed_voices(lang):
        raise HTTPException(400, "صوت غير متاح")
    rate = clean_rate(payload.get("rate"), pl.LANGUAGES[lang]["rate"])
    if job["previews"] >= MAX_PREVIEWS:
        raise HTTPException(429, "وصلت حد المعاينات لهذه المهمة")
    job["previews"] += 1
    out = DATA / job_id / f"preview_{uuid.uuid4().hex[:8]}.mp3"
    try:
        pl.preview_tts(text, voice, rate, out)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, str(e))
    return {"url": f"/files/{job_id}/{out.name}"}


@app.post("/api/jobs/{job_id}/render")
def start_render(job_id: str, payload: dict = Body(...)):
    job = get_job(job_id)
    if job["status"] != "review":
        raise HTTPException(409, "المهمة ليست في مرحلة المراجعة")
    texts_in = payload.get("texts") or {}
    n = len(job["_segments"])
    texts: dict[str, list[str]] = {}
    for key in job["langs"]:
        arr = texts_in.get(key)
        if not isinstance(arr, list) or len(arr) != n:
            raise HTTPException(400, f"عدد الجمل غير مطابق للغة {key}")
        texts[key] = [str(t).strip()[:MAX_LINE_CHARS] or str(job["_lines"][key][i]) for i, t in enumerate(arr)]
    voices_in = payload.get("voices") or {}
    voices = {k: v for k, v in voices_in.items() if k in job["langs"] and v in pl.allowed_voices(k)}
    rate = clean_rate(payload.get("rate"), None)
    job["status"] = "running"  # يمنع الضغط المزدوج
    threading.Thread(target=render, args=(job_id, texts, voices, rate), daemon=True).start()
    return {"ok": True}


def clean_rate(value, default):
    if isinstance(value, str) and re.fullmatch(r"[+-]\d{1,2}%", value):
        return value
    return default


@app.get("/api/languages")
def languages():
    return {
        k: {
            "label": v["label"],
            "voices": [{"id": i, "label": lbl, "gender": g} for i, lbl, g in pl.VOICE_OPTIONS[k]],
        }
        for k, v in pl.LANGUAGES.items()
    }


@app.get("/health")
def health():
    return {"ok": True}


app.mount("/files", StaticFiles(directory=DATA), name="files")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
