"""LuminDub - منصة دبلجة فيديو بالذكاء الاصطناعي (نسخة MVP مجانية)."""
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import pipeline as pl

BASE = Path(__file__).resolve().parent.parent
DATA = Path("/tmp/lumindub")
DATA.mkdir(parents=True, exist_ok=True)

MAX_MB = 50
MAX_SECONDS = 180
JOB_TTL = 3600  # حذف الملفات بعد ساعة (قرص Render المجاني مؤقت)

app = FastAPI(title="LuminDub")
JOBS: dict[str, dict] = {}


def cleanup() -> None:
    now = time.time()
    for jid in list(JOBS):
        if now - JOBS[jid]["created"] > JOB_TTL:
            shutil.rmtree(DATA / jid, ignore_errors=True)
            JOBS.pop(jid, None)


def process(job_id: str, video: Path, langs: list[str], gender: str) -> None:
    job = JOBS[job_id]
    folder = DATA / job_id
    try:
        job.update(status="running", step="استخراج الصوت", progress=5)
        total = pl.video_duration(video)
        if total > MAX_SECONDS:
            raise RuntimeError(f"الحد الأقصى للمدة {MAX_SECONDS} ثانية في النسخة المجانية")
        audio = folder / "audio.mp3"
        pl.extract_audio(video, audio)

        job.update(step="التعرف على الكلام", progress=15)
        src_lang, segments = pl.transcribe(audio)
        job["source_language"] = src_lang

        n = len(langs)
        for i, key in enumerate(langs):
            label = pl.LANGUAGES[key]["label"]
            base = 20 + int(75 * i / n)
            job.update(step=f"ترجمة وتكييف: {label}", progress=base)
            texts = pl.translate(segments, key, src_lang)

            job.update(step=f"توليد الصوت: {label}", progress=base + int(30 / n))
            voice = pl.LANGUAGES[key]["voices"][gender]
            dub = pl.build_dub_track(segments, texts, voice, folder / key, total)

            job.update(step=f"دمج الفيديو: {label}", progress=base + int(60 / n))
            out = folder / f"{key}.mp4"
            pl.mux(video, dub, out)
            job["outputs"][key] = {
                "label": label,
                "url": f"/files/{job_id}/{key}.mp4",
                "text": " ".join(texts),
            }
        job.update(status="done", step="اكتمل", progress=100)
    except Exception as e:  # noqa: BLE001
        job.update(status="error", step="فشل", error=str(e))


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    langs: str = Form(...),
    gender: str = Form("male"),
    consent: bool = Form(False),
):
    cleanup()
    if not consent:
        raise HTTPException(400, "يجب الإقرار بأنك تملك حقوق الفيديو والصوت")
    keys = [k for k in langs.split(",") if k in pl.LANGUAGES]
    if not keys:
        raise HTTPException(400, "اختر لغة واحدة على الأقل")
    if gender not in ("male", "female"):
        gender = "male"

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
    }
    threading.Thread(target=process, args=(job_id, video, keys, gender), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "المهمة غير موجودة أو انتهت صلاحيتها")
    return {k: v for k, v in job.items() if k != "created"}


@app.get("/api/languages")
def languages():
    return {k: v["label"] for k, v in pl.LANGUAGES.items()}


@app.get("/health")
def health():
    return {"ok": True}


app.mount("/files", StaticFiles(directory=DATA), name="files")


@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")
