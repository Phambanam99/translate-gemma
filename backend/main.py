"""
FastAPI backend for CSV Arabic to Vietnamese translation.
Supports asynchronous CSV queueing with single-GPU execution.
"""
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Literal, Optional

import asyncio
import os
import uuid

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from gemma_translator import get_gemma_translator, get_supported_languages

# Load environment variables from .env file
try:
    from load_env import load_env
    load_env()
except ImportError:
    pass  # load_env.py is optional

app = FastAPI(title="CSV Translator API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage for translation jobs
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Job status tracking
jobs: Dict[str, dict] = {}
jobs_lock = Lock()

# CSV async queue (single worker for single GPU)
csv_job_queue: asyncio.Queue = asyncio.Queue()
queued_job_ids: List[str] = []
csv_worker_task: Optional[asyncio.Task] = None


def _get_env_int(name: str, default: int, minimum: int = 1) -> int:
    """Read integer env var safely with lower bound."""
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except (TypeError, ValueError):
        return default


CPU_WORKERS = _get_env_int("CSV_CPU_WORKERS", min(8, os.cpu_count() or 4))
CSV_PREPROCESS_CHUNK_SIZE = _get_env_int("CSV_PREPROCESS_CHUNK_SIZE", 1000, minimum=100)
CSV_GPU_BATCH_SIZE = _get_env_int("CSV_GPU_BATCH_SIZE", 20, minimum=1)

# Request/Response models
class TextTranslateRequest(BaseModel):
    text: str
    source_lang: str = "ar"
    target_lang: str = "vi"
    method: Literal["helsinki", "gemma"] = "gemma"

class ImageTranslateRequest(BaseModel):
    image: str  # base64 or URL
    source_lang: str = "ar"
    target_lang: str = "vi"

class TranslateResponse(BaseModel):
    translated_text: str
    method: str


def _set_job(job_id: str, payload: dict) -> None:
    with jobs_lock:
        jobs[job_id] = payload


def _update_job(job_id: str, **fields) -> None:
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(fields)


def _get_job(job_id: str) -> Optional[dict]:
    with jobs_lock:
        job = jobs.get(job_id)
        return dict(job) if job else None


def _update_queue_positions() -> None:
    for index, queued_job_id in enumerate(queued_job_ids, start=1):
        _update_job(queued_job_id, queue_position=index)


def _write_binary_file(path: str, content: bytes) -> None:
    with open(path, "wb") as file_obj:
        file_obj.write(content)


def _normalize_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _normalize_chunk(chunk: List[object]) -> List[str]:
    return [_normalize_cell(value) for value in chunk]


def _normalize_texts_parallel(values: List[object]) -> List[str]:
    if not values:
        return []

    if CPU_WORKERS <= 1 or len(values) < CSV_PREPROCESS_CHUNK_SIZE:
        return _normalize_chunk(values)

    chunks = [
        values[i : i + CSV_PREPROCESS_CHUNK_SIZE]
        for i in range(0, len(values), CSV_PREPROCESS_CHUNK_SIZE)
    ]
    max_workers = min(CPU_WORKERS, len(chunks))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        normalized_chunks = list(pool.map(_normalize_chunk, chunks))

    normalized: List[str] = []
    for chunk in normalized_chunks:
        normalized.extend(chunk)
    return normalized


def _sanitize_output_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if "\n" not in text and "\r" not in text:
        return text
    return text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def _sanitize_translated_texts(values: List[object]) -> List[str]:
    return [_sanitize_output_text(value) for value in values]


def _process_translation_sync(
    job_id: str,
    input_path: str,
    method: str = "gemma",
    source_lang: str = "ar",
    target_lang: str = "vi",
) -> None:
    """CPU/GPU-heavy CSV translation body executed in worker thread."""
    try:
        _update_job(job_id, message="Đang đọc file CSV...", progress=0)
        df = pd.read_csv(input_path, delimiter="|", encoding="utf-8")

        if "Text" not in df.columns:
            _update_job(
                job_id,
                status="error",
                error="Column 'Text' not found in CSV",
                message="Thiếu cột Text trong file CSV.",
            )
            return

        texts = _normalize_texts_parallel(df["Text"].tolist())
        total = len(texts)
        effective_method = "gemma"

        if method != "gemma":
            print("[Warning] Helsinki method requested but disabled. Using Gemma instead.")

        _update_job(
            job_id,
            total=total,
            method=effective_method,
            processed=0,
            message=f"Đang dịch {total} dòng với {effective_method.upper()}...",
        )

        def update_progress(progress: int, processed: int) -> None:
            processed_count = max(0, min(processed, total))
            _update_job(
                job_id,
                progress=max(0, min(progress, 100)),
                processed=processed_count,
                message=f"Đang dịch {processed_count}/{total} dòng với {effective_method.upper()}...",
            )

        translator = get_gemma_translator()
        translated = translator.translate_batch(
            texts,
            source_lang,
            target_lang,
            progress_callback=update_progress,
            batch_size=CSV_GPU_BATCH_SIZE,
        )

        df["Text"] = _sanitize_translated_texts(translated)
        output_path = os.path.join(OUTPUT_DIR, f"{job_id}_translated.csv")
        df.to_csv(output_path, sep="|", index=False, encoding="utf-8")

        _update_job(
            job_id,
            status="completed",
            progress=100,
            processed=total,
            message="Hoàn thành!",
            output_file=output_path,
            queue_position=0,
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="error",
            error=str(exc),
            message="Xử lý thất bại.",
        )


async def _csv_worker_loop() -> None:
    """Single CSV worker to guarantee only one active GPU CSV task."""
    while True:
        job_payload = await csv_job_queue.get()
        job_id = job_payload["job_id"]

        try:
            if job_id in queued_job_ids:
                queued_job_ids.remove(job_id)
                _update_queue_positions()

            _update_job(
                job_id,
                status="processing",
                progress=0,
                processed=0,
                queue_position=0,
                message="Đang chuẩn bị dữ liệu để dịch...",
            )

            await asyncio.to_thread(
                _process_translation_sync,
                job_id,
                job_payload["input_path"],
                job_payload["method"],
                job_payload["source_lang"],
                job_payload["target_lang"],
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _update_job(
                job_id,
                status="error",
                error=str(exc),
                message="Worker gặp lỗi không mong muốn.",
            )
        finally:
            csv_job_queue.task_done()

@app.on_event("startup")
async def startup_event():
    """Preload translation models on startup"""
    print("Server starting...")
    
    # Check offline mode
    import os
    offline_mode = os.environ.get("HF_HUB_OFFLINE", "false").lower() == "true" or \
                   os.environ.get("TRANSFORMERS_OFFLINE", "false").lower() == "true"
    
    if offline_mode:
        print("="*60)
        print("OFFLINE MODE ENABLED")
        print("Using local model cache only (no internet required)")
        print("="*60)
    
    # Load Helsinki-NLP (AR->EN->VI)
    # print("Helsinki-NLP: Loading...")
    # loop = asyncio.get_event_loop()
    # await loop.run_in_executor(None, get_translator)
    # print("Helsinki-NLP: Ready!")
    
    # Load TranslateGemma
    # NOTE:
    # - For gated models, you must be authenticated (HF_TOKEN) OR have all required files already cached locally.
    # - In offline mode, if cache is incomplete, preloading would crash the whole server.
    print("TranslateGemma 4B: Loading...")
    try:
        await asyncio.to_thread(lambda: get_gemma_translator()._ensure_loaded())
        print("TranslateGemma 4B: Ready!")
    except Exception as e:
        print("=" * 60)
        print("[Startup] WARNING: TranslateGemma failed to preload.")
        print(f"[Startup] Error: {e}")
        if offline_mode:
            print("[Startup] Offline mode is enabled: this usually means the local cache is incomplete for a gated model.")
            print("[Startup] Server will still start, but translation requests will fail until the model cache is fixed.")
        else:
            print("[Startup] If this is a gated model, set HF_TOKEN (Hugging Face access token) in environment.")
        print("=" * 60)

    global csv_worker_task
    if csv_worker_task is None or csv_worker_task.done():
        csv_worker_task = asyncio.create_task(_csv_worker_loop())
        print("[Queue] CSV worker started (GPU concurrency = 1)")


@app.on_event("shutdown")
async def shutdown_event():
    global csv_worker_task
    if csv_worker_task and not csv_worker_task.done():
        csv_worker_task.cancel()
        try:
            await csv_worker_task
        except asyncio.CancelledError:
            pass
    csv_worker_task = None

@app.get("/api/languages")
async def get_languages():
    """Get list of supported languages for TranslateGemma"""
    return {
        "languages": get_supported_languages(),
        "default_source": "ar",
        "default_target": "vi"
    }

@app.post("/api/translate-text", response_model=TranslateResponse)
async def translate_text(request: TextTranslateRequest):
    """Translate a single text string"""
    if request.method != "gemma":
        print(f"[Warning] Helsinki method requested but disabled. Using Gemma instead.")

    translator = get_gemma_translator()
    result = await asyncio.to_thread(
        translator.translate_text,
        request.text,
        request.source_lang,
        request.target_lang,
    )

    return TranslateResponse(translated_text=result, method="gemma")

@app.post("/api/translate-image", response_model=TranslateResponse)
async def translate_image(request: ImageTranslateRequest):
    """Extract and translate text from an image"""
    translator = get_gemma_translator()
    result = await asyncio.to_thread(
        translator.translate_image,
        request.image,
        request.source_lang,
        request.target_lang,
    )
    return TranslateResponse(translated_text=result, method="gemma")

@app.post("/api/upload")
async def upload_csv(
    file: UploadFile,
    method: Literal["helsinki", "gemma"] = Query(default="gemma"),
    source_lang: str = Query(default="ar"),
    target_lang: str = Query(default="vi")
):
    """Upload CSV file and start translation"""
    filename = file.filename or ""
    if not filename.lower().endswith(".csv"):
        raise HTTPException(400, "File must be a CSV")

    job_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}_input.csv")
    content = await file.read()
    await asyncio.to_thread(_write_binary_file, input_path, content)

    job_state = {
        "status": "queued",
        "progress": 0,
        "processed": 0,
        "total": 0,
        "message": "Đã nhận file, đang chờ đến lượt xử lý GPU...",
        "output_file": None,
        "error": None,
        "method": "gemma",
        "queue_position": 0,
        "source_lang": source_lang,
        "target_lang": target_lang,
    }

    _set_job(job_id, job_state)
    queued_job_ids.append(job_id)
    _update_queue_positions()

    await csv_job_queue.put(
        {
            "job_id": job_id,
            "input_path": input_path,
            "method": method,
            "source_lang": source_lang,
            "target_lang": target_lang,
        }
    )

    job_snapshot = _get_job(job_id) or {}
    queue_position = job_snapshot.get("queue_position", 0)
    return {
        "job_id": job_id,
        "message": "Translation queued",
        "method": "gemma",
        "queue_position": queue_position,
    }

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get translation job status"""
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job

@app.get("/api/download/{job_id}")
async def download_file(job_id: str):
    """Download translated CSV"""
    job = _get_job(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    if job["status"] != "completed":
        raise HTTPException(400, "Translation not completed")

    if not job["output_file"] or not os.path.exists(job["output_file"]):
        raise HTTPException(404, "Output file not found")

    return FileResponse(
        job["output_file"],
        media_type="text/csv",
        filename="translated.csv"
    )

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "methods": ["helsinki", "gemma"],
        "queue_depth": csv_job_queue.qsize(),
        "cpu_workers": CPU_WORKERS,
        "gpu_concurrency": 1,
        "csv_gpu_batch_size": CSV_GPU_BATCH_SIZE,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

