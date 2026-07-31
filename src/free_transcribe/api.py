"""Optional HTTP API over the same local transcription core."""

import asyncio
import hmac
import importlib.util
import json
import os
import platform
import re
import shutil
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .core import (
    AVAILABLE_ENGINES,
    DEFAULT_ENGINE,
    SUPPORTED_FORMATS,
    save_transcript,
    transcribe_file,
)

WEB_ROOT = Path(__file__).with_name("web")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _runtime_capabilities() -> dict[str, Any]:
    """Report lightweight backend readiness without loading model frameworks."""
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    qwen_module = "mlx_qwen3_asr" if apple_silicon else "qwen_asr"
    parakeet_module = "mlx_audio" if apple_silicon else "nemo"
    engines = {
        "qwen": _module_available(qwen_module),
        "parakeet": _module_available(parakeet_module),
    }
    return {
        "engines": engines,
        "speakers": _module_available("pyannote.audio"),
    }


@dataclass
class _Job:
    id: str
    work_dir: Path
    source_path: Path
    status: str = "queued"
    stage: str = "queued"
    message: str = "Waiting for an inference slot"
    created_at: str = ""
    started_at: str | None = None
    completed_at: str | None = None
    result_path: Path | None = None
    error: str | None = None
    progress_percent: float | None = None

    def public(self) -> dict[str, Any]:
        progress: dict[str, Any] = {"stage": self.stage, "message": self.message}
        if self.progress_percent is not None:
            progress["percent"] = self.progress_percent
        payload: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "progress": progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }
        if self.status == "succeeded":
            payload["result_url"] = f"/v1/transcriptions/{self.id}/result"
        if self.error:
            payload["error"] = self.error
        return payload


def _bearer_token(request: Request) -> str:
    scheme, _, value = request.headers.get("authorization", "").partition(" ")
    return value if scheme.casefold() == "bearer" else ""


def create_app(
    *,
    token: str | None = None,
    concurrency: int | None = None,
    max_upload_mb: int | None = None,
    max_queue: int | None = None,
    require_cuda: bool | None = None,
) -> FastAPI:
    """Create a self-contained API app; model dependencies remain lazy."""
    configured_token = token if token is not None else os.getenv("FT_API_TOKEN", "")
    worker_count = (
        concurrency
        if concurrency is not None
        else int(os.getenv("FT_API_CONCURRENCY", "1"))
    )
    upload_mb = (
        max_upload_mb
        if max_upload_mb is not None
        else int(os.getenv("FT_MAX_UPLOAD_MB", "4096"))
    )
    upload_limit = upload_mb * 1024 * 1024
    queue_limit = (
        max_queue
        if max_queue is not None
        else int(os.getenv("FT_API_MAX_QUEUE", "20"))
    )
    cuda_required = (
        require_cuda
        if require_cuda is not None
        else os.getenv("FT_REQUIRE_CUDA", "").casefold() in {"1", "true", "yes"}
    )
    if worker_count < 1:
        raise ValueError("API concurrency must be at least 1")
    if upload_limit < 1:
        raise ValueError("upload limit must be positive")
    if queue_limit < 0:
        raise ValueError("queue limit cannot be negative")

    jobs: dict[str, _Job] = {}
    tasks: dict[str, asyncio.Task[None]] = {}
    subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
    inference_slots = asyncio.Semaphore(worker_count)
    admission_lock = asyncio.Lock()
    uploads_in_progress = 0

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if cuda_required:
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("FT_REQUIRE_CUDA is set but PyTorch is missing") from exc
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "FT_REQUIRE_CUDA is set but no NVIDIA CUDA device is available"
                )
        yield
        for task in tasks.values():
            task.cancel()
        for job in jobs.values():
            shutil.rmtree(job.work_dir, ignore_errors=True)

    app = FastAPI(
        title="Free Transcribe",
        version=__version__,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    app.mount("/assets", StaticFiles(directory=WEB_ROOT), name="web-assets")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.url.path == "/":
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; style-src 'self'; script-src 'self'; "
                "connect-src 'self'; img-src 'self' blob:"
            )
        return response

    async def authorize(request: Request) -> None:
        if configured_token and not hmac.compare_digest(
            _bearer_token(request), configured_token
        ):
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    def find_job(job_id: str) -> _Job:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Transcription not found")
        return job

    def public_job(job: _Job) -> dict[str, Any]:
        payload = job.public()
        if job.status == "queued":
            waiting = [candidate for candidate in jobs.values() if candidate.status == "queued"]
            position = waiting.index(job) + 1
            payload["queue_position"] = position
            payload["progress"] = {
                "stage": "queued",
                "message": f"Queued · position {position}",
            }
        return payload

    def publish(job: _Job) -> None:
        payload = public_job(job)
        for queue in subscribers.get(job.id, set()):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(payload)

    def publish_waiting() -> None:
        for candidate in jobs.values():
            if candidate.status == "queued":
                publish(candidate)

    async def execute(
        job: _Job,
        *,
        engine: str,
        model: str | None,
        language: str | None,
        prompt: str | None,
        speakers: bool,
        speaker_count: int | None,
    ) -> None:
        async with inference_slots:
            job.status = "running"
            job.started_at = _now()
            publish(job)
            publish_waiting()
            loop = asyncio.get_running_loop()

            def progress(stage: str, message: str) -> None:
                loop.call_soon_threadsafe(update_progress, stage, message)

            def update_progress(stage: str, message: str) -> None:
                job.stage = stage
                job.message = message
                job.progress_percent = None
                match = re.search(r"\b(\d{1,3}(?:\.\d+)?)%", message)
                if match:
                    job.progress_percent = min(100.0, float(match.group(1)))
                publish(job)

            try:
                result = await asyncio.to_thread(
                    transcribe_file,
                    str(job.source_path),
                    model_name=model,
                    language=language,
                    prompt=prompt,
                    on_progress=progress,
                    engine=engine,
                    diarize=speakers,
                    num_speakers=speaker_count,
                )
                result_path = job.work_dir / "transcript.md"
                await asyncio.to_thread(
                    save_transcript,
                    result,
                    str(job.source_path),
                    str(result_path),
                )
                job.result_path = result_path
                job.status = "succeeded"
                job.stage = "complete"
                job.message = "Transcription complete"
                job.progress_percent = 100.0
                publish(job)
            # This is the boundary of a background job: model/framework errors
            # must become observable job failures instead of orphaned tasks.
            except Exception as exc:  # noqa: BLE001
                job.status = "failed"
                job.stage = "failed"
                job.message = "Transcription failed"
                job.error = str(exc)
                publish(job)
            finally:
                job.completed_at = _now()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        queued = sum(job.status == "queued" for job in jobs.values())
        running = sum(job.status == "running" for job in jobs.values())
        return {
            "status": "ok",
            "service": "free-transcribe",
            "version": __version__,
            "authentication": bool(configured_token),
            "concurrency": worker_count,
            "cuda_required": cuda_required,
            "ready": _runtime_capabilities(),
            "queue": {
                "queued": queued,
                "running": running,
                "max_waiting": queue_limit,
            },
        }

    @app.get("/", include_in_schema=False, response_class=FileResponse)
    async def web_ui() -> FileResponse:
        return FileResponse(WEB_ROOT / "index.html", media_type="text/html")

    @app.post(
        "/v1/transcriptions",
        status_code=202,
        dependencies=[Depends(authorize)],
    )
    async def submit(
        file: Annotated[UploadFile, File()],
        engine: Annotated[str, Form()] = DEFAULT_ENGINE,
        model: Annotated[str | None, Form()] = None,
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str | None, Form()] = None,
        speakers: Annotated[bool, Form()] = False,
        speaker_count: Annotated[int | None, Form()] = None,
    ) -> dict[str, Any]:
        nonlocal uploads_in_progress
        engine = engine.casefold()
        if engine not in AVAILABLE_ENGINES:
            raise HTTPException(
                status_code=422,
                detail=f"engine must be one of: {', '.join(AVAILABLE_ENGINES)}",
            )
        if speaker_count is not None and speaker_count < 1:
            raise HTTPException(status_code=422, detail="speaker_count must be positive")

        suffix = Path(file.filename or "").suffix.casefold()
        if suffix not in SUPPORTED_FORMATS:
            raise HTTPException(status_code=415, detail="Unsupported media format")

        async with admission_lock:
            active = sum(
                job.status in {"queued", "running"} for job in jobs.values()
            )
            if active + uploads_in_progress >= worker_count + queue_limit:
                await file.close()
                raise HTTPException(
                    status_code=429,
                    detail="Transcription queue is full",
                    headers={"Retry-After": "30"},
                )
            uploads_in_progress += 1

        job_id = uuid.uuid4().hex
        work_dir = Path(tempfile.mkdtemp(prefix=f"free-transcribe-{job_id}-"))
        source_path = work_dir / f"source{suffix}"
        size = 0
        try:
            with source_path.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > upload_limit:
                        raise HTTPException(status_code=413, detail="Upload is too large")
                    output.write(chunk)
        except Exception:
            shutil.rmtree(work_dir, ignore_errors=True)
            async with admission_lock:
                uploads_in_progress -= 1
            raise
        finally:
            await file.close()
        if size == 0:
            shutil.rmtree(work_dir, ignore_errors=True)
            async with admission_lock:
                uploads_in_progress -= 1
            raise HTTPException(status_code=422, detail="Upload is empty")

        job = _Job(
            id=job_id,
            work_dir=work_dir,
            source_path=source_path,
            created_at=_now(),
        )
        jobs[job_id] = job
        async with admission_lock:
            uploads_in_progress -= 1
        task = asyncio.create_task(
            execute(
                job,
                engine=engine,
                model=model,
                language=language,
                prompt=prompt,
                speakers=speakers or speaker_count is not None,
                speaker_count=speaker_count,
            )
        )
        tasks[job_id] = task
        task.add_done_callback(lambda _task: tasks.pop(job_id, None))
        return public_job(job)

    @app.get("/v1/transcriptions/{job_id}", dependencies=[Depends(authorize)])
    async def status(job_id: str) -> dict[str, Any]:
        return public_job(find_job(job_id))

    @app.get(
        "/v1/transcriptions/{job_id}/events",
        response_class=StreamingResponse,
        dependencies=[Depends(authorize)],
    )
    async def events(job_id: str, request: Request) -> StreamingResponse:
        job = find_job(job_id)
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=4)
        subscribers.setdefault(job_id, set()).add(queue)

        async def stream() -> AsyncIterator[str]:
            payload = public_job(job)
            try:
                while True:
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    if payload["status"] in {"succeeded", "failed"}:
                        break
                    if await request.is_disconnected():
                        break
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=15)
                    except TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                listeners = subscribers.get(job_id)
                if listeners is not None:
                    listeners.discard(queue)
                    if not listeners:
                        subscribers.pop(job_id, None)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get(
        "/v1/transcriptions/{job_id}/result",
        response_class=FileResponse,
        dependencies=[Depends(authorize)],
    )
    async def result(job_id: str) -> FileResponse:
        job = find_job(job_id)
        if job.status != "succeeded" or job.result_path is None:
            raise HTTPException(status_code=409, detail=f"Job is {job.status}")
        return FileResponse(
            job.result_path,
            media_type="text/markdown; charset=utf-8",
            filename="transcript.md",
        )

    @app.delete(
        "/v1/transcriptions/{job_id}",
        status_code=204,
        dependencies=[Depends(authorize)],
    )
    async def delete(job_id: str) -> None:
        job = find_job(job_id)
        if job.status in {"queued", "running"}:
            raise HTTPException(status_code=409, detail=f"Job is {job.status}")
        jobs.pop(job_id)
        shutil.rmtree(job.work_dir, ignore_errors=True)

    return app


def run(*, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the HTTP API with uvicorn."""
    if host not in {"127.0.0.1", "localhost", "::1"} and not os.getenv(
        "FT_API_TOKEN"
    ):
        raise RuntimeError(
            "FT_API_TOKEN is required when binding the API beyond localhost"
        )
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install the 'api' extra to run the HTTP server") from exc
    uvicorn.run(create_app(), host=host, port=port)


def main() -> None:
    """Standalone `free-transcribe-api` entry point."""
    run(
        host=os.getenv("FT_API_HOST", "127.0.0.1"),
        port=int(os.getenv("FT_API_PORT", "8000")),
    )
