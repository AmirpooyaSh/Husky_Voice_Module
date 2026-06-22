#!/usr/bin/env python3
import os
import time
import tempfile
import logging
from typing import Optional

import torch
import whisper
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("openai_whisper_cuda126_server")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base.en")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")
REQUEST_MODEL_NAME = os.getenv("REQUEST_MODEL_NAME", "whisper-1")
WHISPER_BEAM = int(os.getenv("WHISPER_BEAM", "1"))

_requested_device = os.getenv("WHISPER_DEVICE", "cuda").lower().strip()
if _requested_device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError(
        "WHISPER_DEVICE=cuda was requested, but torch.cuda.is_available() is False. "
        "Check: nvidia-smi, NVIDIA Container Toolkit, and docker run --gpus all."
    )

DEVICE = _requested_device if _requested_device in ("cuda", "cpu") else ("cuda" if torch.cuda.is_available() else "cpu")
FP16 = os.getenv("WHISPER_FP16", "true").lower() in ("1", "true", "yes", "y") and DEVICE == "cuda"

app = FastAPI(title="Local OpenAI Whisper API", version="1.0")
_model = None


@app.on_event("startup")
def load_model() -> None:
    global _model
    log.info("Loading OpenAI Whisper model '%s' on device=%s fp16=%s", WHISPER_MODEL, DEVICE, FP16)
    log.info("Torch CUDA available=%s torch CUDA version=%s", torch.cuda.is_available(), torch.version.cuda)
    _model = whisper.load_model(WHISPER_MODEL, device=DEVICE)
    log.info("Whisper model loaded successfully")


@app.get("/")
def root():
    return health()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "api_model": REQUEST_MODEL_NAME,
        "backend_model": WHISPER_MODEL,
        "device": DEVICE,
        "fp16": FP16,
        "language": WHISPER_LANGUAGE,
        "beam_size": WHISPER_BEAM,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {
                "id": REQUEST_MODEL_NAME,
                "object": "model",
                "owned_by": "local-openai-whisper",
                "backend_model": WHISPER_MODEL,
            }
        ],
    }


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(default="whisper-1"),
    language: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="json"),
):
    if _model is None:
        raise HTTPException(status_code=503, detail="Whisper model is not loaded yet")

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp_path = None

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        start = time.time()
        result = _model.transcribe(
            tmp_path,
            language=language or WHISPER_LANGUAGE,
            fp16=FP16,
            beam_size=WHISPER_BEAM,
            condition_on_previous_text=False,
            verbose=False,
        )
        elapsed = time.time() - start
        text = (result.get("text") or "").strip()

        log.info("Transcribed request_model=%s backend_model=%s elapsed=%.3fs text=%r", model, WHISPER_MODEL, elapsed, text)

        if (response_format or "json").lower() == "text":
            return PlainTextResponse(text)

        return {"text": text}

    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Transcription failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
