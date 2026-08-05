from __future__ import annotations

import asyncio
import io
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response
from loguru import logger
from pydantic import ValidationError

from fish_speech.utils.schema import ServeTTSRequest
from tools.server.model_manager import ModelManager

DEVICE = os.getenv("DEVICE", "cuda").strip()
HALF = os.getenv("HALF", "1").strip().lower() in {"1", "true", "yes", "on"}
COMPILE = os.getenv("COMPILE", "0").strip().lower() in {"1", "true", "yes", "on"}
MODEL_PATH = os.getenv("LLAMA_CHECKPOINT_PATH", "/app/checkpoints/fish-speech-1.5")
DECODER_PATH = os.getenv(
    "DECODER_CHECKPOINT_PATH",
    "/app/checkpoints/fish-speech-1.5/firefly-gan-vq-fsq-8x1024-21hz-generator.pth",
)
DECODER_CONFIG = os.getenv("DECODER_CONFIG_NAME", "firefly_gan_vq")
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "1000"))
WS_CHUNK_SIZE = int(os.getenv("WS_CHUNK_SIZE", str(64 * 1024)))


class ServerState:
    model_manager: ModelManager | None = None
    ready: bool = False
    startup_error: str | None = None
    model_load_ms: float | None = None


state = ServerState()
inference_lock = threading.Lock()


def validate_environment() -> None:
    if not DEVICE.startswith("cuda"):
        raise RuntimeError(f"This project is configured for GPU inference, but DEVICE={DEVICE!r}")


def validate_request(request: ServeTTSRequest) -> None:
    text = request.text.strip()
    if not text:
        raise ValueError("Text cannot be empty.")
    if MAX_TEXT_LENGTH > 0 and len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Text exceeds MAX_TEXT_LENGTH={MAX_TEXT_LENGTH}.")
    if request.chunk_length < 100 or request.chunk_length > 300:
        raise ValueError("chunk_length must be between 100 and 300.")


def get_engine() -> Any:
    if not state.ready or state.model_manager is None:
        raise RuntimeError(state.startup_error or "Fish Speech model is not ready.")
    return state.model_manager.tts_inference_engine


def get_sample_rate(engine: Any) -> int:
    decoder = engine.decoder_model
    if hasattr(decoder, "spec_transform"):
        return int(decoder.spec_transform.sample_rate)
    if hasattr(decoder, "sample_rate"):
        return int(decoder.sample_rate)
    raise RuntimeError("Could not determine decoder sample rate.")


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    output = np.asarray(audio, dtype=np.float32)
    output = np.squeeze(output)
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
    output = np.clip(output, -1.0, 1.0)
    if output.ndim != 1:
        raise RuntimeError(f"Expected mono audio, received shape={output.shape}.")
    return output


def encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    output = normalize_audio(audio)
    buffer = io.BytesIO()
    sf.write(buffer, output, sample_rate, format="WAV", subtype="PCM_16")
    wav_bytes = buffer.getvalue()
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise RuntimeError("Generated output is not a valid WAV.")
    return wav_bytes


def generate_complete_wav(request: ServeTTSRequest) -> tuple[bytes, dict[str, float | int]]:
    engine = get_engine()
    inference_request = request.model_copy(update={"streaming": False, "format": "wav"})
    inference_started_at = time.perf_counter()
    first_result_at: float | None = None
    final_audio: np.ndarray | None = None
    collected_segments: list[np.ndarray] = []
    sample_rate = get_sample_rate(engine)

    with inference_lock:
        for result in engine.inference(inference_request):
            now = time.perf_counter()
            if first_result_at is None:
                first_result_at = now
            if result.code == "error":
                raise RuntimeError(str(result.error))
            if result.code == "segment" and isinstance(result.audio, tuple):
                collected_segments.append(np.asarray(result.audio[1]))
            if result.code == "final" and isinstance(result.audio, tuple):
                result_sample_rate, result_audio = result.audio
                if result_sample_rate:
                    sample_rate = int(result_sample_rate)
                final_audio = np.asarray(result_audio)
                break

    inference_finished_at = time.perf_counter()
    if final_audio is None:
        if not collected_segments:
            raise RuntimeError("Fish Speech generated no audio.")
        final_audio = np.concatenate(collected_segments)

    encoding_started_at = time.perf_counter()
    wav_bytes = encode_wav(final_audio, sample_rate)
    encoding_finished_at = time.perf_counter()

    return wav_bytes, {
        "sample_rate": sample_rate,
        "inference_first_result_ms": round(((first_result_at or inference_finished_at) - inference_started_at) * 1000, 2),
        "inference_latency_ms": round((inference_finished_at - inference_started_at) * 1000, 2),
        "wav_encoding_latency_ms": round((encoding_finished_at - encoding_started_at) * 1000, 2),
        "audio_bytes": len(wav_bytes),
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    validate_environment()
    started = time.perf_counter()
    logger.info("Loading Fish Speech 1.5 from {}", MODEL_PATH)
    try:
        manager = await asyncio.to_thread(
            ModelManager,
            mode="tts",
            device=DEVICE,
            half=HALF,
            compile=COMPILE,
            asr_enabled=False,
            llama_checkpoint_path=MODEL_PATH,
            decoder_checkpoint_path=DECODER_PATH,
            decoder_config_name=DECODER_CONFIG,
        )
        state.model_manager = manager
        state.ready = True
        state.startup_error = None
        state.model_load_ms = round((time.perf_counter() - started) * 1000, 2)
        logger.info("Fish Speech 1.5 loaded in {} ms", state.model_load_ms)
    except Exception as exc:
        state.ready = False
        state.startup_error = str(exc)
        logger.exception("Fish Speech startup failed: {}", exc)
        raise
    yield
    state.ready = False


app = FastAPI(title="Fish Speech 1.5 HTTP and WebSocket Server", version="1.0.0", lifespan=lifespan)


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "service": "Fish Speech 1.5",
        "device": DEVICE,
        "http_tts": "/v1/tts",
        "websocket_tts": "/ws/tts",
        "health": "/health",
        "docs": "/docs",
    }


@app.get("/health")
@app.get("/v1/health")
async def health() -> JSONResponse:
    payload = {
        "status": "ok" if state.ready else "loading",
        "ready": state.ready,
        "model": "fishaudio/fish-speech-1.5",
        "device": DEVICE,
        "half": HALF,
        "compile": COMPILE,
        "model_load_ms": state.model_load_ms,
        "http_tts": "/v1/tts",
        "websocket_tts": "/ws/tts",
        "port": 8000,
        "startup_error": state.startup_error,
    }
    return JSONResponse(payload, status_code=200 if state.ready else 503)


@app.post("/v1/tts")
async def http_tts(request: ServeTTSRequest) -> Response:
    request_started_at = time.perf_counter()
    try:
        validate_request(request)
        if not state.ready:
            raise HTTPException(status_code=503, detail=state.startup_error or "Model is loading.")
        wav_bytes, metrics = await asyncio.to_thread(generate_complete_wav, request)
        total_latency_ms = round((time.perf_counter() - request_started_at) * 1000, 2)
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "Content-Disposition": 'attachment; filename="fish-speech-15.wav"',
                "X-Total-Latency-Ms": str(total_latency_ms),
                "X-Inference-First-Result-Ms": str(metrics["inference_first_result_ms"]),
                "X-Inference-Latency-Ms": str(metrics["inference_latency_ms"]),
                "X-Wav-Encoding-Latency-Ms": str(metrics["wav_encoding_latency_ms"]),
                "X-Audio-Sample-Rate": str(metrics["sample_rate"]),
                "X-Audio-Bytes": str(metrics["audio_bytes"]),
                "Cache-Control": "no-store",
            },
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("HTTP TTS failed: {}", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket) -> None:
    connection_started_at = time.perf_counter()
    await websocket.accept()
    request_id = str(uuid.uuid4())
    try:
        if not state.ready:
            await websocket.send_json({"type": "error", "request_id": request_id, "message": state.startup_error or "Model is loading."})
            await websocket.close(code=1013)
            return

        payload = await websocket.receive_json()
        request_received_at = time.perf_counter()
        payload.pop("latency", None)
        payload["streaming"] = False
        payload["format"] = "wav"
        try:
            request = ServeTTSRequest.model_validate(payload)
            validate_request(request)
        except (ValidationError, ValueError) as exc:
            await websocket.send_json({"type": "error", "request_id": request_id, "message": str(exc)})
            await websocket.close(code=1008)
            return

        await websocket.send_json({
            "type": "accepted",
            "request_id": request_id,
            "device": DEVICE,
            "model": "fishaudio/fish-speech-1.5",
            "format": "wav",
            "delivery": "complete-wav-chunked",
        })

        generation_started_at = time.perf_counter()
        wav_bytes, metrics = await asyncio.to_thread(generate_complete_wav, request)
        generation_finished_at = time.perf_counter()
        first_binary_sent_at: float | None = None
        chunk_count = 0
        byte_count = 0

        for offset in range(0, len(wav_bytes), WS_CHUNK_SIZE):
            chunk = wav_bytes[offset : offset + WS_CHUNK_SIZE]
            if first_binary_sent_at is None:
                first_binary_sent_at = time.perf_counter()
                await websocket.send_json({
                    "type": "ttfa",
                    "request_id": request_id,
                    "server_ttfa_ms": round((first_binary_sent_at - request_received_at) * 1000, 2),
                    "inference_first_result_ms": metrics["inference_first_result_ms"],
                    "inference_latency_ms": metrics["inference_latency_ms"],
                })
            await websocket.send_bytes(chunk)
            chunk_count += 1
            byte_count += len(chunk)

        finished_at = time.perf_counter()
        await websocket.send_json({
            "type": "done",
            "request_id": request_id,
            "server_ttfa_ms": round((first_binary_sent_at - request_received_at) * 1000, 2) if first_binary_sent_at else None,
            "inference_first_result_ms": metrics["inference_first_result_ms"],
            "inference_latency_ms": metrics["inference_latency_ms"],
            "wav_encoding_latency_ms": metrics["wav_encoding_latency_ms"],
            "generation_latency_ms": round((generation_finished_at - generation_started_at) * 1000, 2),
            "server_total_latency_ms": round((finished_at - request_received_at) * 1000, 2),
            "connection_to_done_ms": round((finished_at - connection_started_at) * 1000, 2),
            "sample_rate": metrics["sample_rate"],
            "chunks": chunk_count,
            "bytes": byte_count,
        })
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        logger.warning("Client disconnected: request_id={}", request_id)
    except Exception as exc:
        logger.exception("WebSocket TTS failed: request_id={}, error={}", request_id, exc)
        try:
            await websocket.send_json({"type": "error", "request_id": request_id, "message": str(exc)})
            await websocket.close(code=1011)
        except Exception:
            pass
