from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import wave
from pathlib import Path
from typing import Any

import sounddevice as sd
import websockets

DEFAULT_URL = os.getenv("FISH_WS_URL", "ws://127.0.0.1:8000/ws/tts")


def validate_wav(path: Path) -> dict[str, int]:
    if not path.exists():
        raise RuntimeError(f"Output file does not exist: {path}")
    size = path.stat().st_size
    if size < 44:
        raise RuntimeError(f"Output is too small to be WAV audio: {size} bytes")
    with path.open("rb") as file:
        header = file.read(12)
    if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise RuntimeError(f"Invalid WAV signature: {header!r}")
    try:
        with wave.open(str(path), "rb") as wav_file:
            return {
                "channels": wav_file.getnchannels(),
                "sample_width": wav_file.getsampwidth(),
                "sample_rate": wav_file.getframerate(),
                "frames": wav_file.getnframes(),
            }
    except wave.Error as exc:
        raise RuntimeError(f"Saved output is not readable WAV: {exc}") from exc


def play_wav(path: Path) -> None:
    info = validate_wav(path)
    if info["sample_width"] != 2:
        raise RuntimeError(f"Playback requires PCM16 WAV; received sample_width={info['sample_width']}.")
    with wave.open(str(path), "rb") as wav_file:
        pcm = wav_file.readframes(wav_file.getnframes())
    if not pcm:
        raise RuntimeError("WAV contains no audio frames.")
    print(f"[playback] sample_rate={info['sample_rate']}, channels={info['channels']}, frames={info['frames']}")
    stream = sd.RawOutputStream(samplerate=info["sample_rate"], channels=info["channels"], dtype="int16", latency="low")
    try:
        stream.start()
        stream.write(pcm)
        stream.stop()
    finally:
        stream.close()


def print_metric(label: str, value: float | int | None) -> None:
    print(f"{label:<30}: {'N/A' if value is None else f'{float(value):.2f} ms'}")


async def run_client(args: argparse.Namespace) -> None:
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "text": args.text,
        "reference_id": args.reference_id,
        "chunk_length": args.chunk_length,
        "format": "wav",
        "seed": args.seed,
        "use_memory_cache": args.memory_cache,
        "normalize": not args.no_normalize,
        "streaming": False,
        "max_new_tokens": args.max_new_tokens,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "temperature": args.temperature,
    }

    process_started_at = time.perf_counter()
    connected_at = request_sent_at = first_binary_at = done_at = None
    bytes_received = chunks_received = 0
    completed = False
    server_metrics: dict[str, Any] = {}
    if output.exists():
        output.unlink()

    print(f"[connect] {args.url}")
    print(f"[output] {output}")
    try:
        async with websockets.connect(
            args.url,
            open_timeout=args.connect_timeout,
            ping_interval=20,
            ping_timeout=None,
            close_timeout=15,
            max_size=None,
        ) as websocket:
            connected_at = time.perf_counter()
            print(f"[connection-latency] {(connected_at - process_started_at) * 1000:.2f} ms")
            await websocket.send(json.dumps(payload))
            request_sent_at = time.perf_counter()
            with output.open("wb") as file:
                async for message in websocket:
                    if isinstance(message, bytes):
                        if first_binary_at is None:
                            first_binary_at = time.perf_counter()
                            print(f"[client-ttfa] {(first_binary_at - request_sent_at) * 1000:.2f} ms")
                        file.write(message)
                        bytes_received += len(message)
                        chunks_received += 1
                        continue
                    event = json.loads(message)
                    event_type = event.get("type", "unknown")
                    print(f"[{event_type}] {json.dumps(event, ensure_ascii=False)}")
                    if event_type == "ttfa":
                        server_metrics.update(event)
                    elif event_type == "done":
                        server_metrics.update(event)
                        done_at = time.perf_counter()
                        completed = True
                    elif event_type == "error":
                        raise RuntimeError(event.get("message", "Unknown server error"))
    except Exception:
        if output.exists() and output.stat().st_size == 0:
            output.unlink()
        raise

    if not completed or request_sent_at is None:
        raise RuntimeError("WebSocket closed before the done event was received.")
    if done_at is None:
        done_at = time.perf_counter()
    wav_info = validate_wav(output)

    print("\n" + "=" * 62 + "\nLATENCY SUMMARY\n" + "=" * 62)
    print_metric("Connection latency", (connected_at - process_started_at) * 1000 if connected_at else None)
    print_metric("Client TTFA", (first_binary_at - request_sent_at) * 1000 if first_binary_at else None)
    print_metric("End-to-end TTFA", (first_binary_at - process_started_at) * 1000 if first_binary_at else None)
    print_metric("Server TTFA", server_metrics.get("server_ttfa_ms"))
    print_metric("Inference first result", server_metrics.get("inference_first_result_ms"))
    print_metric("Inference latency", server_metrics.get("inference_latency_ms"))
    print_metric("WAV encoding latency", server_metrics.get("wav_encoding_latency_ms"))
    print_metric("Generation latency", server_metrics.get("generation_latency_ms"))
    print_metric("Client total latency", (done_at - request_sent_at) * 1000)
    print_metric("End-to-end latency", (done_at - process_started_at) * 1000)
    print_metric("Server total latency", server_metrics.get("server_total_latency_ms"))
    print("=" * 62)
    print(f"Saved WAV          : {output}")
    print(f"File size          : {output.stat().st_size:,} bytes")
    print(f"Received bytes     : {bytes_received:,}")
    print(f"Received chunks    : {chunks_received}")
    print(f"Sample rate        : {wav_info['sample_rate']} Hz")
    print(f"Channels           : {wav_info['channels']}")
    print(f"Frames             : {wav_info['frames']}")
    print("=" * 62)

    if args.play:
        await asyncio.to_thread(play_wav, output)
        print("[playback] completed")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fish Speech 1.5 WebSocket client")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--text", default=None)
    parser.add_argument("--output", default="fish-speech-15.wav")
    parser.add_argument("--reference-id", default=None)
    parser.add_argument("--chunk-length", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--memory-cache", choices=["on", "off"], default="off")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--connect-timeout", type=float, default=120)
    playback = parser.add_mutually_exclusive_group()
    playback.add_argument("--play", dest="play", action="store_true")
    playback.add_argument("--no-play", dest="play", action="store_false")
    parser.set_defaults(play=True)
    args = parser.parse_args()
    if not args.text:
        args.text = input("Text: ").strip()
    if not args.text:
        parser.error("Text cannot be empty.")
    return args


def main() -> None:
    try:
        asyncio.run(run_client(parse_arguments()))
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130)
    except sd.PortAudioError as exc:
        print(f"[audio-device-error] {exc}", file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
