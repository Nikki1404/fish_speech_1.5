from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave
from pathlib import Path
from typing import Any

import sounddevice as sd
import websockets


# Change only this line when using a different server.
SERVER_URL = "ws://127.0.0.1:8000/ws/tts"

# Example remote server:
# SERVER_URL = "ws://10.90.126.61:8000/ws/tts"


def validate_wav(path: Path) -> dict[str, int]:
    if not path.exists():
        raise RuntimeError(f"Output file does not exist: {path}")

    file_size = path.stat().st_size

    if file_size < 44:
        raise RuntimeError(
            f"Output is too small to be valid WAV audio: {file_size} bytes"
        )

    with path.open("rb") as file:
        header = file.read(12)

    if header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise RuntimeError(
            f"Invalid WAV header. First 12 bytes: {header!r}"
        )

    try:
        with wave.open(str(path), "rb") as wav_file:
            return {
                "channels": wav_file.getnchannels(),
                "sample_width": wav_file.getsampwidth(),
                "sample_rate": wav_file.getframerate(),
                "frames": wav_file.getnframes(),
            }

    except wave.Error as exc:
        raise RuntimeError(
            f"Saved file is not a readable WAV file: {exc}"
        ) from exc


def play_wav(path: Path) -> None:
    wav_info = validate_wav(path)

    if wav_info["sample_width"] != 2:
        raise RuntimeError(
            "Playback currently supports only PCM16 WAV files. "
            f"Received sample width: {wav_info['sample_width']} bytes"
        )

    with wave.open(str(path), "rb") as wav_file:
        pcm_audio = wav_file.readframes(wav_file.getnframes())

    if not pcm_audio:
        raise RuntimeError("The WAV file contains no audio frames.")

    print(
        "[playback] "
        f"sample_rate={wav_info['sample_rate']} Hz, "
        f"channels={wav_info['channels']}, "
        f"frames={wav_info['frames']}"
    )

    stream = sd.RawOutputStream(
        samplerate=wav_info["sample_rate"],
        channels=wav_info["channels"],
        dtype="int16",
        latency="low",
    )

    try:
        stream.start()
        stream.write(pcm_audio)
        stream.stop()
    finally:
        stream.close()


def print_metric(
    label: str,
    value: float | int | None,
) -> None:
    if value is None:
        print(f"{label:<30}: N/A")
    else:
        print(f"{label:<30}: {float(value):.2f} ms")


async def run_client(args: argparse.Namespace) -> None:
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    connected_at: float | None = None
    request_sent_at: float | None = None
    first_audio_at: float | None = None
    done_received_at: float | None = None

    bytes_received = 0
    chunks_received = 0
    completed = False

    server_metrics: dict[str, Any] = {}

    if output_path.exists():
        output_path.unlink()

    print(f"[connect] {SERVER_URL}")
    print(f"[output] {output_path}")
    print(f"[playback] {'enabled' if args.play else 'disabled'}")

    try:
        async with websockets.connect(
            SERVER_URL,
            open_timeout=args.connect_timeout,
            ping_interval=20,
            ping_timeout=None,
            close_timeout=15,
            max_size=None,
        ) as websocket:
            connected_at = time.perf_counter()

            connection_latency_ms = (
                connected_at - process_started_at
            ) * 1000

            print(
                f"[connection-latency] "
                f"{connection_latency_ms:.2f} ms"
            )

            await websocket.send(json.dumps(payload))
            request_sent_at = time.perf_counter()

            with output_path.open("wb") as output_file:
                async for message in websocket:
                    if isinstance(message, bytes):
                        if first_audio_at is None:
                            first_audio_at = time.perf_counter()

                            client_ttfa_ms = (
                                first_audio_at - request_sent_at
                            ) * 1000

                            print(
                                f"[client-ttfa] "
                                f"{client_ttfa_ms:.2f} ms"
                            )

                        output_file.write(message)
                        output_file.flush()

                        bytes_received += len(message)
                        chunks_received += 1
                        continue

                    event = json.loads(message)
                    event_type = event.get("type", "unknown")

                    print(
                        f"[{event_type}] "
                        f"{json.dumps(event, ensure_ascii=False)}"
                    )

                    if event_type == "ttfa":
                        server_metrics.update(event)

                    elif event_type == "done":
                        server_metrics.update(event)
                        done_received_at = time.perf_counter()
                        completed = True

                    elif event_type == "error":
                        raise RuntimeError(
                            event.get(
                                "message",
                                "Unknown server error",
                            )
                        )

    except Exception:
        if output_path.exists() and output_path.stat().st_size == 0:
            output_path.unlink()

        raise

    if not completed:
        raise RuntimeError(
            "WebSocket closed before receiving the done event."
        )

    if request_sent_at is None:
        raise RuntimeError("Request timing was not initialized.")

    if done_received_at is None:
        done_received_at = time.perf_counter()

    wav_info = validate_wav(output_path)

    connection_latency_ms = (
        (connected_at - process_started_at) * 1000
        if connected_at is not None
        else None
    )

    client_ttfa_ms = (
        (first_audio_at - request_sent_at) * 1000
        if first_audio_at is not None
        else None
    )

    end_to_end_ttfa_ms = (
        (first_audio_at - process_started_at) * 1000
        if first_audio_at is not None
        else None
    )

    client_total_latency_ms = (
        done_received_at - request_sent_at
    ) * 1000

    end_to_end_latency_ms = (
        done_received_at - process_started_at
    ) * 1000

    print()
    print("=" * 64)
    print("LATENCY SUMMARY")
    print("=" * 64)

    print_metric(
        "Connection latency",
        connection_latency_ms,
    )

    print_metric(
        "Client TTFA",
        client_ttfa_ms,
    )

    print_metric(
        "End-to-end TTFA",
        end_to_end_ttfa_ms,
    )

    print_metric(
        "Server TTFA",
        server_metrics.get("server_ttfa_ms"),
    )

    print_metric(
        "Inference first result",
        server_metrics.get("inference_first_result_ms"),
    )

    print_metric(
        "Inference latency",
        server_metrics.get("inference_latency_ms"),
    )

    print_metric(
        "WAV encoding latency",
        server_metrics.get("wav_encoding_latency_ms"),
    )

    print_metric(
        "Generation latency",
        server_metrics.get("generation_latency_ms"),
    )

    print_metric(
        "Client total latency",
        client_total_latency_ms,
    )

    print_metric(
        "End-to-end latency",
        end_to_end_latency_ms,
    )

    print_metric(
        "Server total latency",
        server_metrics.get("server_total_latency_ms"),
    )

    print_metric(
        "Connection to done",
        server_metrics.get("connection_to_done_ms"),
    )

    print("=" * 64)
    print(f"Saved WAV                    : {output_path}")
    print(f"File size                    : {output_path.stat().st_size:,} bytes")
    print(f"Received bytes               : {bytes_received:,}")
    print(f"Received chunks              : {chunks_received}")
    print(f"Sample rate                  : {wav_info['sample_rate']} Hz")
    print(f"Channels                     : {wav_info['channels']}")
    print(f"Sample width                 : {wav_info['sample_width']} bytes")
    print(f"Frames                       : {wav_info['frames']}")
    print("=" * 64)

    if args.play:
        await asyncio.to_thread(play_wav, output_path)
        print("[playback] completed")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fish Speech 1.5 WebSocket test client"
    )

    parser.add_argument(
        "--text",
        default=None,
        help="Text to synthesize",
    )

    parser.add_argument(
        "--output",
        default="fish-speech-15.wav",
        help="Output WAV filename",
    )

    parser.add_argument(
        "--reference-id",
        default=None,
        help="Optional saved reference voice ID",
    )

    parser.add_argument(
        "--chunk-length",
        type=int,
        default=100,
        help="Chunk length between 100 and 300",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--memory-cache",
        choices=["on", "off"],
        default="off",
    )

    parser.add_argument(
        "--no-normalize",
        action="store_true",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
    )

    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=120,
    )

    playback_group = parser.add_mutually_exclusive_group()

    playback_group.add_argument(
        "--play",
        dest="play",
        action="store_true",
        help="Play the saved audio after generation",
    )

    playback_group.add_argument(
        "--no-play",
        dest="play",
        action="store_false",
        help="Save the audio without playing it",
    )

    parser.set_defaults(play=True)

    args = parser.parse_args()

    if not 100 <= args.chunk_length <= 300:
        parser.error(
            "--chunk-length must be between 100 and 300"
        )

    if not args.text:
        args.text = input("Text: ").strip()

    if not args.text:
        parser.error("Text cannot be empty.")

    return args


def main() -> None:
    args = parse_arguments()

    try:
        asyncio.run(run_client(args))

    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(130)

    except sd.PortAudioError as exc:
        print(
            f"[audio-device-error] {exc}",
            file=sys.stderr,
        )
        print(
            "Run client.py on your local Windows machine with an "
            "audio output device, not on the EC2 server.",
            file=sys.stderr,
        )
        print(
            'Check devices using: python -c "import sounddevice as sd; '
            'print(sd.query_devices())"',
            file=sys.stderr,
        )
        raise SystemExit(1)

    except Exception as exc:
        print(
            f"[error] {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
