# Fish Speech 1.5 FastAPI + WebSocket

Endpoints:

- `GET http://localhost:8000/health`
- `POST http://localhost:8000/v1/tts`
- `WS ws://localhost:8000/ws/tts`

## Build

```bash
DOCKER_BUILDKIT=1 docker build --progress=plain -t fish-speech-15:latest .
```

## Run

```bash
docker run -d \
  --name fish-speech-15 \
  --restart unless-stopped \
  --gpus all \
  --shm-size=8g \
  -p 8000:8000 \
  -e DEVICE=cuda \
  -e HALF=1 \
  -e COMPILE=0 \
  fish-speech-15:latest
```

## Logs and health

```bash
docker logs -f fish-speech-15
curl http://127.0.0.1:8000/health
```

## HTTP test

```bash
curl -D headers.txt \
  -X POST http://127.0.0.1:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello from Fish Speech.",
    "format": "wav",
    "streaming": false,
    "chunk_length": 100,
    "max_new_tokens": 256,
    "top_p": 0.7,
    "temperature": 0.7,
    "repetition_penalty": 1.2
  }' \
  --output fish-http.wav
```

## Local WebSocket client

```bash
python -m venv .client-venv
source .client-venv/bin/activate
pip install -r requirements-client.txt
python client.py --text "Hello there." --output fish-15-test.wav --play
```

For a remote server, set:

```bash
export FISH_WS_URL="ws://10.90.126.61:8000/ws/tts"
```
