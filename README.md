# Urdu AI Voice Agent

**Open‑Source Python project — LiveKit voice agent that speaks only in Urdu**

A lightweight, production-minded voice assistant implemented in Python. The agent uses LiveKit for real‑time audio transport, a Hugging Face `alif-3b` open‑weight language model for Urdu responses, Whisper (CPU int8) for STT, and Piper for high‑quality TTS. This README explains architecture, setup, running, and deployment notes so you can get the project working end‑to‑end.

---

## Table of contents

* [Project overview](#project-overview)
* [Features](#features)
* [Architecture](#architecture)
* [Prerequisites](#prerequisites)
* [Quick start (development)](#quick-start-development)
* [Configuration](#configuration)
* [How it works (request flow)](#how-it-works-request-flow)
* [Troubleshooting & tips](#troubleshooting--tips)
* [Contributing](#contributing)
* [License & credits](#license--credits)

---

## Project overview

This repository contains a real‑time Urdu voice agent designed to be used with LiveKit rooms. Users speak to the agent through the LiveKit audio track; the agent converts speech to text (STT), sends the text to a local/hosted `alif-3b` language model tuned for Urdu, then synthesizes the reply to audio using Piper and streams the audio back to participants in the LiveKit room.

Key goals:

* **Urdu‑only** conversational agent (validation + enforcement that output is in Urdu).
* Use **open weight models** only: `alif-3b` (Hugging Face style), Whisper CPU-int8 for STT, and Piper for TTS.
* **No cloud paywalled APIs** required (but you may optionally run model inference on cloud GPU hosts).
* Real‑time user experience via LiveKit and streaming audio responses.

---

## Features

* LiveKit integration for real‑time voice calls/rooms.
* STT: Whisper (int8/CPU friendly variant via whisper.cpp / ggml build recommended).
* LLM: `alif-3b` (local Hugging Face checkpoint or hosted endpoint). Conversational context management included.
* TTS: Piper (open model) producing natural Urdu speech.
* Simple Flask/FastAPI server to glue LiveKit events, inference, and audio output.
* Basic evaluation and logging for utterances.
* Dockerfile and examples to run locally or on a cloud VM.

---

## Architecture

1. **LiveKit** (client/browser or native) — holds a voice room and sends audio tracks to the server agent.
2. **Agent server** (this repo) — receives audio, performs STT, uses `alif-3b` to create a reply, synthesizes reply with Piper, then publishes the audio back into the LiveKit room.
3. **Model runtimes** — optional GPU host or local CPU runtime for Whisper, alif-3b, and Piper.

Flow: `LiveKit -> Agent (audio) -> STT (Whisper) -> LLM (alif-3b) -> TTS (Piper) -> Agent -> LiveKit (audio)`

---

## Prerequisites

* Python 3.10+ (3.11 recommended)
* `pip` and virtualenv or conda
* A LiveKit deployment or LiveKit Cloud credentials (API key + secret) for development
* Enough RAM/CPU for local inference; for `alif-3b` a GPU is strongly recommended but quantized CPU setups are possible (will be slower).
* `whisper.cpp` (ggml) or a CPU-friendly Whisper alternative compiled for int8 inference (recommended for host machines without GPU)
* Piper TTS runtime (available as open weights and runtime)

Note: Running `alif-3b` on CPU may be slow. Consider a GPU or using quantized runtime (4-bit/8-bit) or hosted inference endpoint.

---

## Quick start (development)

### 1. Clone

```bash
git clone https://github.com/<yourname>/urdu-livekit-voice-agent.git
cd urdu-livekit-voice-agent
```

### 2. Create virtual environment & install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`requirements.txt` should include (example):

```
fastapi
uvicorn[standard]
websockets
livekit-server-sdk # or livekit-client if needed
transformers
accelerate
torch
soundfile
pydub
whisper (optional)
# add libraries for Piper runtime
```

> If you plan to use `whisper.cpp` or `ggml` builds, follow their compile instructions and point the server to the binary.

### 3. Add configuration (.env)

Create a `.env` file in project root (see [Configuration](#configuration)). Example:

```
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
LIVEKIT_URL=wss://your.livekit.server
ALIF_MODEL_PATH=/path/to/alif-3b
WHISPER_BACKEND=whispercpp   # or whisper_python
PIPER_RUNTIME=/path/to/piper/runtime
AGENT_LANGUAGE=ur
LISTEN_PORT=8000
```

### 4. Download models

* Put `alif-3b` weights in `models/alif-3b/` (or set `ALIF_MODEL_PATH`) — follow Hugging Face instructions for downloading.
* Prepare Whisper int8 files (e.g., ggml format) or install `whisper.cpp`.
* Prepare Piper TTS model files.

### 5. Run the server

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Connect a LiveKit client

Use the LiveKit web SDK example (or provided `examples/`) to join the same room. When you speak, the agent will respond in Urdu.

---

## Configuration

All runtime configuration is read from environment variables or a `config.yaml` file. Key variables:

* `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, `LIVEKIT_URL` — LiveKit access credentials.
* `ALIF_MODEL_PATH` — local path to `alif-3b` model or remote URL.
* `ALIF_DEVICE` — `cpu` or `cuda`.
* `WHISPER_BACKEND` — `whispercpp` or `whisper_python`.
* `WHISPER_MODEL_PATH` — path to whisper int8 ggml file.
* `PIPER_RUNTIME` — path or URL of Piper TTS runtime/endpoint.
* `AGENT_LANGUAGE` — enforced language code (set to `ur`).

Example `.env` snippet:

```bash
LIVEKIT_API_KEY=abc123
LIVEKIT_API_SECRET=xyz987
LIVEKIT_URL=wss://livekit.example.com
ALIF_MODEL_PATH=/home/models/alif-3b
ALIF_DEVICE=cuda
WHISPER_BACKEND=whispercpp
WHISPER_MODEL_PATH=/home/models/whisper-ggml-int8.bin
PIPER_RUNTIME=http://localhost:5002
AGENT_LANGUAGE=ur
```

---

## How it works (request flow)

1. Client joins LiveKit room and publishes audio track.
2. Server subscribes to the participant's audio track (or receives a push from client depending on setup).
3. Incoming audio frames are buffered into short utterances and passed to the STT backend (Whisper int8 recommended for CPU).
4. STT returns an Urdu text string. The server performs light normalization and validates language (if the text is not Urdu, the agent will ask politely to speak Urdu only).
5. The text is sent as input to the `alif-3b` model with a system prompt that constrains replies to Urdu and a friendly assistant persona.
6. The LLM reply text is fed to Piper for TTS; the resulting audio is streamed/published back into the room as the agent's audio track.

Important: Manage short context windows for `alif-3b` (3B models have limited context length). The repo includes a lightweight conversation manager that keeps the last N turns.

---

## Example code snippet — minimal Flask route

```python
from fastapi import FastAPI
from livekit import LiveKitServerClient

app = FastAPI()

@app.post('/webhook/audio')
async def on_audio_chunk(payload: dict):
    # 1. receive/upload audio chunk
    # 2. pass to STT backend (whisper)
    # 3. call LLM for response
    # 4. call Piper for TTS
    # 5. publish audio back to LiveKit
    return {'ok': True}
```

(See `server/` for complete implementation.)

---
---

## Deployment notes

* **Production LiveKit**: Run the agent as a dedicated service that connects to the LiveKit cluster and publishes a participant representing the agent. Use LiveKit server SDKs to manage participants programmatically.
* **Scaling**: Each agent instance can serve many rooms depending on CPU/GPU available. For heavy LLM usage, scale using Kubernetes and route requests to GPU-backed pods.
* **Security**: Keep LiveKit API keys secret. Use role‑based access control on LiveKit tokens for clients.
* **Latency**: For lower latency, use GPU for `alif-3b` and Piper, and use `whisper.cpp` (ggml int8) on CPU for STT.

---

## Troubleshooting & tips

* **Audio quality**: Ensure sample rate is 16kHz or 48kHz depending on the STT runtime. The repo's `audio_utils` does resampling automatically.
* **Whisper CPU slow**: Use `whisper.cpp`/ggml int8 builds or move STT to a GPU host.
* **LLM hallucination**: Constrain the `system` prompt to require Urdu output and instruct the model to admit uncertainty rather than invent facts.
* **TTS voice & prosody**: Piper provides different voices and config — experiment with duration multipliers and pitch if needed.
* **Language enforcement**: After STT, run a language check (fast heuristic or language detection library) and if the input is not Urdu respond with a brief Urdu-only instruction.

---

## Contributing

Contributions welcome! Please open issues for bugs, feature requests, or model compatibility questions. Pull requests should include tests and follow PEP8 formatting.

Guidelines:

1. Fork the repository
2. Create a branch `feature/your-change`
3. Add tests where possible
4. Submit a PR

---

## License & credits

Third‑party projects used:

* LiveKit (real‑time audio/video)
* Whisper / whisper.cpp (STT)
* Hugging Face Transformers (alif-3b)
* Piper TTS

---

## Example system prompt (enforced Urdu)

```
System: You are an assistant that speaks only in Urdu. Always reply in Urdu. If the user asks to change language, politely say you only speak Urdu and offer to continue in Urdu. Keep replies friendly and concise.
```

---

If you'd like, I can also:

* Produce a short `docker-compose.yml` and `Dockerfile` optimized for CPU inference.
* Create the LiveKit web demo client that demonstrates joining a room and talking to the agent.
* Add a sample `system_prompts/` directory with ready‑to‑use prompts in Urdu.

---

*End of README*
