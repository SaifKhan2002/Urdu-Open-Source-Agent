from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
import soundfile as sf
import io
from kokoro import generate

app = FastAPI(title="Kokoro TTS Server")

class SpeechRequest(BaseModel):
    model: str = "kokoro"
    input: str
    voice: str = "af_bella"
    format: str = "wav"

@app.post("/v1/audio/speech")
async def audio_speech(req: SpeechRequest):
    text = req.input.strip()
    if not text:
        return Response(content=b"", media_type="audio/wav")

    # Generate speech (Kokoro API call)
    audio, sample_rate = generate(text, voice=req.voice)

    # Write WAV to bytes
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return Response(content=buf.getvalue(), media_type="audio/wav")

@app.get("/v1/models")
async def models():
    return {"data": [{"id": "kokoro", "object": "model", "owned_by": "local"}]}
