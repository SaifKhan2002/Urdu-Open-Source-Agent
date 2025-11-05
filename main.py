import os, io, asyncio, logging
import numpy as np
from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, WorkerOptions
from livekit.plugins import silero, openai, azure
from livekit.agents.stt import (
    STT, STTCapabilities, SpeechEvent, SpeechEventType, SpeechData, StreamAdapter,
)
from faster_whisper import WhisperModel

load_dotenv(".env")
logging.basicConfig(level=logging.INFO)

# =========================================================
# Globals
# =========================================================
_WHISPER = None
_VAD = None

# =========================================================
# Helpers
# =========================================================
def _resample_to_16k_mono(int16_pcm: np.ndarray, sr: int, channels: int) -> np.ndarray:
    if channels > 1:
        int16_pcm = int16_pcm.reshape(-1, channels).mean(axis=1).astype(np.int16)
    audio = int16_pcm.astype(np.float32) / 32768.0
    target_sr = 16000
    new_len = int(round(len(audio) * (target_sr / sr)))
    xp = np.linspace(0, len(audio) - 1, num=len(audio), dtype=np.float32)
    x_new = np.linspace(0, len(audio) - 1, num=new_len, dtype=np.float32)
    return np.interp(x_new, xp, audio).astype(np.float32)

def _ensure_multilingual(name: str) -> str:
    return name[:-3] if name.endswith(".en") else name

def _mostly_ascii(s: str) -> bool:
    stripped = "".join(ch for ch in s if not ch.isspace() and ch not in ".,?!،۔")
    if not stripped:
        return True
    ascii_count = sum(1 for ch in stripped if ord(ch) < 128)
    return ascii_count / max(1, len(stripped)) > 0.6

def _maybe_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"  # pure CPU path (int8 quant)

# Urdu guidance for Whisper
_URDU_HOTWORDS = "میں آپ یہ وہ کیا کیوں کیسے کہا ٹھیک جی ہاں نہیں شکریہ السلام علیکم لاہور کراچی اسلام آباد"
_URDU_PROMPT = (
    "براہِ کرم مکمل طور پر اردو رسم الخط میں ٹرانسکرائب کریں۔ "
    "انگریزی یا رومن اردو استعمال نہ کریں۔ واضح اور درست اُردو لکھیں۔ "
    "اعداد و شمار اور نام بھی اردو میں لکھیں۔"
)

def _load_whisper_if_needed():
    """Lazy-load the Whisper model if _WHISPER is None."""
    global _WHISPER
    if _WHISPER is not None:
        return _WHISPER

    # Keep CPU responsive on i3: reduce threads a bit
    cpu_threads = max(1, (os.cpu_count() or 4) // 2)
    os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))

    model_name = _ensure_multilingual(os.getenv("WHISPER_MODEL", "tiny"))
    device, compute_type = _maybe_gpu()
    logging.info(f"[Urdu STT] Loading Whisper '{model_name}' on {device} ({compute_type}), threads={cpu_threads}")
    _WHISPER = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,   # "int8" on CPU is fine
        cpu_threads=cpu_threads,
        num_workers=1,
        download_root=os.getenv("WHISPER_CACHE_DIR", None),
    )
    return _WHISPER

# =========================================================
# STT: Faster-Whisper (Urdu-focused, CPU friendly)
# =========================================================
class UrduWhisperSTT(STT):
    """
    - Urdu-only (language='ur')
    - Fast pass: greedy (beam_size=1). If ASCIIish or empty -> single stronger retry.
    - Urdu initial prompt + hotwords to avoid Roman Urdu.
    - Works well on CPU-only boxes.
    """
    def __init__(self):
        super().__init__(capabilities=STTCapabilities(streaming=False, interim_results=False))
        self._model = _load_whisper_if_needed()

    def _decode(self, audio: np.ndarray, beam_size: int, best_of: int, temperature: float):
        kwargs = dict(
            language="ur",
            task="transcribe",
            initial_prompt=_URDU_PROMPT,
            vad_filter=False,                # external VAD handles segmentation
            suppress_blank=True,
            without_timestamps=True,
            word_timestamps=False,
            no_speech_threshold=0.35,
            log_prob_threshold=-1.2,
            compression_ratio_threshold=2.6,
            condition_on_previous_text=False,
            # chunk params (keep memory small and latency low)
            chunk_length=15,                 # seconds
            hallucination_silence_threshold=0.2,
            temperature=temperature,
            beam_size=beam_size,
            best_of=best_of,
        )
        if beam_size and beam_size > 1:
            kwargs["patience"] = 1.0

        try:
            segments, _ = self._model.transcribe(audio, hotwords=_URDU_HOTWORDS, **kwargs)
        except TypeError:
            segments, _ = self._model.transcribe(audio, **kwargs)

        return " ".join(s.text.strip() for s in segments).strip()

    async def _recognize_impl(self, buffer, *, language=None, **kwargs) -> SpeechEvent:
        # Always Urdu
        wav_bytes = buffer.to_wav_bytes()
        import wave
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr, ch, _sw, n_frames = wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getnframes()
            raw = wf.readframes(n_frames)

        pcm = np.frombuffer(raw, dtype=np.int16, count=n_frames * ch)
        audio = _resample_to_16k_mono(pcm, sr=sr, channels=ch)

        # Pass 1: fastest (greedy)
        text = self._decode(audio, beam_size=1, best_of=1, temperature=0.0)

        # If looks Roman/ASCII or empty, retry once with beam for quality
        if not text or _mostly_ascii(text):
            logging.info("[Urdu STT] ASCII-heavy/empty — retrying with beam…")
            text = self._decode(audio, beam_size=4, best_of=1, temperature=0.0)

        logging.info(f"STT ▶ {text}")
        data = SpeechData(language="ur", text=text, confidence=0.0)
        return SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=[data])

# =========================================================
# Agent (Urdu style)
# =========================================================
class TalkAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=(
            "آپ ایک ہلکے وزن کا اردو معاون ہیں۔ "
            "ہمیشہ درست، مختصر اور مکمل طور پر اردو رسم الخط میں جواب دیں۔ "
            "رومن اردو یا غیر ضروری انگریزی استعمال نہ کریں۔"
        ))

# =========================================================
# Prewarm (optional)
# =========================================================
def prewarm(proc: agents.JobProcess):
    global _WHISPER, _VAD
    try:
        _VAD = silero.VAD.load()
    except Exception:
        _VAD = None
    _load_whisper_if_needed()

# =========================================================
# Entrypoint
# =========================================================
async def entrypoint(ctx: agents.JobContext):
    # -------- LLM (Open-source, CPU-light) --------
    # Qwen2.5 1.5B (Q4) via Ollama: fast on CPU, decent Urdu when instructed.
    llm = openai.LLM(
        model=os.getenv("LLM_MODEL", "qwen2.5:1.5b-instruct-q4_K_M"),
        base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.getenv("LLM_API_KEY", "ollama"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        max_completion_tokens=int(os.getenv("LLM_MAX_TOKENS", "120")),
    )

    # (Optional) If you insist on Alif when available (slower on i3):
    # llm = openai.LLM(
    #     model="alif:3b-instruct-q4_K_M",
    #     base_url="http://localhost:11434/v1",
    #     api_key="ollama",
    #     temperature=0.2,
    #     max_completion_tokens=120,
    # )

    # -------- TTS (unchanged: Azure) --------
    tts = azure.TTS(voice=os.getenv("AZURE_TTS_VOICE", "en-US-AmandaMultilingualNeural"))

    # -------- VAD + STT --------
    vad = _VAD or silero.VAD.load()
    stt = StreamAdapter(stt=UrduWhisperSTT(), vad=vad)

    # -------- Urdu Agent --------
    agent = TalkAgent()

    async with AgentSession(stt=stt, llm=llm, tts=tts) as session:
        await session.start(room=ctx.room, agent=agent, room_input_options=RoomInputOptions())
        await session.say("السلام علیکم! میں حاضر ہوں—آپ اُردو میں بات کر سکتے ہیں۔", allow_interruptions=True)
        while True:
            await asyncio.sleep(1)

# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    agents.cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
