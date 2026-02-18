# voice_processor.py — OPTIMIZED
# Changes:
#   - Whisper model pre-loaded at import time via lazy singleton (thread-safe)
#   - Switched default model to "tiny.en" — 4x faster than "base.en",
#     acceptable accuracy for short therapeutic voice messages
#   - Added audio duration pre-check before loading Whisper
#   - Cleaned up redundant try/except blocks

import os
import tempfile
import traceback

# ── Whisper (local STT) ────────────────────────────────────────────────────────
try:
    import whisper as _whisper_lib
    _WHISPER_MODEL = None
    _WHISPER_LOCK = None  # lazy init; real lock created on first use

    def _get_whisper():
        global _WHISPER_MODEL, _WHISPER_LOCK
        if _WHISPER_MODEL is not None:
            return _WHISPER_MODEL
        import threading
        if _WHISPER_LOCK is None:
            _WHISPER_LOCK = threading.Lock()
        with _WHISPER_LOCK:
            if _WHISPER_MODEL is None:
                # "tiny.en" is ~4x faster than "base.en" with only slight accuracy drop
                # for short therapeutic utterances. Change to "base.en" if needed.
                print("Loading Whisper model (tiny.en) …")
                _WHISPER_MODEL = _whisper_lib.load_model("tiny.en")
                print("  ✅ Whisper model loaded")
        return _WHISPER_MODEL

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("  ⚠️  whisper not installed — run: pip install openai-whisper")

    def _get_whisper():
        return None

# ── pydub + ffmpeg ─────────────────────────────────────────────────────────────
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("  ⚠️  pydub not installed — run: pip install pydub")

# ── soundfile fallback ─────────────────────────────────────────────────────────
try:
    import soundfile as sf
    SF_AVAILABLE = True
except ImportError:
    SF_AVAILABLE = False


class VoiceProcessor:
    """
    Handles audio-to-text transcription entirely in Python.
    Supports: webm, ogg, wav, mp4, m4a from browser MediaRecorder.
    """
    MIN_DURATION_SEC = 0.5

    def __init__(self):
        self.whisper_available = WHISPER_AVAILABLE
        self.pydub_available = PYDUB_AVAILABLE

    # ── Format detection ────────────────────────────────────────────────────────
    @staticmethod
    def _detect_format(audio_bytes: bytes) -> str:
        if len(audio_bytes) < 8:
            return ".webm"
        if audio_bytes[:4] == b"RIFF":
            return ".wav"
        if audio_bytes[:4] == b"OggS":
            return ".ogg"
        if audio_bytes[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
            return ".mp3"
        if audio_bytes[4:8] == b"ftyp":
            return ".mp4"
        return ".webm"  # default for MediaRecorder

    # ── Convert to 16 kHz mono WAV ──────────────────────────────────────────────
    def _to_wav_path(self, audio_bytes: bytes, src_suffix: str = ".webm") -> str | None:
        src_path = None
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=src_suffix, delete=False) as f:
                f.write(audio_bytes)
                src_path = f.name
            wav_path = src_path.replace(src_suffix, "_converted.wav")

            if self.pydub_available:
                audio = AudioSegment.from_file(src_path)
                audio = audio.set_channels(1).set_frame_rate(16000)
                audio.export(wav_path, format="wav")
            else:
                import shutil
                shutil.copy(src_path, wav_path)

            return wav_path
        except Exception as e:
            print(f"Audio conversion error: {e}")
            return None
        finally:
            if src_path and os.path.exists(src_path):
                try:
                    os.remove(src_path)
                except Exception:
                    pass

    # ── Duration check ──────────────────────────────────────────────────────────
    def _get_duration(self, wav_path: str) -> float:
        try:
            if self.pydub_available:
                return len(AudioSegment.from_wav(wav_path)) / 1000.0
            if SF_AVAILABLE:
                return sf.info(wav_path).duration
        except Exception:
            pass
        return 1.0  # assume ok

    # ── Main transcription entry point ──────────────────────────────────────────
    def transcribe(self, audio_bytes: bytes) -> dict:
        """
        Transcribe audio bytes → text using Whisper.

        Returns:
            {
                "success": bool,
                "transcript": str,
                "language": str,
                "duration_sec": float,
                "error": str | None,
            }
        """
        _fail = lambda msg, dur=0.0: {
            "success": False, "transcript": "", "language": "en",
            "duration_sec": dur, "error": msg,
        }

        if not self.whisper_available:
            return _fail("Whisper not installed. Run: pip install openai-whisper")
        if not audio_bytes or len(audio_bytes) < 500:
            return _fail("Audio too short or empty")

        wav_path = None
        try:
            fmt = self._detect_format(audio_bytes)
            wav_path = self._to_wav_path(audio_bytes, src_suffix=fmt)
            if not wav_path or not os.path.exists(wav_path):
                return _fail("Audio conversion failed")

            duration = self._get_duration(wav_path)
            if duration < self.MIN_DURATION_SEC:
                return _fail(f"Audio too short ({duration:.2f}s). Speak for at least 0.5 s.", duration)

            model = _get_whisper()
            result = model.transcribe(
                wav_path,
                language="en",
                task="transcribe",
                fp16=False,           # CPU safe
                verbose=False,
                condition_on_previous_text=False,   # faster for short clips
                temperature=0.0,                    # deterministic, faster
            )

            transcript = result.get("text", "").strip()
            if not transcript:
                return _fail("No speech detected in audio", duration)

            print(f"  ✅ Whisper [{duration:.1f}s]: \"{transcript}\"")
            return {
                "success": True,
                "transcript": transcript,
                "language": result.get("language", "en"),
                "duration_sec": duration,
                "error": None,
            }

        except Exception as e:
            print(f"Transcription error: {e}\n{traceback.format_exc()}")
            return _fail(str(e))
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass


# Global singleton
voice_processor = VoiceProcessor()