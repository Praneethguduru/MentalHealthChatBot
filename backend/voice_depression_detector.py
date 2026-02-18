# voice_depression_detector.py — OPTIMIZED
# Changes:
#   - Feature extraction wrapped in try/except per-feature group so one
#     failure doesn't kill the whole analysis
#   - Added quick audio length check before loading librosa
#   - Reuses a single global model instance (was already done, confirmed)
#   - Removed redundant temp file writes where possible

import numpy as np
import tempfile
import os
from improved_depression_model import ImprovedDepressionModel


class VoiceDepressionDetector:
    def __init__(self, model_path: str = "models/depression_model_improved.pkl"):
        try:
            self.model = ImprovedDepressionModel()
            self.model.load_model(model_path)
            self.model_loaded = True
            print("  ✅ Voice depression detection model loaded")
        except Exception as e:
            print(f"  ⚠️  Could not load depression model: {e}")
            self.model_loaded = False

    # ── Feature extraction ──────────────────────────────────────────────────────
    def extract_features_from_audio(self, audio_path: str) -> dict | None:
        """Extract acoustic features from a WAV file path."""
        try:
            import librosa
        except ImportError:
            print("  ⚠️  librosa not installed — run: pip install librosa")
            return None

        try:
            y, sr = librosa.load(audio_path, sr=16000)
        except Exception as e:
            print(f"  ⚠️  librosa load error: {e}")
            return None

        if len(y) < sr * 0.5:
            print("  ⚠️  Audio too short for analysis")
            return None

        features: dict = {}

        # Helper — wraps each group so one failure doesn't stop others
        def safe(fn):
            try:
                fn()
            except Exception as e:
                print(f"  ⚠️  Feature group error: {e}")

        # PITCH
        def _pitch():
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            vals = [
                pitches[magnitudes[:, t].argmax(), t]
                for t in range(pitches.shape[1])
                if pitches[magnitudes[:, t].argmax(), t] > 0
            ]
            if vals:
                arr = np.array(vals)
                features.update({
                    "covarep_pitch_mean": float(arr.mean()),
                    "covarep_pitch_std": float(arr.std()),
                    "covarep_pitch_min": float(arr.min()),
                    "covarep_pitch_max": float(arr.max()),
                    "covarep_pitch_median": float(np.median(arr)),
                })
        safe(_pitch)

        # ENERGY
        def _energy():
            rms = librosa.feature.rms(y=y)[0]
            features.update({
                "covarep_energy_mean": float(rms.mean()),
                "covarep_energy_std": float(rms.std()),
                "covarep_energy_min": float(rms.min()),
                "covarep_energy_max": float(rms.max()),
                "covarep_energy_median": float(np.median(rms)),
            })
        safe(_energy)

        # SPECTRAL CENTROID
        def _centroid():
            sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features.update({
                "covarep_spectral_centroid_mean": float(sc.mean()),
                "covarep_spectral_centroid_std": float(sc.std()),
                "covarep_spectral_centroid_min": float(sc.min()),
                "covarep_spectral_centroid_max": float(sc.max()),
                "covarep_spectral_centroid_median": float(np.median(sc)),
            })
        safe(_centroid)

        # ZCR
        def _zcr():
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.update({
                "covarep_zcr_mean": float(zcr.mean()),
                "covarep_zcr_std": float(zcr.std()),
                "covarep_zcr_min": float(zcr.min()),
                "covarep_zcr_max": float(zcr.max()),
                "covarep_zcr_median": float(np.median(zcr)),
            })
        safe(_zcr)

        # MFCCs
        def _mfcc():
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                row = mfccs[i]
                features.update({
                    f"covarep_mfcc{i}_mean": float(row.mean()),
                    f"covarep_mfcc{i}_std": float(row.std()),
                    f"covarep_mfcc{i}_min": float(row.min()),
                    f"covarep_mfcc{i}_max": float(row.max()),
                    f"covarep_mfcc{i}_median": float(np.median(row)),
                })
        safe(_mfcc)

        # FORMANT-LIKE (rolloff + bandwidth)
        def _formant():
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features.update({
                "formant_rolloff_mean": float(rolloff.mean()),
                "formant_rolloff_std": float(rolloff.std()),
                "formant_rolloff_min": float(rolloff.min()),
                "formant_rolloff_max": float(rolloff.max()),
                "formant_rolloff_median": float(np.median(rolloff)),
                "formant_bandwidth_mean": float(bw.mean()),
                "formant_bandwidth_std": float(bw.std()),
                "formant_bandwidth_min": float(bw.min()),
                "formant_bandwidth_max": float(bw.max()),
                "formant_bandwidth_median": float(np.median(bw)),
            })
        safe(_formant)

        # TEMPO
        def _tempo():
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features["covarep_tempo_mean"] = float(tempo)
        safe(_tempo)

        # HARMONICS
        def _harmonics():
            harmonic, percussive = librosa.effects.hpss(y)
            features.update({
                "covarep_harmonic_mean": float(np.mean(np.abs(harmonic))),
                "covarep_harmonic_std": float(np.std(np.abs(harmonic))),
                "covarep_percussive_mean": float(np.mean(np.abs(percussive))),
            })
        safe(_harmonics)

        return features if features else None

    # ── Analyze raw audio bytes ─────────────────────────────────────────────────
    def analyze_audio_bytes(self, audio_bytes: bytes) -> dict:
        """Analyze raw audio bytes and return depression prediction dict."""
        if not self.model_loaded:
            return {"error": "Model not loaded", "depression_detected": None, "available": False}

        if not audio_bytes or len(audio_bytes) < 500:
            return {"error": "Audio too short", "depression_detected": None, "available": True}

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name

            features = self.extract_features_from_audio(temp_path)
            if not features:
                return {
                    "error": "Could not extract features — audio too short or silent",
                    "depression_detected": None,
                    "available": True,
                }

            result = self.model.predict(features)
            result["available"] = True
            result["error"] = None
            return result

        except Exception as e:
            print(f"  ⚠️  Analysis error: {e}")
            return {"error": str(e), "depression_detected": None, "available": True}
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    # ── Therapeutic context string ──────────────────────────────────────────────
    def get_therapeutic_context(self, result: dict | None) -> str:
        if not result or result.get("error") or result.get("depression_detected") is None:
            return ""

        probability = result.get("probability", 0)
        risk_level = result.get("risk_level", "Low")

        if risk_level == "Low":
            return (
                f"\nVOICE ANALYSIS CONTEXT (for therapist awareness only):\n"
                f"- Low depression markers (probability: {probability:.1%})\n"
                f"- No special adjustments needed based on voice\n"
            )
        elif risk_level == "Moderate":
            return (
                f"\nVOICE ANALYSIS CONTEXT (for therapist awareness only):\n"
                f"- Moderate depression markers (probability: {probability:.1%})\n"
                f"- Pay extra attention to emotional undertones\n"
                f"- Gently explore how user has been feeling lately\n"
                f"- Be more supportive and validating in responses\n"
            )
        else:  # High
            return (
                f"\nVOICE ANALYSIS CONTEXT (HIGH CONCERN — for therapist awareness only):\n"
                f"- Significant depression markers (probability: {probability:.1%})\n"
                f"- Be especially warm, supportive, and present\n"
                f"- Gently assess current emotional state\n"
                f"- Consider recommending professional support\n"
                f"- Watch for any crisis indicators\n"
            )


# Global singleton
voice_detector = VoiceDepressionDetector()