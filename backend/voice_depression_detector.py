# voice_depression_detector.py
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
from improved_depression_model import ImprovedDepressionModel

class VoiceDepressionDetector:
    def __init__(self, model_path='models/depression_model_improved.pkl'):
        """Initialize with trained model"""
        try:
            self.model = ImprovedDepressionModel()
            self.model.load_model(model_path)
            self.model_loaded = True
            print("✓ Voice depression detection model loaded")
        except Exception as e:
            print(f"⚠️  Could not load depression model: {e}")
            self.model_loaded = False

    def extract_features_from_audio(self, audio_path):
        """Extract acoustic features from audio file"""
        try:
            y, sr = librosa.load(audio_path, sr=16000)

            if len(y) < sr * 0.5:
                print("⚠️  Audio too short for analysis")
                return None

            features = {}

            # ---- PITCH FEATURES ----
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if len(pitch_values) > 0:
                features['covarep_pitch_mean']   = float(np.mean(pitch_values))
                features['covarep_pitch_std']    = float(np.std(pitch_values))
                features['covarep_pitch_min']    = float(np.min(pitch_values))
                features['covarep_pitch_max']    = float(np.max(pitch_values))
                features['covarep_pitch_median'] = float(np.median(pitch_values))

            # ---- ENERGY FEATURES ----
            rms = librosa.feature.rms(y=y)[0]
            features['covarep_energy_mean']   = float(np.mean(rms))
            features['covarep_energy_std']    = float(np.std(rms))
            features['covarep_energy_min']    = float(np.min(rms))
            features['covarep_energy_max']    = float(np.max(rms))
            features['covarep_energy_median'] = float(np.median(rms))

            # ---- SPECTRAL FEATURES ----
            spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            features['covarep_spectral_centroid_mean']   = float(np.mean(spec_centroid))
            features['covarep_spectral_centroid_std']    = float(np.std(spec_centroid))
            features['covarep_spectral_centroid_min']    = float(np.min(spec_centroid))
            features['covarep_spectral_centroid_max']    = float(np.max(spec_centroid))
            features['covarep_spectral_centroid_median'] = float(np.median(spec_centroid))

            # ---- ZERO CROSSING RATE ----
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features['covarep_zcr_mean']   = float(np.mean(zcr))
            features['covarep_zcr_std']    = float(np.std(zcr))
            features['covarep_zcr_min']    = float(np.min(zcr))
            features['covarep_zcr_max']    = float(np.max(zcr))
            features['covarep_zcr_median'] = float(np.median(zcr))

            # ---- MFCCs ----
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'covarep_mfcc{i}_mean']   = float(np.mean(mfccs[i]))
                features[f'covarep_mfcc{i}_std']    = float(np.std(mfccs[i]))
                features[f'covarep_mfcc{i}_min']    = float(np.min(mfccs[i]))
                features[f'covarep_mfcc{i}_max']    = float(np.max(mfccs[i]))
                features[f'covarep_mfcc{i}_median'] = float(np.median(mfccs[i]))

            # ---- FORMANT-LIKE FEATURES ----
            spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            features['formant_rolloff_mean']   = float(np.mean(spec_rolloff))
            features['formant_rolloff_std']    = float(np.std(spec_rolloff))
            features['formant_rolloff_min']    = float(np.min(spec_rolloff))
            features['formant_rolloff_max']    = float(np.max(spec_rolloff))
            features['formant_rolloff_median'] = float(np.median(spec_rolloff))

            spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            features['formant_bandwidth_mean']   = float(np.mean(spec_bandwidth))
            features['formant_bandwidth_std']    = float(np.std(spec_bandwidth))
            features['formant_bandwidth_min']    = float(np.min(spec_bandwidth))
            features['formant_bandwidth_max']    = float(np.max(spec_bandwidth))
            features['formant_bandwidth_median'] = float(np.median(spec_bandwidth))

            # ---- TEMPO ----
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['covarep_tempo_mean'] = float(tempo)

            # ---- HARMONICS ----
            harmonic, percussive = librosa.effects.hpss(y)
            features['covarep_harmonic_mean'] = float(np.mean(np.abs(harmonic)))
            features['covarep_harmonic_std']  = float(np.std(np.abs(harmonic)))
            features['covarep_percussive_mean'] = float(np.mean(np.abs(percussive)))

            return features

        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def analyze_audio_bytes(self, audio_bytes):
        """
        Analyze raw audio bytes and detect depression markers.
        Used by the FastAPI endpoint.
        """
        if not self.model_loaded:
            return {
                'error': 'Model not loaded',
                'depression_detected': None,
                'available': False
            }

        temp_path = None
        try:
            # Save audio bytes to temp file
            with tempfile.NamedTemporaryFile(
                suffix='.wav', delete=False
            ) as tmp:
                tmp.write(audio_bytes)
                temp_path = tmp.name

            # Extract features
            features = self.extract_features_from_audio(temp_path)

            if features is None:
                return {
                    'error': 'Could not extract features - audio too short',
                    'depression_detected': None,
                    'available': True
                }

            # Predict
            result = self.model.predict(features)
            result['available'] = True
            result['error'] = None
            return result

        except Exception as e:
            print(f"Analysis error: {e}")
            return {
                'error': str(e),
                'depression_detected': None,
                'available': True
            }
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    def get_therapeutic_context(self, result):
        """
        Get therapeutic context based on voice analysis result.
        Used to inform the chatbot response.
        """
        if not result or result.get('error') or result.get('depression_detected') is None:
            return ""

        probability = result.get('probability', 0)
        risk_level = result.get('risk_level', 'Low')

        if risk_level == 'Low':
            return f"""
VOICE ANALYSIS CONTEXT (for therapist awareness only):
- Voice analysis suggests low depression markers (probability: {probability:.1%})
- User appears to be communicating clearly
- No special adjustments needed based on voice
"""
        elif risk_level == 'Moderate':
            return f"""
VOICE ANALYSIS CONTEXT (for therapist awareness only):
- Voice analysis suggests moderate depression markers (probability: {probability:.1%})
- Pay extra attention to emotional undertones
- Gently explore how user has been feeling lately
- Be more supportive and validating in responses
"""
        else:  # High
            return f"""
VOICE ANALYSIS CONTEXT (HIGH CONCERN - for therapist awareness only):
- Voice analysis suggests significant depression markers (probability: {probability:.1%})
- User's voice patterns indicate potential distress
- Be especially warm, supportive, and present
- Gently assess current emotional state
- Consider recommending professional support
- Watch for any crisis indicators
"""

# Global instance
voice_detector = VoiceDepressionDetector()