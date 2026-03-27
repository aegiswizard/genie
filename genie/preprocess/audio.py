"""
Genie 🧞‍♀️ — Audio Preprocessing
Normalize, resample, trim, and extract acoustic features.
Works with scipy alone — librosa is optional but improves quality.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

def _try_librosa():
    try:
        import librosa
        return librosa
    except ImportError:
        return None


def _try_scipy():
    try:
        from scipy.io import wavfile
        from scipy import signal as scipy_signal
        return wavfile, scipy_signal
    except ImportError:
        return None, None


# ---------------------------------------------------------------------------
# Format conversion via ffmpeg
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_to_wav(input_path: str, output_path: str, sample_rate: int = 16000) -> bool:
    """
    Convert any audio/video format to mono WAV via ffmpeg.
    Returns True on success. Falls back gracefully if ffmpeg unavailable.
    """
    if not _ffmpeg_available():
        return False
    try:
        subprocess.run([
            "ffmpeg", "-y",
            "-i", input_path,
            "-ac", "1",                         # mono
            "-ar", str(sample_rate),            # resample
            "-acodec", "pcm_s16le",             # standard PCM
            output_path,
        ], capture_output=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


# ---------------------------------------------------------------------------
# Load audio
# ---------------------------------------------------------------------------

def load_audio(
    path: str,
    sample_rate: int = 16000,
    temp_dir: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], int, list]:
    """
    Load audio from any supported format into a float32 numpy array.

    Returns:
        (waveform, actual_sample_rate, warnings)
        waveform is None if loading failed completely.
    """
    warnings = []
    path = str(path)
    ext = Path(path).suffix.lower()

    # If not WAV, try converting via ffmpeg first
    if ext not in (".wav",):
        if _ffmpeg_available():
            tmp = tempfile.NamedTemporaryFile(
                suffix=".wav", dir=temp_dir, delete=False
            )
            tmp.close()
            if convert_to_wav(path, tmp.name, sample_rate):
                path = tmp.name
            else:
                warnings.append("ffmpeg conversion failed — trying direct load")
        else:
            warnings.append("ffmpeg not available — only .wav files are natively supported")

    # Try librosa first (best quality)
    librosa = _try_librosa()
    if librosa is not None:
        try:
            waveform, sr = librosa.load(path, sr=sample_rate, mono=True)
            return waveform.astype(np.float32), sr, warnings
        except Exception as e:
            warnings.append(f"librosa load failed: {e}")

    # Fallback: scipy wavfile
    wavfile, _ = _try_scipy()
    if wavfile is not None:
        try:
            sr, data = wavfile.read(path)
            if data.ndim > 1:
                data = data.mean(axis=1)  # to mono
            if data.dtype != np.float32:
                data = data.astype(np.float32)
                if np.abs(data).max() > 1.0:
                    data /= 32768.0  # normalize int16
            return data, sr, warnings
        except Exception as e:
            warnings.append(f"scipy load failed: {e}")

    warnings.append("Could not load audio — no working loader found")
    return None, sample_rate, warnings


# ---------------------------------------------------------------------------
# Silence detection and trimming
# ---------------------------------------------------------------------------

def detect_silence_ratio(waveform: np.ndarray, threshold_db: float = -40.0) -> float:
    """Return fraction of frames that are silence (0.0 = all speech, 1.0 = all silent)."""
    if len(waveform) == 0:
        return 1.0
    rms = np.sqrt(np.mean(waveform ** 2))
    if rms == 0:
        return 1.0
    frame_size = 512
    frames = [waveform[i:i+frame_size] for i in range(0, len(waveform), frame_size)]
    silent = 0
    threshold_linear = 10 ** (threshold_db / 20.0)
    for frame in frames:
        if len(frame) == 0:
            continue
        frame_rms = np.sqrt(np.mean(frame ** 2))
        if frame_rms < threshold_linear:
            silent += 1
    return silent / max(len(frames), 1)


def trim_silence(waveform: np.ndarray, sample_rate: int, threshold_db: float = -40.0) -> np.ndarray:
    """Trim leading and trailing silence."""
    librosa = _try_librosa()
    if librosa is not None:
        try:
            trimmed, _ = librosa.effects.trim(waveform, top_db=abs(threshold_db))
            return trimmed
        except Exception:
            pass
    # Manual trim: find first and last non-silent sample
    threshold_linear = 10 ** (threshold_db / 20.0)
    abs_wave = np.abs(waveform)
    non_silent = np.where(abs_wave > threshold_linear)[0]
    if len(non_silent) == 0:
        return waveform
    return waveform[non_silent[0]:non_silent[-1]+1]


# ---------------------------------------------------------------------------
# Acoustic feature extraction (core — no ML needed)
# ---------------------------------------------------------------------------

def extract_acoustic_features(
    waveform: np.ndarray,
    sample_rate: int,
) -> dict:
    """
    Extract interpretable acoustic features from raw waveform.
    These map to state labels without any ML model.

    Features:
        energy_rms          — overall loudness
        zcr_mean            — zero crossing rate (voice quality proxy)
        zcr_std             — ZCR variance (agitation proxy)
        spectral_centroid   — brightness of voice (tense=high)
        speech_rate_estimate — syllables per second estimate
        silence_ratio       — fraction of clip with silence
        energy_variation    — coefficient of variation in energy
        pause_count         — number of pause events detected
        pitch_variation     — if librosa available, pitch std deviation
    """
    if waveform is None or len(waveform) == 0:
        return {}

    features = {}

    # RMS energy
    rms = float(np.sqrt(np.mean(waveform ** 2)))
    features["energy_rms"] = rms

    # Zero crossing rate
    signs = np.sign(waveform)
    zcr = float(np.mean(np.abs(np.diff(signs))) / 2)
    features["zcr_mean"] = zcr

    # ZCR variation (frame-level)
    frame_size = int(sample_rate * 0.025)  # 25ms frames
    hop_size   = int(sample_rate * 0.010)  # 10ms hop
    frames = [waveform[i:i+frame_size]
              for i in range(0, len(waveform) - frame_size, hop_size)]
    if frames:
        zcr_per_frame = np.array([
            float(np.mean(np.abs(np.diff(np.sign(f))) / 2))
            for f in frames if len(f) == frame_size
        ])
        features["zcr_std"] = float(np.std(zcr_per_frame)) if len(zcr_per_frame) > 0 else 0.0

        # Energy per frame
        energy_per_frame = np.array([
            float(np.sqrt(np.mean(f ** 2)))
            for f in frames if len(f) == frame_size
        ])
        if len(energy_per_frame) > 0 and np.mean(energy_per_frame) > 0:
            features["energy_variation"] = float(
                np.std(energy_per_frame) / np.mean(energy_per_frame)
            )
        else:
            features["energy_variation"] = 0.0

    # Spectral centroid (approximate via FFT)
    try:
        fft_mag = np.abs(np.fft.rfft(waveform[:sample_rate]))  # first second
        freqs   = np.fft.rfftfreq(sample_rate, 1.0/sample_rate)
        if fft_mag.sum() > 0:
            centroid = float(np.sum(freqs * fft_mag) / fft_mag.sum())
            features["spectral_centroid"] = centroid
    except Exception:
        features["spectral_centroid"] = 0.0

    # Silence ratio
    features["silence_ratio"] = detect_silence_ratio(waveform)

    # Speech duration estimate
    speech_duration = len(waveform) / sample_rate * (1.0 - features["silence_ratio"])
    features["speech_duration_s"] = speech_duration

    # Pause detection (consecutive silent frames)
    threshold_linear = 10 ** (-40.0 / 20.0)
    if frames:
        is_silent = energy_per_frame < threshold_linear
        pause_count = 0
        in_pause = False
        for s in is_silent:
            if s and not in_pause:
                pause_count += 1
                in_pause = True
            elif not s:
                in_pause = False
        features["pause_count"] = pause_count
        # Speech rate estimate: rough syllable count from energy peaks
        if speech_duration > 0:
            # Energy peaks as proxy for syllable nuclei
            from_mean = energy_per_frame - np.mean(energy_per_frame)
            peaks = 0
            for i in range(1, len(from_mean) - 1):
                if from_mean[i] > from_mean[i-1] and from_mean[i] > from_mean[i+1] and from_mean[i] > 0:
                    peaks += 1
            # Convert frame peaks to per-second estimate
            total_duration = len(waveform) / sample_rate
            features["speech_rate_estimate"] = peaks / max(total_duration, 1)
        else:
            features["speech_rate_estimate"] = 0.0

    # Pitch variation (librosa only)
    librosa = _try_librosa()
    if librosa is not None:
        try:
            f0, voiced, _ = librosa.pyin(
                waveform,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=sample_rate,
            )
            f0_voiced = f0[voiced] if f0 is not None and voiced is not None else np.array([])
            if len(f0_voiced) > 1:
                features["pitch_mean"] = float(np.mean(f0_voiced))
                features["pitch_std"]  = float(np.std(f0_voiced))
                features["pitch_variation"] = float(
                    np.std(f0_voiced) / max(np.mean(f0_voiced), 1)
                )
            else:
                features["pitch_variation"] = 0.0
        except Exception:
            features["pitch_variation"] = 0.0

    return features


# ---------------------------------------------------------------------------
# SNR estimate
# ---------------------------------------------------------------------------

def estimate_snr(waveform: np.ndarray, sample_rate: int) -> float:
    """
    Rough SNR estimate in dB.
    Compares high-energy (speech) frames to low-energy (noise floor) frames.
    """
    if waveform is None or len(waveform) == 0:
        return 0.0
    frame_size = int(sample_rate * 0.025)
    frames = [waveform[i:i+frame_size]
              for i in range(0, len(waveform) - frame_size, frame_size)]
    if not frames:
        return 0.0
    energies = np.array([np.sqrt(np.mean(f**2)) for f in frames if len(f) == frame_size])
    energies = energies[energies > 0]
    if len(energies) < 4:
        return 0.0
    energies_sorted = np.sort(energies)
    noise_level  = np.mean(energies_sorted[:max(1, len(energies_sorted)//10)])
    signal_level = np.mean(energies_sorted[-max(1, len(energies_sorted)//5):])
    if noise_level == 0:
        return 60.0
    snr = 20 * np.log10(signal_level / noise_level)
    return float(np.clip(snr, 0.0, 60.0))
