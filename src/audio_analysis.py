"""
Audio analysis utilities for rule-based prediction adjustments.

These heuristics help improve the audio model predictions based on
acoustic characteristics commonly associated with dog age:

- Puppies: more tonal/harmonic barks (low flatness), higher tempo, bursts of energy
- Adults: balanced characteristics, less silence, moderate energy
- Seniors: lots of silence/pauses (>50%), lower sustained energy
"""
import numpy as np
import librosa


def analyze_audio(y: np.ndarray, sr: int = 16000) -> dict:
    """
    Analyze audio characteristics relevant to dog age estimation.
    
    Args:
        y: Audio time series
        sr: Sample rate
        
    Returns:
        Dictionary with audio analysis results
    """
    duration = len(y) / sr
    
    # RMS energy (loudness)
    rms = librosa.feature.rms(y=y)[0]
    mean_rms = float(np.mean(rms))
    max_rms = float(np.max(rms))
    
    # Spectral centroid (brightness/pitch indicator)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = float(np.mean(spectral_centroids))
    
    # Zero crossing rate (roughness/noisiness)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    mean_zcr = float(np.mean(zcr))
    
    # Spectral rolloff (frequency below which 85% of energy is contained)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
    mean_rolloff = float(np.mean(rolloff))
    
    # Tempo/rhythm analysis
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0]
    
    # Silence ratio (proportion of quiet frames)
    silence_threshold = 0.01
    silence_ratio = float(np.mean(rms < silence_threshold))
    
    # Spectral flatness - KEY FEATURE for puppies
    # Lower flatness = more tonal/harmonic = typical of puppy barks
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    mean_flatness = float(np.mean(flatness))
    
    # Spectral bandwidth - energy spread across frequencies
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    mean_bandwidth = float(np.mean(bandwidth))
    
    # Peak-to-mean ratio - puppies have bursty energy patterns
    peak_to_mean = max_rms / mean_rms if mean_rms > 0 else 1.0
    
    # === Classification flags based on CALIBRATED thresholds ===
    
    # Puppy indicators (based on actual puppy samples)
    is_short = duration < 1.5
    is_tonal = mean_flatness < 0.02  # Low flatness = harmonic puppy barks
    is_high_tempo = tempo > 125  # Fast bark rhythm
    is_bursty = peak_to_mean > 4.0  # High peaks relative to mean = energetic bursts
    
    # Senior indicators (based on actual senior samples)  
    has_lots_of_silence = silence_ratio > 0.50  # Only high silence (>50%) indicates senior
    is_very_quiet = mean_rms < 0.03  # Much stricter threshold
    is_low_energy = max_rms < 0.15  # Stricter threshold
    
    # Adult indicators
    has_moderate_silence = 0.15 < silence_ratio < 0.35  # Low silence = adult activity
    is_high_zcr = mean_zcr > 0.12  # Adults have rougher vocalizations
    
    return {
        'duration': duration,
        'mean_rms': mean_rms,
        'max_rms': max_rms,
        'mean_centroid': mean_centroid,
        'mean_zcr': mean_zcr,
        'mean_rolloff': mean_rolloff,
        'tempo': tempo,
        'silence_ratio': silence_ratio,
        'mean_flatness': mean_flatness,
        'mean_bandwidth': mean_bandwidth,
        'peak_to_mean': peak_to_mean,
        # Derived flags - puppy
        'is_short': is_short,
        'is_tonal': is_tonal,
        'is_high_tempo': is_high_tempo,
        'is_bursty': is_bursty,
        # Derived flags - senior
        'has_lots_of_silence': has_lots_of_silence,
        'is_very_quiet': is_very_quiet,
        'is_low_energy': is_low_energy,
        # Derived flags - adult
        'has_moderate_silence': has_moderate_silence,
        'is_high_zcr': is_high_zcr,
    }


def apply_audio_heuristics(probs: np.ndarray, analysis: dict) -> np.ndarray:
    """
    Apply rule-based adjustments to audio model probabilities.
    
    These manual adjustments compensate for limitations in the trained
    audio model by incorporating domain knowledge about dog vocalizations.
    
    Args:
        probs: numpy array of shape (3,) with [young, adult, senior] probabilities
        analysis: dict from analyze_audio()
        
    Returns:
        Adjusted probabilities (normalized to sum to 1)
    """
    adjusted = probs.copy().astype(float)
    
    # === PUPPY INDICATORS ===
    
    if analysis['is_tonal']:
        # Low spectral flatness = harmonic/tonal barks = typical puppy
        adjusted[0] *= 1.4  # Strong boost for puppy
        adjusted[2] *= 0.6  # Reduce senior
    
    if analysis['is_high_tempo']:
        # Fast rhythm = energetic puppy
        adjusted[0] *= 1.3
        adjusted[2] *= 0.7
    
    if analysis['is_bursty']:
        # High peak-to-mean ratio = bursty energy = puppy
        adjusted[0] *= 1.2
        adjusted[1] *= 1.1  # Adults also can be bursty
        adjusted[2] *= 0.7
    
    if analysis['is_short']:
        # Short barks more common in puppies
        adjusted[0] *= 1.2
        adjusted[2] *= 0.8
    
    # === ADULT INDICATORS ===
    
    if analysis['has_moderate_silence']:
        # Low silence = sustained vocalization = adult
        adjusted[1] *= 1.3
        adjusted[0] *= 0.9
        adjusted[2] *= 0.9
    
    if analysis['is_high_zcr']:
        # Higher zero-crossing = rougher vocalization = adult
        adjusted[1] *= 1.2
    
    # === SENIOR INDICATORS ===
    # (Only strong indicators - avoid false positives)
    
    if analysis['has_lots_of_silence']:
        # >50% silence is a strong senior indicator
        adjusted[2] *= 1.5
        adjusted[0] *= 0.6
    
    if analysis['is_very_quiet'] and analysis['is_low_energy']:
        # Both quiet AND low energy = senior
        adjusted[2] *= 1.3
        adjusted[0] *= 0.7
    
    # Normalize to ensure probabilities sum to 1
    adjusted = adjusted / adjusted.sum()
    
    return adjusted


def get_audio_confidence_weight(analysis: dict) -> float:
    """
    Calculate a confidence weight for the audio prediction based on signal quality.
    
    Args:
        analysis: dict from analyze_audio()
        
    Returns:
        Weight between 0.3 and 1.0 (lower for poor quality audio)
    """
    base_weight = 1.0
    
    # Reduce confidence for very short clips
    if analysis['duration'] < 0.5:
        base_weight *= 0.5
    elif analysis['duration'] < 1.0:
        base_weight *= 0.7
    
    # Reduce confidence for very quiet audio
    if analysis['mean_rms'] < 0.02:
        base_weight *= 0.6
    
    # Reduce confidence for mostly silence
    if analysis['silence_ratio'] > 0.5:
        base_weight *= 0.7
    
    return max(0.3, min(1.0, base_weight))
