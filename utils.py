import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_audio(file_path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return the signal and sample rate.
    
    Parameters
    ----------
    file_path : str
        Path to the audio file
    sr : int, optional
        Target sampling rate, by default 22050
    
    Returns
    -------
    Tuple[np.ndarray, int]
        Audio signal and sampling rate
    """
    try:
        signal, sr = librosa.load(file_path, sr=sr)
        return signal, sr
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {str(e)}")
        raise

def extract_features(signal: np.ndarray, sr: int, n_mfcc: int = 13, 
                    n_mels: int = 40) -> Dict[str, np.ndarray]:
    """
    Extract audio features from the signal.
    
    Parameters
    ----------
    signal : np.ndarray
        Audio signal
    sr : int
        Sampling rate
    n_mfcc : int, optional
        Number of MFCCs to extract, by default 13
    n_mels : int, optional
        Number of Mel bands to generate, by default 40
    
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing different audio features
    """
    try:
        # Extract various features
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        mel_spect = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
        chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
        
        # Calculate statistics
        features = {
            'mfcc': mfccs,
            'mel_spectrogram': librosa.power_to_db(mel_spect),
            'chroma': chroma,
            'spectral_contrast': spectral_contrast
        }
        
        return features
    
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise

def normalize_features(features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Normalize the extracted features.
    
    Parameters
    ----------
    features : Dict[str, np.ndarray]
        Dictionary containing different audio features
    
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing normalized features
    """
    normalized_features = {}
    
    for feature_name, feature_data in features.items():
        # Normalize each feature to range [-1, 1]
        feature_mean = np.mean(feature_data)
        feature_std = np.std(feature_data)
        if feature_std != 0:
            normalized_features[feature_name] = (feature_data - feature_mean) / feature_std
        else:
            normalized_features[feature_name] = feature_data - feature_mean
            
    return normalized_features

def segment_audio(signal: np.ndarray, sr: int, segment_duration: float = 3.0) -> List[np.ndarray]:
    """
    Segment audio signal into fixed-length segments.
    
    Parameters
    ----------
    signal : np.ndarray
        Audio signal
    sr : int
        Sampling rate
    segment_duration : float, optional
        Duration of each segment in seconds, by default 3.0
    
    Returns
    -------
    List[np.ndarray]
        List of audio segments
    """
    segment_length = int(sr * segment_duration)
    segments = []
    
    # Pad signal if it's shorter than segment_length
    if len(signal) < segment_length:
        signal = np.pad(signal, (0, segment_length - len(signal)))
    
    # Split signal into segments
    for i in range(0, len(signal), segment_length):
        segment = signal[i:i + segment_length]
        if len(segment) == segment_length:
            segments.append(segment)
    
    return segments 