import numpy as np
import librosa
import soundfile as sf
from typing import Tuple, List, Dict, Optional
import logging
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_audio(file_path: str, config_path: str = "config.yaml") -> Tuple[np.ndarray, int]:
    """
    Load audio file using librosa.
    
    Parameters
    ----------
    file_path : str
        Path to audio file
    config_path : str
        Path to configuration file
        
    Returns
    -------
    Tuple[np.ndarray, int]
        Audio signal and sample rate
    """
    config = load_config(config_path)
    try:
        signal, sr = librosa.load(file_path, sr=config['features']['sample_rate'])
        return signal, sr
    except Exception as e:
        logger.error(f"Error loading audio file {file_path}: {str(e)}")
        raise

def extract_features(signal: np.ndarray, sr: int, config_path: str = "config.yaml") -> Dict[str, np.ndarray]:
    """
    Extract audio features using librosa.
    
    Parameters
    ----------
    signal : np.ndarray
        Audio signal
    sr : int
        Sample rate
    config_path : str
        Path to configuration file
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of extracted features
    """
    try:
        config = load_config(config_path)
        features_config = config['features']
        
        # Initialize features dictionary
        features = {}
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=features_config['n_mfcc'],
            n_fft=features_config['n_fft'],
            hop_length=features_config['hop_length'],
            n_mels=features_config['n_mels']
        )
        features['mfcc'] = mfcc
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_fft=features_config['n_fft'],
            hop_length=features_config['hop_length'],
            win_length=features_config['win_length'],
            window=features_config['window'],
            n_mels=features_config['n_mels']
        )
        features['mel_spectrogram'] = librosa.power_to_db(mel_spec)
        
        # Extract chroma features
        chroma = librosa.feature.chroma_stft(
            y=signal,
            sr=sr,
            n_chroma=features_config['n_chroma'],
            n_fft=features_config['n_fft'],
            hop_length=features_config['hop_length']
        )
        features['chroma'] = chroma
        
        # Extract spectral contrast
        contrast = librosa.feature.spectral_contrast(
            y=signal,
            sr=sr,
            n_fft=features_config['n_fft'],
            hop_length=features_config['hop_length']
        )
        features['spectral_contrast'] = contrast
        
        # Extract additional spectral features
        if 'spectral_centroid' in features_config['spectral_features']:
            centroid = librosa.feature.spectral_centroid(
                y=signal,
                sr=sr,
                n_fft=features_config['n_fft'],
                hop_length=features_config['hop_length']
            )
            features['spectral_centroid'] = centroid
            
        if 'spectral_bandwidth' in features_config['spectral_features']:
            bandwidth = librosa.feature.spectral_bandwidth(
                y=signal,
                sr=sr,
                n_fft=features_config['n_fft'],
                hop_length=features_config['hop_length']
            )
            features['spectral_bandwidth'] = bandwidth
            
        if 'spectral_rolloff' in features_config['spectral_features']:
            rolloff = librosa.feature.spectral_rolloff(
                y=signal,
                sr=sr,
                n_fft=features_config['n_fft'],
                hop_length=features_config['hop_length']
            )
            features['spectral_rolloff'] = rolloff
            
        if 'spectral_flatness' in features_config['spectral_features']:
            flatness = librosa.feature.spectral_flatness(
                y=signal,
                n_fft=features_config['n_fft'],
                hop_length=features_config['hop_length']
            )
            features['spectral_flatness'] = flatness
            
        if 'zero_crossing_rate' in features_config['spectral_features']:
            zcr = librosa.feature.zero_crossing_rate(
                y=signal,
                hop_length=features_config['hop_length']
            )
            features['zero_crossing_rate'] = zcr
        
        # Extract rhythm features
        tempo, beats = librosa.beat.beat_track(
            y=signal,
            sr=sr,
            hop_length=features_config['tempo_hop_length']
        )
        features['tempo'] = np.array([[tempo]])
        
        # Extract onset features
        onset_env = librosa.onset.onset_strength(
            y=signal,
            sr=sr,
            hop_length=features_config['onset_hop_length']
        )
        features['onset_strength'] = onset_env.reshape(1, -1)
        
        # Extract tempogram
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=features_config['tempo_hop_length']
        )
        features['tempogram'] = tempogram
        
        return features
        
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise

def normalize_features(features: Dict[str, np.ndarray], config_path: str = "config.yaml") -> Dict[str, np.ndarray]:
    """
    Normalize extracted features.
    
    Parameters
    ----------
    features : Dict[str, np.ndarray]
        Dictionary of extracted features
    config_path : str
        Path to configuration file
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary of normalized features
    """
    try:
        config = load_config(config_path)
        if not config['features']['normalize_features']:
            return features
            
        normalized = {}
        min_val, max_val = config['features']['normalization_range']
        
        for name, feature in features.items():
            if feature.size > 1:  # Skip single-value features like tempo
                feature_min = np.min(feature)
                feature_max = np.max(feature)
                if feature_max > feature_min:
                    normalized[name] = (feature - feature_min) * (max_val - min_val) / (feature_max - feature_min) + min_val
                else:
                    normalized[name] = feature
            else:
                normalized[name] = feature
                
        return normalized
        
    except Exception as e:
        logger.error(f"Error normalizing features: {str(e)}")
        raise

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