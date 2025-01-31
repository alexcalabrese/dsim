"""
CREMA-D dataset loader and preprocessing module
Handles data loading, parsing, and transformation pipelines
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np

class AudioDataset(Dataset):
    """PyTorch dataset for loading and processing CREMA-D audio files
    
    Attributes:
        root_dir (str): Root directory of dataset
        df (pd.DataFrame): Metadata dataframe
        feature_params (dict): Audio feature extraction parameters
        emotion_map (dict): Emotion label encoding mapping
        transform (callable): Optional transform to apply to samples
    """
    
    EMOTION_CODES = {
        'ANG': 'Anger',
        'DIS': 'Disgust',
        'FEA': 'Fear',
        'HAP': 'Happy',
        'NEU': 'Neutral',
        'SAD': 'Sad'
    }

    def __init__(self, root_dir, metadata_path, feature_params, transform=None):
        """
        Args:
            root_dir (str): Path to dataset root directory
            metadata_path (str): Path to processed metadata CSV
            feature_params (dict): Parameters for feature extraction
            transform (callable, optional): Optional transform
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(metadata_path)
        self.feature_params = feature_params
        self.transform = transform
        self._validate_dataset()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Load and process a single audio sample"""
        try:
            audio_path = self._get_audio_path(idx)
            waveform, sr = librosa.load(audio_path, sr=self.feature_params['sample_rate'])
            
            # Extract features
            features = self._extract_features(waveform, sr)
            
            # Get label
            emotion = self._parse_emotion_label(audio_path)
            label = self._emotion_to_index(emotion)
            
            sample = {
                'features': features,
                'label': label,
                'audio_path': audio_path
            }

            return self.transform(sample) if self.transform else sample
            
        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            return None

    def _extract_features(self, waveform, sr):
        """Extract audio features using librosa"""
        features = {}
        
        # MFCCs
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sr,
            n_mfcc=self.feature_params['n_mfcc'],
            hop_length=self.feature_params['hop_length']
        )
        features['mfcc'] = mfcc
        
        # Add other features (melspectrogram, chroma, etc)
        if self.feature_params.get('include_mel', False):
            mel = librosa.feature.melspectrogram(
                y=waveform,
                sr=sr,
                n_mels=self.feature_params.get('n_mels', 128)
            )
            features['mel'] = librosa.power_to_db(mel)
            
        return features

    def _parse_emotion_label(self, audio_path):
        """Extract emotion label from filename"""
        filename = os.path.basename(audio_path)
        emotion_code = filename.split('_')[2]
        return self.EMOTION_CODES.get(emotion_code, 'Unknown')

    def _emotion_to_index(self, emotion):
        """Convert emotion string to class index"""
        return list(self.EMOTION_CODES.values()).index(emotion)

    def _validate_dataset(self):
        """Check dataset integrity"""
        # Implement validation checks here
        pass

def create_data_loader(dataset, batch_size=32, shuffle=True, num_workers=4):
    """Create PyTorch DataLoader with proper collation"""
    def collate_fn(batch):
        # Handle None samples from error cases
        batch = [b for b in batch if b is not None]
        return {
            'features': torch.stack([torch.tensor(b['features']) for b in batch]),
            'labels': torch.tensor([b['label'] for b in batch]),
            'paths': [b['audio_path'] for b in batch]
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    ) 