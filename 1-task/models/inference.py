"""
Inference pipeline with audio preprocessing and model prediction
"""

import torch
import numpy as np
import librosa
import os

class EmotionPredictor:
    """End-to-end emotion prediction pipeline
    
    Features:
        - Real-time audio processing
        - Batch prediction support
        - Confidence thresholding
    """
    
    def __init__(self, model_path, feature_params, device='cuda'):
        """
        Args:
            model_path (str): Path to trained model checkpoint
            feature_params (dict): Feature extraction parameters
            device (str): Computation device
        """
        self.device = torch.device(device)
        self.model = self._load_model(model_path).to(self.device)
        self.feature_params = feature_params
        self.classes = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']
        
    def predict(self, audio_path, threshold=0.5):
        """Predict emotion from audio file"""
        try:
            # Preprocess audio
            features = self._preprocess_audio(audio_path)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(features.to(self.device))
                probs = torch.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                
            if conf.item() < threshold:
                return "Unknown", conf.item()
                
            return self.classes[pred.item()], conf.item()
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return "Error", 0.0
            
    def _preprocess_audio(self, audio_path):
        """Process audio file into model input format"""
        waveform, sr = librosa.load(
            audio_path,
            sr=self.feature_params['sample_rate']
        )
        features = self._extract_features(waveform, sr, self.feature_params)
        return torch.tensor(features).unsqueeze(0)  # Add batch dim
        
    def _load_model(self, model_path):
        """Load trained model with safety checks"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        model = torch.load(model_path)
        model.eval()
        return model

    def _extract_features(self, waveform, sr, feature_params):
        """Extract audio features using librosa"""
        features = {}
        
        # MFCCs
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sr,
            n_mfcc=feature_params['n_mfcc'],
            hop_length=feature_params['hop_length']
        )
        features['mfcc'] = mfcc
        
        # Add other features (melspectrogram, chroma, etc)
        if feature_params.get('include_mel', False):
            mel = librosa.feature.melspectrogram(
                y=waveform,
                sr=sr,
                n_mels=feature_params.get('n_mels', 128)
            )
            features['mel'] = librosa.power_to_db(mel)
            
        return features 