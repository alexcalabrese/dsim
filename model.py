import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from typing import Dict, List, Tuple, Optional
import logging
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDetectionModel:
    """
    A class for emotion detection from audio features using a CNN architecture.
    """
    
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int = 6):
        """
        Initialize the emotion detection model.
        
        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Shape of input features
        num_classes : int, optional
            Number of emotion classes, by default 6
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.emotion_mapping = {
            'ANG': 'anger',
            'DIS': 'disgust',
            'FEA': 'fear',
            'HAP': 'happy',
            'NEU': 'neutral',
            'SAD': 'sad'
        }
        
    def _build_model(self) -> models.Model:
        """
        Build and compile the CNN model for emotion detection.
        
        Returns
        -------
        models.Model
            Compiled Keras model
        """
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def save_evaluation_plots(self, 
                            history: Dict,
                            y_true: Optional[np.ndarray] = None,
                            y_pred: Optional[np.ndarray] = None,
                            base_dir: str = "./evaluation") -> None:
        """
        Save evaluation plots including training history and confusion matrix.
        
        Parameters
        ----------
        history : Dict
            Training history dictionary
        y_true : Optional[np.ndarray]
            True labels for confusion matrix
        y_pred : Optional[np.ndarray]
            Predicted labels for confusion matrix
        base_dir : str
            Base directory to save evaluation plots
        """
        try:
            # Create evaluation directory with timestamp
            eval_dir = Path(base_dir) / f"evaluation_{self.timestamp}"
            eval_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving evaluation plots to {eval_dir}")
            
            # Close any existing plots
            plt.close('all')
            
            # Plot training history
            history_fig = plt.figure(figsize=(12, 4))
            
            # Plot accuracy
            plt.subplot(1, 2, 1)
            plt.plot(history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in history:
                plt.plot(history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Plot loss
            plt.subplot(1, 2, 2)
            plt.plot(history['loss'], label='Training Loss')
            if 'val_loss' in history:
                plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            plt.tight_layout()
            history_path = eval_dir / 'training_history.png'
            history_fig.savefig(history_path)
            plt.close(history_fig)
            
            # Save confusion matrix if validation data is provided
            if y_true is not None and y_pred is not None:
                cm_fig = plt.figure(figsize=(10, 8))
                cm = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=list(self.emotion_mapping.values()),
                           yticklabels=list(self.emotion_mapping.values()))
                plt.title('Confusion Matrix')
                plt.xlabel('Predicted')
                plt.ylabel('True')
                
                cm_path = eval_dir / 'confusion_matrix.png'
                cm_fig.savefig(cm_path, bbox_inches='tight')
                plt.close(cm_fig)
                
                # Save classification report
                report = classification_report(
                    y_true.argmax(axis=1),
                    y_pred.argmax(axis=1),
                    target_names=list(self.emotion_mapping.values())
                )
                report_path = eval_dir / 'classification_report.txt'
                with open(report_path, 'w') as f:
                    f.write(report)
            
            logger.info(f"Successfully saved evaluation plots to {eval_dir}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation plots: {str(e)}")
            raise
        finally:
            # Ensure all plots are closed
            plt.close('all')
    
    def train(self, 
              X_train: np.ndarray, 
              y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 50,
              batch_size: int = 32,
              save_plots: bool = True) -> Dict:
        """
        Train the emotion detection model.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training labels
        X_val : Optional[np.ndarray], optional
            Validation features, by default None
        y_val : Optional[np.ndarray], optional
            Validation labels, by default None
        epochs : int, optional
            Number of training epochs, by default 50
        batch_size : int, optional
            Batch size for training, by default 32
        save_plots : bool, optional
            Whether to save evaluation plots, by default True
            
        Returns
        -------
        Dict
            Training history
        """
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )
        
        if save_plots:
            # Get predictions for confusion matrix if validation data exists
            y_pred = self.predict(X_val) if X_val is not None else None
            self.save_evaluation_plots(
                history.history,
                y_true=y_val,
                y_pred=y_pred
            )
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict emotions from input features.
        
        Parameters
        ----------
        X : np.ndarray
            Input features
            
        Returns
        -------
        np.ndarray
            Predicted emotion probabilities
        """
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk with timestamp.
        
        Parameters
        ----------
        filepath : str
            Path where to save the model
        """
        try:
            # Remove .keras extension if present
            if filepath.endswith('.keras'):
                filepath = filepath[:-6]
            
            # Add timestamp and .keras extension
            filepath = f"{filepath}_{self.timestamp}.keras"
            
            # Create directory if it doesn't exist
            save_dir = Path(filepath).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving model to {filepath}")
            self.model.save(filepath)
            
            # Save model configuration and metadata
            metadata = {
                'timestamp': self.timestamp,
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'emotion_mapping': self.emotion_mapping
            }
            
            # Save metadata to JSON
            metadata_path = filepath.replace('.keras', '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Model and metadata saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load_model(cls, filepath: str) -> 'EmotionDetectionModel':
        """
        Load a saved model from disk.
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
            
        Returns
        -------
        EmotionDetectionModel
            Loaded model instance
        """
        try:
            loaded_model = models.load_model(filepath)
            instance = cls(loaded_model.input_shape[1:], loaded_model.output_shape[-1])
            instance.model = loaded_model
            return instance
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

def prepare_features_for_model(features: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Prepare extracted features for model input.
    
    Parameters
    ----------
    features : Dict[str, np.ndarray]
        Dictionary of extracted features
        
    Returns
    -------
    np.ndarray
        Processed features ready for model input
    """
    # Define fixed sizes for features
    FIXED_TIME_STEPS = 128
    FIXED_FREQ_BINS = 128
    
    # Stack features along the channel dimension
    feature_list = [
        features['mfcc'],
        features['mel_spectrogram'],
        features['chroma'],
        features['spectral_contrast']
    ]
    
    # Resize all features to fixed dimensions
    resized_features = []
    for feature in feature_list:
        # Get current dimensions
        freq_bins, time_steps = feature.shape
        
        # Create grid points
        orig_freq = np.linspace(0, 1, freq_bins)
        orig_time = np.linspace(0, 1, time_steps)
        
        # Create interpolator
        interpolator = RegularGridInterpolator(
            (orig_freq, orig_time),
            feature,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        
        # Create target grid
        new_freq = np.linspace(0, 1, FIXED_FREQ_BINS)
        new_time = np.linspace(0, 1, FIXED_TIME_STEPS)
        grid_freq, grid_time = np.meshgrid(new_freq, new_time, indexing='ij')
        points = np.stack([grid_freq.ravel(), grid_time.ravel()], axis=1)
        
        # Interpolate and reshape
        resized = interpolator(points).reshape(FIXED_FREQ_BINS, FIXED_TIME_STEPS)
        resized_features.append(resized)
    
    # Stack features
    stacked_features = np.stack(resized_features, axis=-1)
    
    # Add batch dimension if needed
    if len(stacked_features.shape) == 3:
        stacked_features = np.expand_dims(stacked_features, axis=0)
        
    return stacked_features 