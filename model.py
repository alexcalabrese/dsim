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
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionDetectionModel:
    """
    A class for emotion detection from audio features using a CNN architecture.
    """
    
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int = 6, config_path: str = "config.yaml"):
        """
        Initialize the emotion detection model.
        
        Parameters
        ----------
        input_shape : Tuple[int, ...]
            Shape of input features
        num_classes : int, optional
            Number of emotion classes, by default 6
        config_path : str, optional
            Path to the configuration file, by default "config.yaml"
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.emotion_mapping = {
            'ANG': 'anger',
            'DIS': 'disgust',
            'FEA': 'fear',
            'HAP': 'happy',
            'NEU': 'neutral',
            'SAD': 'sad'
        }
        
        self.model = self._build_model()
        
    def _build_model(self) -> models.Model:
        """
        Build and compile the CNN model for emotion detection, with adjustments to reduce overfitting.
        
        Returns
        -------
        models.Model
            Compiled Keras model
        """
        l2_reg = tf.keras.regularizers.l2(1e-4)  # L2 regularization factor
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=self.input_shape),
            
            # First Convolutional Block with L2 regularization and increased dropout
            layers.Conv2D(self.config['model']['conv_filters'][0], (3, 3), 
                          activation='relu', padding='same', kernel_regularizer=l2_reg),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.config['model'].get('dropout_rate', 0.3)),  # increased dropout
            
            # Second Convolutional Block with L2 regularization and increased dropout
            layers.Conv2D(self.config['model']['conv_filters'][1], (3, 3), 
                          activation='relu', padding='same', kernel_regularizer=l2_reg),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.config['model'].get('dropout_rate', 0.3)),
            
            # Third Convolutional Block with L2 regularization and increased dropout
            layers.Conv2D(self.config['model']['conv_filters'][2], (3, 3), 
                          activation='relu', padding='same', kernel_regularizer=l2_reg),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(self.config['model'].get('dropout_rate', 0.3)),
            
            # Global Average Pooling to reduce overfitting compared to Flattening
            layers.GlobalAveragePooling2D(),
            
            # Dense Layers with additional regularization
            layers.Dense(self.config['model']['dense_units'][0], activation='relu', kernel_regularizer=l2_reg),
            layers.BatchNormalization(),
            layers.Dropout(self.config['model'].get('dropout_rate', 0.3) ),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile model with learning rate from config
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config['training']['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def save_evaluation_plots(self, 
                            history: Dict,
                            y_true: Optional[np.ndarray] = None,
                            y_pred: Optional[np.ndarray] = None,
                            eval_dir: Optional[Path] = None) -> None:
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
        eval_dir : Optional[Path]
            Directory to save evaluation plots, if None uses config path
        """
        try:
            # Use provided eval_dir if available, otherwise fallback to config path
            if eval_dir is None:
                eval_dir = Path(self.config['paths']['evaluation_dir']) / f"evaluation_{self.timestamp}"
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
              save_plots: bool = True,
              eval_dir: Optional[Path] = None) -> Dict:
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
            Training metrics and history
        """
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=1
        )
        
        # Ensure that the history is correctly captured
        metrics = {
            'train_accuracy': float(history.history['accuracy'][-1]),
            'accuracy_history': history.history['accuracy'],
            'loss_history': history.history['loss']
        }
        
        if validation_data:
            val_pred = self.model.predict(X_val)
            metrics.update({
                'val_accuracy': float(history.history['val_accuracy'][-1]),
                'val_accuracy_history': history.history['val_accuracy'],
                'val_loss_history': history.history['val_loss']
            })
            
            # Calculate per-class metrics
            if y_val is not None:  # Type guard for mypy
                y_val_true = np.argmax(y_val, axis=1)
                y_val_pred = np.argmax(val_pred, axis=1)
                
                # Get confusion matrix
                cm = confusion_matrix(y_val_true, y_val_pred)
                # Store confusion matrix values
                for i in range(len(self.emotion_mapping)):
                    for j in range(len(self.emotion_mapping)):
                        metrics[f'confusion_matrix_{i}_{j}'] = int(cm[i, j])
                
                # Get classification report for detailed metrics
                report = classification_report(
                    y_val_true,
                    y_val_pred,
                    target_names=list(self.emotion_mapping.values()),
                    output_dict=True,
                    zero_division=0
                )
                
                # Add overall metrics
                metrics.update({
                    'precision': report['weighted avg']['precision'],
                    'recall': report['weighted avg']['recall'],
                    'f1': report['weighted avg']['f1-score']
                })
                
                # Add per-class metrics
                for emotion in self.emotion_mapping.values():
                    metrics[f'{emotion}_precision'] = report[emotion]['precision']
                    metrics[f'{emotion}_recall'] = report[emotion]['recall']
                    metrics[f'{emotion}_f1'] = report[emotion]['f1-score']
        
        # Save evaluation plots if required
        if save_plots and eval_dir:
            self.save_evaluation_plots(
                history.history,
                y_true=y_val,
                y_pred=val_pred if validation_data else None,
                eval_dir=eval_dir
            )
        
        return metrics
    
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