import argparse
import logging
import numpy as np
from pathlib import Path
import yaml
from typing import Optional, Dict, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from numpy.typing import NDArray
import json
from datetime import datetime

from data_processor import DataProcessor
from model import EmotionDetectionModel, prepare_features_for_model
from utils import load_audio, extract_features, normalize_features
from model_comparison import BaselineModel, compare_models

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file
        
    Returns
    -------
    Dict
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def train_model(data_dir: str, model_save_path: Optional[str] = None, config_path: str = "config.yaml"):
    """
    Train both CNN and baseline models on the CREMA-D dataset.
    
    Parameters
    ----------
    data_dir : str
        Path to the data directory
    model_save_path : Optional[str]
        Path to save the trained model
    config_path : str
        Path to the configuration file
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Print training configuration fields
        logger.info("Training configuration:")
        for key, value in config['training'].items():
            print(f"{key}: {value}")
        
        # Create evaluation directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_dir = Path(config['paths']['evaluation_dir']) / f"evaluation_{timestamp}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created evaluation directory: {eval_dir}")
        
        logger.info("Initializing data processor...")
        processor = DataProcessor(data_dir)
        
        logger.info("Preparing dataset...")
        dataset = processor.prepare_dataset(
            sample_size=config['training']['dataset_percentage']
        )
        
        # Get emotion mapping
        emotion_mapping = processor.emotion_mapping
        
        # Train CNN model
        logger.info("Training CNN model...")
        cnn_metrics = train_cnn_model(
            dataset=dataset,
            model_save_path=model_save_path,
            config=config,
            config_path=config_path,
            eval_dir=eval_dir,
            timestamp=timestamp
        )
        
        # Train baseline model
        logger.info("Training baseline model...")
        baseline_metrics = train_baseline_model(
            dataset=dataset,
            config=config,
            emotion_mapping=emotion_mapping,
            eval_dir=eval_dir
        )
        
        # Compare models
        logger.info("Comparing models...")
        compare_models(
            cnn_metrics,
            baseline_metrics,
            save_dir=eval_dir
        )
        
        # Save final metrics
        metrics = {
            'timestamp': timestamp,
            'cnn': cnn_metrics,
            'baseline': baseline_metrics,
            'config': config
        }
        metrics_file = eval_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Saved metrics to {metrics_file}")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

def train_cnn_model(dataset: Dict[str, Union[NDArray, Any]], 
                   model_save_path: Optional[str], 
                   config: Dict[str, Any],
                   config_path: str,
                   eval_dir: Path,
                   timestamp: str) -> Dict[str, float]:
    """
    Train the CNN model.
    
    Parameters
    ----------
    dataset : Dict[str, Union[NDArray, Any]]
        Dataset dictionary containing features and labels
    model_save_path : Optional[str]
        Path to save the trained model
    config : Dict[str, Any]
        Configuration dictionary
    config_path : str
        Path to the configuration file
    eval_dir : Path
        Evaluation directory
    timestamp : str
        Timestamp for the evaluation directory
        
    Returns
    -------
    Dict[str, float]
        Training metrics
    """
    # Get input shape from first training example
    sample_features = prepare_features_for_model(dataset['train_features'][0])
    input_shape = sample_features.shape[1:]
    
    logger.info("Initializing CNN model...")
    model = EmotionDetectionModel(
        input_shape=input_shape,
        num_classes=6,  # Number of emotions
        config_path=config_path
    )
    
    logger.info("Preparing training data...")
    X_train = np.vstack([prepare_features_for_model(f) for f in dataset['train_features']])
    X_val = np.vstack([prepare_features_for_model(f) for f in dataset['val_features']])
    
    logger.info("Starting CNN model training...")
    metrics = model.train(
        X_train=X_train,
        y_train=dataset['train_labels'],
        X_val=X_val,
        y_val=dataset['val_labels'],
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        eval_dir=eval_dir
    )
    
    if model_save_path:
        logger.info(f"Saving CNN model to {model_save_path}")
        model.save_model(model_save_path)
    
    return metrics

def train_baseline_model(dataset: Dict[str, Union[NDArray, Any]], 
                        config: Dict[str, Any],
                        emotion_mapping: Dict[str, str],
                        eval_dir: Path) -> Dict[str, float]:
    """
    Train the baseline logistic regression model.
    
    Parameters
    ----------
    dataset : Dict[str, Union[NDArray, Any]]
        Dataset dictionary containing features and labels
    config : Dict[str, Any]
        Configuration dictionary
    emotion_mapping : Dict[str, str]
        Mapping of emotion codes to emotion names
    eval_dir : Path
        Evaluation directory
        
    Returns
    -------
    Dict[str, float]
        Training metrics
    """
    logger.info("Initializing baseline model...")
    baseline = BaselineModel()
    
    logger.info("Training baseline model...")
    metrics = baseline.train(
        X_train=dataset['train_features'],
        y_train=dataset['train_labels'],
        X_val=dataset['val_features'],
        y_val=dataset['val_labels'],
        emotion_mapping=emotion_mapping,
        base_dir=eval_dir
    )
    
    return metrics

def plot_training_history(history: dict):
    """
    Plot training history metrics.
    
    Parameters
    ----------
    history : dict
        Training history dictionary
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def evaluate_model(model: EmotionDetectionModel, 
                  X_test: np.ndarray, 
                  y_test: np.ndarray,
                  label_mapping: dict):
    """
    Evaluate the model and print performance metrics.
    
    Parameters
    ----------
    model : EmotionDetectionModel
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    label_mapping : dict
        Mapping between numeric labels and emotion names
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Print classification report
    target_names = [label_mapping[i] for i in range(len(label_mapping))]
    print("\nClassification Report:")
    print(classification_report(y_true_classes, y_pred_classes, target_names=target_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    # plt.savefig('confusion_matrix.png')
    plt.close()

def predict_emotion(model_path: str, audio_file: str, config_path: str = "config.yaml"):
    """
    Predict emotion for a single audio file.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model
    audio_file : str
        Path to the audio file
    config_path : str
        Path to the configuration file
    """
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Load model
        model = EmotionDetectionModel.load_model(model_path)
        
        # Process audio file
        signal, sr = load_audio(audio_file)
        features = extract_features(signal, sr)
        normalized_features = normalize_features(features)
        
        # Prepare features for model
        X = prepare_features_for_model(normalized_features)
        
        # Get prediction
        prediction = model.predict(X)
        emotion_idx = np.argmax(prediction[0])
        confidence = prediction[0][emotion_idx]
        
        logger.info(f"Predicted emotion: {emotion_idx} with confidence: {confidence:.2f}")
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Audio Emotion Detection')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the models')
    train_parser.add_argument('--data_dir', type=str, required=True,
                             help='Path to the data directory')
    train_parser.add_argument('--model_save_path', type=str,
                             help='Path to save the trained CNN model')
    train_parser.add_argument('--config', type=str, default='config.yaml',
                             help='Path to the configuration file')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict emotion from audio')
    predict_parser.add_argument('--model_path', type=str, required=True,
                              help='Path to the saved model')
    predict_parser.add_argument('--audio_file', type=str, required=True,
                              help='Path to the audio file')
    predict_parser.add_argument('--config', type=str, default='config.yaml',
                              help='Path to the configuration file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args.data_dir, args.model_save_path, args.config)
    elif args.command == 'predict':
        predict_emotion(args.model_path, args.audio_file, args.config)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 