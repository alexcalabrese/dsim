import argparse
import logging
from pathlib import Path
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from data_processor import DataProcessor
from model import EmotionDetectionModel, prepare_features_for_model
from utils import load_audio, extract_features, normalize_features

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(data_dir: str, model_save_path: Optional[str] = None):
    """
    Train the emotion detection model on the CREMA-D dataset.
    
    Parameters
    ----------
    data_dir : str
        Path to the data directory
    model_save_path : Optional[str], optional
        Path to save the trained model, by default None
    """
    try:
        logger.info("Initializing data processor...")
        processor = DataProcessor(data_dir)
        
        logger.info("Preparing dataset...")
        dataset = processor.prepare_dataset()
        
        # Get input shape from first training example
        sample_features = prepare_features_for_model(dataset['train_features'][0])
        input_shape = sample_features.shape[1:]
        
        logger.info("Initializing model...")
        model = EmotionDetectionModel(input_shape=input_shape)
        
        logger.info("Preparing training data...")
        X_train = np.vstack([prepare_features_for_model(f) for f in dataset['train_features']])
        X_val = np.vstack([prepare_features_for_model(f) for f in dataset['val_features']])
        
        logger.info("Starting model training...")
        history = model.train(
            X_train=X_train,
            y_train=dataset['train_labels'],
            X_val=X_val,
            y_val=dataset['val_labels'],
            epochs=5
        )
        
        # Plot training history
        plot_training_history(history)
        
        if model_save_path:
            logger.info(f"Saving model to {model_save_path}")
            model.save_model(model_save_path)
        
        # Evaluate model
        logger.info("Evaluating model on test set...")
        X_test = np.vstack([prepare_features_for_model(f) for f in dataset['test_features']])
        evaluate_model(model, X_test, dataset['test_labels'], processor.get_label_mapping())
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        raise

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
    # plt.savefig('training_history.png')
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

def predict_emotion(model_path: str, audio_file: str):
    """
    Predict emotion for a single audio file.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model
    audio_file : str
        Path to the audio file
    """
    try:
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
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_dir', type=str, required=True,
                             help='Path to the data directory')
    train_parser.add_argument('--model_save_path', type=str,
                             help='Path to save the trained model')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict emotion from audio')
    predict_parser.add_argument('--model_path', type=str, required=True,
                              help='Path to the saved model')
    predict_parser.add_argument('--audio_file', type=str, required=True,
                              help='Path to the audio file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args.data_dir, args.model_save_path)
    elif args.command == 'predict':
        predict_emotion(args.model_path, args.audio_file)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 