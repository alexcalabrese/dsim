"""
End-to-end training and evaluation pipeline for audio emotion detection
"""

import os
import yaml
import torch
import logging
from torch.utils.data import random_split
from sklearn.metrics import classification_report
from data.dataset import AudioDataset, create_data_loader
from models.model import EmotionCNN
from models.train import EmotionTrainingSystem
from utils.visualization import plot_confusion_matrix  # Implement this in visualization.py
import pytorch_lightning as pl

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "data": {
        "root_dir": "path/to/CREMA-D/AudioWAV",
        "metadata_path": "path/to/processedResults/tabulatedVotes.csv",
        "test_size": 0.2,
        "val_size": 0.1,
        "batch_size": 64
    },
    "features": {
        "sample_rate": 16000,
        "n_mfcc": 40,
        "hop_length": 512,
        "include_mel": True,
        "n_mels": 128
    },
    "training": {
        "max_epochs": 50,
        "learning_rate": 3e-4,
        "early_stop_patience": 5,
        "model_save_path": "models/best_model.pth"
    }
}

def main():
    logger.info("Starting training pipeline")
    logger.info(f"Using configuration: {CONFIG}")

    # Initialize dataset
    logger.info("Initializing dataset...")
    dataset = AudioDataset(
        root_dir=CONFIG["data"]["root_dir"],
        metadata_path=CONFIG["data"]["metadata_path"],
        feature_params=CONFIG["features"]
    )

    # Split dataset
    logger.info("Splitting dataset into train/val/test sets")
    test_size = int(len(dataset) * CONFIG["data"]["test_size"])
    val_size = int(len(dataset) * CONFIG["data"]["val_size"])
    train_size = len(dataset) - test_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    logger.info(f"Dataset splits - Train: {train_size}, Val: {val_size}, Test: {test_size}")

    # Create data loaders
    logger.info("Creating data loaders")
    train_loader = create_data_loader(
        train_dataset,
        batch_size=CONFIG["data"]["batch_size"],
        num_workers=4
    )
    val_loader = create_data_loader(
        val_dataset,
        batch_size=CONFIG["data"]["batch_size"],
        shuffle=False
    )
    test_loader = create_data_loader(
        test_dataset,
        batch_size=CONFIG["data"]["batch_size"],
        shuffle=False
    )

    # Model setup
    logger.info("Initializing model")
    input_shape = (1, CONFIG["features"]["n_mfcc"], 128)  # Adjust based on your feature size
    model = EmotionCNN(input_shape, num_classes=6)
    system = EmotionTrainingSystem(model, learning_rate=CONFIG["training"]["learning_rate"])

    # Training
    logger.info("Setting up trainer")
    trainer = pl.Trainer(
        max_epochs=CONFIG["training"]["max_epochs"],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[
            pl.callbacks.EarlyStopping(
                monitor="val_acc",
                patience=CONFIG["training"]["early_stop_patience"],
                mode="max"
            ),
            pl.callbacks.ModelCheckpoint(
                monitor="val_acc",
                mode="max",
                filename="best-checkpoint"
            )
        ]
    )
    
    logger.info("Starting model training")
    trainer.fit(system, train_loader, val_loader)

    # Load best model
    logger.info("Loading best model checkpoint")
    best_model = EmotionCNN.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    torch.save(best_model.state_dict(), CONFIG["training"]["model_save_path"])
    logger.info(f"Saved best model to {CONFIG['training']['model_save_path']}")

    # Evaluation
    logger.info("Starting evaluation on test set")
    trainer.test(best_model, test_loader)
    
    # Generate predictions
    logger.info("Generating predictions for detailed metrics")
    true_labels = []
    pred_labels = []
    best_model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"]
            labels = batch["labels"]
            outputs = best_model(features)
            preds = torch.argmax(outputs, dim=1)
            
            true_labels.extend(labels.tolist())
            pred_labels.extend(preds.tolist())

    # Classification report
    logger.info("Generating classification report")
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=dataset.EMOTION_CODES.values()
    )
    logger.info(f"\nClassification Report:\n{report}")

    # Confusion matrix
    logger.info("Generating confusion matrix plot")
    plot_confusion_matrix(
        true_labels,
        pred_labels,
        classes=list(dataset.EMOTION_CODES.values())
    )
    logger.info("Training pipeline completed successfully")

if __name__ == "__main__":
    main() 