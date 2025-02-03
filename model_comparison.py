import numpy as np
import logging
from typing import Dict, Tuple, List, Optional, Any
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_running_in_colab() -> bool:
    """Check if the code is running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def setup_drive_directory(base_path: str) -> Path:
    """
    Set up Google Drive directory for saving results if running in Colab.
    
    Parameters
    ----------
    base_path : str
        Base path for saving results
        
    Returns
    -------
    Path
        Path object pointing to the save directory
    """
    if is_running_in_colab():
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Create base directory in Google Drive
            drive_base_path = Path('/content/drive/MyDrive/emotion_detection')
            drive_base_path.mkdir(parents=True, exist_ok=True)
            
            # Create specific directory
            save_dir = drive_base_path / base_path
            save_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Set up Google Drive directory at: {save_dir}")
            return save_dir
        except Exception as e:
            logger.error(f"Error setting up Google Drive: {str(e)}")
            raise
    else:
        # Use local directory
        save_dir = Path(base_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

class BaseModel:
    """Base class for all traditional ML models."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.history = {
            'accuracy_history': [],
            'val_accuracy_history': []
        }
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def preprocess_features(self, features_dict: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Preprocess features for the model.
        
        Parameters
        ----------
        features_dict : List[Dict[str, np.ndarray]]
            List of dictionaries containing features
            
        Returns
        -------
        np.ndarray
            Flattened and concatenated features
        """
        processed_features = []
        
        for feature_dict in features_dict:
            flat_features = []
            for feature_name, feature_array in feature_dict.items():
                if feature_array.size > 1:
                    stats = [
                        np.mean(feature_array),
                        np.std(feature_array),
                        np.min(feature_array),
                        np.max(feature_array),
                        np.median(feature_array),
                        np.percentile(feature_array, 25),
                        np.percentile(feature_array, 75)
                    ]
                    flat_features.extend(stats)
            
            processed_features.append(flat_features)
            
        return np.array(processed_features)

    def train(self,
             X_train: List[Dict[str, np.ndarray]],
             y_train: np.ndarray,
             X_val: Optional[List[Dict[str, np.ndarray]]] = None,
             y_val: Optional[np.ndarray] = None,
             emotion_mapping: Optional[Dict[str, str]] = None,
             base_dir: Optional[str] = None) -> Dict[str, float]:
        """Train the model and track metrics."""
        logger.info(f"Preprocessing features for {self.__class__.__name__}...")
        X_train_processed = self.preprocess_features(X_train)
        X_train_scaled = self.scaler.fit_transform(X_train_processed)
        
        logger.info(f"Training {self.__class__.__name__}...")
        self.model.fit(X_train_scaled, np.argmax(y_train, axis=1))
        
        train_pred = self.model.predict_proba(X_train_scaled)
        train_accuracy = accuracy_score(np.argmax(y_train, axis=1), 
                                     np.argmax(train_pred, axis=1))
        self.history['accuracy_history'].append(train_accuracy)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'accuracy_history': self.history['accuracy_history'],
            'loss_history': []
        }
        
        if X_val is not None and y_val is not None:
            X_val_processed = self.preprocess_features(X_val)
            X_val_scaled = self.scaler.transform(X_val_processed)
            
            val_pred = self.model.predict_proba(X_val_scaled)
            val_accuracy = accuracy_score(np.argmax(y_val, axis=1), 
                                       np.argmax(val_pred, axis=1))
            self.history['val_accuracy_history'].append(val_accuracy)
            
            metrics.update({
                'val_accuracy': val_accuracy,
                'val_accuracy_history': self.history['val_accuracy_history'],
                'val_loss_history': []
            })
            
            y_val_true = np.argmax(y_val, axis=1)
            y_val_pred = np.argmax(val_pred, axis=1)
            
            report = classification_report(
                y_val_true,
                y_val_pred,
                target_names=list(emotion_mapping.values()) if emotion_mapping else None,
                output_dict=True,
                zero_division=0
            )
            
            metrics.update({
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1': report['weighted avg']['f1-score']
            })
            
            if emotion_mapping:
                for emotion in emotion_mapping.values():
                    metrics[f'{emotion}_precision'] = report[emotion]['precision']
                    metrics[f'{emotion}_recall'] = report[emotion]['recall']
                    metrics[f'{emotion}_f1'] = report[emotion]['f1-score']
            
            if base_dir:
                # Set up save directory (local or Google Drive)
                save_dir = setup_drive_directory(base_dir)
                model_dir = save_dir / self.__class__.__name__.lower()
                
                self.save_evaluation_plots(
                    y_val_true,
                    y_val_pred,
                    metrics,
                    emotion_mapping,
                    model_dir
                )
        
        return metrics
    
    def predict(self, X: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """Make predictions with the model."""
        X_processed = self.preprocess_features(X)
        X_scaled = self.scaler.transform(X_processed)
        return self.model.predict_proba(X_scaled)

    def save_evaluation_plots(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            metrics: Dict[str, Any],
                            emotion_mapping: Dict[str, str],
                            save_dir: Path) -> None:
        """Save evaluation plots for the model."""
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Plot confusion matrices
            plt.figure(figsize=(20, 8))
            
            # Raw counts
            plt.subplot(1, 2, 1)
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=list(emotion_mapping.values()),
                       yticklabels=list(emotion_mapping.values()))
            plt.title(f'{self.__class__.__name__} Confusion Matrix (Counts)')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            # Percentages
            plt.subplot(1, 2, 2)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='RdYlBu_r',
                       xticklabels=list(emotion_mapping.values()),
                       yticklabels=list(emotion_mapping.values()))
            plt.title(f'{self.__class__.__name__} Confusion Matrix (Percentages)')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            plt.tight_layout()
            plt.savefig(save_dir / 'confusion_matrices.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            # Save per-class metrics
            plt.figure(figsize=(12, 6))
            metrics_data = {
                'Precision': [metrics[f'{emotion}_precision'] for emotion in emotion_mapping.values()],
                'Recall': [metrics[f'{emotion}_recall'] for emotion in emotion_mapping.values()],
                'F1-Score': [metrics[f'{emotion}_f1'] for emotion in emotion_mapping.values()]
            }
            
            x = np.arange(len(emotion_mapping))
            width = 0.25
            multiplier = 0
            
            for metric, scores in metrics_data.items():
                offset = width * multiplier
                plt.bar(x + offset, scores, width, label=metric)
                multiplier += 1
            
            plt.xlabel('Emotions')
            plt.ylabel('Score')
            plt.title(f'{self.__class__.__name__} Per-class Performance Metrics')
            plt.xticks(x + width, emotion_mapping.values(), rotation=45)
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_dir / 'per_class_metrics.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            # Save metrics to JSON
            with open(save_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            
        except Exception as e:
            logger.error(f"Error saving evaluation plots: {str(e)}")
            raise

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )

class DecisionTreeModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            min_samples_split=5
        )

class RandomForestModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )

class SVMModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )

class NaiveBayesModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = GaussianNB()

class GradientBoostingModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )

class XGBoostModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42,
            n_jobs=-1
        )

def compare_models(mlp_metrics: Dict[str, float],
                  traditional_models_metrics: Dict[str, Dict[str, float]],
                  save_dir: Path) -> None:
    """
    Compare and visualize the performance of MLP and traditional ML models.
    
    Parameters
    ----------
    mlp_metrics : Dict[str, float]
        Metrics from the MLP model
    traditional_models_metrics : Dict[str, Dict[str, float]]
        Dictionary of metrics from traditional ML models
    save_dir : Path
        Directory to save comparison plots
    """
    try:
        # Set up save directory (local or Google Drive)
        save_dir = setup_drive_directory(str(save_dir))
        
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'font.family': 'sans-serif'
        })
        
        # Combine all metrics
        all_metrics = {'MLP': mlp_metrics}
        all_metrics.update(traditional_models_metrics)
        
        # 1. Overall Accuracy Comparison
        plt.figure(figsize=(15, 8))
        
        models = list(all_metrics.keys())
        train_acc = [metrics.get('train_accuracy', 0) for metrics in all_metrics.values()]
        val_acc = [metrics.get('val_accuracy', 0) for metrics in all_metrics.values()]
        
        x = np.arange(len(models))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 16))
        
        # Training vs Validation Accuracy
        bars1 = ax1.bar(x - width/2, train_acc, width, label='Training',
                       color='#2ecc71', alpha=0.8)
        bars2 = ax1.bar(x + width/2, val_acc, width, label='Validation',
                       color='#3498db', alpha=0.8)
        
        ax1.set_ylabel('Accuracy', fontsize=13)
        ax1.set_title('Model Accuracy Comparison', pad=20, fontsize=15, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
        ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        # Add value labels
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.1%}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=10)
        
        autolabel(bars1)
        autolabel(bars2)
        
        # 2. F1-Scores per Emotion
        emotions = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
        f1_scores = {
            model_name: [metrics.get(f'{emotion}_f1', 0) for emotion in emotions]
            for model_name, metrics in all_metrics.items()
        }
        
        x = np.arange(len(emotions))
        width = 0.8 / len(models)
        
        for i, (model_name, scores) in enumerate(f1_scores.items()):
            offset = width * i - width * (len(models) - 1) / 2
            bars = ax2.bar(x + offset, scores, width, label=model_name, alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           rotation=90,
                           fontsize=8)
        
        ax2.set_ylabel('F1-Score', fontsize=13)
        ax2.set_title('Per-emotion F1-Scores Comparison', pad=20, fontsize=15, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(emotions, rotation=45, ha='right', fontsize=12)
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        # Save the comparison plot to the appropriate directory
        plt.savefig(save_dir / 'model_comparison.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save comparison metrics to the appropriate directory
        comparison = {
            'models': {
                model_name: {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in metrics.items()
                    if not isinstance(v, list)
                }
                for model_name, metrics in all_metrics.items()
            },
            'performance_summary': {
                'best_model_train': max(models, key=lambda m: all_metrics[m]['train_accuracy']),
                'best_model_val': max(models, key=lambda m: all_metrics[m]['val_accuracy']),
                'train_accuracy_range': {
                    'min': min(train_acc),
                    'max': max(train_acc)
                },
                'val_accuracy_range': {
                    'min': min(val_acc),
                    'max': max(val_acc)
                }
            }
        }
        
        with open(save_dir / 'model_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=4)
        
        logger.info(f"Saved model comparison results to {save_dir}")
        
    except Exception as e:
        logger.error(f"Error saving model comparison: {str(e)}")
        raise 