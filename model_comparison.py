import numpy as np
import logging
from typing import Dict, Tuple, List, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineModel:
    """
    A baseline logistic regression model for emotion classification.
    """
    
    def __init__(self):
        self.model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.history = {
            'accuracy_history': [],
            'val_accuracy_history': []
        }
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def preprocess_features(self, features_dict: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Preprocess features for the baseline model.
        
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
            # Flatten and concatenate all features
            flat_features = []
            for feature_name, feature_array in feature_dict.items():
                if feature_array.size > 1:  # Skip single-value features
                    # Calculate statistical features
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
        """
        Train the baseline model and track metrics.
        """
        logger.info("Preprocessing features...")
        X_train_processed = self.preprocess_features(X_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_processed)
        
        logger.info("Training baseline model...")
        self.model.fit(X_train_scaled, np.argmax(y_train, axis=1))
        
        # Calculate training metrics
        train_pred = self.model.predict_proba(X_train_scaled)
        train_accuracy = accuracy_score(np.argmax(y_train, axis=1), 
                                     np.argmax(train_pred, axis=1))
        self.history['accuracy_history'].append(train_accuracy)
        
        metrics = {
            'train_accuracy': train_accuracy,
            'accuracy_history': self.history['accuracy_history'],
            'loss_history': []  # Baseline model doesn't compute loss
        }
        
        # Calculate validation metrics if validation data is provided
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
                'val_loss_history': []  # Baseline model doesn't compute loss
            })
            
            # Calculate per-class metrics
            y_val_true = np.argmax(y_val, axis=1)
            y_val_pred = np.argmax(val_pred, axis=1)
            
            # Get classification report for detailed metrics
            report = classification_report(
                y_val_true,
                y_val_pred,
                target_names=list(emotion_mapping.values()) if emotion_mapping else None,
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
            if emotion_mapping:
                for emotion in emotion_mapping.values():
                    metrics[f'{emotion}_precision'] = report[emotion]['precision']
                    metrics[f'{emotion}_recall'] = report[emotion]['recall']
                    metrics[f'{emotion}_f1'] = report[emotion]['f1-score']
            
            # Save evaluation plots
            if base_dir:
                self.save_evaluation_plots(
                    y_val_true,
                    y_val_pred,
                    metrics,
                    emotion_mapping,
                    Path(base_dir)
                )
        
        return metrics
    
    def save_evaluation_plots(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            metrics: Dict,
                            emotion_mapping: Dict[str, str],
                            base_dir: Path) -> None:
        """
        Save evaluation plots for the baseline model.
        
        Parameters
        ----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        metrics : Dict
            Model metrics
        emotion_mapping : Dict[str, str]
            Mapping of emotion labels
        base_dir : Path
            Directory to save plots
        """
        try:
            # Create baseline subdirectory
            eval_dir = base_dir / 'baseline'
            eval_dir.mkdir(parents=True, exist_ok=True)
            
            # Set style for better visualizations
            plt.style.use('default')  # Reset to default style
            plt.rcParams.update({
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'axes.grid': True,
                'grid.alpha': 0.3,
                'axes.labelsize': 11,
                'axes.titlesize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'font.family': 'sans-serif'
            })
            
            # 1. Save confusion matrix with improved styling
            plt.figure(figsize=(12, 10))
            cm = confusion_matrix(y_true, y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            # Plot both raw counts and percentages
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Raw counts
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=list(emotion_mapping.values()),
                       yticklabels=list(emotion_mapping.values()),
                       ax=ax1, square=True, cbar_kws={'shrink': .8})
            ax1.set_title('Confusion Matrix (Counts)', pad=20, fontsize=12, fontweight='bold')
            ax1.set_xlabel('Predicted', fontsize=11)
            ax1.set_ylabel('True', fontsize=11)
            
            # Percentages
            sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='RdYlBu_r',
                       xticklabels=list(emotion_mapping.values()),
                       yticklabels=list(emotion_mapping.values()),
                       ax=ax2, square=True, cbar_kws={'shrink': .8})
            ax2.set_title('Confusion Matrix (Percentages)', pad=20, fontsize=12, fontweight='bold')
            ax2.set_xlabel('Predicted', fontsize=11)
            ax2.set_ylabel('True', fontsize=11)
            
            plt.suptitle('Baseline Model Confusion Matrices', fontsize=14, fontweight='bold', y=1.05)
            plt.tight_layout()
            plt.savefig(eval_dir / 'confusion_matrices.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            # 2. Per-class metrics visualization
            report = classification_report(
                y_true,
                y_pred,
                target_names=list(emotion_mapping.values()),
                output_dict=True,
                zero_division=0  # Handle undefined metrics gracefully
            )
            
            # Create per-class metrics plot
            metrics_fig, ax = plt.subplots(figsize=(12, 6))
            emotions = list(emotion_mapping.values())
            metrics_data = {
                'Precision': [report[emotion]['precision'] for emotion in emotions],
                'Recall': [report[emotion]['recall'] for emotion in emotions],
                'F1-Score': [report[emotion]['f1-score'] for emotion in emotions]
            }
            
            x = np.arange(len(emotions))
            width = 0.25
            multiplier = 0
            colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
            
            for metric, scores in metrics_data.items():
                offset = width * multiplier
                ax.bar(x + offset, scores, width, label=metric, 
                      color=colors[multiplier], alpha=0.8)
                multiplier += 1
            
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title('Per-class Performance Metrics', fontsize=12, fontweight='bold', pad=20)
            ax.set_xticks(x + width)
            ax.set_xticklabels(emotions, rotation=45, ha='right')
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for i in range(len(emotions)):
                for j, metric in enumerate(metrics_data.keys()):
                    height = metrics_data[metric][i]
                    ax.text(i + j * width, height, f'{height:.2f}',
                           ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            plt.savefig(eval_dir / 'per_class_metrics.png', bbox_inches='tight', dpi=300)
            plt.close()
            
            # Save metrics and report
            with open(eval_dir / 'classification_report.json', 'w') as f:
                json.dump(report, f, indent=4)
            
            with open(eval_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"Saved baseline evaluation results to {eval_dir}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation plots: {str(e)}")
            raise
    
    def predict(self, X: List[Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Make predictions with the baseline model.
        
        Parameters
        ----------
        X : List[Dict[str, np.ndarray]]
            Input features
            
        Returns
        -------
        np.ndarray
            Predicted probabilities
        """
        X_processed = self.preprocess_features(X)
        X_scaled = self.scaler.transform(X_processed)
        return self.model.predict_proba(X_scaled)

def compare_models(cnn_metrics: Dict[str, float],
                  baseline_metrics: Dict[str, float],
                  save_dir: Path) -> None:
    """
    Compare and visualize the performance of CNN and baseline models.
    """
    try:
        # Set style for better visualizations
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
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. Overall Accuracy Comparison (Left)
        models = ['CNN', 'Baseline']
        train_acc = [cnn_metrics.get('train_accuracy', 0), baseline_metrics.get('train_accuracy', 0)]
        val_acc = [cnn_metrics.get('val_accuracy', 0), baseline_metrics.get('val_accuracy', 0)]
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, train_acc, width, label='Training', 
                       color='#2ecc71', alpha=0.8)
        bars2 = ax1.bar(x + width/2, val_acc, width, label='Validation', 
                       color='#3498db', alpha=0.8)
        
        ax1.set_ylabel('Accuracy', fontsize=13)
        ax1.set_title('Overall Model Accuracy', pad=20, fontsize=15, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontweight='bold', fontsize=12)
        ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.1%}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=11,
                           fontweight='bold')
        
        autolabel(bars1)
        autolabel(bars2)
        
        # 2. Per-class Performance Comparison (Right)
        emotions = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad']
        cnn_per_class = [cnn_metrics.get(f'{emotion}_f1', 0) for emotion in emotions]
        baseline_per_class = [baseline_metrics.get(f'{emotion}_f1', 0) for emotion in emotions]
        
        x = np.arange(len(emotions))
        width = 0.35
        
        bars3 = ax2.bar(x - width/2, cnn_per_class, width, label='CNN', 
                       color='#2ecc71', alpha=0.8)
        bars4 = ax2.bar(x + width/2, baseline_per_class, width, label='Baseline', 
                       color='#3498db', alpha=0.8)
        
        ax2.set_ylabel('F1-Score', fontsize=13)
        ax2.set_title('Per-class Performance (F1-Score)', pad=20, fontsize=15, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(emotions, rotation=30, ha='right', fontsize=11)
        ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
        
        def autolabel_small(bars):
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.1%}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=10)
        
        autolabel_small(bars3)
        autolabel_small(bars4)
        
        # Add overall title and adjust layout
        fig.suptitle('Model Performance Comparison', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save the comparison plot
        plt.savefig(save_dir / 'model_comparison.png', bbox_inches='tight', dpi=300)
        plt.close()
        
        # Save comparison metrics
        comparison = {
            'cnn': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                   for k, v in cnn_metrics.items()},
            'baseline': baseline_metrics,
            'performance_gap': {
                'train_accuracy_gap': float(cnn_metrics.get('train_accuracy', 0) - 
                                         baseline_metrics.get('train_accuracy', 0)),
                'val_accuracy_gap': float(cnn_metrics.get('val_accuracy', 0) - 
                                       baseline_metrics.get('val_accuracy', 0)),
                'per_class_gaps': {
                    emotion: float(cnn_metrics.get(f'{emotion}_f1', 0) - 
                                 baseline_metrics.get(f'{emotion}_f1', 0))
                    for emotion in emotions
                }
            }
        }
        
        with open(save_dir / 'model_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=4)
        
        logger.info(f"Saved model comparison results to {save_dir}")
        
    except Exception as e:
        logger.error(f"Error saving model comparison: {str(e)}")
        raise 