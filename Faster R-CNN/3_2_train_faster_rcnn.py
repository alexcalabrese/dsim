import os
import json
import time
import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import yaml
import shutil
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Compose:
    """Custom Compose class that can handle both image and target for object detection."""
    
    def __init__(self, transforms: List[Callable]):
        """Initialize with list of transforms.
        
        Parameters
        ----------
        transforms : List[Callable]
            List of transforms to apply
        """
        self.transforms = transforms
        
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[torch.Tensor, Dict]:
        """Apply transforms to both image and target.
        
        Parameters
        ----------
        image : PIL.Image
            Input image
        target : Dict
            Target dictionary containing boxes and labels
            
        Returns
        -------
        Tuple[torch.Tensor, Dict]
            Transformed image and target
        """
        # Apply augmentations first (while image is still PIL)
        for t in self.transforms:
            if isinstance(t, RandomHorizontalFlip):
                image, target = t(image, target)
                
        # Convert to tensor last
        for t in self.transforms:
            if not isinstance(t, RandomHorizontalFlip):
                image = t(image)
                
        return image, target

class RandomHorizontalFlip:
    """Custom horizontal flip that handles both image and bounding boxes."""
    
    def __init__(self, prob: float = 0.5):
        """Initialize with flip probability.
        
        Parameters
        ----------
        prob : float
            Probability of flipping
        """
        self.prob = prob
        
    def __call__(self, image: Image.Image, target: Dict) -> Tuple[Image.Image, Dict]:
        """Apply horizontal flip to image and adjust bounding boxes.
        
        Parameters
        ----------
        image : PIL.Image
            Input image
        target : Dict
            Target dictionary containing boxes and labels
            
        Returns
        -------
        Tuple[PIL.Image, Dict]
            Flipped image and adjusted target
        """
        if torch.rand(1) < self.prob:
            # Get image width (must be done before converting to tensor)
            width = image.size[0]
            
            # Flip image
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Flip boxes
            if 'boxes' in target:
                boxes = target['boxes'].clone()  # Create a copy to avoid modifying original
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]  # Swap and invert x coordinates
                target['boxes'] = boxes
                
        return image, target

class WebUIDataset(torch.utils.data.Dataset):
    """Custom Dataset for WebUI elements detection."""
    
    def __init__(
        self,
        root_dir: str,
        split: str,
        transforms: Optional[nn.Module] = None
    ):
        """Initialize the dataset.
        
        Parameters
        ----------
        root_dir : str
            Root directory containing the dataset
        split : str
            Dataset split ('train', 'val', or 'test')
        transforms : Optional[nn.Module]
            Transforms to be applied to images and targets
        """
        self.root = Path(root_dir)
        self.split = split
        self.transforms = transforms
        
        # Load annotations
        with open(self.root / split / 'annotations' / f'{split}_annotations.json', 'r') as f:
            self.data = json.load(f)
            
        self.images = self.data['images']
        self.categories = {cat['id']: cat['name'] for cat in self.data['categories']}
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get an item from the dataset.
        
        Parameters
        ----------
        idx : int
            Index of the item to get
            
        Returns
        -------
        Tuple[torch.Tensor, Dict]
            Image tensor and target dictionary containing boxes and labels
        """
        # Get image info
        img_info = self.images[idx]
        img_path = self.root / self.split / 'images' / img_info['file_name']
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Prepare target
        target = {
            'boxes': torch.tensor([ann['bbox'] for ann in img_info['annotations']], dtype=torch.float32),
            'labels': torch.tensor([ann['category_id'] for ann in img_info['annotations']], dtype=torch.int64),
            'image_id': torch.tensor([img_info['id']]),
            'area': torch.tensor([ann['area'] for ann in img_info['annotations']], dtype=torch.float32),
            'iscrowd': torch.tensor([ann['iscrowd'] for ann in img_info['annotations']], dtype=torch.int64)
        }
        
        # Convert boxes from [x, y, w, h] to [x1, y1, x2, y2]
        boxes = target['boxes']
        boxes[:, 2:] += boxes[:, :2]
        target['boxes'] = boxes
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        return img, target
    
    def __len__(self) -> int:
        """Get the length of the dataset.
        
        Returns
        -------
        int
            Number of images in the dataset
        """
        return len(self.images)

def is_colab() -> bool:
    """Check if the code is running in Google Colab.
    
    Returns
    -------
    bool
        True if running in Colab, False otherwise
    """
    try:
        import google.colab
        return True
    except ImportError:
        return False

def setup_output_dir(config: Dict) -> Tuple[Path, bool]:
    """Setup output directory, handling both local and Colab environments.
    
    Parameters
    ----------
    config : Dict
        Configuration dictionary
        
    Returns
    -------
    Tuple[Path, bool]
        Output root directory path and whether using Google Drive
    """
    using_drive = False
    
    if is_colab():
        try:
            from google.colab import drive
            # Mount Google Drive
            drive.mount('/content/drive')
            logger.info("Google Drive mounted successfully")
            
            # Create output directory in Drive
            drive_output = Path('/content/drive/MyDrive/faster_rcnn_experiments')
            drive_output.mkdir(parents=True, exist_ok=True)
            
            # Update config to use Drive path
            config['paths']['output_root'] = str(drive_output)
            using_drive = True
            
            logger.info(f"Output will be saved to Google Drive: {drive_output}")
            return drive_output, using_drive
            
        except Exception as e:
            logger.warning(f"Failed to mount Google Drive: {str(e)}")
            logger.warning("Falling back to local storage")
    
    # Use local storage
    output_dir = Path(config['paths']['output_root'])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, using_drive

def create_experiment_folder(config: Dict) -> Path:
    """Create a timestamped experiment folder.
    
    Parameters
    ----------
    config : Dict
        Configuration dictionary
        
    Returns
    -------
    Path
        Path to the experiment folder
    """
    # Setup output directory
    output_root, using_drive = setup_output_dir(config)
    
    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment name
    exp_name = f"{config['paths']['model_name']}_{timestamp}"
    
    # Create experiment folder
    exp_dir = output_root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'metrics').mkdir(exist_ok=True)
    (exp_dir / 'predictions').mkdir(exist_ok=True)
    (exp_dir / 'plots').mkdir(exist_ok=True)
    
    # Save config
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Log storage location
    if using_drive:
        logger.info(f"Experiment folder created in Google Drive: {exp_dir}")
    else:
        logger.info(f"Experiment folder created locally: {exp_dir}")
    
    return exp_dir

def get_transform(config: Dict, train: bool = True) -> Compose:
    """Get transforms for training or validation.
    
    Parameters
    ----------
    config : Dict
        Configuration dictionary
    train : bool
        Whether to get transforms for training
        
    Returns
    -------
    Compose
        Composed transforms
    """
    transforms = []
    
    # Add augmentations first (while image is still PIL)
    if train:
        transforms.append(RandomHorizontalFlip(config['augmentation']['horizontal_flip_prob']))
    
    # Convert PIL image to tensor last
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
        
    return Compose(transforms)

def create_model(num_classes: int) -> nn.Module:
    """Create Faster R-CNN model.
    
    Parameters
    ----------
    num_classes : int
        Number of classes (including background)
        
    Returns
    -------
    nn.Module
        Initialized Faster R-CNN model
    """
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')
    
    # Replace the classifier with a new one for our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter
) -> float:
    """Train the model for one epoch.
    
    Parameters
    ----------
    model : nn.Module
        Model to train
    optimizer : torch.optim.Optimizer
        Optimizer to use
    data_loader : DataLoader
        Training data loader
    device : torch.device
        Device to train on
    epoch : int
        Current epoch number
    writer : SummaryWriter
        TensorBoard writer
        
    Returns
    -------
    float
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for images, targets in tqdm(data_loader, desc=f'Training epoch {epoch}'):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        total_loss += losses.item()
        
        # Log losses
        for k, v in loss_dict.items():
            writer.add_scalar(f'Loss/{k}', v.item(), epoch)
    
    avg_loss = total_loss / len(data_loader)
    writer.add_scalar('Loss/total', avg_loss, epoch)
    
    return avg_loss

@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate the model.
    
    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    data_loader : DataLoader
        Validation/test data loader
    device : torch.device
        Device to evaluate on
        
    Returns
    -------
    Dict[str, float]
        Dictionary containing evaluation metrics
    """
    model.eval()
    
    try:
        # Try to use torchmetrics MeanAveragePrecision
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        metric = MeanAveragePrecision(box_format='xyxy')  # Specify box format
        
        for images, targets in tqdm(data_loader, desc='Evaluating'):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            predictions = model(images)
            
            # Format predictions and targets for metric computation
            preds = []
            targs = []
            for pred, target in zip(predictions, targets):
                preds.append({
                    'boxes': pred['boxes'].cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu()
                })
                targs.append({
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                })
                
            metric.update(preds, targs)
        
        # Compute metrics
        metrics = metric.compute()
        
        return {
            'map': metrics['map'].item(),
            'map_50': metrics['map_50'].item(),
            'map_75': metrics['map_75'].item()
        }
        
    except (ImportError, ModuleNotFoundError) as e:
        logger.warning(f"Could not use torchmetrics MeanAveragePrecision: {str(e)}")
        logger.warning("Using simple evaluation metrics instead")
        
        # Simple evaluation: average confidence score and detection count
        total_confidence = 0.0
        total_detections = 0
        total_images = 0
        
        for images, targets in tqdm(data_loader, desc='Evaluating'):
            images = list(img.to(device) for img in images)
            predictions = model(images)
            
            for pred in predictions:
                # Filter predictions with confidence > 0.5
                mask = pred['scores'] > 0.5
                scores = pred['scores'][mask]
                
                if len(scores) > 0:
                    total_confidence += scores.mean().item()
                    total_detections += len(scores)
                total_images += 1
        
        # Compute average metrics
        avg_confidence = total_confidence / total_images if total_images > 0 else 0
        avg_detections = total_detections / total_images if total_images > 0 else 0
        
        return {
            'avg_confidence': avg_confidence,
            'avg_detections': avg_detections,
            'total_images': total_images
        }

def visualize_predictions(
    image: torch.Tensor,
    prediction: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    class_names: Dict[int, str],
    config: Dict
) -> np.ndarray:
    """Visualize model predictions on an image.
    
    Parameters
    ----------
    image : torch.Tensor
        Input image tensor [C, H, W]
    prediction : Dict[str, torch.Tensor]
        Model predictions containing 'boxes', 'labels', and 'scores'
    target : Dict[str, torch.Tensor]
        Ground truth containing 'boxes' and 'labels'
    class_names : Dict[int, str]
        Mapping from class IDs to names
    config : Dict
        Configuration dictionary
        
    Returns
    -------
    np.ndarray
        Visualization image with predictions and ground truth
    """
    # Convert image tensor to numpy array
    image = image.cpu().permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Get visualization parameters
    conf_threshold = config['visualization']['confidence_threshold']
    box_thickness = config['visualization']['box_thickness']
    font_scale = config['visualization']['font_scale']
    font_thickness = config['visualization']['font_thickness']
    default_color = tuple(config['visualization']['colors']['default'])
    
    # Draw ground truth boxes
    for box, label in zip(target['boxes'].cpu(), target['labels'].cpu()):
        x1, y1, x2, y2 = box.int().tolist()
        class_name = class_names[label.item()]
        
        # Draw box in green for ground truth
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), box_thickness)
        cv2.putText(image, f"GT: {class_name}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
    
    # Draw predicted boxes
    for box, label, score in zip(prediction['boxes'].cpu(),
                               prediction['labels'].cpu(),
                               prediction['scores'].cpu()):
        if score < conf_threshold:
            continue
            
        x1, y1, x2, y2 = box.int().tolist()
        class_name = class_names[label.item()]
        
        # Draw box in red for predictions
        cv2.rectangle(image, (x1, y1), (x2, y2), default_color, box_thickness)
        cv2.putText(image, f"Pred: {class_name} ({score:.2f})", (x1, y1 - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, default_color, font_thickness)
    
    return image

def save_prediction_examples(
    model: nn.Module,
    data_loader: DataLoader,
    class_names: Dict[int, str],
    config: Dict,
    exp_dir: Path,
    device: torch.device
) -> None:
    """Save example predictions from the model.
    
    Parameters
    ----------
    model : nn.Module
        Trained model
    data_loader : DataLoader
        Data loader for test set
    class_names : Dict[int, str]
        Mapping from class IDs to names
    config : Dict
        Configuration dictionary
    exp_dir : Path
        Experiment directory
    device : torch.device
        Device to run predictions on
    """
    logger.info("Saving prediction examples...")
    model.eval()
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(data_loader):
            if i >= config['visualization']['num_examples']:
                break
                
            # Get predictions
            images = list(img.to(device) for img in images)
            predictions = model(images)
            
            # Visualize each image in the batch
            for j, (image, pred, target) in enumerate(zip(images, predictions, targets)):
                # Create visualization
                vis_image = visualize_predictions(
                    image,
                    pred,
                    target,
                    class_names,
                    config
                )
                
                # Save visualization
                save_path = exp_dir / 'predictions' / f'example_{i}_{j}.jpg'
                cv2.imwrite(str(save_path), vis_image)
                
    logger.info(f"Saved {config['visualization']['num_examples']} prediction examples")

def plot_training_metrics(
    metrics_history: Dict[str, List[float]],
    exp_dir: Path
) -> None:
    """Create and save training metric plots.
    
    Parameters
    ----------
    metrics_history : Dict[str, List[float]]
        Dictionary containing metric names and their values over epochs
    exp_dir : Path
        Experiment directory to save plots
    """
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Set style
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")
    
    # Plot training loss
    if 'loss' in metrics_history:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_history['loss'], marker='o')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig(plots_dir / 'training_loss.png')
        plt.close()
    
    # Plot mAP metrics
    map_metrics = ['map', 'map_50', 'map_75']
    available_maps = [m for m in map_metrics if m in metrics_history]
    
    if available_maps:
        plt.figure(figsize=(10, 6))
        for metric in available_maps:
            plt.plot(metrics_history[metric], marker='o', label=f'mAP@{metric.split("_")[-1] if "_" in metric else "all"}')
        plt.title('mAP Metrics Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'map_metrics.png')
        plt.close()
    
    # Plot detection metrics
    if 'avg_detections' in metrics_history:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_history['avg_detections'], marker='o', label='Avg Detections')
        if 'avg_confidence' in metrics_history:
            plt.plot(metrics_history['avg_confidence'], marker='s', label='Avg Confidence')
        plt.title('Detection Metrics Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(plots_dir / 'detection_metrics.png')
        plt.close()

def plot_class_distribution(dataset: WebUIDataset, exp_dir: Path) -> None:
    """Plot class distribution in the dataset.
    
    Parameters
    ----------
    dataset : WebUIDataset
        Dataset to analyze
    exp_dir : Path
        Experiment directory to save plots
    """
    plots_dir = exp_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Count instances per class
    class_counts = defaultdict(int)
    for img_info in dataset.images:
        for ann in img_info['annotations']:
            class_id = ann['category_id']
            class_counts[dataset.categories[class_id]] += 1
    
    # Sort by count
    sorted_counts = dict(sorted(class_counts.items(), key=lambda x: x[1], reverse=True))
    
    # Plot class distribution
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(sorted_counts)), list(sorted_counts.values()))
    plt.xticks(range(len(sorted_counts)), list(sorted_counts.keys()), rotation=45, ha='right')
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Number of Instances')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'class_distribution.png')
    plt.close()

def main():
    """Main training function."""
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create experiment folder (handles Colab/local automatically)
    exp_dir = create_experiment_folder(config)
    
    # Set device
    if config['hardware']['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{config['hardware']['cuda_device']}")
    else:
        device = torch.device('cpu')
    logger.info(f'Using device: {device}')
    
    # If in Colab, check if using GPU
    if is_colab():
        if torch.cuda.is_available():
            logger.info("GPU is available in Colab")
            try:
                import subprocess
                gpu_info = subprocess.check_output(['nvidia-smi']).decode('utf-8')
                logger.info("GPU Info:\n" + gpu_info)
            except Exception as e:
                logger.warning(f"Failed to get GPU info: {str(e)}")
        else:
            logger.warning("No GPU available in Colab. Training might be slow.")
    
    # Set paths
    data_dir = Path(config['dataset']['root_dir'])
    
    # Load dataset configuration
    with open(data_dir / 'config.json', 'r') as f:
        dataset_config = json.load(f)
    
    num_classes = dataset_config['num_classes'] + 1  # Add background class
    
    # Create datasets and data loaders
    dataset_train = WebUIDataset(data_dir, 'train', get_transform(config, train=True))
    dataset_val = WebUIDataset(data_dir, 'val', get_transform(config, train=False))
    
    data_loader_train = DataLoader(
        dataset_train, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        collate_fn=lambda x: tuple(zip(*x))
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Create model
    model = create_model(num_classes)
    model.to(device)
    
    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config['training']['learning_rate'],
        momentum=config['training']['momentum'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['lr_scheduler']['step_size'],
        gamma=config['training']['lr_scheduler']['gamma']
    )
    
    # Create TensorBoard writer
    writer = SummaryWriter(exp_dir / 'logs')
    
    # Setup logging file
    log_file = exp_dir / 'logs' / 'training.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Training parameters
    num_epochs = config['training']['num_epochs']
    best_map = 0.0
    
    # Initialize metrics history
    metrics_history = defaultdict(list)
    
    # Training loop
    logger.info('Starting training...')
    for epoch in range(num_epochs):
        # Train for one epoch
        avg_loss = train_one_epoch(model, optimizer, data_loader_train, device, epoch, writer)
        logger.info(f'Epoch {epoch} - Average loss: {avg_loss:.4f}')
        
        # Store loss
        metrics_history['loss'].append(avg_loss)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        metrics = evaluate(model, data_loader_val, device)
        logger.info(f'Validation metrics: {metrics}')
        
        # Store metrics
        for k, v in metrics.items():
            metrics_history[k].append(v)
        
        # Log metrics
        for k, v in metrics.items():
            writer.add_scalar(f'Metrics/{k}', v, epoch)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, exp_dir / 'checkpoints' / 'latest.pth')
        
        # Save best model
        if metrics['map'] > best_map:
            best_map = metrics['map']
            torch.save(checkpoint, exp_dir / 'checkpoints' / 'best.pth')
            logger.info(f'Saved new best model with mAP: {best_map:.4f}')
    
    writer.close()
    logger.info('Training completed!')
    
    # Plot training metrics
    logger.info('Generating metric plots...')
    plot_training_metrics(metrics_history, exp_dir)
    
    # Plot class distribution
    logger.info('Generating class distribution plot...')
    plot_class_distribution(dataset_train, exp_dir)
    
    # Evaluate on test set
    logger.info('Evaluating on test set...')
    dataset_test = WebUIDataset(data_dir, 'test', get_transform(config, train=False))
    data_loader_test = DataLoader(
        dataset_test,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    test_metrics = evaluate(model, data_loader_test, device)
    logger.info(f'Test metrics: {test_metrics}')
    
    # Save test metrics
    metrics_file = exp_dir / 'metrics' / 'test_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # After training, save prediction examples
    logger.info("Generating prediction examples...")
    save_prediction_examples(
        model,
        data_loader_test,
        dataset_test.categories,
        config,
        exp_dir,
        device
    )
    
    logger.info(f'All results saved in: {exp_dir}')

if __name__ == '__main__':
    main() 