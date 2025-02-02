import os
import yaml
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YOLOtoRCNNConverter:
    """Convert YOLO format dataset to Faster R-CNN format.
    
    This class handles the conversion of a YOLO format dataset to a format suitable
    for training Faster R-CNN models. The conversion includes:
    1. Converting normalized YOLO coordinates to absolute pixel coordinates
    2. Creating annotation files in COCO-like format
    3. Organizing the dataset structure as required by Faster R-CNN
    """
    
    def __init__(
        self,
        yolo_dataset_path: str,
        output_path: str,
        min_box_size: int = 10,
        min_visibility: float = 0.3
    ):
        """Initialize the converter.
        
        Parameters
        ----------
        yolo_dataset_path : str
            Path to the root of the YOLO format dataset
        output_path : str
            Path where the converted dataset will be saved
        min_box_size : int
            Minimum size (in pixels) for a bounding box to be included
        min_visibility : float
            Minimum visibility (area ratio) for a bounding box to be included
        """
        self.yolo_path = Path(yolo_dataset_path)
        self.output_path = Path(output_path)
        self.min_box_size = min_box_size
        self.min_visibility = min_visibility
        
        # Load YOLO dataset config
        with open(self.yolo_path / 'dataset.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Create class mapping
        self.classes = self.config['names']
        self.num_classes = len(self.classes)
        
        # Create output directories
        self._create_directories()
        
    def _create_directories(self) -> None:
        """Create the necessary directory structure for Faster R-CNN."""
        logger.info("Creating directory structure...")
        
        # Create main directories
        for split in ['train', 'val', 'test']:
            (self.output_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_path / split / 'annotations').mkdir(parents=True, exist_ok=True)
            
    def _convert_yolo_to_absolute(
        self,
        box: List[float],
        image_width: int,
        image_height: int
    ) -> List[int]:
        """Convert YOLO format coordinates to absolute pixel coordinates.
        
        Parameters
        ----------
        box : List[float]
            YOLO format box [x_center, y_center, width, height]
        image_width : int
            Width of the image in pixels
        image_height : int
            Height of the image in pixels
            
        Returns
        -------
        List[int]
            Absolute coordinates [x_min, y_min, x_max, y_max]
        """
        x_center, y_center, width, height = box
        
        # Convert to absolute coordinates
        x_min = int((x_center - width/2) * image_width)
        y_min = int((y_center - height/2) * image_height)
        x_max = int((x_center + width/2) * image_width)
        y_max = int((y_center + height/2) * image_height)
        
        # Ensure coordinates are within image bounds
        x_min = max(0, min(x_min, image_width))
        y_min = max(0, min(y_min, image_height))
        x_max = max(0, min(x_max, image_width))
        y_max = max(0, min(y_max, image_height))
        
        return [x_min, y_min, x_max, y_max]
    
    def _is_valid_box(
        self,
        box: List[int],
        image_width: int,
        image_height: int
    ) -> bool:
        """Check if a bounding box is valid.
        
        Parameters
        ----------
        box : List[int]
            Absolute coordinates [x_min, y_min, x_max, y_max]
        image_width : int
            Width of the image in pixels
        image_height : int
            Height of the image in pixels
            
        Returns
        -------
        bool
            True if the box is valid, False otherwise
        """
        x_min, y_min, x_max, y_max = box
        
        # Check box size
        width = x_max - x_min
        height = y_max - y_min
        
        if width < self.min_box_size or height < self.min_box_size:
            return False
            
        # Check box area ratio
        box_area = width * height
        image_area = image_width * image_height
        if box_area / image_area < self.min_visibility:
            return False
            
        return True
    
    def _create_annotation_dict(
        self,
        image_id: int,
        filename: str,
        width: int,
        height: int,
        boxes: List[List[int]],
        class_ids: List[int]
    ) -> Dict:
        """Create an annotation dictionary for an image.
        
        Parameters
        ----------
        image_id : int
            Unique identifier for the image
        filename : str
            Name of the image file
        width : int
            Width of the image in pixels
        height : int
            Height of the image in pixels
        boxes : List[List[int]]
            List of bounding boxes in [x_min, y_min, x_max, y_max] format
        class_ids : List[int]
            List of class IDs corresponding to each box
            
        Returns
        -------
        Dict
            Annotation dictionary in COCO-like format
        """
        annotations = []
        
        for box, class_id in zip(boxes, class_ids):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            
            annotation = {
                'bbox': [x_min, y_min, width, height],
                'category_id': class_id,
                'area': width * height,
                'iscrowd': 0
            }
            annotations.append(annotation)
            
        return {
            'file_name': filename,
            'height': height,
            'width': width,
            'id': image_id,
            'annotations': annotations
        }
    
    def convert_split(self, split: str) -> None:
        """Convert a dataset split from YOLO to Faster R-CNN format.
        
        Parameters
        ----------
        split : str
            Dataset split to convert ('train', 'val', or 'test')
        """
        logger.info(f"Converting {split} split...")
        
        # Get paths
        yolo_images_dir = self.yolo_path / split / 'images'
        yolo_labels_dir = self.yolo_path / split / 'labels'
        rcnn_images_dir = self.output_path / split / 'images'
        rcnn_annotations_dir = self.output_path / split / 'annotations'
        
        # Initialize annotation data
        annotations = []
        image_id = 0
        
        # Process each image
        image_files = list(yolo_images_dir.glob('*.jpg'))
        for image_file in tqdm(image_files, desc=f"Processing {split} images"):
            # Load image
            image = Image.open(image_file)
            width, height = image.size
            
            # Get corresponding label file
            label_file = yolo_labels_dir / f"{image_file.stem}.txt"
            if not label_file.exists():
                logger.warning(f"No label file found for {image_file.name}")
                continue
                
            # Read and convert annotations
            boxes = []
            class_ids = []
            
            with open(label_file, 'r') as f:
                for line in f:
                    class_id, *box = map(float, line.strip().split())
                    abs_box = self._convert_yolo_to_absolute(box, width, height)
                    
                    if self._is_valid_box(abs_box, width, height):
                        boxes.append(abs_box)
                        class_ids.append(int(class_id))
            
            if boxes:  # Only include images with valid boxes
                # Copy image
                shutil.copy2(image_file, rcnn_images_dir / image_file.name)
                
                # Create annotation
                annotation = self._create_annotation_dict(
                    image_id,
                    image_file.name,
                    width,
                    height,
                    boxes,
                    class_ids
                )
                annotations.append(annotation)
                image_id += 1
        
        # Save annotations
        annotation_file = rcnn_annotations_dir / f"{split}_annotations.json"
        with open(annotation_file, 'w') as f:
            json.dump({
                'images': annotations,
                'categories': [
                    {'id': idx, 'name': name}
                    for idx, name in self.classes.items()
                ]
            }, f, indent=2)
            
        logger.info(f"Converted {len(annotations)} images for {split} split")
    
    def convert_dataset(self) -> None:
        """Convert the entire dataset from YOLO to Faster R-CNN format."""
        logger.info("Starting dataset conversion...")
        
        # Convert each split
        for split in ['train', 'val', 'test']:
            self.convert_split(split)
            
        logger.info("Dataset conversion completed successfully!")
        
        # Save dataset configuration
        config = {
            'name': 'webui_elements',
            'num_classes': self.num_classes,
            'classes': self.classes,
            'train_path': str(self.output_path / 'train'),
            'val_path': str(self.output_path / 'val'),
            'test_path': str(self.output_path / 'test')
        }
        
        with open(self.output_path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

def main():
    """Main function to run the conversion."""
    # Set paths
    yolo_dataset_path = "Data/3_WebUI_7k/yolo_dataset"
    output_path = "Data/3_WebUI_7k/rcnn_dataset"
    
    # Create converter and run conversion
    converter = YOLOtoRCNNConverter(
        yolo_dataset_path=yolo_dataset_path,
        output_path=output_path,
        min_box_size=10,
        min_visibility=0.3
    )
    
    converter.convert_dataset()

if __name__ == "__main__":
    main() 