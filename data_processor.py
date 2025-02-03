from tqdm import tqdm  # Import tqdm for progress bar
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import tensorflow as tf
from utils import load_audio, extract_features, normalize_features

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    A class to handle loading and processing of the CREMA-D dataset.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the data processor.
        
        Parameters
        ----------
        data_dir : str
            Path to the data directory containing audio files and metadata
        """
        self.data_dir = Path(data_dir)
        self.label_encoder = LabelEncoder()
        self.emotion_mapping = {
            'ANG': 'anger',
            'DIS': 'disgust',
            'FEA': 'fear',
            'HAP': 'happy',
            'NEU': 'neutral',
            'SAD': 'sad'
        }
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load and process the dataset metadata.
        
        Returns
        -------
        pd.DataFrame
            Processed metadata
        """
        try:
            # Load tabulated votes
            votes_df = pd.read_csv(self.data_dir / 'processedResults' / 'tabulatedVotes.csv')
            
            # Extract emotion from filename
            votes_df['emotion'] = votes_df['fileName'].apply(
                lambda x: self.emotion_mapping[x.split('_')[2]]
            )
            
            return votes_df
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            raise
    
    def process_audio_file(self, file_path: str) -> Dict[str, np.ndarray]:
        """
        Process a single audio file and extract features.
        
        Parameters
        ----------
        file_path : str
            Path to the audio file
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of extracted and normalized features
        """
        try:
            # Try different file extensions
            base_path = file_path.rsplit('.', 1)[0]
            for ext in ['.wav', '.WAV', '.mp3', '.MP3']:
                try_path = base_path + ext
                if Path(try_path).exists():
                    file_path = try_path
                    break
            
            # Load and process audio with error handling
            try:
                signal, sr = load_audio(file_path)
            except Exception as e:
                logger.warning(f"Failed to load {file_path} with librosa, trying soundfile...")
                import soundfile as sf
                signal, sr = sf.read(file_path)
            
            features = extract_features(signal, sr)
            normalized_features = normalize_features(features)
            # Print the first few characters of the features and the shape of each column
            # features_shape = {key: value.shape for key, value in normalized_features.items()}
            # logger.info(f"Features: {normalized_features}, Shape: {features_shape}")
            
            return normalized_features
        except Exception as e:
            logger.error(f"Error processing audio file {file_path}: {str(e)}")
            raise
    
    def prepare_dataset(self, 
                       test_size: float = 0.2,
                       val_size: float = 0.1,
                       sample_size: float = 0.01,
                       random_state: int = 42) -> Tuple[Dict[str, np.ndarray], ...]:
        """
        Prepare the dataset for training.
        
        Parameters
        ----------
        test_size : float, optional
            Proportion of dataset to include in the test split, by default 0.2
        val_size : float, optional
            Proportion of dataset to include in the validation split, by default 0.1
        sample_size : float, optional
            Proportion of total dataset to use, by default 0.01 (1%)
        random_state : int, optional
            Random state for reproducibility, by default 42
            
        Returns
        -------
        Tuple[Dict[str, np.ndarray], ...]
            Training, validation, and test sets with their labels
        """
        try:
            # Load metadata
            logger.info("Loading metadata from the dataset.")
            metadata_df = self.load_metadata()
            logger.info(f"Metadata loaded successfully with {len(metadata_df)} entries.")
            
            # Shuffle the metadata DataFrame to ensure randomness in the dataset
            metadata_df = shuffle(metadata_df, random_state=random_state)
            logger.info("Shuffled the dataset for randomness.")
            
            # Sample only a portion of the data
            if sample_size < 1.0:
                metadata_df = metadata_df.sample(
                    frac=sample_size, 
                    random_state=random_state
                ).reset_index(drop=True)
                logger.info(f"Using {len(metadata_df)} samples ({sample_size*100:.1f}% of the dataset) for training.")
            
            # Encode labels
            logger.info("Encoding labels for the emotions.")
            labels = self.label_encoder.fit_transform(metadata_df['emotion'])
            labels_onehot = tf.keras.utils.to_categorical(labels)
            logger.info("Labels encoded successfully.")
            
            # Process audio files and extract features
            features_list = []
            logger.info("Processing audio files to extract features.")
            for file_name in tqdm(metadata_df['fileName'], desc="Processing files", unit="file"):
                file_path = str(self.data_dir / 'AudioWAV' / f"{file_name}.wav")
                features = self.process_audio_file(file_path)
                features_list.append(features)
                logger.debug(f"Processed features for file: {file_name}.")
            
            # Split dataset
            logger.info("Splitting the dataset into training, validation, and test sets.")
            train_idx, test_idx = train_test_split(
                np.arange(len(features_list)),
                test_size=test_size,
                random_state=random_state,
                stratify=labels
            )
            logger.info(f"Train-test split completed. Train size: {len(train_idx)}, Test size: {len(test_idx)}.")
            
            train_idx, val_idx = train_test_split(
                train_idx,
                test_size=val_size/(1-test_size),
                random_state=random_state,
                stratify=labels[train_idx]
            )
            logger.info(f"Train-validation split completed. Validation size: {len(val_idx)}.")
            
            # Prepare data splits
            splits = {
                'train': train_idx,
                'val': val_idx,
                'test': test_idx
            }
            
            dataset = {}
            for split_name, indices in splits.items():
                split_features = [features_list[i] for i in indices]
                split_labels = labels_onehot[indices]
                
                dataset[f'{split_name}_features'] = split_features
                dataset[f'{split_name}_labels'] = split_labels
                logger.info(f"Prepared {split_name} dataset with {len(split_features)} features and {len(split_labels)} labels.")
            
            logger.info("Dataset preparation completed successfully.")
            return dataset
        
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            raise
    
    def get_label_mapping(self) -> Dict[int, str]:
        """
        Get the mapping between numeric labels and emotion names.
        
        Returns
        -------
        Dict[int, str]
            Dictionary mapping numeric labels to emotion names
        """
        return dict(enumerate(self.label_encoder.classes_)) 