import argparse
import logging
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set up logging with INFO level for detailed output.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_metadata(data_dir: Path) -> pd.DataFrame:
    """
    Load and process the dataset metadata for exploration.
    
    Parameters
    ----------
    data_dir : Path
        Path to the data directory containing metadata file
    
    Returns
    -------
    pd.DataFrame
        Processed metadata DataFrame with emotion labels mapped to their names
    """
    try:
        metadata_path = data_dir / "processedResults" / "tabulatedVotes.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        
        df = pd.read_csv(metadata_path)
        # Apply emotion mapping similar to the train codebase.
        emotion_mapping = {
            'ANG': 'anger',
            'DIS': 'disgust',
            'FEA': 'fear',
            'HAP': 'happy',
            'NEU': 'neutral',
            'SAD': 'sad'
        }
        # Extract emotion code from filename and map it to full emotion name.
        # Assumes the filename format contains the emotion code as the third underscore-delimited token.
        df['emotion'] = df['fileName'].apply(lambda x: emotion_mapping.get(x.split('_')[2], 'unknown'))
        logger.info("Successfully loaded and processed metadata.")
        return df
    except Exception as e:
        logger.error(f"Error loading metadata: {str(e)}")
        raise

def analyze_emotion_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the emotion distribution within the metadata.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the metadata with an 'emotion' column
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the counts and percentages for each emotion
    """
    counts = df['emotion'].value_counts().reset_index()
    counts.columns = ['emotion', 'count']
    counts['percentage'] = counts['count'] / counts['count'].sum() * 100
    logger.info("Emotion distribution analysis complete.")
    return counts

def plot_emotion_distribution(df_stats: pd.DataFrame, output_path: Path = None) -> None:
    """
    Plot the emotion distribution as a bar chart.
    
    Parameters
    ----------
    df_stats : pd.DataFrame
        DataFrame containing emotion counts and percentages
    output_path : Path, optional
        Path to save the plot image. If None, plot is shown interactively.
    """
    plt.figure(figsize=(8, 6))
    sns.barplot(data=df_stats, x='emotion', y='count', palette='viridis')
    plt.title('Emotion Distribution')
    plt.xlabel('Emotion')
    plt.ylabel('Count')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        logger.info(f"Plot saved to {output_path}")
    else:
        plt.show()
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Explore dataset metadata and check for class balance."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to the data directory (default: data)"
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default=None,
        help="Path to save the generated plot. If not provided, plot is displayed interactively."
    )
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    df_metadata = load_metadata(data_dir)
    
    # Display a summary of the metadata.
    logger.info("Dataset head:")
    logger.info(df_metadata.head())
    
    # Analyze emotion distribution.
    distribution_stats = analyze_emotion_distribution(df_metadata)
    logger.info("Emotion distribution stats:")
    logger.info(distribution_stats.to_string(index=False))
    
    # Plot distribution.
    output_path = Path(args.output_plot) if args.output_plot else None
    plot_emotion_distribution(distribution_stats, output_path)

if __name__ == "__main__":
    main()