"""
Data loading and preprocessing module.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Optional

try:
    from .config import DATASET_PATH, ensure_directories
except ImportError:
    from config import DATASET_PATH, ensure_directories

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure directories exist
ensure_directories()

# For backward compatibility
DATA_PATH = Path('train_model/')
DATA_PATH.mkdir(parents=True, exist_ok=True)


def load_dataset(dataset_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load dataset from JSON file.
    
    Args:
        dataset_path: Path to the dataset JSON file. If None, uses default path.
        
    Returns:
        DataFrame containing the dataset.
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist.
        ValueError: If dataset format is invalid.
    """
    if dataset_path is None:
        dataset_path = DATASET_PATH
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    try:
        logger.info(f"Loading dataset from {dataset_path}")
        dataset = pd.read_json(dataset_path)
        data_df = pd.DataFrame(dataset)
        
        # Validate required columns
        required_columns = ['instruction', 'response']
        missing_columns = [col for col in required_columns if col not in data_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Dataset loaded: {data_df.shape[0]} rows, {data_df.shape[1]} columns")
        return data_df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset.
    
    Args:
        df: Input DataFrame with 'instruction' and 'response' columns.
        
    Returns:
        Preprocessed DataFrame.
    """
    logger.info("Preprocessing data...")
    
    # Create a copy to avoid modifying original
    work_data = df.copy()
    
    # Remove newlines from response
    work_data['response'] = work_data['response'].str.replace('\\n', ' ', regex=False)
    work_data['response'] = work_data['response'].str.replace('\n', ' ', regex=False)
    
    # Remove empty responses
    initial_count = len(work_data)
    work_data = work_data[work_data['response'].str.strip().astype(bool)]
    removed_count = initial_count - len(work_data)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} rows with empty responses")
    
    logger.info(f"Preprocessing complete: {len(work_data)} rows remaining")
    return work_data


# Load and preprocess data (for backward compatibility)
try:
    data_df = load_dataset()
    work_data = preprocess_data(data_df)
    unique_instruction_ru = work_data['instruction'].unique().tolist()
    
    logger.info(f"Dataset ready: {len(work_data)} samples")
    logger.info(f"Unique instructions: {len(unique_instruction_ru)}")
    
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    data_df = pd.DataFrame()
    work_data = pd.DataFrame()
    unique_instruction_ru = []

