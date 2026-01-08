"""
Configuration file for the fine-tuning project.
Centralizes all paths, settings, and constants.
"""
from pathlib import Path
from typing import Optional

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / 'data'
DATASET_PATH = DATA_DIR / 'dataset.json'
TEST_DATASET_PATH = DATA_DIR / '11dataset_test.json'

# Model paths
TRAIN_MODEL_DIR = PROJECT_ROOT / 'train_model'
MODEL_CACHE_DIR = TRAIN_MODEL_DIR / 'model_cache'
LOGS_DIR = TRAIN_MODEL_DIR / 'logs'
TRAIN_DATASET_FILE = TRAIN_MODEL_DIR / 'my_train_dataset.txt'

# Model configuration
DEFAULT_MODEL_NAME = 'gpt2'
DEFAULT_OUTPUT_MODEL_NAME = 'my_ft_model_gpt_2'

# Training configuration
DEFAULT_TRAINING_CONFIG = {
    'num_train_epochs': 4,
    'per_device_train_batch_size': 4,
    'learning_rate': 5e-5,
    'save_steps': 10_000,
    'block_size': 256,
    'save_total_limit': 2,
}

# Generation configuration
DEFAULT_GENERATION_CONFIG = {
    'max_length': 100,
    'num_return_sequences': 1,
    'temperature': 0.95,
    'top_k': 10,
    'top_p': 0.95,
    'do_sample': True,
    'no_repeat_ngram_size': 2,
}

# Special tokens
INSTRUCTION_TOKEN = '<instruction>'
RESPONSE_TOKEN = '<response>'
EOS_TOKEN = '<|endoftext|>'

def ensure_directories():
    """Create necessary directories if they don't exist."""
    TRAIN_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
