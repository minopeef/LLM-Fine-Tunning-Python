"""
Training script for fine-tuning GPT-2 model.
"""
import logging
import sys
from pathlib import Path

try:
    from .fune_tuner import FineTuner
    from .load_data import load_dataset, preprocess_data
    from .config import DEFAULT_OUTPUT_MODEL_NAME, ensure_directories
except ImportError:
    from fune_tuner import FineTuner
    from load_data import load_dataset, preprocess_data
    from config import DEFAULT_OUTPUT_MODEL_NAME, ensure_directories

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    try:
        # Ensure directories exist
        ensure_directories()
        
        logger.info("Starting model training process...")
        
        # Load and preprocess data
        logger.info("Loading dataset...")
        data_df = load_dataset()
        work_data = preprocess_data(data_df)
        
        if len(work_data) == 0:
            logger.error("No data available for training")
            sys.exit(1)
        
        # Initialize fine-tuner
        logger.info("Initializing fine-tuner...")
        finetuner = FineTuner()
        
        # Prepare data
        logger.info("Preparing training data...")
        dataset_path = finetuner.prepare_data(work_data)
        
        # Fine-tune model
        logger.info("Starting fine-tuning...")
        finetuner.fine_tune(dataset_path, output_name=DEFAULT_OUTPUT_MODEL_NAME)
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()