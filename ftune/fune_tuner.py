"""
Fine-tuning module for GPT-2 models.
"""
import pandas as pd
import logging
import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    TextDataset, 
    DataCollatorForLanguageModeling, 
    Trainer, 
    TrainingArguments
)
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from .config import (
        TRAIN_MODEL_DIR, 
        MODEL_CACHE_DIR, 
        LOGS_DIR, 
        TRAIN_DATASET_FILE,
        DEFAULT_MODEL_NAME,
        DEFAULT_TRAINING_CONFIG,
        INSTRUCTION_TOKEN,
        RESPONSE_TOKEN
    )
except ImportError:
    from config import (
        TRAIN_MODEL_DIR, 
        MODEL_CACHE_DIR, 
        LOGS_DIR, 
        TRAIN_DATASET_FILE,
        DEFAULT_MODEL_NAME,
        DEFAULT_TRAINING_CONFIG,
        INSTRUCTION_TOKEN,
        RESPONSE_TOKEN
    )

logger = logging.getLogger(__name__)


class FineTuner:
    """
    Fine-tuner class for GPT-2 language models.
    """
    
    def __init__(self, 
                 model_name: str = DEFAULT_MODEL_NAME, 
                 cache_dir: Optional[Path] = None,
                 data_path: Optional[Path] = None,
                 device: Optional[str] = None):
        """
        Initialize the fine-tuner.
        
        Args:
            model_name: Name of the pre-trained model to use.
            cache_dir: Directory for model cache. If None, uses default.
            data_path: Path for training outputs. If None, uses default.
            device: Device to use ('cuda', 'cpu', or None for auto-detection).
        """
        self.data_path = Path(data_path) if data_path else TRAIN_MODEL_DIR
        self.cache_dir = Path(cache_dir) if cache_dir else MODEL_CACHE_DIR
        self.model_name = model_name
        
        # Device management
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing model '{model_name}' on device '{self.device}'")
        
        # Initialize tokenizer and model
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(
                model_name, 
                cache_dir=str(self.cache_dir)
            )
            
            # Set pad_token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = GPT2LMHeadModel.from_pretrained(
                model_name, 
                cache_dir=str(self.cache_dir)
            )
            self.model.to(self.device)
            
            logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def prepare_data(self, df: pd.DataFrame, output_path: Optional[Path] = None) -> Path:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with 'instruction' and 'response' columns.
            output_path: Path to save the prepared dataset. If None, uses default.
            
        Returns:
            Path to the prepared dataset file.
        """
        logger.info("Preparing data for training...")
        
        if output_path is None:
            output_path = TRAIN_DATASET_FILE
        
        # Validate required columns
        required_columns = ['instruction', 'response']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Use vectorized operations instead of apply for better performance
        eos_token = self.tokenizer.eos_token
        df_input = df['instruction'].astype(str)
        df_output = df['response'].astype(str)
        
        # Create input and output strings using vectorized operations
        inputs = f"{INSTRUCTION_TOKEN} " + df_input + f" {eos_token}"
        outputs = f" {RESPONSE_TOKEN} " + df_output + f" {eos_token}"
        
        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as file:
            for input_text, output_text in zip(inputs, outputs):
                file.write(f"{input_text} {output_text}\n")
        
        logger.info(f"Prepared dataset saved to {output_path} ({len(df)} samples)")
        return output_path

    def fine_tune(self, 
                  dataset_path: Path,
                  output_name: str = 'my_ft_model', 
                  training_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Fine-tune the model on the given dataset.
        
        Args:
            dataset_path: Path to the training dataset file.
            output_name: Name for the output model directory.
            training_config: Dictionary with training parameters. If None, uses defaults.
        """
        if training_config is None:
            training_config = DEFAULT_TRAINING_CONFIG.copy()
        
        logger.info(f"Starting fine-tuning with config: {training_config}")
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        try:
            # Create dataset
            train_dataset = TextDataset(
                tokenizer=self.tokenizer,
                file_path=str(dataset_path),
                block_size=training_config.get('block_size', 256)
            )

            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, 
                mlm=False
            )

            # Setup training arguments
            output_dir = self.data_path / output_name
            training_args = TrainingArguments(
                output_dir=str(output_dir),
                overwrite_output_dir=True,
                num_train_epochs=training_config.get('num_train_epochs', 4),
                per_device_train_batch_size=training_config.get('per_device_train_batch_size', 4),
                save_steps=training_config.get('save_steps', 10_000),
                learning_rate=training_config.get('learning_rate', 5e-5),
                save_total_limit=training_config.get('save_total_limit', 2),
                logging_dir=str(LOGS_DIR),
                logging_steps=100,
                report_to=None,  # Disable wandb/tensorboard by default
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
            )

            logger.info("Training started...")
            trainer.train()
            
            # Save model and tokenizer
            logger.info(f"Saving model to {output_dir}")
            self.model.save_pretrained(str(output_dir))
            self.tokenizer.save_pretrained(str(output_dir))
            
            logger.info("Fine-tuning completed successfully")
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            raise