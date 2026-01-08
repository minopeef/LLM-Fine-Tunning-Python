"""
Text generation module for fine-tuned GPT-2 models.
"""
import re
import logging
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    from .config import (
        TRAIN_MODEL_DIR,
        DEFAULT_OUTPUT_MODEL_NAME,
        DEFAULT_GENERATION_CONFIG,
        INSTRUCTION_TOKEN,
        RESPONSE_TOKEN
    )
except ImportError:
    from config import (
        TRAIN_MODEL_DIR,
        DEFAULT_OUTPUT_MODEL_NAME,
        DEFAULT_GENERATION_CONFIG,
        INSTRUCTION_TOKEN,
        RESPONSE_TOKEN
    )

logger = logging.getLogger(__name__)


class TextGenerator:
    """
    Text generator class for fine-tuned GPT-2 models.
    """
    
    def __init__(self, 
                 model_name: str = DEFAULT_OUTPUT_MODEL_NAME, 
                 data_path: Optional[Path] = None,
                 device: Optional[str] = None):
        """
        Initialize the text generator.
        
        Args:
            model_name: Name of the fine-tuned model directory.
            data_path: Path to the model directory. If None, uses default.
            device: Device to use ('cuda', 'cpu', or None for auto-detection).
        """
        if data_path is None:
            data_path = TRAIN_MODEL_DIR
        
        model_path = Path(data_path) / model_name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Device management
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Loading model from {model_path} on device '{self.device}'")
        
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
            self.model = GPT2LMHeadModel.from_pretrained(str(model_path))
            self.model.to(self.device)
            self.model.eval()
            
            # Ensure pad_token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    @staticmethod
    def remove_tags(text: str) -> str:
        """
        Remove HTML/XML tags from text.
        
        Args:
            text: Input text that may contain tags.
            
        Returns:
            Text with tags removed.
        """
        return re.sub(r'<.*?>', '', text)

    def generate_text(self, 
                     instruction: str, 
                     max_length: int = 100, 
                     num_return_sequences: int = 1, 
                     temperature: float = 1.0, 
                     top_k: int = 0, 
                     top_p: float = 1.0, 
                     do_sample: bool = False,
                     generation_config: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """
        Generate text based on instruction.
        
        Args:
            instruction: Input instruction/question.
            max_length: Maximum length of generated text.
            num_return_sequences: Number of sequences to generate.
            temperature: Sampling temperature (controls randomness).
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.
            do_sample: Whether to use sampling.
            generation_config: Optional dictionary to override generation parameters.
            
        Returns:
            Dictionary with 'full_texts' and 'generated_texts' lists.
        """
        if generation_config:
            max_length = generation_config.get('max_length', max_length)
            num_return_sequences = generation_config.get('num_return_sequences', num_return_sequences)
            temperature = generation_config.get('temperature', temperature)
            top_k = generation_config.get('top_k', top_k)
            top_p = generation_config.get('top_p', top_p)
            do_sample = generation_config.get('do_sample', do_sample)
        
        # Build prompt
        eos_token = self.tokenizer.eos_token
        prompt_text = f"{INSTRUCTION_TOKEN} {instruction} {eos_token} {RESPONSE_TOKEN} "
        
        # Encode input
        encoded_input = self.tokenizer.encode(prompt_text, return_tensors='pt')
        encoded_input = encoded_input.to(self.device)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                encoded_input,
                max_length=max_length + len(encoded_input[0]),
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=top_k if top_k > 0 else 50,
                top_p=top_p,
                do_sample=do_sample,
                no_repeat_ngram_size=2,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode results
        all_texts = [
            self.tokenizer.decode(output, skip_special_tokens=True) 
            for output in outputs
        ]
        
        # Extract generated text (remove prompt)
        prompt_length = len(self.tokenizer.decode(encoded_input[0], skip_special_tokens=True))
        trimmed_texts = [text[prompt_length:].strip() for text in all_texts]
        
        # Remove tags
        cleaned_texts = [self.remove_tags(text).strip() for text in trimmed_texts]
        
        return {
            "full_texts": all_texts,
            "generated_texts": cleaned_texts
        }


