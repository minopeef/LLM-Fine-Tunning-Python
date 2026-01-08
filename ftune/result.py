"""
Result testing script for the fine-tuned model.
"""
import logging
try:
    from .load_data import load_dataset, preprocess_data, unique_instruction_ru
    from .text_generator import TextGenerator
    from .config import DEFAULT_OUTPUT_MODEL_NAME, DEFAULT_GENERATION_CONFIG
except ImportError:
    from load_data import load_dataset, preprocess_data, unique_instruction_ru
    from text_generator import TextGenerator
    from config import DEFAULT_OUTPUT_MODEL_NAME, DEFAULT_GENERATION_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main testing function."""
    try:
        logger.info("Loading model and dataset...")
        
        # Load dataset to get unique instructions
        if not unique_instruction_ru:
            data_df = load_dataset()
            work_data = preprocess_data(data_df)
            unique_instructions = work_data['instruction'].unique().tolist()
        else:
            unique_instructions = unique_instruction_ru
        
        if len(unique_instructions) == 0:
            logger.error("No instructions available for testing")
            return
        
        logger.info(f"Found {len(unique_instructions)} unique instructions")
        logger.info(f"Sample instructions: {unique_instructions[:10]}")
        
        # Initialize generator
        generator = TextGenerator(model_name=DEFAULT_OUTPUT_MODEL_NAME)
        
        # Test with first instruction
        if len(unique_instructions) > 1:
            test_instruction = unique_instructions[1]
        else:
            test_instruction = unique_instructions[0]
        
        logger.info(f"\nTesting with instruction: {test_instruction}")
        logger.info("=" * 80)
        
        # Generate multiple responses
        generated_texts = generator.generate_text(
            instruction=test_instruction,
            max_length=100,
            num_return_sequences=3,
            do_sample=True,
            temperature=0.95,
            top_k=10,
            top_p=0.95
        )
        
        for i, text in enumerate(generated_texts['generated_texts'], 1):
            logger.info(f"Generated Text {i}: {text}")
        
        # Test with specific question
        logger.info("\n" + "=" * 80)
        logger.info("Testing with specific question: 'How do I build a Flutter APK?'")
        result = generator.generate_text(
            "How do I build a Flutter APK?", 
            num_return_sequences=1,
            **DEFAULT_GENERATION_CONFIG
        )
        logger.info(f"Response: {result['generated_texts'][0]}")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)


if __name__ == "__main__":
    main()