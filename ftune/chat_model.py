"""
Interactive chat interface for the fine-tuned model.
"""
import logging
import sys
try:
    from .text_generator import TextGenerator
    from .config import DEFAULT_OUTPUT_MODEL_NAME, DEFAULT_GENERATION_CONFIG
except ImportError:
    from text_generator import TextGenerator
    from config import DEFAULT_OUTPUT_MODEL_NAME, DEFAULT_GENERATION_CONFIG

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce logging noise in chat interface
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main chat function."""
    try:
        print("=" * 80)
        print("Chat with the fine-tuned model")
        print("Type 'exit' or 'quit' to end the chat")
        print("=" * 80)
        print()
        
        # Initialize generator (load model once)
        print("Loading model...")
        try:
            generator = TextGenerator(model_name=DEFAULT_OUTPUT_MODEL_NAME)
            print("Model loaded successfully!\n")
        except FileNotFoundError as e:
            print(f"Error: Model not found. Please train the model first.")
            print(f"Details: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Chat loop
        while True:
            try:
                # Get user input
                instruction = input("You: ").strip()
                
                # Check for exit commands
                if instruction.lower() in ['exit', 'quit', 'q']:
                    print("\nExiting the chat. Goodbye!")
                    break
                
                # Skip empty inputs
                if not instruction:
                    continue
                
                # Generate response
                generated_texts = generator.generate_text(
                    instruction=instruction,
                    generation_config=DEFAULT_GENERATION_CONFIG
                )
                
                # Display response
                response = generated_texts['generated_texts'][0]
                print(f"Model: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nExiting the chat. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error generating response: {e}")
                print(f"Error: {e}\n")
                
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
