# Fine-Tuning LLM Project

A Python project for fine-tuning GPT-2 language models on custom instruction-response datasets. This project enables you to train a GPT-2 model on your own data and interact with it through a chat interface.

## Project Overview

This project fine-tunes a GPT-2 model using instruction-response pairs. The model learns to generate responses based on given instructions, making it suitable for question-answering and conversational tasks.

## Project Structure

```
fune_tunning_llm/
├── ftune/                  # Main package directory
│   ├── fune_tuner.py      # Fine-tuning class and logic
│   ├── load_data.py       # Data loading and preprocessing
│   ├── train_model.py     # Training script
│   ├── text_generator.py  # Text generation class
│   ├── result.py          # Result testing script
│   └── chat_model.py      # Interactive chat interface
├── data/                   # Data directory
│   ├── dataset.json       # Training dataset (instruction-response pairs)
│   └── 11dataset_test.json
├── scripts/                # Alternative scripts directory
├── requirements.txt        # Python dependencies
└── ReadMe.md              # This file
```

## Features

- Fine-tune GPT-2 models on custom datasets
- Instruction-response format training
- Text generation with configurable parameters
- Interactive chat interface
- Data preprocessing and formatting

## Requirements

- Python 3.7+
- PyTorch
- Transformers library
- Pandas
- See requirements.txt for complete dependency list

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Format

The dataset should be a JSON file with the following structure:

```json
[
    {
        "instruction": "Your question or instruction here",
        "response": "Expected response here"
    },
    ...
]
```

The dataset file should be located at `data/dataset.json`.

## Usage

### Step 1: Prepare Your Environment

1. Ensure you have a `ftune` directory (or create it)
2. Ensure your dataset is in `data/dataset.json`
3. Make sure the `train_model` directory is empty or doesn't contain a previously trained model (if you want to train a new model)
4. Ensure `my_train_dataset.txt` doesn't exist (it will be created during training)

### Step 2: Train the Model

Run the training script to fine-tune the GPT-2 model:

```bash
python ftune/train_model.py
```

This will:
- Load and preprocess the dataset
- Prepare training data in the required format
- Fine-tune the GPT-2 model
- Save the trained model to `train_model/my_ft_model_gpt_2/`

Training parameters can be adjusted in `ftune/train_model.py`:
- `num_train_epochs`: Number of training epochs (default: 4)
- `per_device_train_batch_size`: Batch size (default: 4)
- `learning_rate`: Learning rate (default: 5e-5)

### Step 3: Test the Model

Generate sample responses using the trained model:

```bash
python ftune/result.py
```

This script will:
- Load the trained model
- Generate responses for sample instructions
- Display the generated text

### Step 4: Chat with the Model

Start an interactive chat session:

```bash
python ftune/chat_model.py
```

Type your questions and the model will generate responses. Type 'exit' to quit.

## Configuration

### Model Settings

The default model used is `gpt2`. You can change this in `ftune/fune_tuner.py`:

```python
FineTuner(model_name='gpt2')  # Change to other GPT-2 variants if needed
```

### Generation Parameters

Text generation parameters can be adjusted in `ftune/chat_model.py` or `ftune/result.py`:

- `max_length`: Maximum length of generated text (default: 100)
- `temperature`: Controls randomness (0.0-1.0, default: 0.95)
- `top_k`: Number of top tokens to consider (default: 10)
- `top_p`: Nucleus sampling threshold (default: 0.95)
- `do_sample`: Enable sampling for diversity (default: True)

## Code Components

### FineTuner Class (`fune_tuner.py`)

Handles model initialization, data preparation, and fine-tuning:
- `prepare_data()`: Formats dataset for training
- `fine_tune()`: Trains the model on prepared data

### TextGenerator Class (`text_generator.py`)

Manages text generation from the trained model:
- `generate_text()`: Generates responses based on instructions
- Supports various generation parameters for customization

### Data Loading (`load_data.py`)

- Loads JSON dataset
- Preprocesses data (removes newlines, formats text)
- Creates necessary directories

## Output Files

After training, the following files will be created:

- `train_model/my_ft_model_gpt_2/`: Directory containing the trained model and tokenizer
- `train_model/my_train_dataset.txt`: Formatted training dataset
- `train_model/logs/`: Training logs directory

## Notes

- Training time depends on dataset size and hardware capabilities
- Ensure sufficient disk space for model storage
- GPU acceleration is recommended for faster training
- The model uses instruction-response format with special tokens (`<instruction>` and `<response>`)

## Troubleshooting

- If training fails, ensure the dataset format is correct
- Check that you have sufficient memory/disk space
- Verify all dependencies are installed correctly
- Make sure the `train_model` directory exists and is writable

## License

This project is provided as-is for educational and research purposes.
