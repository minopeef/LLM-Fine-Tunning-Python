import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from pathlib import Path

from load_data import DATA_PATH

class FineTuner:
    def __init__(self, 
                 model_name='gpt2', 
                 cache_dir='model_cache',
                 data_path=DATA_PATH):
        self.data_path = Path(data_path)
        
        # Инициализация токенизатора и модели
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name, cache_dir=str(self.data_path / cache_dir))
        self.model = GPT2LMHeadModel.from_pretrained(model_name, cache_dir=str(self.data_path / cache_dir))

    def prepare_data(self, df):
        """
        Подготовка данных для обучения
        """
        # Объединение типа проблемы и исходного текста в одну строку входных данных
        df['input'] = df.apply(
            lambda row: f"<instruction> {row['instruction']} {self.tokenizer.eos_token}", axis=1
            )
        
        # Добавление к целевому тексту токена окончания строки
        df['output'] = df.apply(lambda row: f" <response> {row['response']} {self.tokenizer.eos_token}", axis=1)
        
        # Подготовка пути для сохранения данных
        dataset_path = self.data_path / 'my_train_dataset.txt'
        # Запись данных в файл
        with dataset_path.open('w', encoding='utf-8') as file:
            for input_text, target_text in zip(df['input'], df['output']):
                file.write(input_text + ' ' + target_text + '\n')
        return dataset_path

    def fine_tune(self, 
                  dataset_path, 
                  output_name='my_ft_model', 
                  num_train_epochs=4, 
                  per_device_train_batch_size=4, 
                  learning_rate=5e-5, 
                  save_steps=10_000):
        """
        Дообучение модели на заданном датасете.
        """
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=str(dataset_path),
            block_size=256
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

        training_args = TrainingArguments(
            output_dir=str(self.data_path / output_name),
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            save_steps=save_steps,
            learning_rate=learning_rate,
            save_total_limit=2,
            logging_dir=str(self.data_path / 'logs'),
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )

        trainer.train()
        # Сохранение обученной модели и токенизатора
        self.model.save_pretrained(str(self.data_path / output_name))
        self.tokenizer.save_pretrained(str(self.data_path / output_name))