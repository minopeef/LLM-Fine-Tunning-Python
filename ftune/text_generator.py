from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
from load_data import DATA_PATH

class TextGenerator:
    def __init__(self, model_name='my_ft_model', data_path=DATA_PATH):
        """
        Инициализация модели и токенизатора.
        Загружаем модель и токенизатор из указанного пути.
        """
        model_path = Path(data_path) / model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
        self.model = GPT2LMHeadModel.from_pretrained(str(model_path))
        self.model.eval()

    def generate_text(self, 
                    instruction: str, 
                    max_length=100, 
                    num_return_sequences=1, 
                    temperature=1.0, 
                    top_k=0, 
                    top_p=1.0, 
                    do_sample=False):
        """
        Генерация текста на основе заданного начального текста (prompt) и параметров.
        
        Параметры:
        - instruction: вопрос.
        - max_length: Максимальная длина сгенерированного текста.
        - num_return_sequences: Количество возвращаемых последовательностей.
        - temperature: Контролирует разнообразие вывода.
        - top_k: Если больше 0, ограничивает количество слов для выборки только k наиболее вероятными словами.
        - top_p: Если меньше 1.0, применяется nucleus sampling.
        - do_sample: Если True, включает случайную выборку для увеличения разнообразия.
        """
        # Формирование prompt
        prompt_text = f"<instruction> {instruction} {self.tokenizer.eos_token} <response> "
        
        # Кодирование текста в формате, пригодном для модели
        encoded_input = self.tokenizer.encode(prompt_text, return_tensors='pt')
        
        # Генерация текстов
        outputs = self.model.generate(
            encoded_input,
            max_length=max_length + len(encoded_input[0]),
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            no_repeat_ngram_size=2
        )
        
        # Декодирование результатов
        all_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        # Удаление входных данных из текстов
        prompt_length = len(self.tokenizer.decode(encoded_input[0], skip_special_tokens=True))
        trimmed_texts = [text[prompt_length:] for text in all_texts]
        
        # Возврат результатов в виде словаря
        return {
            "full_texts": all_texts,
            "generated_texts": trimmed_texts
        }