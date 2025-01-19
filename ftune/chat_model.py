from load_data import DATA_PATH
from text_generator import TextGenerator

# Инициализация модели
generator = TextGenerator(
    model_name='my_ft_model_gpt_2',  # Название вашей модели
    data_path=DATA_PATH             # Путь к данным
)

print("=== Chat with the fine-tuned model ===")
print("Type 'exit' to quit the chat.\n")

while True:
    # Ввод вопроса от пользователя
    instruction = input("You: ")
    
    # Выход из чата
    if instruction.lower() == 'exit':
        print("Exiting the chat. Goodbye!")
        break
    
    # Генерация ответа
    generated_texts = generator.generate_text(
        instruction=instruction,
        max_length=100,        # Длина ответа
        num_return_sequences=1, # Генерируем только 1 ответ
        do_sample=True,         # Активируем выбор случайных слов
        temperature=0.95,       # Настройка уверенности модели
        top_k=10,               # Количество рассматриваемых вариантов
        top_p=0.95              # Параметр ядерного сэмплинга
    )
    
    # Вывод ответа
    response = generated_texts['generated_texts'][0]
    print(f"Model: {response}\n")
