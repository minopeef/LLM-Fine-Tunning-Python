from load_data import DATA_PATH, unique_name_ru, unique_rubrics
from text_generator import TextGenerator

print ( "10 НАПЗВАНИЙ",unique_name_ru[:10])
print ("10 КАТЕГОРИЙ", unique_rubrics[:10])

name_ru = unique_name_ru[1]
rubrics = unique_rubrics[1]
rating = 1

generator = TextGenerator(
    model_name='ft_model_gpt_2',
    data_path=DATA_PATH
)

generated_texts = generator.generate_text(
    name_ru=name_ru,
    rubrics=rubrics,
    rating=rating,
    max_length=100,
    num_return_sequences=3,
    do_sample=True,
    temperature=0.95,  # Слегка уменьшаем уверенность
    top_k=10,         # Уменьшаем количество рассматриваемых верхних k слов
    top_p=0.95        # Уменьшаем "ядерность" распределения
)
for i, text in enumerate(generated_texts['generated_texts']):
    print(f"Generated Text {i+1}: {text}")