from datasets import load_dataset
import pandas as pd

from pathlib import Path

DATA_PATH = Path('data/')
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Load the dataset

dataset = load_dataset("d0rj/geo-reviews-dataset-2023", cache_dir=DATA_PATH / 'model_cache')

# После успешной загрузки датасета, содержимое подмножества train (тренировочные данные)
# преобразуется в DataFrame с помощью библиотеки pandas. 
# Это упрощает обработку данных, 
# так как DataFrame предоставляет множество удобных функций для манипуляции и анализа данных.
data_df = pd.DataFrame(dataset['train'])
print("Number of rows and columns in the data set:", data_df.shape)
print(data_df.head(5))
print(data_df.info())


#Препроцессинг данных
#Удаление пропущенных значений
work_data = data_df.dropna(subset=['text', 'name_ru', 'rating'])
#Удаление дубликатов
work_data = work_data.drop_duplicates(subset=['text']).reset_index(drop=True)
#Замена переносов строк
work_data['text'] = work_data['text'].str.replace('\\n', ' ')
#Ограничение объема данных
work_data = work_data[:100]
print("Вывод первого текста из датасета:", work_data['text'][0]) 

unique_name_ru = work_data['name_ru'].unique().tolist()

print("Вывод первого названия из датасета:", unique_name_ru[0])
unique_rubrics = work_data['rubrics'].unique().tolist()
print("Вывод первого категории из датасета:", unique_rubrics[0])
