
import pandas as pd

from pathlib import Path

DATA_PATH = Path('train_model/')
DATA_PATH.mkdir(parents=True, exist_ok=True)

# Load the dataset
# Загрузка JSON файла
json_path = "../data/dataset.json"  # Укажите путь к вашему JSON-файлу

dataset = pd.read_json(json_path)

# После успешной загрузки датасета, содержимое подмножества train (тренировочные данные)
# преобразуется в DataFrame с помощью библиотеки pandas. 
# Это упрощает обработку данных, 
# так как DataFrame предоставляет множество удобных функций для манипуляции и анализа данных.
data_df = pd.DataFrame(dataset)
print("Number of rows and columns in the data set:", data_df.shape)
print(data_df.head(5))
print(data_df.info())


#Препроцессинг данных

work_data = data_df
#Замена переносов строк
work_data['response'] = data_df['response'].str.replace('\\n', ' ')
#Ограничение объема данных

print("Вывод первого текста из датасета:", work_data['response'][0]) 

unique_instruction_ru = work_data['instruction'].unique().tolist()

print("Вывод первого инструкции из датасета:", unique_instruction_ru[0])


