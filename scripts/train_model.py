from fune_tuner import FineTuner
from load_data import work_data



finetuner = FineTuner()
dataset_path = finetuner.prepare_data(work_data)
finetuner.fine_tune(dataset_path, output_name='ft_model_gpt_2')
print("Модель успешно обучена и сохранена.")