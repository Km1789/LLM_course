# config.py
from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "gpt2"          # Название модели токенизатора
    text_file: str = "simple_llm/cat_story.txt"  # Путь к тексту
    max_length: int = 4               # Размер контекстного окна
    stride: int = 1                   # Шаг скольжения окна
    batch_size: int = 2               # Размер пакета для обучения
    seed: int = 42                    # Для воспроизводимости результатов

# Создаем экземпляр настроек, чтобы импортировать его везде
settings = Config()