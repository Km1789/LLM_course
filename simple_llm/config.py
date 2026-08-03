# config.py
from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "gpt2"          # Название модели токенизатора
    text_file: str = "simple_llm/input.txt"  # Путь к тексту
    max_length: int = 4               # Размер контекстного окна
    stride: int = 1                   # Шаг скольжения окна
    batch_size: int = 2               # Размер пакета для обучения
    seed: int = 42                    # Для воспроизводимости результатов

    # Новые параметры для эмбеддингов
    embed_dim: int = 16               # Размер вектора слова
    # vocab_size мы узнаем динамически из токенизатора, но можно задать жестко, например 50257 для GPT-2

    num_epochs: int = 10              # Количество эпох обучения

# Создаем экземпляр настроек, чтобы импортировать его везде
settings = Config()