from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "gpt2"
    text_file: str = "simple_llm/cat_story.txt"
    max_length: int = 4               # Размер контекстного окна
    stride: int = 1
    batch_size: int = 2
    seed: int = 42
    
    # Новые параметры для эмбеддингов
    embed_dim: int = 16               # Размер вектора слова
    # vocab_size мы узнаем динамически из токенизатора, но можно задать жестко, например 50257 для GPT-2

settings = Config()