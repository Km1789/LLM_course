# main.py
from config import settings
from loader import create_dataloader
from tokenizer_utils import get_tokenizer
from embedding import EmbeddingLayer
import torch


def main():
    print(f"Запуск проекта: {settings.text_file}")

    # 1. Читаем текст и создаем DataLoader
    with open(settings.text_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    dataloader = create_dataloader(raw_text, shuffle=False)

    # 2. Инициализируем токенизатор для получения размера словаря
    tokenizer = get_tokenizer()
    vocab_size = tokenizer.vocab_size

    # 3. Создаём слой эмбеддингов
    embedding_layer = EmbeddingLayer(
        vocab_size=vocab_size,
        max_length=settings.max_length,
        embed_dim=settings.embed_dim,
    )

    print(f"Размер словаря: {vocab_size}")
    print(f"Размер эмбеддинга: {settings.embed_dim}")

    # 4. Прогоняем первый батч через эмбеддинги
    print("\n=== Проверка слоя Embedding ===")
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        # batch_x имеет форму [batch_size, seq_length] -> [2, 4]
        print(f"Входные IDs (X): {batch_x.shape}")

        # Пропускаем через слой
        embedded_x = embedding_layer(batch_x)

        # embedded_x должен иметь форму [batch_size, seq_length, embed_dim] -> [2, 4, 16]
        print(f"Векторные представления: {embedded_x.shape}")

        # Проверка: действительно ли мы сложили токены и позиции?
        # Возьмем первый элемент первого примера
        first_vector = embedded_x[0, 0, :]
        print(
            f"Первый вектор (пример): {first_vector[:5]}..."
        )  # Покажем первые 5 чисел

        # Проверка градиентов (важно для обучения!)
        # Если мы хотим обучать модель, тензор должен требовать градиенты
        print(f"Требует градиентов: {embedded_x.requires_grad}")

        break  # Только первый батч


if __name__ == "__main__":
    main()
