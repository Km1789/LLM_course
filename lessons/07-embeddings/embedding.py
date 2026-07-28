import torch
import torch.nn as nn
from config import settings


class TokenEmbedding(nn.Embedding):
    """
    Слой эмбеддингов токенов.
    Преобразует индексы токенов (целые числа) в плотные векторы фиксированной размерности.
    Размер таблицы: [размер_словаря, размерность_вектора].
    """

    def __init__(self, vocab_size, embed_dim):
        """
        Инициализация слоя.
        :param vocab_size: Количество уникальных токенов в словаре (напр. 50257 для GPT-2).
        :param embed_dim: Размерность вектора для каждого токена (напр. 768).
        """
        super().__init__(vocab_size, embed_dim)


class PositionalEmbedding(nn.Module):
    """
    Обучаемые позиционные эмбеддинги.
    В отличие от статических (синусоидальных), эти веса подстраиваются в процессе обучения.
    Помогают модели понять порядок слов в последовательности, так как трансформер сам по себе
    не обладает информацией о позиции токенов.
    """

    def __init__(self, max_length, embed_dim):
        """
        :param max_length: Максимально возможная длина контекста (окна).
        :param embed_dim: Размерность вектора (должна совпадать с TokenEmbedding).
        """
        super().__init__()
        # Создаём обучаемую таблицу эмбеддингов для каждой позиции в окне
        self.embedding = nn.Embedding(max_length, embed_dim)

    def forward(self, x):
        """
        Прямой проход слоя.
        :param x: Входной тензор индексов токенов [batch_size, seq_length].
        :return: Тензор позиционных векторов [1, seq_length, embed_dim].
        """
        batch_size, seq_length = x.shape

        # Генерируем индексы позиций от 0 до seq_length-1
        # unsqueeze(0) добавляет размерность батча (делает тензор 2D)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0)

        # Извлекаем соответствующие векторы из таблицы
        return self.embedding(positions)

    def get_num_parameters(self):
        """Возвращает количество параметров в слое."""
        return sum(p.numel() for p in self.parameters())


class EmbeddingLayer(nn.Module):
    """
    Комплексный слой эмбеддингов, объединяющий информацию о токенах и их позициях.
    Это "входные ворота" для любой модели на базе Transformer.
    """

    def __init__(self, vocab_size, max_length, embed_dim):
        """
        :param vocab_size: Общий размер словаря.
        :param max_length: Максимальное количество токенов в окне.
        :param embed_dim: Размерность скрытого пространства (embedding dimension).
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(max_length, embed_dim)

        # Важный этап: правильная инициализация весов для стабильного старта обучения
        self._init_weights()

    def _init_weights(self):
        """
        Применяет инициализацию весов, характерную для GPT моделей.
        Используется нормальное распределение с малым стандартным отклонением.
        """
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                # Инициализация весов средним 0.0 и std 0.02 (стандарт OpenAI/HuggingFace)
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Вычисляет суммарный эмбеддинг для входной последовательности.
        :param x: Тензор индексов токенов [batch_size, seq_length].
        :return: Сумма токенных и позиционных эмбеддингов [batch_size, seq_length, embed_dim].
        """
        # 1. Получаем семантические векторы токенов
        # Результат: [batch_size, seq_length, embed_dim]
        token_emb = self.token_embedding(x)

        # 2. Получаем векторы позиций
        # Результат: [1, seq_length, embed_dim]
        pos_emb = self.position_embedding(x)

        # 3. Складываем их (PyTorch автоматически расширит pos_emb до размера батча)
        # Суммирование — это стандартный способ объединения информации в Transformer
        return token_emb + pos_emb

    def get_num_parameters(self):
        """Возвращает количество параметров в слое."""
        return sum(p.numel() for p in self.parameters())
