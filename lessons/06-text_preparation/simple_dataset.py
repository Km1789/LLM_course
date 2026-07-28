import torch

from torch.utils.data import Dataset, DataLoader


class SimpleDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):

        # Сохраняем токенизатор и максимальную длину контекста

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.stride = stride

        # Токенизируем весь текст сразу

        self.input_ids = tokenizer.encode(txt)

        # Важный момент: для обучения нам нужно много перекрывающихся кусков.

        # Но мы не создаем их все сразу в памяти, а генерируем "на лету" в __getitem__

        # Это экономит оперативную память.

    def __len__(self):

        # Рассчитываем, сколько всего кусков (примеров) мы можем вырезать из текста

        # Формула: (Всего токенов - Размер окна) / Шаг сдвига + 1

        return (len(self.input_ids) - self.max_length - 1) // self.stride + 1

    def __getitem__(self, idx):

        # Вычисляем начало и конец текущего куска текста

        start_idx = idx * self.stride

        end_idx = start_idx + self.max_length

        # Вырезаем кусок токенов для входа (X)

        chunk = self.input_ids[start_idx:end_idx]

        # Цель (Y) - это тот же кусок, но сдвинутый на 1 токен вправо

        # Модель видит токены [0, 1, 2], должна предсказать [1, 2, 3]

        targets = self.input_ids[start_idx + 1 : end_idx + 1]

        # Превращаем списки Python в тензоры PyTorch

        # dtype=torch.long важен, так как это индексы (целые числа)

        return torch.tensor(chunk, dtype=torch.long), torch.tensor(
            targets, dtype=torch.long
        )
