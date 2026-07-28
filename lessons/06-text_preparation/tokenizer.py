from transformers import AutoTokenizer

from torch.utils.data import DataLoader
from simple_dataset import SimpleDataset


# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Читаем наш текст
with open("lessons/06-text_preparation/cat_story.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Токенизируем
encoder_text = tokenizer.encode(raw_text)

print(f"Всего токенов: {len(encoder_text)}")
print(f"Первые 20 токенов: {encoder_text[:20]}")


# Декодируем первые 5 токенов обратно в текст
first_tokens = encoder_text[:5]
decoded_tokens = tokenizer.decode(first_tokens)
print(f"Токены: {first_tokens}")
print(f"Текст: {decoded_tokens}")


# Определяем размер окна
window_context = 3


print("=" * 60)
print("Пары 'Вход → Цель' для обучения модели")
print("=" * 60)

# Генерируем пары "вход → цель" для обучения модели
for i in range(1, window_context + 1):
    # Определяем контекст и желаемый токен
    context = encoder_text[:i]
    # Показываем контекст
    desired = encoder_text[i]

    # Показываем и числа, и текст
    context_text = tokenizer.decode(context)
    # Показываем текст
    desired_text = tokenizer.decode([desired])

    # Показываем пары
    print(f"{context_text:40} → {desired_text}")
    # Показываем пары в числовом формате
    print(f"{str(context):40} → {desired}")


# Остальной код

# Настройки
max_length = 4  # Модель будет смотреть на 4 токена
stride = 1  # Сдвигаем окно на 1 токен (для максимального покрытия)
batch_size = 2  # В одном пакете будет 2 примера

# Создаем датасет
dataset = SimpleDataset(raw_text, tokenizer, max_length, stride)

# Создаем загрузчик
# Параметр: shuffle перемешивает примеры перед каждой эпохой, чтобы модель не запоминала порядок
# Для демонстрации используем shuffle=False, чтобы вывод был предсказуемым
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Проверка: посмотрим, что внутри
print("\n=== Проверка DataLoader ===")
for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
    print(f"\nПакет №{batch_idx}:")
    print(f"Вход (X): {batch_x.shape}")
    print(f"Цель (Y): {batch_y.shape}")

    # Выведем сами числа
    print(f"X: {batch_x}")
    print(f"Y: {batch_y}")

    # текст (для отладки):
    print(f"Текст X: {tokenizer.decode(batch_x[0])}")
    print(f"Текст Y: {tokenizer.decode(batch_y[0])}")

    # Остановимся после первого пакета, чтобы не захламлять вывод
    break
