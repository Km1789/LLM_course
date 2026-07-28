# main.py
from config import settings
from loader import create_dataloader
from tokenizer_utils import get_tokenizer

def main():
    print(f"Запуск проекта: {settings.text_file}")
    print(f"Конфигурация: окно={settings.max_length}, батч={settings.batch_size}")
    
    # Читаем текст
    with open(settings.text_file, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    # Создаем загрузчик данных
    dataloader = create_dataloader(raw_text, shuffle=False)
    
    # Проверяем данные (валидация пайплайна)
    print("\n=== Проверка первого батча ===")
    for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
        print(f"Пакет №{batch_idx}")
        print(f"Вход (X): {batch_x}")
        print(f"Цель (Y): {batch_y}")
        
        # Для наглядности декодируем первый пример в пакете
        tokenizer = get_tokenizer()
        
        print(f"Текст X: {tokenizer.decode(batch_x[0])}")
        print(f"Текст Y: {tokenizer.decode(batch_y[0])}")
        
        # Проверяем только первый батч
        break

if __name__ == "__main__":
    main()