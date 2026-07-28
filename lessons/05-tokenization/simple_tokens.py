text = "Кот очень любит свежую сметану"
tokens = text.split()
print(tokens)

# Создаём псевдо словарь: токен(слово) -> ID
word_vocab = {"Кот": 2, "любит": 3, "сметану": 4, "Пробел": 99, "[UNK]": 100}

# Функция для токенизации по словам с обработкой неизвестных токенов и пробелов
def tokenize_words_with_unk(text, vocab):
    # Разбиваем текст на слова
    words = text.split()
    # Инициализируем список для хранения идентификаторов слов
    ids = []
    # Проходим по каждому слову и добавляем его идентификатор в список
    for i, word in enumerate(words):
        # Если слово есть в словаре, добавляем его ID, иначе добавляем ID для [UNK]
        if word in vocab:
            ids.append(vocab[word])
        else:
            ids.append(vocab["[UNK]"])
        # Добавляем токен пробела между словами (но не после последнего слова)
        if i < len(words) - 1:
            ids.append(vocab["Пробел"])
    return ids

word_ids = tokenize_words_with_unk(text, word_vocab)
print(f"Идентификаторы слов: {word_ids}")

# Функция для ДеТокенизации
def detokenize_words(ids, id_to_word):
    # Преобразуем список идентификаторов обратно в список слов
    words = [id_to_word[i] for i in ids if id_to_word[i] != "Пробел"]
    # Объединяем слова в строку с пробелами
    return ' '.join(words)

# Создаём обратный словарь: ID -> слово
id_to_word = {idx: word for word, idx in word_vocab.items()}
restored_text = detokenize_words(word_ids, id_to_word)
print(f"Восстановленный текст: {restored_text}")