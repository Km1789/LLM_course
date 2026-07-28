from transformers import AutoTokenizer

# Загружаем BPE-токенизатор для русской GPT-модели
tokenizer = AutoTokenizer.from_pretrained("ai-forever/rugpt3small_based_on_gpt2")

# Наша фраза
text = "Кот очень любит свежую сметану"

# Токенизация
encoded = tokenizer(text)
ids = encoded['input_ids']

# Декодируем каждый токен отдельно для читаемого вывода
readable_tokens = [tokenizer.decode([id]) for id in ids]

print("Токены:", readable_tokens)
print("ID:", ids)

decoded = tokenizer.decode(encoded['input_ids'], skip_special_tokens=True)
print("Декодированный текст:", decoded)