# tokenizer_utils.py
from transformers import AutoTokenizer
from config import settings

def get_tokenizer():
    """
    Загружает и возвращает токенизатор на основе настроек.
    """
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    # Устанавливаем pad_token, если его нет (важно для многих моделей)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def encode_text(tokenizer, text):
    """
    Превращает текст в список ID токенов.
    """
    return tokenizer.encode(text)