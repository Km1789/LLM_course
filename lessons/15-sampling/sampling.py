# sampling.py
import torch


def sample_next_token(last_logits, do_sample=False, temperature=1.0,
                      top_k=None):
    # Жадный режим: всегда самый вероятный токен
    if not do_sample:
        return torch.argmax(last_logits, dim=-1, keepdim=True)

    # 1. Температура: меняем резкость распределения
    logits = last_logits / temperature

    # 2. Отсекаем хвост распределения
    if top_k is not None:
        logits = filter_top_k(logits, top_k)

    # 3. Логиты -> вероятности
    probs = torch.softmax(logits, dim=-1)

    # 4. Тянем случайный токен пропорционально вероятностям
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def filter_top_k(logits, k):
    """
    Top-k: оставляет k самых больших логитов, остальным ставит -inf.

    После softmax токены с логитом -inf получают вероятность 0,
    то есть полностью выбывают из выбора.

    Args:
        logits: [batch_size, vocab_size]
        k: сколько кандидатов оставить (не больше размера словаря)

    Returns:
        logits той же формы, где всё, кроме топ-k, равно -inf
    """
    # Просить кандидатов больше, чем есть слов в словаре, бессмысленно
    k = min(k, logits.size(-1))

    top_values, _ = torch.topk(logits, k)

    # Порог - самый маленький логит из топ-k
    threshold = top_values[:, -1].unsqueeze(-1)

    # Всё, что ниже порога, выключаем
    return logits.masked_fill(logits < threshold, float("-inf"))


def filter_top_p(logits, p):
    """
    Top-p (nucleus): оставляет минимальное ядро самых вероятных токенов,
    чья суммарная вероятность достигает p. Остальным ставит -inf.

    Args:
        logits: [batch_size, vocab_size]
        p: порог накопленной вероятности, число от 0 до 1

    Returns:
        logits той же формы, где все токены вне ядра равны -inf
    """
    # 1. Сортируем логиты по убыванию
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    # 2. Вероятности отсортированных токенов и накопленная сумма
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # 3. Токен выбывает, если сумма ДО него уже достигла p.
    #    Токен, на котором порог пересекли, остаётся в ядре.
    remove = (cumulative - sorted_probs) >= p

    # Самый вероятный токен не выбрасываем никогда: ядро не бывает пустым
    remove[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

    # 4. Возвращаем логиты на их исходные места в словаре
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, sorted_indices, sorted_logits)

    return filtered