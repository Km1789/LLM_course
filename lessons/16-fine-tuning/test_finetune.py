def test_prompt_format():
    """Промпт собирается по шаблону Question/Answer."""
    prompt = build_prompt("What is the cat name?")
    assert prompt == "Question: What is the cat name?\nAnswer:", prompt
    print("Тест 1 (формат промпта): OK")


def test_prompt_is_masked():
    """Токены вопроса закрыты меткой -100, токены ответа - нет."""
    inputs, labels = dataset[0]
    prompt_len = len(tokenizer.encode(build_prompt(QA_PAIRS[0][0])))

    masked = labels[: prompt_len - 1]
    assert (masked == IGNORE_INDEX).all(), "промпт должен быть замаскирован"

    answer_part = labels[prompt_len - 1:]
    assert (answer_part != IGNORE_INDEX).any(), "ответ маскировать нельзя"
    print("Тест 3 (промпт замаскирован, ответ нет): OK")


def test_answer_ends_with_eos():
    """Каждый ответ заканчивается токеном EOS: модель учится молчать."""
    for idx in range(len(dataset)):
        _, labels = dataset[idx]
        real = labels[labels != IGNORE_INDEX]
        assert real[-1].item() == tokenizer.eos_token_id, f"пример {idx} без EOS"
    print("Тест 4 (ответ заканчивается EOS): OK")