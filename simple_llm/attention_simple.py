import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSelfAttention(nn.Module):
    """Механизм самовнимания без обучаемых весов (Q = K = V = x)."""

    def __init__(self):
        super().__init__()
        # В этой версии нет обучаемых параметров, поэтому __init__ пустой

    def forward(self, x):
        # Если вход без батча [seq_len, embed_dim], добавим ось батча
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        # Шаг 1: Query, Key, Value = входные векторы (без преобразований)
        Q = x
        K = x
        V = x

        # Шаг 2: Attention Scores = Q @ K^T  →  [batch, seq_len, seq_len]
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))

        # Шаг 3: Attention Weights = softmax по последней оси (по строкам)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Шаг 4: Context Vectors = weights @ V
        context_vectors = torch.matmul(attention_weights, V)

        if squeeze_output:
            context_vectors = context_vectors.squeeze(0)
            attention_weights = attention_weights.squeeze(0)
            attention_scores = attention_scores.squeeze(0)

        return context_vectors, attention_weights, attention_scores


def compute_attention_step_by_step(x, tokenizer=None, token_names=None):
    seq_length, embed_dim = x.shape
    if token_names is None:
        token_names = [f"Token_{i}" for i in range(seq_length)]

    # ШАГ 1: Attention Scores — скалярное произведение всех пар
    attention_scores = torch.empty(seq_length, seq_length)
    for i in range(seq_length):
        for j in range(seq_length):
            attention_scores[i, j] = torch.dot(x[i], x[j])
    print("Матрица scores:")
    header = "         " + "".join([f"{name:>10}" for name in token_names])
    print(header)
    for i, name in enumerate(token_names):
        row = f"{name:>8}" + "".join([f"{attention_scores[i, j]:>10.4f}" for j in range(seq_length)])
        print(row)

    # ШАГ 2: Attention Weights — softmax по строкам
    attention_weights = F.softmax(attention_scores, dim=-1)
    print("\nМатрица weights:")
    print(header)
    for i, name in enumerate(token_names):
        row = f"{name:>8}" + "".join([f"{attention_weights[i, j]:>10.4f}" for j in range(seq_length)])
        print(row)

    # ШАГ 3: Context Vectors — взвешенная сумма
    context_vectors = attention_weights @ x
    print("\nКонтекстные векторы:")
    for i, name in enumerate(token_names):
        values = ", ".join([f"{v:.4f}" for v in context_vectors[i].tolist()])
        print(f"  {name:>8}: [{values}]")

    return {
        "input": x,
        "attention_scores": attention_scores,
        "attention_weights": attention_weights,
        "context_vectors": context_vectors,
    }