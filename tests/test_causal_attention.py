import torch

from src.causal_attention import CausalSelfAttention


def test_causal_attention_shape():

    attention = CausalSelfAttention(
        hidden_size=64,
        num_heads=4,
    )

    x = torch.randn(2, 8, 64)

    output, weights = attention(x)

    assert output.shape == (2, 8, 64)

    assert weights.shape == (
        2,
        4,
        8,
        8,
    )


def test_future_tokens_are_masked():

    attention = CausalSelfAttention(
        hidden_size=64,
        num_heads=4,
    )

    x = torch.randn(1, 6, 64)

    _, weights = attention(x)

    future_weights = torch.triu(
        weights,
        diagonal=1,
    )

    assert torch.allclose(
        future_weights,
        torch.zeros_like(future_weights),
        atol=1e-6,
    )