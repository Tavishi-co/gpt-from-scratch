import torch

from src.gpt import GPT


def test_gpt_forward():

    model = GPT(
        vocab_size=100,
        max_position_embeddings=16,
        hidden_size=64,
        num_heads=4,
        num_layers=2,
        intermediate_size=256,
    )

    input_ids = torch.randint(
        0,
        100,
        (2, 8),
    )

    targets = torch.randint(
        0,
        100,
        (2, 8),
    )

    outputs = model(
        input_ids,
        targets,
    )

    assert outputs["logits"].shape == (
        2,
        8,
        100,
    )

    assert outputs["loss"].ndim == 0


def test_gpt_generation():

    model = GPT(
        vocab_size=50,
        max_position_embeddings=16,
        hidden_size=32,
        num_heads=4,
        num_layers=2,
        intermediate_size=128,
    )

    input_ids = torch.tensor([
        [1, 2, 3]
    ])

    generated = model.generate(
        input_ids,
        max_new_tokens=5,
    )

    assert generated.shape == (1, 8)