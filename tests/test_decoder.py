import torch

from src.decoder import GPTDecoderBlock, GPTDecoder


def test_decoder_block():

    block = GPTDecoderBlock(
        hidden_size=64,
        num_heads=4,
        intermediate_size=256,
    )

    x = torch.randn(2, 8, 64)

    output, weights = block(x)

    assert output.shape == (2, 8, 64)

    assert weights.shape == (
        2,
        4,
        8,
        8,
    )


def test_decoder_stack():

    decoder = GPTDecoder(
        hidden_size=64,
        num_heads=4,
        num_layers=3,
        intermediate_size=256,
    )

    x = torch.randn(2, 8, 64)

    output, attention_weights = decoder(x)

    assert output.shape == (2, 8, 64)

    assert len(attention_weights) == 3