import torch


def precompute_rope(
    seq_len,
    head_dim,
    base=10000,
    device=None,
):

    theta = 1.0 / (
        base ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=device,
            ).float()
            / head_dim
        )
    )

    positions = torch.arange(
        seq_len,
        device=device,
    ).float()

    frequencies = torch.outer(
        positions,
        theta,
    )

    cos = torch.cos(frequencies)
    sin = torch.sin(frequencies)

    return cos, sin


def apply_rope(x, cos, sin):

    # x:
    # [batch, heads, seq_len, head_dim]

    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    rotated_even = (
        x_even * cos
        - x_odd * sin
    )

    rotated_odd = (
        x_even * sin
        + x_odd * cos
    )

    output = torch.stack(
        [rotated_even, rotated_odd],
        dim=-1,
    )

    return output.flatten(-2)