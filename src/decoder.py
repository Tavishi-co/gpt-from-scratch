import torch.nn as nn

from .causal_attention import CausalSelfAttention


class GPTDecoderBlock(nn.Module):

    def __init__(
        self,
        hidden_size=128,
        num_heads=4,
        intermediate_size=512,
        dropout=0.0,
    ):
        super().__init__()

        # Pre-LayerNorm
        self.ln_1 = nn.LayerNorm(hidden_size)

        self.attention = CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.ln_2 = nn.LayerNorm(hidden_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(
                hidden_size,
                intermediate_size,
            ),
            nn.GELU(),
            nn.Linear(
                intermediate_size,
                hidden_size,
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x):

        # Pre-LN attention + residual
        attention_input = self.ln_1(x)

        attention_output, attention_weights = (
            self.attention(attention_input)
        )

        x = x + attention_output

        # Pre-LN FFN + residual
        ffn_input = self.ln_2(x)

        ffn_output = self.feed_forward(
            ffn_input
        )

        x = x + ffn_output

        return x, attention_weights


class GPTDecoder(nn.Module):

    def __init__(
        self,
        hidden_size=128,
        num_heads=4,
        num_layers=4,
        intermediate_size=512,
        dropout=0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            GPTDecoderBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(
            hidden_size
        )

    def forward(self, x):

        all_attention_weights = []

        for layer in self.layers:

            x, attention_weights = layer(x)

            all_attention_weights.append(
                attention_weights
            )

        x = self.final_norm(x)

        return x, all_attention_weights