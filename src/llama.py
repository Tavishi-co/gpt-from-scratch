import torch
import torch.nn as nn
import torch.nn.functional as F

from .rmsnorm import RMSNorm
from .llama_attention import LlamaAttention
from .swiglu import SwiGLU


class LlamaDecoderBlock(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_size,
        max_position_embeddings,
        dropout=0.0,
    ):
        super().__init__()

        self.attention_norm = RMSNorm(
            hidden_size
        )

        self.attention = LlamaAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
        )

        self.ffn_norm = RMSNorm(
            hidden_size
        )

        self.feed_forward = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(self, x):

        attention_input = self.attention_norm(x)

        attention_output, weights = (
            self.attention(attention_input)
        )

        x = x + attention_output

        ffn_input = self.ffn_norm(x)

        x = x + self.feed_forward(
            ffn_input
        )

        return x, weights


class LlamaModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_size=128,
        num_heads=4,
        num_layers=4,
        intermediate_size=448,
        max_position_embeddings=128,
    ):
        super().__init__()

        self.token_embeddings = nn.Embedding(
            vocab_size,
            hidden_size,
        )

        self.layers = nn.ModuleList([
            LlamaDecoderBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                max_position_embeddings=max_position_embeddings,
            )
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(
            hidden_size
        )

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )

        # Weight tying
        self.lm_head.weight = (
            self.token_embeddings.weight
        )

    def forward(self, input_ids, targets=None):

        x = self.token_embeddings(
            input_ids
        )

        all_weights = []

        for layer in self.layers:

            x, weights = layer(x)

            all_weights.append(weights)

        x = self.norm(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                targets.reshape(-1),
            )

        return {
            "logits": logits,
            "loss": loss,
            "attention_weights": all_weights,
        }