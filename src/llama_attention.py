import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import (
    precompute_rope,
    apply_rope,
)


class LlamaAttention(nn.Module):

    def __init__(
        self,
        hidden_size,
        num_heads,
        max_position_embeddings,
        dropout=0.0,
    ):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.k_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.v_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.o_proj = nn.Linear(
            hidden_size,
            hidden_size,
            bias=False,
        )

        self.dropout = nn.Dropout(dropout)

        cos, sin = precompute_rope(
            max_position_embeddings,
            self.head_dim,
        )

        self.register_buffer(
            "cos",
            cos,
            persistent=False,
        )

        self.register_buffer(
            "sin",
            sin,
            persistent=False,
        )

    def forward(self, x):

        batch_size, seq_len, _ = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        Q = apply_rope(Q, cos, sin)
        K = apply_rope(K, cos, sin)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1),
        )

        scores = scores / math.sqrt(
            self.head_dim
        )

        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
            )
        )

        scores = scores.masked_fill(
            causal_mask == 0,
            float("-inf"),
        )

        weights = F.softmax(
            scores,
            dim=-1,
        )

        weights = self.dropout(weights)

        output = torch.matmul(
            weights,
            V,
        )

        output = output.transpose(
            1,
            2,
        ).contiguous()

        output = output.view(
            batch_size,
            seq_len,
            self.hidden_size,
        )

        output = self.o_proj(output)

        return output, weights