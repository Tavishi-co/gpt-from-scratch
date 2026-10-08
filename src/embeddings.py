import torch
import torch.nn as nn


class GPTEmbeddings(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_size,
        max_position_embeddings,
        dropout=0.0,
    ):
        super().__init__()

        self.token_embeddings = nn.Embedding(
            vocab_size,
            hidden_size,
        )

        self.position_embeddings = nn.Embedding(
            max_position_embeddings,
            hidden_size,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.shape

        if seq_len > self.position_embeddings.num_embeddings:
            raise ValueError(
                "Sequence length exceeds maximum context length."
            )

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        )

        positions = positions.unsqueeze(0).expand(
            batch_size,
            seq_len,
        )

        token_embeddings = self.token_embeddings(
            input_ids
        )

        position_embeddings = self.position_embeddings(
            positions
        )

        x = token_embeddings + position_embeddings

        return self.dropout(x)