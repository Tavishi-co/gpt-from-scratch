import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import GPTEmbeddings
from .decoder import GPTDecoder


class GPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_position_embeddings=128,
        hidden_size=128,
        num_heads=4,
        num_layers=4,
        intermediate_size=512,
        dropout=0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_position_embeddings = (
            max_position_embeddings
        )

        self.embeddings = GPTEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
        )

        self.decoder = GPTDecoder(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            dropout=dropout,
        )

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )

        # Tie output projection to token embeddings.
        self.lm_head.weight = (
            self.embeddings.token_embeddings.weight
        )

    def forward(
        self,
        input_ids,
        targets=None,
    ):

        x = self.embeddings(input_ids)

        hidden_states, attention_weights = (
            self.decoder(x)
        )

        logits = self.lm_head(hidden_states)

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states,
            "attention_weights": attention_weights,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens,
        temperature=1.0,
    ):

        self.eval()

        for _ in range(max_new_tokens):

            # Keep only the available context window.
            input_context = input_ids[
                :,
                -self.max_position_embeddings:
            ]

            outputs = self(
                input_context
            )

            logits = outputs["logits"]

            # Only the final position predicts
            # the next token.
            next_token_logits = logits[:, -1, :]

            next_token_logits = (
                next_token_logits / temperature
            )

            probabilities = F.softmax(
                next_token_logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1,
            )

        return input_ids