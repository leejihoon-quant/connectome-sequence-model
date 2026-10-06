"""Configurable GRU skeleton for ordered sequence prediction."""

from __future__ import annotations

import torch
from torch import nn


class SequenceGRU(nn.Module):
    """Map feature sequences to output sequences while exposing recurrent state."""

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence, next_hidden = self.gru(features, hidden)
        return self.head(sequence), next_hidden
