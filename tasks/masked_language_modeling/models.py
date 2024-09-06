import torch
import torch.nn as nn
from typing import List
from typing import TypeVar

from transformers.encoders import Encoder


class MLM(nn.Module):
    def __init__(
        self,
        embeddings: nn.Module,
        positional_embeddings: nn.Module,
        encoder_blocks: List[Encoder],
        header: nn.Module,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.positional_embeddings = positional_embeddings
        self.encoder_blocks = encoder_blocks
        self.header = header

    @classmethod
    def create(cls) -> "MLM":
        return cls()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        pass
