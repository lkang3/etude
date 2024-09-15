from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class LoRAConfig:
    input_dim: int
    output_dim: int
    rank: int
    alpha: float
    seed: int


def freeze_model_parameters(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = False


class LoRALayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int,
        alpha: float,
        seed: Optional[int] = 123,
    ):
        torch.manual_seed(seed)
        super().__init__()
        self.validate_rank(input_dim, output_dim, rank)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.alpha = alpha
        self.seed = seed
        self.matrix_a = nn.Parameter(torch.zeros(rank, output_dim))
        self.matrix_b = nn.Parameter(torch.rand(input_dim, rank))

    @staticmethod
    def validate_rank(input_dim: int, output_dim: int, rank: int) -> None:
        if rank > min(input_dim, output_dim):
            msg = (
                f"rank value: {rank} should be <= "
                f"min(input_dim: {input_dim}, output_dim: {output_dim})"
            )
            raise ValueError(msg)

    @classmethod
    def from_config(cls, config: LoRAConfig) -> "LoRALayer":
        return cls(
            config.input_dim,
            config.output_dim,
            config.rank,
            config.alpha,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.alpha * inputs @ self.matrix_b @ self.matrix_a


class LoRAModel(nn.Module):
    def __init__(self, model_to_freeze: nn.Module, lora_config: LoRAConfig):
        super().__init__()
        self.model_to_freeze = model_to_freeze
        freeze_model_parameters(self.model_to_freeze)
        self.lora_layer = LoRALayer.from_config(lora_config)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model_to_freeze(inputs) + self.lora_layer(inputs)
