from enum import Enum
from enum import auto
from typing import TypeVar

import torch
from torch import nn


class BaseDistanceLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, input_vector_one: torch.Tensor, input_vector_two: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError()


class L1DistanceLayer(BaseDistanceLayer):
    def forward(
        self, input_vector_one: torch.Tensor, input_vector_two: torch.Tensor
    ) -> torch.Tensor:
        return torch.linalg.norm(input_vector_one - input_vector_two, ord=1)


class L2DistanceLayer(BaseDistanceLayer):
    def forward(
        self, input_vector_one: torch.Tensor, input_vector_two: torch.Tensor
    ) -> torch.Tensor:
        return torch.linalg.norm(input_vector_one - input_vector_two, ord=2)


TypeDistanceLayer = TypeVar("TypeDistanceLayer", bound=BaseDistanceLayer)


class DistanceType(Enum):
    L1 = auto()
    L2 = auto()

    @classmethod
    def create_contrastive_layer(
        cls, distance_type: "DistanceType"
    ) -> TypeDistanceLayer:
        if distance_type == cls.L1:
            return L1DistanceLayer()
        elif distance_type == cls.L2:
            return L2DistanceLayer()
        raise ValueError(f"Only {list(cls)} is supported")
