from enum import Enum, auto
from typing import Optional

import torch
from torch import nn

from tasks.similarity_learning.entities import ContrastiveOutput
from tasks.similarity_learning.entities import TripletOutput


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: Optional[torch.Tensor] = torch.tensor(0.5)):
        super().__init__()
        self.margin = margin

    def forward(self, contrastive_output: ContrastiveOutput) -> torch.Tensor:
        distance = contrastive_output.distance
        is_positive_pair = torch.Tensor(not contrastive_output.is_contrastive).long()

        return (
            is_positive_pair * torch.pow(distance, 2)
            + (1 - is_positive_pair)
            * torch.pow(torch.max(torch.tensor(0.0), self.margin - distance), 2)
        )


class TripletLoss(nn.Module):
    def __init__(self, margin: Optional[torch.Tensor] = torch.tensor(0.5)):
        super().__init__()
        self.margin = margin

    def forward(self, triplet_output: TripletOutput) -> torch.Tensor:
        return torch.max(
            (
                torch.pow(triplet_output.anchor_distance, 2)
                - torch.pow(triplet_output.contrastive_distance, 2)
                + self.margin
            ),
            torch.tensor(0.0)
        )


class LossType(Enum):
    CONTRASTIVE = auto()
    TRIPLET = auto()

    @classmethod
    def create_loss_func(cls, loss_type: "LossType") -> nn.Module:
        if loss_type == cls.CONTRASTIVE:
            return ContrastiveLoss()
        elif loss_type == cls.TRIPLET:
            return TripletLoss()
        raise ValueError(f"Only {list(cls)} is supported")
