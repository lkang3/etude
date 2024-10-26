from dataclasses import dataclass

import torch


@dataclass
class ContrastiveOutput:
    distance: torch.Tensor
    is_contrastive: bool


@dataclass
class TripletOutput:
    contrastive_distance: torch.Tensor
    anchor_distance: torch.Tensor
