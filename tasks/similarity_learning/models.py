from dataclasses import dataclass

import torch
from torch import nn

from tasks.similarity_learning.distance import TypeDistanceLayer, DistanceType
from tasks.similarity_learning.entities import ContrastiveOutput
from tasks.similarity_learning.entities import TripletOutput
from tasks.similarity_learning.loss import LossType


@dataclass
class SimilarityModelingConfig:
    input_embedding_size: int
    hidden_embedding_size: int
    distance_type: DistanceType
    loss_type: LossType


class SiameseModelWithPositiveAndNegativeInputs(nn.Module):
    def __init__(self, mlp: nn.Module, distance_layer: TypeDistanceLayer):
        super().__init__()
        self.mlp = mlp
        self.distance_layer = distance_layer

    @classmethod
    def from_config(
        cls, config: SimilarityModelingConfig,
    ) -> "SiameseModelWithPositiveAndNegativeInputs":
        return cls(
            nn.Linear(config.input_embedding_size, config.hidden_embedding_size),
            DistanceType.create_contrastive_layer(config.distance_type),
        )

    def forward(
        self,
        input_vector_one: torch.Tensor,
        input_vector_two: torch.Tensor,
        is_positive_vector_pair: bool,
    ) -> ContrastiveOutput:
        input_vector_one = self.mlp(input_vector_one)
        input_vector_two = self.mlp(input_vector_two)
        vector_distance = self.distance_layer(input_vector_one, input_vector_two)

        return ContrastiveOutput(
            distance=vector_distance, is_contrastive=is_positive_vector_pair,
        )


class SiameseModelWithTripletInputs(nn.Module):
    def __init__(
        self,
        mlp: nn.Module,
        distance_layer: TypeDistanceLayer,
    ):
        super().__init__()
        self.mlp = mlp
        self.distance_layer = distance_layer

    @classmethod
    def from_config(
        cls, config: SimilarityModelingConfig,
    ) -> "SiameseModelWithTripletInputs":
        return cls(
            nn.Linear(config.input_embedding_size, config.hidden_embedding_size),
            DistanceType.create_contrastive_layer(config.distance_type),
        )

    def forward(
        self,
        positive_input_vector: torch.Tensor,
        negative_input_vector: torch.Tensor,
        anchor_input_vector: torch.Tensor,
    ) -> TripletOutput:
        positive_input_vector = self.mlp(positive_input_vector)
        negative_input_vector = self.mlp(negative_input_vector)
        anchor_input_vector = self.mlp(anchor_input_vector)
        contrastive_distance = self.distance_layer(
            positive_input_vector, negative_input_vector,
        )
        anchor_distance = self.distance_layer(
            positive_input_vector, anchor_input_vector,
        )

        return TripletOutput(
            contrastive_distance=contrastive_distance, anchor_distance=anchor_distance,
        )
