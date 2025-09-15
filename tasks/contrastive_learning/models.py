from dataclasses import dataclass
from typing import List

import torch
from torch import nn

from tasks.common.distance import DistanceType
from tasks.common.distance import TypeDistanceLayer
from tasks.contrastive_learning.entities import ContrastiveOutput
from tasks.contrastive_learning.entities import NegativeInputPair
from tasks.contrastive_learning.entities import PositiveInputPair
from tasks.contrastive_learning.entities import TripletOutput
from tasks.contrastive_learning.entities import TypeContrastiveLearningInputBatch
from tasks.contrastive_learning.loss import LossType


class ContrastiveLearningModel(nn.Module):
    def __init__(
        self,
        model_to_retrain: nn.Module,
        distance_type: DistanceType,
    ):
        super().__init__()
        self.model_to_pretrain = model_to_retrain
        self.distance_type = distance_type
        self.distance_layer: DistanceType = DistanceType.create_contrastive_layer(self.distance_type)

    def get_anchor_embeddings(
        self, input_pair: NegativeInputPair | PositiveInputPair
    ) -> torch.Tensor:
        return self.model_to_pretrain(input_pair.anchor)

    def get_other_embeddings(
        self, input_pair: NegativeInputPair | PositiveInputPair
    ) -> torch.Tensor:
        return self.model_to_pretrain(input_pair.other)

    def forward_with_one_input_pair(
        self,
        input_pair: NegativeInputPair | PositiveInputPair,
    ) -> ContrastiveOutput:
        anchor_embeddings = self.get_anchor_embeddings(input_pair)
        other_embeddings = self.get_other_embeddings(input_pair)
        return ContrastiveOutput(
            distance=self.distance_layer(anchor_embeddings, other_embeddings),
            is_negative_input_pairs=isinstance(input_pair, NegativeInputPair),
        )

    def forward(
        self, data_in_batch: TypeContrastiveLearningInputBatch
    ) -> List[ContrastiveOutput]:
        model_input_pairs = data_in_batch.get_input_pairs()
        outputs: List[ContrastiveOutput] = []
        for model_input_pair in model_input_pairs:
            outputs.append(
                self.forward_with_one_input_pair(
                    self.model_to_pretrain(model_input_pair)
                )
            )
        return outputs


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
        cls,
        config: SimilarityModelingConfig,
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
            distance=vector_distance,
            is_negative_input_pairs=is_positive_vector_pair,
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
        cls,
        config: SimilarityModelingConfig,
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
            positive_input_vector,
            negative_input_vector,
        )
        anchor_distance = self.distance_layer(
            positive_input_vector,
            anchor_input_vector,
        )

        return TripletOutput(
            contrastive_distance=contrastive_distance,
            anchor_distance=anchor_distance,
        )
