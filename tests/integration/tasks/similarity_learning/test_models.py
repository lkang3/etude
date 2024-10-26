from unittest.mock import Mock

import torch

from tasks.similarity_learning.models import SimilarityModelingConfig
from tasks.similarity_learning.distance import DistanceType
from tasks.similarity_learning.entities import ContrastiveOutput
from tasks.similarity_learning.entities import TripletOutput
from tasks.similarity_learning.models import SiameseModelWithPositiveAndNegativeInputs
from tasks.similarity_learning.models import SiameseModelWithTripletInputs

from tests.utils import is_scalar


class TestSiameseModelWithPositiveAndNegativeInputs:
    def test_forward(self) -> None:
        input_embedding_size = 13
        hidden_embedding_size = 5
        config = SimilarityModelingConfig(
            input_embedding_size=input_embedding_size,
            hidden_embedding_size=hidden_embedding_size,
            distance_type=DistanceType.L2,
            loss_type=Mock(),
        )
        model = SiameseModelWithPositiveAndNegativeInputs.from_config(config)
        vector_size = 14
        input_vector_one = torch.rand((vector_size, input_embedding_size))
        input_vector_two = torch.rand((vector_size, input_embedding_size))
        is_positive_pair = False
        model_output: ContrastiveOutput = model(
            input_vector_one, input_vector_two, is_positive_pair,
        )

        assert isinstance(model_output, ContrastiveOutput)
        assert is_scalar(model_output.distance)
        assert model_output.is_contrastive is is_positive_pair


class TestSiameseModelWithTripletInputs:
    def test_forward(self) -> None:
        input_embedding_size = 13
        hidden_embedding_size = 5
        config = SimilarityModelingConfig(
            input_embedding_size=input_embedding_size,
            hidden_embedding_size=hidden_embedding_size,
            distance_type=DistanceType.L2,
            loss_type=Mock(),
        )
        model = SiameseModelWithTripletInputs.from_config(config)
        vector_size = 14
        positive_input_vector = torch.rand((vector_size, input_embedding_size))
        negative_input_vector = torch.rand((vector_size, input_embedding_size))
        anchor_input_vector = torch.rand((vector_size, input_embedding_size))

        model_output: TripletOutput = model(
            positive_input_vector, negative_input_vector, anchor_input_vector,
        )

        assert isinstance(model_output, TripletOutput)
        assert is_scalar(model_output.contrastive_distance)
        assert is_scalar(model_output.anchor_distance)
