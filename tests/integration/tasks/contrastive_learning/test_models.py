from unittest.mock import Mock

import torch
import torch.nn as nn

from tasks.common.distance import DistanceType
from tasks.contrastive_learning.entities import ContrastiveLearningSentenceInputBatch
from tasks.contrastive_learning.entities import ContrastiveOutput
from tasks.contrastive_learning.entities import TripletOutput
from tasks.contrastive_learning.models import ContrastiveLearningModel
from tasks.contrastive_learning.models import SiameseModelWithPositiveAndNegativeInputs
from tasks.contrastive_learning.models import SiameseModelWithTripletInputs
from tasks.contrastive_learning.models import SimilarityModelingConfig
from tests.utils import is_scalar


class DummyModelToPretrain(nn.Module):
    def forward(self, data: torch.Tensor):
        return data


class TestContrastiveLearningModel:
    def test_forward(self) -> None:
        model_to_pre_train = DummyModelToPretrain()
        model = ContrastiveLearningModel(model_to_pre_train, DistanceType.L2)
        batch_size = 3
        data_size_one = 4
        data_size_two = 5
        batch_data = torch.rand((batch_size, data_size_one, data_size_two))
        model_inputs = ContrastiveLearningSentenceInputBatch(batch_data)

        outputs = model(model_inputs)

        assert len(outputs) == batch_size * 2
        assert all(isinstance(output, ContrastiveOutput) for output in outputs)
        assert (
            len([output for output in outputs if output.is_negative_input_pairs])
            == batch_size
        )
        assert (
            len([output for output in outputs if not output.is_negative_input_pairs])
            == batch_size
        )


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
            input_vector_one,
            input_vector_two,
            is_positive_pair,
        )

        assert isinstance(model_output, ContrastiveOutput)
        assert is_scalar(model_output.distance)
        assert model_output.is_negative_input_pairs is is_positive_pair


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
            positive_input_vector,
            negative_input_vector,
            anchor_input_vector,
        )

        assert isinstance(model_output, TripletOutput)
        assert is_scalar(model_output.contrastive_distance)
        assert is_scalar(model_output.anchor_distance)
