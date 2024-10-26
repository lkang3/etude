import pytest
from tasks.similarity_learning.models import SiameseModelWithPositiveAndNegativeInputs
from tasks.similarity_learning.models import SiameseModelWithTripletInputs
from tasks.similarity_learning.models import SimilarityModelingConfig
from tasks.similarity_learning.distance import DistanceType
from tasks.similarity_learning.loss import LossType
from tasks.similarity_learning.loss import ContrastiveLoss
from tasks.similarity_learning.loss import TripletLoss
from trainer.models import Trainer


import torch as nn


class TestTrainingSiameseModelWithPositiveAndNegativeInputs:
    @pytest.fixture
    def model(self) -> SiameseModelWithPositiveAndNegativeInputs:
        input_embedding_size = 13
        hidden_embedding_size = 5
        config = SimilarityModelingConfig(
            input_embedding_size=input_embedding_size,
            hidden_embedding_size=hidden_embedding_size,
            distance_type=DistanceType.L2,
            loss_type=LossType.CONTRASTIVE,
        )
        return SiameseModelWithPositiveAndNegativeInputs.from_config(config)

    @pytest.fixture
    def loss(self) -> ContrastiveLoss:
        return ContrastiveLoss()

    def test_train(
        self, model: SiameseModelWithPositiveAndNegativeInputs, loss: ContrastiveLoss,
    ) -> None:
        trainer = Trainer(model=model, loss_func=loss)
        trainer.train()

    def test_inference(self) -> None:
        pass


class TestTrainingTestSiameseModelWithTripletInputs:
    def test_train(self) -> None:
        pass

    def test_inference(self) -> None:
        pass