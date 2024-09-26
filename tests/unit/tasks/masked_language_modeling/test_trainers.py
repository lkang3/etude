from unittest.mock import Mock

from tasks.masked_language_modeling.trainers import MLMTrainer


class TestMLMTrainer:
    def test_train(self) -> None:
        model = MLMTrainer(Mock())
        model.train(Mock(), Mock())


