from dataclasses import dataclass

import torch

from tasks.masked_language_modeling.models import MLM
from tasks.masked_language_modeling.models import MLMConfig


@dataclass
class MaskedTokenOutput:
    pass


class MLMTrainer:
    def __init__(self, model: MLM):
        self.model = model

    @classmethod
    def from_config(cls, config: MLMConfig) -> "MLMTrainer":
        cls(MLM.from_config(config))

    def train(self, input_sentence: torch.Tensor, target_sentence: torch.Tensor) -> torch.Tensor:
        pass

    def inference(self, input_sentence: torch.Tensor) -> MaskedTokenOutput:
        pass
