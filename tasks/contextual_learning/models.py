from typing import List

import torch
from torch import nn

from tasks.contextual_learning.contextual_token_generator import ContextualModelOutput
from tasks.contextual_learning.contextual_token_generator import (
    NLPContextualModelConfig,
)
from tasks.contextual_learning.contextual_token_generator import TypeBaseContextualModel


class ContextualLearningNLPModel(nn.Module):
    def __init__(
        self,
        model_to_pretrain: nn.Module,
        model_config: NLPContextualModelConfig,
    ):
        super().__init__()
        self.model_to_pretrain = model_to_pretrain
        self.contextual_model: TypeBaseContextualModel = model_config.create_model()

    def forward(self, batch_data: torch.Tensor) -> List[ContextualModelOutput]:
        data_embeddings = self.model_to_pretrain(batch_data)
        return self.contextual_model(data_embeddings)
