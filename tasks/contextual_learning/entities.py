from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import TypeVar

import torch


class BaseContextualLearningInputBatch(ABC):

    @abstractmethod
    def batch_size(self) -> int:
        pass

    @abstractmethod
    def get_batch_data(self) -> torch.Tensor:
        pass


@dataclass
class ContextualLearningSentenceInputBatch(BaseContextualLearningInputBatch):
    data_in_batch: torch.Tensor

    def batch_size(self) -> int:
        return len(self.data_in_batch)

    def get_batch_data(self) -> torch.Tensor:
        return self.data_in_batch


TypeContextualLearningInputBatch = TypeVar(
    "TypeContextualLearningInputBatch",
    bound=BaseContextualLearningInputBatch,
)
