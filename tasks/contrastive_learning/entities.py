from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from typing import List
from typing import TypeVar

import torch


@dataclass
class PositiveInputPair:
    anchor: torch.Tensor
    other: torch.Tensor


@dataclass
class NegativeInputPair:
    anchor: torch.Tensor
    other: torch.Tensor


class BaseContrastiveLearningInputBatch(ABC):

    @abstractmethod
    def batch_size(self) -> int:
        pass

    @abstractmethod
    def get_batch_data(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_positive_input_pairs(self) -> List[PositiveInputPair]:
        pass

    @abstractmethod
    def get_negative_input_pairs(self) -> List[NegativeInputPair]:
        pass

    @abstractmethod
    def get_input_pairs(self) -> List[PositiveInputPair | NegativeInputPair]:
        pass


@dataclass
class ContrastiveLearningSentenceInputBatch(BaseContrastiveLearningInputBatch):
    data_in_batch: torch.Tensor

    def batch_size(self) -> int:
        return len(self.data_in_batch)

    def get_batch_data(self) -> torch.Tensor:
        return self.data_in_batch

    def get_positive_input_pairs(self) -> List[PositiveInputPair]:
        input_pairs = []
        for index in range(0, self.batch_size()):
            input_pairs.append(
                # TODO
                PositiveInputPair(
                    anchor=self.data_in_batch[index, ...],
                    other=self.data_in_batch[index, ...],
                )
            )
        return input_pairs

    def get_negative_input_pairs(self) -> List[NegativeInputPair]:
        input_pairs = []
        for index in range(0, self.batch_size()):
            input_pairs.append(
                # TODO
                NegativeInputPair(
                    anchor=self.data_in_batch[index, ...],
                    other=self.data_in_batch[index, ...],
                )
            )
        return input_pairs

    def get_input_pairs(self) -> List[PositiveInputPair | NegativeInputPair]:
        outputs: List[PositiveInputPair | NegativeInputPair] = []
        outputs.extend(self.get_positive_input_pairs())
        outputs.extend(self.get_negative_input_pairs())
        return outputs


TypeContrastiveLearningInputBatch = TypeVar(
    "TypeContrastiveLearningInputBatch",
    bound=BaseContrastiveLearningInputBatch,
)


@dataclass
class ContrastiveOutput:
    distance: torch.Tensor
    is_negative_input_pairs: bool


@dataclass
class TripletOutput:
    contrastive_distance: torch.Tensor
    anchor_distance: torch.Tensor
