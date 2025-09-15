from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from typing import TypeVar

import numpy as np
import torch

torch.manual_seed(123)


class BaseMaskedPredictionInput(ABC):
    @abstractmethod
    def get_data(self) -> torch.Tensor:
        pass

    @abstractmethod
    def get_masked_data(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _get_mask_indices(self) -> np.ndarray:
        pass

    @abstractmethod
    def _apply_mask_value(self, data: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def filter_data_with_mask_indices(self, data: torch.Tensor) -> torch.Tensor:
        pass


@dataclass
class MaskedPredictionWithSequenceInput(BaseMaskedPredictionInput):
    data: torch.Tensor
    mask_value: float
    mask_percent: float
    mask_indices: np.ndarray = field(init=False)
    masked_data: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.mask_indices = self._get_mask_indices()
        self.masked_data = self._apply_mask_value(self.get_data())

    def _get_mask_indices(self) -> np.ndarray:
        rnd = np.random.RandomState(123)
        input_size = len(self.data)
        num_of_mask_elements = int(input_size * self.mask_percent)
        output = rnd.choice(range(input_size), num_of_mask_elements, replace=False)
        return output

    def _apply_mask_value(self, data: torch.Tensor) -> torch.Tensor:
        data_copy = torch.clone(data).detach()
        data_copy[self.mask_indices] = self.mask_value
        return data_copy

    def get_data(self) -> torch.Tensor:
        return self.data

    def get_masked_data(self) -> torch.Tensor:
        return self.masked_data

    def filter_data_with_mask_indices(self, data: torch.Tensor) -> torch.Tensor:
        return data[self.mask_indices, :]


TypeMaskedPredictionInput = TypeVar(
    "TypeMaskedPredictionInput", bound=BaseMaskedPredictionInput
)
