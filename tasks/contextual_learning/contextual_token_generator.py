from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Iterator
from typing import List
from typing import Optional
from typing import TypeVar

import torch
from torch import nn

torch.manual_seed(123)


@dataclass
class SentenceData:
    data_in_tensor: torch.Tensor


@dataclass
class ContextualAndTargetIndices:
    contextual_data_indices: torch.Tensor
    target_data_indices: torch.Tensor


@dataclass
class ContextualModelInput:
    contextual_data: torch.Tensor
    target_data: torch.Tensor


@dataclass
class ContextualModelOutput:
    projected_target_data: torch.Tensor
    expected_target_data: torch.Tensor

    @staticmethod
    def concatenate_projected_contextual_data_in_batch(
        model_outputs: List["ContextualModelOutput"],
    ) -> torch.Tensor:
        outputs: List[torch.Tensor] = [
            model_output.projected_target_data for model_output in model_outputs
        ]
        return torch.concat(outputs, dim=0)

    @staticmethod
    def concatenate_projected_target_data_in_batch(
        model_outputs: List["ContextualModelOutput"],
    ) -> torch.Tensor:
        outputs: List[torch.Tensor] = [
            model_output.projected_target_data for model_output in model_outputs
        ]
        return torch.concat(outputs, dim=0)


class BaseContextualModel(nn.Module):
    @abstractmethod
    def get_model_input(
        self, batch_data: torch.Tensor
    ) -> Iterator[ContextualModelInput]:
        pass


TypeBaseContextualModel = TypeVar("TypeBaseContextualModel", bound=BaseContextualModel)


def get_contextual_and_target_data_indices(
    num_of_contextual_samples: int, num_of_target_samples: int
) -> ContextualAndTargetIndices:
    population = num_of_contextual_samples + num_of_target_samples
    population_permutation = torch.randperm(population)
    contextual_indices = population_permutation[:num_of_contextual_samples]
    target_indices = population_permutation[-num_of_target_samples:]
    return ContextualAndTargetIndices(contextual_indices, target_indices)


class ManyToOneContextualModel(BaseContextualModel):
    def __init__(self, embedding_size: int, num_of_contextual_samples: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_of_contextual_samples = num_of_contextual_samples
        self.num_of_target_samples = 1
        self.contextual_and_target_indices = get_contextual_and_target_data_indices(
            self.num_of_contextual_samples,
            self.num_of_target_samples,
        )

        self.projector = nn.Linear(self.embedding_size, self.embedding_size)

    def get_contextual_sample_indices(self) -> torch.Tensor:
        return torch.randperm(self.num_of_contextual_samples)

    def get_model_input(
        self, batch_data: torch.Tensor
    ) -> Iterator[ContextualModelInput]:
        for data_in_batch in batch_data:
            yield ContextualModelInput(
                contextual_data=data_in_batch[
                    self.contextual_and_target_indices.contextual_data_indices, ...
                ],
                target_data=data_in_batch[
                    self.contextual_and_target_indices.target_data_indices, ...
                ],
            )

    @staticmethod
    def aggregate_contextual_data_in_batch(data_in_batch: torch.Tensor) -> torch.Tensor:
        return torch.sum(data_in_batch, dim=0).unsqueeze(0)

    def forward(self, data_in_batch: torch.Tensor) -> List[ContextualModelOutput]:
        outputs: List[ContextualModelOutput] = []
        for model_input in self.get_model_input(data_in_batch):
            combined_data = self.aggregate_contextual_data_in_batch(
                model_input.contextual_data
            )
            projected_target_data = self.projector(combined_data)
            outputs.append(
                ContextualModelOutput(projected_target_data, model_input.target_data)
            )
        return outputs


class OneToManyContextualModel(BaseContextualModel):
    def __init__(self, embedding_size: int, num_of_target_samples: int):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_of_contextual_samples = 1
        self.num_of_target_samples = num_of_target_samples
        self.contextual_and_target_indices = get_contextual_and_target_data_indices(
            self.num_of_contextual_samples,
            self.num_of_target_samples,
        )
        self.project_headers = [
            nn.Linear(self.embedding_size, self.embedding_size)
            for _ in range(self.num_of_target_samples)
        ]

    def get_model_input(
        self, batch_data: torch.Tensor
    ) -> Iterator[ContextualModelInput]:
        for data_in_batch in batch_data:
            yield ContextualModelInput(
                contextual_data=data_in_batch[
                    self.contextual_and_target_indices.contextual_data_indices, ...
                ],
                target_data=data_in_batch[
                    self.contextual_and_target_indices.target_data_indices, ...
                ],
            )

    def project_contextual_data_in_batch(
        self, data_in_batch: torch.Tensor
    ) -> torch.Tensor:
        embeddings = [
            project_header(data_in_batch) for project_header in self.project_headers
        ]
        return torch.cat(embeddings, dim=0)

    def forward(self, batch_data: torch.Tensor) -> List[ContextualModelOutput]:
        outputs: List[ContextualModelOutput] = []
        for model_input in self.get_model_input(batch_data):
            projected_target_data = self.project_contextual_data_in_batch(
                model_input.contextual_data
            )
            outputs.append(
                ContextualModelOutput(projected_target_data, model_input.target_data)
            )
        return outputs


class NLPContextualModelType(Enum):
    C_BOW = auto()
    SKIP_GRAM = auto()


@dataclass
class NLPContextualModelConfig:
    model_type: NLPContextualModelType
    embedding_size: int
    num_of_contextual_samples: int = 1
    num_of_target_samples: int = 1

    def create_model(self) -> TypeBaseContextualModel:
        if self.model_type == NLPContextualModelType.C_BOW:
            return ManyToOneContextualModel(
                self.embedding_size, self.num_of_contextual_samples
            )
        if self.model_type == NLPContextualModelType.SKIP_GRAM:
            return OneToManyContextualModel(
                self.embedding_size, self.num_of_target_samples
            )
        raise ValueError(f"Only {list(NLPContextualModelType)} is supported")
