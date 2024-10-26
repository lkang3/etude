from abc import abstractmethod
from typing import List
from typing import Tuple

import torch
import torch.nn as nn


def get_classification_accuracy(
    logits: torch.Tensor, expected_class_labels: torch.Tensor,
) -> torch.Tensor:
    _, class_labels = torch.max(logits, dim=-1)
    return (class_labels == expected_class_labels).sum() / len(expected_class_labels)


class Expert(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: int = 0.02):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        data = self.net(inputs)
        return torch.softmax(data, dim=-1)


class Gate(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activate_layer: nn.Module):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            activate_layer,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        data = self.net(inputs)
        return torch.softmax(data, dim=-1)


class MoE(nn.Module):
    def __init__(
        self,
        experts: List[Expert],
        gate: nn.Module,
        freeze_expert: bool = False,
    ):
        super().__init__()
        if freeze_expert:
            for expert in experts:
                for param in expert.parameters():
                    param.requires_grad = False
        self.experts = nn.ModuleList(experts)
        self.out_dim = self.experts[0].out_dim
        self.gate = gate

    @abstractmethod
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: (num_of_tokens, embedding_size)
        :return: (num_of_tokens, output_embedding_size)
        """
        raise NotImplemented()


class MoEWeightedAverage(MoE):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = self.gate(inputs)
        outputs = torch.stack([expert(inputs) for expert in self.experts], dim=-1)
        weights = weights.unsqueeze(1).expand_as(outputs)

        return torch.sum(outputs * weights, dim=-1)


class MoETopX(MoE):
    def __init__(
        self,
        experts: List[Expert],
        gate: nn.Module,
        freeze_expert: bool = False,
        top_x: int = 2,
    ):
        super().__init__(experts, gate, freeze_expert)
        self.top_x = top_x

    @staticmethod
    def get_weights_of_top_experts(
        weights: torch.Tensor,
        top_x: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        top_expert_weights, top_expert_indices = torch.topk(weights, top_x)
        top_expert_weights = torch.softmax(top_expert_weights, dim=-1)
        return top_expert_weights, top_expert_indices

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weights = self.gate(inputs)
        weights, top_expert_indices = self.get_weights_of_top_experts(weights, self.top_x)
        weights = torch.softmax(weights, dim=-1)

        outputs = torch.zeros((inputs.shape[0], self.out_dim))
        for expert_id, expert in enumerate(self.experts):
            token_indices, expert_weight_indices = torch.where(top_expert_indices == expert_id)
            outputs[token_indices] += (
                expert(inputs[token_indices])
                * weights[token_indices, expert_weight_indices].unsqueeze(-1)
            )
        return outputs
