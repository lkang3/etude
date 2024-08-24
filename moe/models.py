from abc import abstractmethod
from typing import Callable
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim


def get_classification_accuracy(
    logits: torch.Tensor, expected_class_labels: torch.Tensor,
) -> torch.Tensor:
    _, class_labels = torch.max(logits, dim=-1)
    return (class_labels == expected_class_labels).sum() / len(expected_class_labels)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_func: Callable,
        accuracy_func: Callable,
        learning_rate: float = 0.01,
        epochs: int = 1,
        batch_size: int = 5,
    ):
        self.model = model
        self.loss_func = loss_func
        self.accuracy_func = accuracy_func
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train_with_one_epoch(self, inputs: torch.Tensor, expected_outputs: torch.Tensor) -> None:
        num_of_samples = inputs.shape[0]
        for batch_start_idx in range(0, num_of_samples, self.batch_size):
            batch_end_idx = min(num_of_samples, batch_start_idx + self.batch_size)
            batch_inputs = inputs[batch_start_idx: batch_end_idx]
            expected_batch_outputs = expected_outputs[batch_start_idx: batch_end_idx]
            self.optimizer.zero_grad()
            batch_outputs = self.model(batch_inputs)
            loss_expert1 = self.loss_func(batch_outputs, expected_batch_outputs)
            loss_expert1.backward()
            self.optimizer.step()

    def train(self, inputs: torch.Tensor, expected_outputs: torch.Tensor) -> nn.Module:
        for epoch in range(self.epochs):
            self.train_with_one_epoch(inputs, expected_outputs)
        return self.model

    @torch.no_grad()
    def evaluate(
        self, inputs: torch.Tensor, expected_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        outputs = self.model(inputs)
        loss = self.loss_func(outputs, expected_outputs)
        accuracy = self.accuracy_func(outputs, expected_outputs)

        return loss, accuracy


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
