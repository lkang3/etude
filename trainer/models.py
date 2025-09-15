from typing import Callable
from typing import Tuple

import torch
from torch import nn
from torch import optim


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_func: Callable,
        learning_rate: float = 0.01,
        epochs: int = 1,
        batch_size: int = 5,
    ):
        self.model = model
        self.loss_func = loss_func
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    @staticmethod
    def get_end_index_of_samples_in_one_batch(
        total_samples: int,
        batch_size: int,
        start_index_of_current_batch: int,
    ) -> int:
        return min(total_samples, start_index_of_current_batch + batch_size)

    @staticmethod
    def get_train_inputs_of_one_epoch(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        sample_start_index: int,
        sample_end_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            inputs[sample_start_index:sample_end_index],
            targets[sample_start_index:sample_end_index],
        )

    def train_with_one_epoch(
        self, inputs: torch.Tensor, expected_outputs: torch.Tensor
    ) -> None:
        num_of_samples = inputs.shape[0]
        for batch_start_idx in range(0, num_of_samples, self.batch_size):
            batch_end_idx = self.get_end_index_of_samples_in_one_batch(
                num_of_samples,
                self.batch_size,
                batch_start_idx,
            )
            (batch_inputs, expected_batch_outputs) = self.get_train_inputs_of_one_epoch(
                inputs,
                expected_outputs,
                batch_start_idx,
                batch_end_idx,
            )
            self.optimizer.zero_grad()
            batch_outputs = self.model(batch_inputs)
            loss = self.loss_func(batch_outputs, expected_batch_outputs)
            loss.backward()
            self.optimizer.step()

    def train(self, inputs: torch.Tensor, expected_outputs: torch.Tensor) -> nn.Module:
        for epoch in range(self.epochs):
            self.train_with_one_epoch(inputs, expected_outputs)
        return self.model

    @torch.no_grad()
    def evaluate(
        self,
        inputs: torch.Tensor,
        expected_outputs: torch.Tensor,
    ) -> torch.Tensor:
        self.model.eval()
        outputs = self.model(inputs)
        loss = self.loss_func(outputs, expected_outputs)
        return loss
