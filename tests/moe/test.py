import numpy as np
import pytest
from typing import Iterator
from unittest.mock import Mock
from unittest.mock import patch

import torch
import torch.nn as nn

from moe.models import Expert
from moe.models import Gate
from moe.models import MoETopX
from moe.models import MoEWeightedAverage


torch.manual_seed(123)


@pytest.fixture
def module_under_test() -> str:
    return "moe.models"


@pytest.fixture
def expert_forward() -> Mock:
    with patch.object(Expert, "forward") as mock_func:
        yield mock_func


@pytest.fixture
def gate_forward() -> Mock:
    with patch.object(Gate, "forward") as mock_func:
        yield mock_func


@pytest.fixture
def get_weights_of_top_experts() -> Iterator[Mock]:
    with patch.object(MoETopX, "get_weights_of_top_experts") as mock_func:
        yield mock_func


@pytest.fixture
def torch_topk() -> Iterator[Mock]:
    with patch.object(torch, "topk") as mock_func:
        yield mock_func


@pytest.fixture
def torch_softmax() -> Iterator[Mock]:
    with patch.object(torch, "softmax") as mock_func:
        yield mock_func


class TestMoETopX:
    def test_forward(self) -> None:
        num_of_tokens = 5
        input_embedding_size = 4
        expert_output_embedding_size = 2
        num_of_experts = 3
        expert = Expert(input_embedding_size, expert_output_embedding_size)
        experts = [expert for _ in range(num_of_experts)]
        gate = Gate(input_embedding_size, num_of_experts, activate_layer=nn.ReLU())

        model = MoETopX(
            experts=experts,
            gate=gate,
            top_x=num_of_experts-1,
        )
        inputs = torch.rand((num_of_tokens, input_embedding_size))
        outputs = model(inputs)

        assert outputs.shape == (num_of_tokens, expert_output_embedding_size)

    def test_get_weights_of_top_experts(
        self,
        torch_topk: Mock,
        torch_softmax: Mock,
    ) -> None:
        inputs = Mock()
        top_x = 2
        expected_top_expert_weights = Mock()
        expected_top_expert_indices = Mock()
        torch_topk.return_value = (expected_top_expert_weights, expected_top_expert_indices)
        outputs = MoETopX.get_weights_of_top_experts(inputs, top_x)

        torch_topk.assert_called_once_with(inputs, top_x)
        torch_softmax.assert_called_once_with(expected_top_expert_weights, dim=-1)
        assert outputs == (
            torch_softmax.return_value,
            expected_top_expert_indices,
        )


class TestMoEWeightedAverage:
    def test_forward(self, expert_forward: Mock, gate_forward: Mock) -> None:
        num_of_tokens = 3
        input_embedding_size = 4
        expert_output_embedding_size = 2
        num_of_experts = 2
        expert_forward.return_value = torch.ones((num_of_tokens, expert_output_embedding_size))
        expert = Expert(input_embedding_size, expert_output_embedding_size)
        experts = [expert] * num_of_experts
        weights_of_experts = [0.2, 0.8]
        gate_forward.return_value = torch.Tensor(weights_of_experts)
        model = MoEWeightedAverage(experts, Gate(input_embedding_size, num_of_experts, nn.ReLU()))

        outputs = model(torch.rand((num_of_tokens, input_embedding_size)))

        assert torch.all(
            outputs == torch.Tensor(
                [np.multiply(weights_of_experts, num_of_experts)] * num_of_tokens
            )
        )
