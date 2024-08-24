import pytest

import torch
from torch import Tensor

from moe.models import Expert
from moe.models import ExpertLayer
from moe.models import Gate
from moe.models import GateLayer
from moe.models import MoE
from moe.models import MoeLayer


torch.manual_seed(123)


class TestMoeLayer:
    @pytest.fixture
    def batch_size(self) -> int:
        return 3

    @pytest.fixture
    def data_size(self) -> int:
        return 5

    @pytest.fixture
    def embedding_size(self) -> int:
        return 768

    @pytest.fixture
    def data(self, batch_size: int, data_size: int, embedding_size: int) -> Tensor:
        return torch.rand((batch_size, data_size, embedding_size))

    def test_forward(self, data: Tensor) -> None:
        embedding_size = data.shape[-1]
        num_of_experts = 5
        experts = [ExpertLayer(embedding_size) for _ in range(num_of_experts)]
        gate = GateLayer(embedding_size, num_of_experts)
        moe_layer = MoeLayer(experts, gate, top_k=2)

        moe_layer(data)


class TestMoE:
    def test_moe(self) -> None:
        embedding_size = 5
        num_of_experts = 3
        experts = [Expert(embedding_size, embedding_size) for _ in range(num_of_experts)]
        gate = Gate(embedding_size, num_of_experts)
        moe = MoE(experts, gate)

        num_of_samples = 7
        inputs = torch.randn(num_of_samples, embedding_size)

        outputs = moe(inputs)
        assert outputs.shape == (num_of_samples, embedding_size)
