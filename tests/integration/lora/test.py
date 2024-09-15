import numpy as np
import pytest
import torch
from torch import nn

from lora.models import LoRAConfig
from lora.models import LoRALayer
from lora.models import LoRAModel
from tests.utils import is_tensor_equal

torch.manual_seed(123)


class TestLoRALayer:
    def test_from_config(self) -> None:
        input_dim = np.random.randint(1, 5)
        output_dim = np.random.randint(1, 5)
        rank = min(input_dim, output_dim)
        alpha = 0.01
        config = LoRAConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            rank=min(input_dim, output_dim),
            alpha=alpha,
            seed=123,
        )
        model = LoRALayer.from_config(config)

        assert model.matrix_a.shape == (rank, output_dim)
        assert model.matrix_b.shape == (input_dim, rank)
        assert torch.count_nonzero(model.matrix_a).item() == 0
        assert torch.count_nonzero(model.matrix_b).item() > 0

    def test_reproducibility(self) -> None:
        input_dim = np.random.randint(1, 5)
        output_dim = np.random.randint(1, 5)
        config = LoRAConfig(
            input_dim=5,
            output_dim=6,
            rank=min(input_dim, output_dim),
            alpha=0.01,
            seed=123,
        )
        model_one = LoRALayer.from_config(config)
        model_two = LoRALayer.from_config(config)

        assert is_tensor_equal(model_one.matrix_a, model_two.matrix_a)
        assert is_tensor_equal(model_one.matrix_b, model_two.matrix_b)

    @pytest.mark.parametrize("input_dim", [5, 6])
    @pytest.mark.parametrize("output_dim", [5, 6])
    def test_zero_starting_point(self, input_dim: int, output_dim: int):
        config = LoRAConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            rank=min(input_dim, output_dim),
            alpha=0.01,
            seed=123,
        )
        model = LoRALayer.from_config(config)

        outputs = model(torch.rand(input_dim))
        assert is_tensor_equal(outputs, torch.zeros(output_dim))


class TestLoRAModel:
    def test_freeze_model_params(self) -> None:
        input_dim = np.random.randint(1, 5)
        output_dim = np.random.randint(1, 5)
        model_to_freeze = nn.Linear(input_dim, output_dim)

        assert all(param.requires_grad for param in model_to_freeze.parameters())
        config = LoRAConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            rank=min(input_dim, output_dim),
            alpha=0.01,
            seed=123,
        )
        model = LoRAModel(model_to_freeze, config)

        assert all(
            not param.requires_grad for param in model.model_to_freeze.parameters()
        )
        assert all(param.requires_grad for param in model.lora_layer.parameters())

    def test_forward(self):
        input_dim = np.random.randint(1, 5)
        output_dim = np.random.randint(1, 5)
        model_to_freeze = nn.Linear(input_dim, output_dim)

        assert all(param.requires_grad for param in model_to_freeze.parameters())
        config = LoRAConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            rank=min(input_dim, output_dim),
            alpha=0.01,
            seed=123,
        )
        model = LoRAModel(model_to_freeze, config)
        batch_size = np.random.randint(1, 7)
        outputs = model(torch.rand(batch_size, input_dim))

        assert outputs.shape == (batch_size, output_dim)
