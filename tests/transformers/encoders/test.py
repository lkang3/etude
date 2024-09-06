import pytest
import torch
from typing import Iterator
from unittest.mock import Mock
from unittest.mock import patch

from transformers.encoders import Encoder

torch.manual_seed(123)


@pytest.fixture
def module_under_test() -> str:
    return "encoder.models"


@pytest.fixture
def residual_connection(module_under_test: str) -> Iterator[Mock]:
    with patch(f"{module_under_test}.residual_connection") as mock_func:
        yield mock_func


class TestEncoder:
    def test_forward_workflow(self, residual_connection: Mock) -> None:
        attention = Mock()
        mlp = Mock()
        norm_after_attention = Mock()
        norm_after_mlp = Mock()
        drop_out_after_attention = Mock()
        drop_out_after_mlp = Mock(0)
        model = Encoder(
            attention,
            mlp,
            norm_after_attention,
            norm_after_mlp,
            drop_out_after_attention,
            drop_out_after_mlp,
        )
        input_data = Mock()
        model(input_data)

    def test_create_from_config(self) -> None:
        seq_length = Mock()
        embedding_size = Mock()
        Encoder.create_from_config(seq_length, embedding_size)


    def test_forward(self) -> None:
        seq_length = 3
        embedding_size = 5
        model = Encoder.create_from_config(seq_length, embedding_size)
        inputs = torch.rand(seq_length, embedding_size)
        outputs = model(inputs)

        assert inputs.shape == outputs.shape
