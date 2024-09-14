from typing import Iterator
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from transformers.models import EncoderAndDecoderLayer
from transformers.models import EncoderAndDecoderLayers
from transformers.models import EncoderOrDecoderLayer
from transformers.models import EncoderOrDecoderLayers

torch.manual_seed(123)


@pytest.fixture
def module_under_test() -> str:
    return "transformers.models"


@pytest.fixture
def residual_connection(module_under_test: str) -> Iterator[Mock]:
    with patch(f"{module_under_test}.residual_connection") as mock_func:
        yield mock_func


@pytest.fixture
def create_attention_layer(module_under_test: str) -> Iterator[Mock]:
    with patch(f"{module_under_test}.create_attention_layer") as mock_func:
        yield mock_func


@pytest.fixture
def create_mlp(module_under_test: str) -> Iterator[Mock]:
    with patch(f"{module_under_test}.create_mlp") as mock_func:
        yield mock_func


@pytest.fixture
def create_norm_layer(module_under_test: str) -> Iterator[Mock]:
    with patch(f"{module_under_test}.create_norm_layer") as mock_func:
        yield mock_func


@pytest.fixture
def encoder_or_decoder_layer_create_block() -> Iterator[Mock]:
    with patch.object(EncoderOrDecoderLayer, "create_block") as mock_func:
        yield mock_func


@pytest.fixture
def encoder_and_decoder_layer_create_block() -> Iterator[Mock]:
    with patch.object(EncoderAndDecoderLayer, "create_block") as mock_func:
        yield mock_func


@pytest.fixture
def create_drop_out_layer(module_under_test: str) -> Iterator[Mock]:
    with patch(f"{module_under_test}.create_drop_out_layer") as mock_func:
        yield mock_func


class TestEncoderOrDecoderLayer:
    @pytest.mark.usefixtures(
        "create_mlp",
        "create_norm_layer",
        "encoder_or_decoder_layer_create_block",
        "create_drop_out_layer",
    )
    def test_from_config(self, create_attention_layer: Mock) -> None:
        config = Mock()
        EncoderOrDecoderLayer.from_config(config)

        create_attention_layer.assert_called_once_with(
            config.max_seq_length, config.embedding_size, config.type
        )


class TestEncoderOrDecoderLayers:
    def test_from_config(self, encoder_or_decoder_layer_create_block: Mock) -> None:
        config = Mock()
        EncoderOrDecoderLayers.from_config(config)

        encoder_or_decoder_layer_create_block.assert_called_once_with(config)


class TestEncoderAndDecoderLayer:
    # TODO pls improve this testing
    @pytest.mark.usefixtures(
        "create_mlp",
        "create_norm_layer",
        "encoder_or_decoder_layer_create_block",
        "create_drop_out_layer",
    )
    def test_from_config(self, create_attention_layer: Mock) -> None:
        config = Mock()
        EncoderAndDecoderLayer.from_config(config)

        create_attention_layer.assert_called_once_with(
            config.max_seq_length, config.embedding_size, config.type
        )


class TestEncoderAndDecoderLayers:
    def test_from_config(self, encoder_and_decoder_layer_create_block: Mock) -> None:
        config = Mock()
        EncoderAndDecoderLayers.from_config(config)

        encoder_and_decoder_layer_create_block.assert_called_once_with(config)
