from typing import Iterator
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch
from torch import nn

from tasks.masked_language_modeling.models import MLM
from transformers.encoders import EncoderBlock

torch.manual_seed(123)


class TestMLM:
    @pytest.fixture
    def max_seq_length(self) -> int:
        return 10

    @pytest.fixture
    def embedding_size(self) -> int:
        return 3

    @pytest.fixture
    def mock_embedding_layer_forward(
        self, max_seq_length: int, embedding_size: int
    ) -> Iterator[Mock]:
        with patch.object(nn.Embedding, "forward") as mock_func:
            mock_func.return_value = torch.rand(max_seq_length, embedding_size)
            yield mock_func

    @pytest.fixture
    def mock_positional_embeddings_layer(
        self, max_seq_length: int, embedding_size: int
    ) -> Mock:
        mock_layer = Mock(return_value=torch.rand(max_seq_length, embedding_size))

        return mock_layer

    @pytest.fixture
    def mock_encoder_block_forward(
        self, max_seq_length: int, embedding_size: int
    ) -> Iterator[Mock]:
        with patch.object(EncoderBlock, "__call__") as mock_func:
            mock_func.return_value = torch.rand(max_seq_length, embedding_size)
            yield mock_func

    @pytest.mark.usefixtures(
        "mock_embedding_layer_forward",
        "mock_encoder_block_forward",
        "mock_positional_embeddings_layer",
    )
    def test_forward(
        self,
        max_seq_length: int,
        embedding_size: int,
    ) -> None:
        num_of_vocabulary = 100
        embedding_size = 3
        num_of_encoders = 3
        model = MLM.create_from_config(
            max_seq_length,
            embedding_size,
            num_of_vocabulary,
            num_of_encoders,
        )
        inputs = torch.rand(max_seq_length, embedding_size)

        outputs = model(inputs)

        assert outputs.shape == (max_seq_length, num_of_vocabulary)
        assert all(torch.isclose(outputs.sum(-1), torch.ones(len(outputs))))
