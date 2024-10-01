from unittest.mock import Mock

import pytest
import torch

from transformers.models import EncoderAndDecoderLayer
from transformers.models import EncoderAndDecoderLayers
from transformers.models import EncoderOrDecoderConfig
from transformers.models import EncoderOrDecoderLayer
from transformers.models import EncoderOrDecoderLayers
from transformers.models import EncoderOrDecoderType

torch.manual_seed(123)


class TestEncoderOrDecoderLayer:
    @pytest.mark.parametrize(
        "layer_type",
        [EncoderOrDecoderType.DECODER, EncoderOrDecoderType.ENCODER],
        ids=str,
    )
    def test_forward(self, layer_type: EncoderOrDecoderType) -> None:
        embedding_size = 5
        seq_length = 4
        config = EncoderOrDecoderConfig(
            type=layer_type,
            source_seq_length=seq_length,
            target_seq_length=seq_length,
            embedding_size=embedding_size,
            num_of_layers=Mock(),
        )
        model = EncoderOrDecoderLayer.from_config(config)
        inputs = torch.rand(seq_length, embedding_size)
        outputs = model(inputs)

        assert outputs.shape == inputs.shape


class TestEncoderOrDecoderLayers:
    @pytest.mark.parametrize(
        "layer_type",
        [EncoderOrDecoderType.DECODER, EncoderOrDecoderType.ENCODER],
        ids=str,
    )
    @pytest.mark.parametrize("num_of_layers", [1, 2], ids=str)
    def test_forward(
        self, layer_type: EncoderOrDecoderType, num_of_layers: int
    ) -> None:
        embedding_size = 5
        seq_length = 4
        config = EncoderOrDecoderConfig(
            type=layer_type,
            source_seq_length=seq_length,
            target_seq_length=seq_length,
            embedding_size=embedding_size,
            num_of_layers=num_of_layers,
        )

        model = EncoderOrDecoderLayers.from_config(config)
        inputs = torch.rand(seq_length, embedding_size)
        outputs = model(inputs)

        assert len(model.layers) == num_of_layers
        assert outputs.shape == inputs.shape


class TestEncoderAndDecoderLayer:
    @pytest.mark.parametrize("embedding_size", [5, 6])
    def test_forward(self, embedding_size: int) -> None:
        source_seq_length = 4
        target_seq_length = 3
        config = EncoderOrDecoderConfig(
            type=EncoderOrDecoderType.HYBRID,
            source_seq_length=source_seq_length,
            target_seq_length=target_seq_length,
            embedding_size=embedding_size,
            num_of_layers=Mock(),
        )
        model = EncoderAndDecoderLayer.from_config(config)
        encoder_inputs = torch.rand(source_seq_length, embedding_size)
        decoder_inputs = torch.rand(target_seq_length, embedding_size)

        outputs = model(decoder_inputs, encoder_inputs)

        assert outputs.shape == decoder_inputs.shape


class TestEncoderAndDecoderLayers:
    @pytest.mark.parametrize("num_of_layers", [1, 2], ids=str)
    def test_forward(self, num_of_layers: int) -> None:
        embedding_size = 5
        source_seq_length = 4
        target_seq_length = 3
        config = EncoderOrDecoderConfig(
            type=EncoderOrDecoderType.HYBRID,
            source_seq_length=source_seq_length,
            target_seq_length=target_seq_length,
            embedding_size=embedding_size,
            num_of_layers=num_of_layers,
        )

        model = EncoderAndDecoderLayers.from_config(config)
        encoder_inputs = torch.rand(source_seq_length, embedding_size)
        decoder_inputs = torch.rand(target_seq_length, embedding_size)
        outputs = model(decoder_inputs, encoder_inputs)

        assert len(model.layers) == num_of_layers
        assert outputs.shape == decoder_inputs.shape
