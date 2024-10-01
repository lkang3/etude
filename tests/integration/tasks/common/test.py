from unittest.mock import Mock

import pytest
import torch

from tasks.common.models import EncoderAndDecoderTransformer
from tasks.common.models import EncoderTransformer
from tasks.common.models import TransformerConfig
from tests.utils import create_sentence_with_tokens

torch.manual_seed(123)


class TestEncoderTransformer:
    def test_forward(self) -> None:
        hidden_embedding_size = 17
        output_embedding_size = 21
        num_of_vocabulary = 53
        seq_length = 10
        config = TransformerConfig(
            source_seq_length=seq_length,
            target_seq_length=Mock(),
            hidden_embedding_size=hidden_embedding_size,
            output_embedding_size=output_embedding_size,
            num_of_vocabulary=num_of_vocabulary,
            num_of_encoder_or_decoder_layers=2,
        )
        model = EncoderTransformer.from_config(config)
        inputs = create_sentence_with_tokens(num_of_vocabulary, seq_length)
        outputs = model(inputs)

        assert outputs.shape == (seq_length, output_embedding_size)


class TestEncoderAndDecoderTransformer:
    @pytest.mark.parametrize(
        "source_seq_length, target_seq_length",
        [(10, 9), (9, 10)],
        ids=str,
    )
    def test_forward(self, source_seq_length: int, target_seq_length: int) -> None:
        hidden_embedding_size = 17
        output_embedding_size = 21
        num_of_vocabulary = 53
        config = TransformerConfig(
            source_seq_length=source_seq_length,
            target_seq_length=target_seq_length,
            hidden_embedding_size=hidden_embedding_size,
            output_embedding_size=output_embedding_size,
            num_of_vocabulary=num_of_vocabulary,
            num_of_encoder_or_decoder_layers=2,
        )
        model = EncoderAndDecoderTransformer.from_config(config)
        target_seq = create_sentence_with_tokens(num_of_vocabulary, target_seq_length)
        encoder_output_of_source_seq = torch.rand(
            source_seq_length, hidden_embedding_size
        )
        outputs = model(target_seq, encoder_output_of_source_seq)

        assert outputs.shape == (target_seq_length, output_embedding_size)
