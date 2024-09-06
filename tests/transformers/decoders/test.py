import pytest

import torch

from transformers.decoders import Decoder

torch.manual_seed(123)


class TestDecoder:
    @pytest.mark.parametrize("embedding_size", [5, 6])
    def test_forward(self, embedding_size: int) -> None:
        source_seq_length = 4
        query_seq_length = 3
        model = Decoder.create_from_config(source_seq_length, query_seq_length, embedding_size)
        encoder_inputs = torch.rand(source_seq_length, embedding_size)
        decoder_inputs = torch.rand(query_seq_length, embedding_size)

        outputs = model(decoder_inputs, encoder_inputs)

        assert outputs.shape == decoder_inputs.shape
