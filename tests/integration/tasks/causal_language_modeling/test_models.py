import pytest
import torch

from tasks.causal_language_modeling.models import CLMConfig
from tasks.causal_language_modeling.models import DecoderOnlyTransformer
from tasks.causal_language_modeling.models import NextToken
from tests.utils import is_normalized


class TestDecoderOnlyTransformer:
    @pytest.mark.parametrize("max_seq_length", [1, 2], ids=str)
    def test_forward(self, max_seq_length: int) -> None:
        embedding_size = 17
        num_of_vocabulary = 53
        config = CLMConfig(
            max_seq_length=max_seq_length,
            embedding_size=embedding_size,
            num_of_vocabulary=num_of_vocabulary,
            num_of_decoders=2,
        )
        model = DecoderOnlyTransformer.from_config(config)
        inputs = torch.randint(0, num_of_vocabulary - 1, (max_seq_length,))
        outputs = model(inputs)

        assert outputs.shape == (max_seq_length, num_of_vocabulary)


class TestNextToken:
    @pytest.mark.parametrize("max_seq_length", [1, 2], ids=str)
    def test_forward(self, max_seq_length: int) -> None:
        embedding_size = 17
        num_of_vocabulary = 53
        config = CLMConfig(
            max_seq_length=max_seq_length,
            embedding_size=embedding_size,
            num_of_vocabulary=num_of_vocabulary,
            num_of_decoders=2,
        )
        model = NextToken.from_config(config)
        inputs = torch.randint(0, num_of_vocabulary - 1, (max_seq_length,))
        outputs = model(inputs)

        assert is_normalized(outputs)
        assert outputs.shape == (1, num_of_vocabulary)
