import pytest
import torch
from torch import nn

from tasks.causal_language_modeling.models import CLMConfig
from tasks.causal_language_modeling.models import DecoderOnlyTransformer
from transformers.models import TransformerConfig
from transformers.models import TransformerType


class TestNextToken:
    @pytest.fixture
    def max_seq_length(self) -> int:
        return 10

    @pytest.fixture
    def embedding_size(self) -> int:
        return 7

    @pytest.fixture
    def num_of_vocabulary(self) -> int:
        return 20

    @pytest.fixture
    def config(
        self,
        max_seq_length: int,
        embedding_size: int,
        num_of_vocabulary: int,
    ) -> CLMConfig:
        return CLMConfig(
            TransformerConfig(
                type=TransformerType.DECODER,
                source_seq_length=max_seq_length,
                target_seq_length=max_seq_length,
                embedding_size=embedding_size,
                num_of_layers=1,
            ),
            num_of_vocabulary,
        )

    def test(self, config: CLMConfig, num_of_vocabulary: int) -> None:
        model = DecoderOnlyTransformer.from_config(config)
        inputs = torch.LongTensor([1, 2, 4, 5])
        max_seq_length = config.transformer_config.max_seq_length
        if inputs.shape[0] < max_seq_length:
            padding = nn.ZeroPad1d((0, max_seq_length - inputs.shape[0]))
            inputs = padding(inputs)
        print(inputs)
        outputs = model(inputs)

        assert outputs.shape == (max_seq_length, num_of_vocabulary)
