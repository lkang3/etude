import pytest

from unittest.mock import Mock

from tasks.masked_language_modeling.models import MLM
from transformers.encoders import Encoder


class TestMLM:
    def test_things(self) -> None:
        embeddings = Mock()
        positional_embeddings = Mock()
        encoder_block = Mock()
        output_sequence_token_logits = Mock()
        model = MLM(

        )

        outputs = model()
