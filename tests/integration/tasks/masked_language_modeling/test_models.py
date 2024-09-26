import torch

from tasks.masked_language_modeling.models import MLM
from tasks.masked_language_modeling.models import MLMConfig
from tests.utils import is_normalized


torch.manual_seed(123)


class TestMLM:
    def test_forward(self) -> None:
        max_seq_length = 11
        num_of_vocabulary = 20
        config = MLMConfig(
            max_seq_length=max_seq_length,
            embedding_size=7,
            num_of_vocabulary=20,
            num_of_encoder_layers=2,
        )
        model = MLM.from_config(config)

        inputs = torch.randint(0, num_of_vocabulary-1, (max_seq_length,))
        outputs = model(inputs)

        assert outputs.shape == (max_seq_length, num_of_vocabulary)
        assert is_normalized(outputs)
