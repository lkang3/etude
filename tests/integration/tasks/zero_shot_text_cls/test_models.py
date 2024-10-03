import torch

from tasks.zero_shot_text_cls.models import ZeroShotTextClassifier
from tasks.zero_shot_text_cls.models import ZeroShotTextClsConfig
from tests.utils import is_normalized


class TestZeroShotTextClassifier:
    def test_forward(self) -> None:
        max_source_seq_length = 9
        max_target_seq_length = 7
        source_hidden_embedding_size = 32
        target_hidden_embedding_size = 16
        max_target_options = 13
        num_of_shared_vocabularies = 53
        num_of_encoder_layers = 2
        config = ZeroShotTextClsConfig(
            max_source_seq_length=max_source_seq_length,
            max_target_seq_length=max_target_seq_length,
            source_hidden_embedding_size=source_hidden_embedding_size,
            target_hidden_embedding_size=target_hidden_embedding_size,
            max_target_options=max_target_options,
            num_of_shared_vocabularies=num_of_shared_vocabularies,
            num_of_encoder_layers=num_of_encoder_layers,
        )
        model = ZeroShotTextClassifier.from_config(config)
        source_sentence = torch.randint(
            0, num_of_shared_vocabularies - 1, (max_source_seq_length,)
        )
        # example (total num of class_* and pad is max_target_options)
        # "class_1[SEP]class_2[SEP]...[SEP]class_n[SPE]pad[SEP]...[SEP]"
        target_sentence = torch.randint(
            0, num_of_shared_vocabularies - 1, (max_target_seq_length,)
        )
        target_probabilities = model(source_sentence, target_sentence)

        assert target_probabilities.shape == (1, max_target_options)
        assert is_normalized(target_probabilities)
