import torch

from tasks.masked_language_modeling.models import MLMConfig
from tasks.masked_language_modeling.trainers import MLMTrainer

from tests.utils import create_sentence_with_tokens


torch.manual_seed(123)


def mask_sentence(
    input_sentence: torch.Tensor, indices_of_masked_tokens: torch.Tensor,
) -> torch.Tensor:
    pass


class TestMLMTrainer:
    def test_train(self) -> None:
        max_seq_length = 11
        num_of_vocabulary = 20
        config = MLMConfig(
            max_seq_length=max_seq_length,
            embedding_size=7,
            num_of_vocabulary=20,
            num_of_encoder_layers=2,
        )
        model = MLMTrainer.from_config(config)

        train_target_sentence = create_sentence_with_tokens(num_of_vocabulary, max_seq_length)
        train_input_sequence = mask_sentence(
            train_target_sentence, torch.randint(0, max_seq_length-1, (2,))
        )
        loss = model.train(train_input_sequence, train_target_sentence)

    def test_inference(self) -> None:
        max_seq_length = 11
        num_of_vocabulary = 20
        config = MLMConfig(
            max_seq_length=max_seq_length,
            embedding_size=7,
            num_of_vocabulary=20,
            num_of_encoder_layers=2,
        )
        model = MLMTrainer.from_config(config)

        input_sentence = create_sentence_with_tokens(num_of_vocabulary, max_seq_length)
        input_sentence = mask_sentence(input_sentence, torch.randint(0, max_seq_length-1, (2,)))
        outputs = model.inference(input_sentence)
