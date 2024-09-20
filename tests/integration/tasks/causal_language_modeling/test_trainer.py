from unittest.mock import Mock

import pytest
import torch

from tasks.causal_language_modeling.models import CLMConfig
from tasks.causal_language_modeling.models import NextToken
from tasks.causal_language_modeling.trainer import NextTokenTrainer
from tasks.causal_language_modeling.trainer import padding_sequence
from tests.utils import is_tensor_equal

torch.manual_seed(123)


def create_sentence_with_tokens(
    num_of_vocabulary: int, max_seq_length: int
) -> torch.Tensor:
    return torch.randint(0, num_of_vocabulary - 1, (max_seq_length,))


def test_padding_sequence() -> None:
    input_sequence = torch.Tensor([1, 2, 3, 4])
    max_sequence_length = len(input_sequence) + 3
    expected_value_to_pad = 0
    output_sequence = padding_sequence(
        input_sequence, expected_value_to_pad, max_sequence_length
    )

    assert is_tensor_equal(
        torch.Tensor([1, 2, 3, 4, 0, 0, 0]),
        output_sequence,
    )


def test_padding_sequence_not_padding() -> None:
    input_sequence = torch.Tensor([1, 2, 3, 4])
    max_sequence_length = len(input_sequence)
    output_sequence = padding_sequence(input_sequence, Mock(), max_sequence_length)

    assert is_tensor_equal(
        input_sequence,
        output_sequence,
    )


def test_padding_sequence_fail() -> None:
    input_sequence = torch.Tensor([1, 2, 3, 4])
    max_sequence_length = len(input_sequence) - 1
    with pytest.raises(ValueError):
        padding_sequence(input_sequence, Mock(), max_sequence_length)


class TestNextTokenTrainer:
    def test_train(self) -> None:
        train_input_sentence_size = 7
        train_target_sentence_size = 5
        max_seq_length = 17
        assert max_seq_length >= train_input_sentence_size + train_target_sentence_size
        num_of_vocabulary = 53
        config = CLMConfig(
            max_seq_length=max_seq_length,
            embedding_size=17,
            num_of_vocabulary=num_of_vocabulary,
            num_of_decoders=2,
        )
        trainer = NextTokenTrainer(
            NextToken.from_config(config), train_target_sentence_size
        )

        train_input_sentence = create_sentence_with_tokens(
            num_of_vocabulary, train_input_sentence_size
        )
        train_target_sentence = create_sentence_with_tokens(
            num_of_vocabulary, train_target_sentence_size
        )
        loss = trainer.train(train_input_sentence, train_target_sentence)
        torch.isclose(loss, torch.tensor(3.973403573036194))

    def test_inference(self) -> None:
        max_input_sentence_size = 6
        max_output_sentence_size = 5
        max_seq_length = max_input_sentence_size + max_output_sentence_size
        num_of_vocabulary = 53
        config = CLMConfig(
            max_seq_length=max_seq_length,
            embedding_size=17,
            num_of_vocabulary=num_of_vocabulary,
            num_of_decoders=2,
        )
        trainer = NextTokenTrainer(
            NextToken.from_config(config), max_output_sentence_size
        )

        input_sentence = create_sentence_with_tokens(
            num_of_vocabulary, max_input_sentence_size
        )
        output_sentence = trainer.inference(input_sentence)

        assert len(output_sentence) <= max_output_sentence_size
        assert output_sentence.dtype == input_sentence.dtype
