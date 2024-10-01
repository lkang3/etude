import torch

from tasks.seq2seq.models import Seq2SeqConfig
from tasks.seq2seq.trainers import NextTokenTrainer
from tests.utils import create_sentence_with_tokens
from tests.utils import is_sentence_token_valid


class TestNextTokenTrainer:
    def test_train(self) -> None:
        source_seq_length = 17
        target_seq_length = 15
        source_seq_num_of_vocabulary = 53
        target_seq_num_of_vocabulary = 27
        config = Seq2SeqConfig(
            source_seq_length=source_seq_length,
            target_seq_length=target_seq_length,
            embedding_size=27,
            source_seq_num_of_vocabulary=source_seq_num_of_vocabulary,
            target_seq_num_of_vocabulary=target_seq_num_of_vocabulary,
            num_of_encoders=2,
            num_of_decoders=3,
        )
        trainer = NextTokenTrainer.from_config(config)

        train_input_sentence = create_sentence_with_tokens(
            source_seq_num_of_vocabulary, source_seq_length
        )
        train_target_sentence = create_sentence_with_tokens(
            target_seq_num_of_vocabulary, target_seq_length
        )
        loss = trainer.train(train_input_sentence, train_target_sentence)

        assert loss.dtype == torch.float

    def test_inference(self) -> None:
        source_seq_length = 17
        target_seq_length = 15
        source_seq_num_of_vocabulary = 53
        target_seq_num_of_vocabulary = 27
        config = Seq2SeqConfig(
            source_seq_length=source_seq_length,
            target_seq_length=target_seq_length,
            embedding_size=27,
            source_seq_num_of_vocabulary=source_seq_num_of_vocabulary,
            target_seq_num_of_vocabulary=target_seq_num_of_vocabulary,
            num_of_encoders=2,
            num_of_decoders=3,
        )
        trainer = NextTokenTrainer.from_config(config)
        input_sentence = create_sentence_with_tokens(
            source_seq_num_of_vocabulary, source_seq_length
        )
        output_sentence = trainer.inference(input_sentence)

        assert output_sentence.dtype == input_sentence.dtype
        assert len(output_sentence) <= target_seq_length
        assert is_sentence_token_valid(
            output_sentence, torch.arange(0, target_seq_num_of_vocabulary)
        )
