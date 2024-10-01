import pytest

from tasks.seq2seq.models import NextToken
from tasks.seq2seq.models import Seq2SeqConfig
from tests.utils import create_sentence_with_tokens
from tests.utils import is_normalized


class TestNextToken:
    @pytest.mark.parametrize(
        "source_seq_length, target_seq_length",
        [(10, 9), (9, 10)],
        ids=str,
    )
    def test_forward(self, source_seq_length: int, target_seq_length: int) -> None:
        embedding_size = 17
        source_seq_num_of_vocabulary = 53
        target_seq_num_of_vocabulary = 27

        config = Seq2SeqConfig(
            source_seq_length=source_seq_length,
            target_seq_length=target_seq_length,
            embedding_size=embedding_size,
            source_seq_num_of_vocabulary=source_seq_num_of_vocabulary,
            target_seq_num_of_vocabulary=target_seq_num_of_vocabulary,
            num_of_decoders=2,
            num_of_encoders=3,
        )
        model = NextToken.from_config(config)
        source_seq = create_sentence_with_tokens(
            source_seq_num_of_vocabulary, source_seq_length
        )
        target_seq = create_sentence_with_tokens(
            target_seq_num_of_vocabulary, target_seq_length
        )

        outputs = model(target_seq, source_seq)

        assert is_normalized(outputs)
        assert outputs.shape == (1, target_seq_num_of_vocabulary)
