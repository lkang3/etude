import numpy as np
import torch

from attentions.models import MultipleHeadAttention
from attentions.models import VanillaAttentionForDecoder
from attentions.models import VanillaAttentionForEncoder


class TestVanillaAttentionForEncoder:
    def test_forward(self) -> None:
        embedding_size = 17
        output_embedding_size = 12
        source_sequence_size = 6
        query_sequence_size = 5
        model = VanillaAttentionForEncoder(
            embedding_size=embedding_size,
            output_embedding_size=output_embedding_size,
            source_sequence_size=source_sequence_size,
            query_sequence_size=query_sequence_size,
        )
        batch_size = 3
        query_sequence = torch.rand(batch_size, query_sequence_size, embedding_size)
        source_sequence = torch.rand(batch_size, source_sequence_size, embedding_size)
        outputs = model(
            key=source_sequence, value=source_sequence, query=query_sequence
        )

        assert outputs.shape == (batch_size, query_sequence_size, output_embedding_size)

    def test_calculate_attention_scores(self) -> None:
        embedding_size = 17
        output_embedding_size = 12
        source_sequence_size = 6
        query_sequence_size = 5
        model = VanillaAttentionForEncoder(
            embedding_size=embedding_size,
            output_embedding_size=output_embedding_size,
            source_sequence_size=source_sequence_size,
            query_sequence_size=query_sequence_size,
        )
        query_sequence = torch.rand(query_sequence_size, embedding_size)
        source_sequence = torch.rand(source_sequence_size, embedding_size)
        attention_scores = model.calculate_attention_scores(
            key=source_sequence, query=query_sequence
        )

        assert len(attention_scores) == query_sequence_size
        assert all(
            len(score.source_seq_token_ids) == len(score.scores)
            for score in attention_scores
        )
        attention_scores_dict = {
            score.query_seq_token_id: score for score in attention_scores
        }
        assert np.all(
            attention_scores_dict[0].source_seq_token_ids
            == np.arange(source_sequence_size)
        )
        assert np.all(
            attention_scores_dict[1].source_seq_token_ids
            == np.arange(source_sequence_size)
        )
        assert np.all(
            attention_scores_dict[2].source_seq_token_ids
            == np.arange(source_sequence_size)
        )


class TestVanillaAttentionForDecoder:
    def test_forward(self) -> None:
        embedding_size = 17
        output_embedding_size = 12
        source_sequence_size = 6
        query_sequence_size = 5
        model = VanillaAttentionForDecoder(
            embedding_size=embedding_size,
            output_embedding_size=output_embedding_size,
            source_sequence_size=source_sequence_size,
            query_sequence_size=query_sequence_size,
        )
        batch_size = 3
        query_sequence = torch.rand(batch_size, query_sequence_size, embedding_size)
        source_sequence = torch.rand(batch_size, source_sequence_size, embedding_size)
        outputs = model(
            key=source_sequence, value=source_sequence, query=query_sequence
        )

        assert outputs.shape == (batch_size, query_sequence_size, output_embedding_size)

    def test_calculate_attention_scores(self) -> None:
        embedding_size = 17
        output_embedding_size = 12
        source_sequence_size = 4
        query_sequence_size = 3
        model = VanillaAttentionForDecoder(
            embedding_size=embedding_size,
            output_embedding_size=output_embedding_size,
            source_sequence_size=source_sequence_size,
            query_sequence_size=query_sequence_size,
        )
        query_sequence = torch.rand(query_sequence_size, embedding_size)
        source_sequence = torch.rand(source_sequence_size, embedding_size)
        attention_scores = model.calculate_attention_scores(
            key=source_sequence, query=query_sequence
        )

        assert len(attention_scores) == query_sequence_size
        assert all(
            len(score.source_seq_token_ids) == len(score.scores)
            for score in attention_scores
        )
        attention_scores_dict = {
            score.query_seq_token_id: score for score in attention_scores
        }
        assert attention_scores_dict[0].source_seq_token_ids == [0]
        assert attention_scores_dict[1].source_seq_token_ids == [0, 1]
        assert attention_scores_dict[2].source_seq_token_ids == [0, 1, 2]


class TestMultipleHeadAttention:
    def test_forward(self) -> None:
        embedding_size = 17
        output_embedding_size = 12
        source_sequence_size = 6
        query_sequence_size = 5
        attention_model = VanillaAttentionForEncoder(
            embedding_size=embedding_size,
            output_embedding_size=output_embedding_size,
            source_sequence_size=source_sequence_size,
            query_sequence_size=query_sequence_size,
        )
        num_of_heads = 2
        model = MultipleHeadAttention([attention_model for _ in range(num_of_heads)])

        batch_size = 3
        query_sequence = torch.rand(batch_size, query_sequence_size, embedding_size)
        source_sequence = torch.rand(batch_size, source_sequence_size, embedding_size)
        outputs = model(
            key=source_sequence, value=source_sequence, query=query_sequence
        )
        assert outputs.shape == (
            batch_size,
            query_sequence_size,
            output_embedding_size * num_of_heads,
        )
