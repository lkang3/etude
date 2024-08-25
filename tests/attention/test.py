import torch

from attentions.entity import AttentionScores
from attentions.entity import AttentionSchema
from attentions.enum import AttentionDirection
from attentions.models import MultipleHeadAttention
from attentions.models import SlideWindowAttention
from attentions.models import VanillaAttentionForDecoder
from tests.utils import is_tensor_equal


torch.manual_seed(123)


class TestAttentionSchema:
    def test_vanilla_attention_mask(self) -> None:
        attention_schema = AttentionSchema(direction=AttentionDirection.BOTH)

        attention_mask = attention_schema.create_attention_mask_based_on_direction(5, 6)
        assert is_tensor_equal(attention_mask, torch.ones(5, 6))

    def test_left_only_attention_mask(self) -> None:
        attention_schema = AttentionSchema(direction=AttentionDirection.LEFT)

        attention_mask = attention_schema.create_attention_mask_based_on_direction(5, 6)
        assert is_tensor_equal(
            attention_mask,
            torch.Tensor(
                [
                    [1., torch.nan, torch.nan, torch.nan, torch.nan, torch.nan],
                    [1., 1., torch.nan, torch.nan, torch.nan, torch.nan],
                    [1., 1., 1., torch.nan, torch.nan, torch.nan],
                    [1., 1., 1., 1., torch.nan, torch.nan],
                    [1., 1., 1., 1., 1., torch.nan]
                ]
            )
        )

    def test_right_only_attention_mask(self) -> None:
        attention_schema = AttentionSchema(direction=AttentionDirection.RIGHT)

        attention_mask = attention_schema.create_attention_mask_based_on_direction(5, 6)
        assert is_tensor_equal(
            attention_mask,
            torch.Tensor(
                [
                    [1., 1., 1., 1., 1., 1.],
                    [torch.nan, 1., 1., 1., 1., 1.],
                    [torch.nan, torch.nan, 1., 1., 1., 1.],
                    [torch.nan, torch.nan, torch.nan, 1., 1., 1.],
                    [torch.nan, torch.nan, torch.nan, torch.nan, 1., 1.],
                 ]
            )
        )

    def test_sliding_window_attention_mask(self) -> None:
        attention_schema = AttentionSchema(direction=AttentionDirection.BOTH)

        attention_mask = attention_schema.create_sliding_window_mask(5, 6, 2)
        assert is_tensor_equal(
            attention_mask,
            torch.Tensor(
                [
                    [1., 1., 1., torch.nan, torch.nan, torch.nan],
                    [1., 1., 1., 1., torch.nan, torch.nan],
                    [1., 1., 1., 1., 1., torch.nan],
                    [torch.nan, 1., 1., 1., 1., 1.],
                    [torch.nan, torch.nan, 1., 1., 1., 1.],
                 ]
            )
        )

    def test_sliding_window_and_left_only_attention_mask(self) -> None:
        attention_schema = AttentionSchema(direction=AttentionDirection.LEFT)

        attention_mask = attention_schema.create_sliding_window_mask(5, 6, 2)
        assert is_tensor_equal(
            attention_mask,
            torch.Tensor(
                [
                    [1., torch.nan, torch.nan, torch.nan, torch.nan, torch.nan],
                    [1., 1., torch.nan, torch.nan, torch.nan, torch.nan],
                    [1., 1., 1., torch.nan, torch.nan, torch.nan],
                    [torch.nan, 1., 1., 1., torch.nan, torch.nan],
                    [torch.nan, torch.nan, 1., 1., 1., torch.nan]
                ]
            )
        )


class TestSlidingWindowAttention:
    def test_calculate_attention_scores(self) -> None:
        embedding_size = 8
        weight_embedding_size = 8
        source_sequence_size = 6
        query_sequence_size = 5
        window_size = 2
        attention = SlideWindowAttention(
            embedding_size=embedding_size,
            output_embedding_size=weight_embedding_size,
            source_sequence_size=source_sequence_size,
            query_sequence_size=query_sequence_size,
            window_size=window_size,
        )
        key = torch.rand((source_sequence_size, embedding_size))
        query = torch.rand((query_sequence_size, embedding_size))
        scores = attention.calculate_attention_scores(key, query)
        assert len(scores) == query_sequence_size
        assert all(
            len(score.scores) == len(score.source_seq_token_ids)
            for score in scores
        )
        assert scores[0] == AttentionScores(query_seq_token_id=0, source_seq_token_ids=[0, 1, 2], scores=torch.Tensor([0.3142, 0.3243, 0.3615]))

    def test_sliding_window_attention(self) -> None:
        embedding_size = 16
        weight_embedding_size = 8
        source_token_size = 9
        target_token_size = 8
        window_size = 2
        swa = SlideWindowAttention(
            embedding_size,
            weight_embedding_size,
            source_token_size,
            target_token_size,
            window_size,
        )
        query = torch.rand(target_token_size, embedding_size)
        key = torch.rand(source_token_size, embedding_size)
        value = torch.rand_like(key)

        outputs = swa(key, value, query)
        assert outputs.shape == (target_token_size, weight_embedding_size)


class TestMultiHeadAttention:
    def test_forward(self) -> None:
        num_of_heads = 5
        embedding_size = 30
        single_head_output_embedding_size = int(embedding_size / num_of_heads)
        source_sequence_size = 6
        query_sequence_size = 5
        head = VanillaAttentionForDecoder(
            embedding_size,
            single_head_output_embedding_size,
            source_sequence_size,
            query_sequence_size,
        )
        heads = [head] * num_of_heads
        attention = MultipleHeadAttention(heads)
        key = torch.rand(source_sequence_size, embedding_size)
        value = torch.rand_like(key)
        query = torch.rand(query_sequence_size, embedding_size)
        outputs = attention(key, value, query)

        assert outputs.shape == (query_sequence_size, embedding_size)
