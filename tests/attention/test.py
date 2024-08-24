import torch

from moe.attention import chunk
from moe.attention import calculate_slide_window_attention_scores
from moe.attention import AttentionScores
from moe.attention import SlideWindowAttention


torch.manual_seed(123)


class TestSlidingWindowAttention:

    def test_calculate_slide_window_attention_scores(self) -> None:
        embedding_size = 256
        num_tokens_source = 9
        num_tokens_target = 8
        k = torch.rand(num_tokens_source, embedding_size)
        q = torch.rand(num_tokens_target, embedding_size)

        window_size = 2
        scores = calculate_slide_window_attention_scores(k, q, window_size)
        assert scores[0] == AttentionScores(target_seq_token_id=0, source_seq_token_ids=[0, 1, 2], scores=torch.Tensor([0.2966, 0.3913, 0.3121]))
        assert scores[-1] == AttentionScores(target_seq_token_id=7, source_seq_token_ids=[5, 6, 7, 8], scores=torch.Tensor([0.2824, 0.2287, 0.2581, 0.2308]))

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

        outputs = swa.forward(key, value, query)
        assert outputs.shape == (target_token_size, weight_embedding_size)


    def test_test(self) -> None:
        data = torch.rand((1, 10, 256))
        chunk(data, 3)