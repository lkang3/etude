import torch

from positional_embeddings.models import SinusoidalPositionalEmbeddings


class TestSinusoidalPositionalEmbeddings:
    def test_forward(self) -> None:
        max_sequence_size = 7
        embedding_size = 13
        embeddings = SinusoidalPositionalEmbeddings(
            max_sequence_size,
            embedding_size,
        )

        batch_size = 5
        data = torch.randint(0, max_sequence_size, (batch_size,))
        output = embeddings(data)
        assert output.shape == (batch_size, max_sequence_size, embedding_size)
