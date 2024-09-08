import torch
from torch import nn

from positional_embeddings.models import SinusoidalPositionalEmbeddings
from transformers.encoders import EncoderBlock


class MLMHeader(nn.Module):
    def __init__(self, embedding_size: int, num_of_tokens: int):
        super().__init__()
        self.mlp = nn.Linear(embedding_size, num_of_tokens)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.mlp(inputs)
        return nn.functional.softmax(outputs, dim=-1)


class MLM(nn.Module):
    def __init__(
        self,
        embeddings: nn.Embedding,
        positional_embeddings: SinusoidalPositionalEmbeddings,
        encoder_block: EncoderBlock,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.positional_embeddings = positional_embeddings
        self.encoder_block = encoder_block
        self.header = MLMHeader(
            self.encoder_block.embedding_size, self.embeddings.num_embeddings
        )

    @classmethod
    def create_from_config(
        cls,
        max_seq_length: int,
        embedding_size,
        num_of_vocabulary: int,
        num_of_encoders: int,
    ) -> "MLM":
        embeddings_layer = nn.Embedding(num_of_vocabulary, embedding_size)
        positional_embeddings_layer = SinusoidalPositionalEmbeddings(
            max_seq_length, embedding_size
        )
        encoder_block = EncoderBlock(max_seq_length, embedding_size, num_of_encoders)
        return cls(embeddings_layer, positional_embeddings_layer, encoder_block)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.embeddings(inputs)
        positional_embeddings = self.positional_embeddings()

        embeddings = embeddings + positional_embeddings
        embeddings = self.encoder_block(embeddings)
        return self.header(embeddings)
