import torch
from dataclasses import dataclass
from torch import nn

from positional_embeddings.models import SinusoidalPositionalEmbeddings
from transformers.models import EncoderOrDecoderLayers
from transformers.models import TransformerConfig
from transformers.models import TransformerType


@dataclass
class MLMConfig:
    max_seq_length: int
    embedding_size: int
    num_of_vocabulary: int
    num_of_encoder_layers: int

    @property
    def encoder_config(self) -> TransformerConfig:
        return TransformerConfig(
            TransformerType.ENCODER,
            self.max_seq_length,
            self.max_seq_length,
            self.embedding_size,
            self.num_of_encoder_layers,
        )


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
        encoder_layers: EncoderOrDecoderLayers,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.positional_embeddings = positional_embeddings
        self.encoder_layers = encoder_layers
        self.header = MLMHeader(
            self.encoder_layers.embedding_size, self.embeddings.num_embeddings
        )

    @classmethod
    def from_config(cls, config: MLMConfig) -> "MLM":
        embeddings_layer = nn.Embedding(config.num_of_vocabulary, config.embedding_size)
        positional_embeddings_layer = SinusoidalPositionalEmbeddings(
            config.max_seq_length, config.embedding_size
        )
        encoder_block = EncoderOrDecoderLayers.from_config(config.encoder_config)
        return cls(embeddings_layer, positional_embeddings_layer, encoder_block)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.embeddings(inputs)
        positional_embeddings = self.positional_embeddings()

        embeddings = embeddings + positional_embeddings
        embeddings = self.encoder_layers(embeddings)
        return self.header(embeddings)
