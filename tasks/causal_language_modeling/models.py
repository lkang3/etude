from dataclasses import dataclass

import torch
from torch import nn

from positional_embeddings.models import SinusoidalPositionalEmbeddings
from transformers.models import EncoderOrDecoderConfig
from transformers.models import EncoderOrDecoderLayers
from transformers.models import EncoderOrDecoderType


@dataclass
class CLMConfig:
    max_seq_length: int
    embedding_size: int
    num_of_vocabulary: int
    num_of_decoders: int


class DecoderOnlyTransformerHeader(nn.Module):
    def __init__(self, embedding_size: int, num_of_tokens: int):
        super().__init__()
        self.mlp = nn.Linear(embedding_size, num_of_tokens)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        embeddings: nn.Embedding,
        positional_embeddings: SinusoidalPositionalEmbeddings,
        transformer_layers: EncoderOrDecoderLayers,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.positional_embeddings = positional_embeddings
        self.transformer_layers = transformer_layers
        self.header = DecoderOnlyTransformerHeader(
            self.transformer_layers.embedding_size, self.embeddings.num_embeddings
        )

    @classmethod
    def from_config(cls, config: CLMConfig) -> "DecoderOnlyTransformer":
        embedding_size = config.embedding_size
        embeddings_layer = nn.Embedding(config.num_of_vocabulary, embedding_size)
        positional_embeddings_layer = SinusoidalPositionalEmbeddings(
            config.max_seq_length,
            embedding_size,
        )
        transformer_config = EncoderOrDecoderConfig(
            type=EncoderOrDecoderType.DECODER,
            source_seq_length=config.max_seq_length,
            target_seq_length=config.max_seq_length,
            embedding_size=embedding_size,
            num_of_layers=config.num_of_decoders,
        )
        transformer_layers = EncoderOrDecoderLayers.from_config(transformer_config)
        return cls(embeddings_layer, positional_embeddings_layer, transformer_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.embeddings(inputs)
        positional_embeddings = self.positional_embeddings()
        embeddings = embeddings + positional_embeddings
        hidden_states = self.transformer_layers(embeddings)

        return self.header(hidden_states)


class NextToken(nn.Module):
    def __init__(self, transformer: nn.Module, max_sequence_length: int):
        super().__init__()
        self.transformer = transformer
        self.max_sequence_length = max_sequence_length

    @classmethod
    def from_config(cls, config: CLMConfig) -> "NextToken":
        return cls(
            DecoderOnlyTransformer.from_config(config),
            config.max_seq_length,
        )

    @staticmethod
    def aggregate_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states[-1, :]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.transformer(inputs)
        hidden_states = self.aggregate_hidden_states(hidden_states)
        outputs = nn.functional.softmax(hidden_states, dim=-1)
        return outputs.unsqueeze(0)
