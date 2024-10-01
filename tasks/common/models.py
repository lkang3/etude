from dataclasses import dataclass

import torch
from torch import nn

from positional_embeddings.models import SinusoidalPositionalEmbeddings
from transformers.models import EncoderAndDecoderLayers
from transformers.models import EncoderOrDecoderConfig
from transformers.models import EncoderOrDecoderLayers
from transformers.models import EncoderOrDecoderType


@dataclass
class TransformerConfig:
    source_seq_length: int
    target_seq_length: int
    hidden_embedding_size: int
    num_of_encoder_or_decoder_layers: int
    num_of_vocabulary: int
    output_embedding_size: int

    @property
    def encoder_layer_config(self) -> EncoderOrDecoderConfig:
        return EncoderOrDecoderConfig(
            EncoderOrDecoderType.ENCODER,
            self.source_seq_length,
            self.source_seq_length,
            self.hidden_embedding_size,
            self.num_of_encoder_or_decoder_layers,
        )

    @property
    def decoder_layer_config(self) -> EncoderOrDecoderConfig:
        return EncoderOrDecoderConfig(
            EncoderOrDecoderType.DECODER,
            self.target_seq_length,
            self.target_seq_length,
            self.hidden_embedding_size,
            self.num_of_encoder_or_decoder_layers,
        )

    @property
    def hybrid_layer_config(self) -> EncoderOrDecoderConfig:
        return EncoderOrDecoderConfig(
            EncoderOrDecoderType.HYBRID,
            self.source_seq_length,
            self.target_seq_length,
            self.hidden_embedding_size,
            self.num_of_encoder_or_decoder_layers,
        )


class Header(nn.Module):
    def __init__(self, hidden_embedding_size: int, output_embedding_size: int):
        super().__init__()
        self.mlp = nn.Linear(hidden_embedding_size, output_embedding_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.mlp(inputs)


class EncoderTransformer(nn.Module):
    def __init__(
        self,
        embeddings: nn.Embedding,
        positional_embeddings: SinusoidalPositionalEmbeddings,
        encoder_layers: EncoderOrDecoderLayers,
        header: nn.Module,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.positional_embeddings = positional_embeddings
        self.encoder_layers = encoder_layers
        self.header = header

    @classmethod
    def from_config(cls, config: TransformerConfig) -> "EncoderTransformer":
        embeddings_layer = nn.Embedding(
            config.num_of_vocabulary, config.hidden_embedding_size
        )
        positional_embeddings_layer = SinusoidalPositionalEmbeddings(
            config.source_seq_length, config.hidden_embedding_size
        )
        encoder_layers = EncoderOrDecoderLayers.from_config(config.encoder_layer_config)
        header = Header(config.hidden_embedding_size, config.output_embedding_size)
        return cls(
            embeddings_layer, positional_embeddings_layer, encoder_layers, header
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.embeddings(inputs)
        positional_embeddings = self.positional_embeddings()
        embeddings = embeddings + positional_embeddings
        embeddings = self.encoder_layers(embeddings)
        return self.header(embeddings)


class EncoderAndDecoderTransformer(nn.Module):
    def __init__(
        self,
        embeddings: nn.Embedding,
        positional_embeddings: SinusoidalPositionalEmbeddings,
        transformer_layers: EncoderAndDecoderLayers,
        header: nn.Module,
    ):
        super().__init__()
        self.embeddings = embeddings
        self.positional_embeddings = positional_embeddings
        self.transformer_layers = transformer_layers
        self.header = header

    @property
    def max_source_seq_size(self) -> int:
        return self.transformer_layers.max_encoder_input_length

    @property
    def max_target_seq_size(self) -> int:
        return self.transformer_layers.max_decoder_input_length

    @classmethod
    def from_config(cls, config: TransformerConfig) -> "EncoderAndDecoderTransformer":
        embedding_size = config.hidden_embedding_size
        embeddings_layer = nn.Embedding(config.num_of_vocabulary, embedding_size)
        positional_embeddings_layer = SinusoidalPositionalEmbeddings(
            config.target_seq_length,
            embedding_size,
        )
        transformer_layers = EncoderAndDecoderLayers.from_config(
            config.hybrid_layer_config
        )
        header = Header(embedding_size, config.output_embedding_size)
        return cls(
            embeddings_layer, positional_embeddings_layer, transformer_layers, header
        )

    def forward(
        self,
        target_seq: torch.Tensor,
        encoder_output_of_source_seq: torch.Tensor,
    ) -> torch.Tensor:
        target_seq_embeddings = self.embeddings(target_seq)
        target_seq_positional_embeddings = self.positional_embeddings()
        target_seq_embeddings = target_seq_embeddings + target_seq_positional_embeddings
        hidden_states = self.transformer_layers(
            target_seq_embeddings, encoder_output_of_source_seq
        )

        return self.header(hidden_states)
