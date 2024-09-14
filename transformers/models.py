from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import TypeVar

import torch
from torch import nn

from attentions.models import BaseAttention
from attentions.models import VanillaAttentionForDecoder
from attentions.models import VanillaAttentionForEncoder
from transformers.utils import create_drop_out_layer
from transformers.utils import create_mlp
from transformers.utils import create_norm_layer
from transformers.utils import residual_connection

TypeAttention = TypeVar("TypeAttention", bound=BaseAttention)


class TransformerType(Enum):
    DECODER = auto
    ENCODER = auto
    HYBRID = auto


@dataclass
class TransformerConfig:
    type: TransformerType
    source_seq_length: int
    target_seq_length: int
    embedding_size: int
    num_of_layers: int

    @property
    def max_seq_length(self) -> int:
        return max(self.source_seq_length, self.target_seq_length)


def create_attention_layer(
    max_seq_size: int,
    embedding_size: int,
    transformer_type: TransformerType,
) -> TypeAttention:
    if transformer_type == TransformerType.DECODER:
        return VanillaAttentionForDecoder(
            embedding_size=embedding_size,
            output_embedding_size=embedding_size,
            source_sequence_size=max_seq_size,
            query_sequence_size=max_seq_size,
        )
    if transformer_type == TransformerType.ENCODER:
        return VanillaAttentionForEncoder(
            embedding_size=embedding_size,
            output_embedding_size=embedding_size,
            source_sequence_size=max_seq_size,
            query_sequence_size=max_seq_size,
        )
    raise ValueError(
        "Only accept TransformerType.DECODER or TransformerType.ENCODER as transformer_type"
    )


class EncoderOrDecoderLayer(nn.Module):
    def __init__(
        self,
        attention: TypeAttention,
        mlp: nn.Module,
        norm_after_self_attention: nn.Module,
        norm_after_mlp: nn.Module,
        drop_out_after_self_attention: nn.Module,
        drop_out_after_mlp: nn.Module,
    ):
        super().__init__()
        self.attention = attention
        self.mlp = mlp
        self.norm_after_self_attention = norm_after_self_attention
        self.norm_after_mlp = norm_after_mlp
        self.drop_out_after_self_attention = drop_out_after_self_attention
        self.drop_out_after_mlp = drop_out_after_mlp

    @classmethod
    def from_config(cls, config: TransformerConfig) -> "EncoderOrDecoderLayer":
        attention = create_attention_layer(
            config.max_seq_length,
            config.embedding_size,
            config.type,
        )
        mlp = create_mlp(config.embedding_size, config.embedding_size)
        norm_after_self_attention = create_norm_layer(config.embedding_size)
        norm_after_mlp = create_norm_layer(config.embedding_size)
        drop_out_after_self_attention = create_drop_out_layer()
        drop_out_after_mlp = create_drop_out_layer()

        return cls(
            attention,
            mlp,
            norm_after_self_attention,
            norm_after_mlp,
            drop_out_after_self_attention,
            drop_out_after_mlp,
        )

    @classmethod
    def create_block(cls, config: TransformerConfig) -> nn.ModuleList:
        return nn.ModuleList([cls.from_config(config)] * config.num_of_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.attention(inputs, inputs, inputs)
        hidden_states = residual_connection(inputs, hidden_states)
        hidden_states = self.norm_after_self_attention(hidden_states)
        hidden_states = self.drop_out_after_self_attention(hidden_states)

        mlp_outputs = self.mlp(hidden_states)
        hidden_states = residual_connection(hidden_states, mlp_outputs)
        hidden_states = self.norm_after_mlp(hidden_states)
        hidden_states = self.drop_out_after_mlp(hidden_states)

        return hidden_states


class EncoderOrDecoderLayers(nn.Module):
    def __init__(
        self,
        max_seq_length: int,
        embedding_size: int,
        layers: nn.ModuleList,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding_size = embedding_size
        self.layers = layers

    @classmethod
    def from_config(cls, config: TransformerConfig) -> "EncoderOrDecoderLayers":
        return cls(
            config.max_seq_length,
            config.embedding_size,
            EncoderOrDecoderLayer.create_block(config),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states


class EncoderAndDecoderLayer(nn.Module):
    def __init__(
        self,
        decoder_self_attention: TypeAttention,
        decoder_encoder_attention: TypeAttention,
        mlp: nn.Module,
        norm_after_decoder_input_attention: nn.Module,
        norm_after_decoder_encoder_attention: nn.Module,
        norm_after_mlp: nn.Module,
        drop_out_after_decoder_input_attention: nn.Module,
        drop_out_after_decoder_encoder_attention: nn.Module,
        drop_out_after_mlp: nn.Module,
    ):
        super().__init__()
        self.decoder_self_attention = decoder_self_attention
        self.decoder_encoder_attention = decoder_encoder_attention
        self.mlp = mlp
        self.norm_after_decoder_input_attention = norm_after_decoder_input_attention
        self.norm_after_decoder_encoder_attention = norm_after_decoder_encoder_attention
        self.norm_after_mlp = norm_after_mlp
        self.drop_out_after_decoder_input_attention = (
            drop_out_after_decoder_input_attention
        )
        self.drop_out_after_decoder_encoder_attention = (
            drop_out_after_decoder_encoder_attention
        )
        self.drop_out_after_mlp = drop_out_after_mlp

    @classmethod
    def from_config(cls, config: TransformerConfig) -> "EncoderAndDecoderLayer":
        decoder_input_attention = VanillaAttentionForDecoder(
            embedding_size=config.embedding_size,
            output_embedding_size=config.embedding_size,
            source_sequence_size=config.target_seq_length,
            query_sequence_size=config.target_seq_length,
        )
        decoder_encoder_attention = VanillaAttentionForEncoder(
            embedding_size=config.embedding_size,
            output_embedding_size=config.embedding_size,
            source_sequence_size=config.source_seq_length,
            query_sequence_size=config.target_seq_length,
        )
        mlp = create_mlp(config.embedding_size, config.embedding_size)
        norm_after_decoder_input_attention = create_norm_layer(
            decoder_input_attention.embedding_size
        )
        norm_after_encoder_decoder_attention = create_norm_layer(
            decoder_encoder_attention.embedding_size
        )
        norm_after_mlp = create_norm_layer(config.embedding_size)

        drop_out_after_decoder_input_attention = create_drop_out_layer()
        drop_out_after_encoder_input_attention = create_drop_out_layer()
        drop_out_after_mlp = create_drop_out_layer()

        return cls(
            decoder_input_attention,
            decoder_encoder_attention,
            mlp,
            norm_after_decoder_input_attention,
            norm_after_encoder_decoder_attention,
            norm_after_mlp,
            drop_out_after_decoder_input_attention,
            drop_out_after_encoder_input_attention,
            drop_out_after_mlp,
        )

    @classmethod
    def create_block(cls, config: TransformerConfig) -> nn.ModuleList:
        return nn.ModuleList([cls.from_config(config)] * config.num_of_layers)

    def forward(
        self, decoder_inputs: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        data = self.decoder_self_attention(
            decoder_inputs, decoder_inputs, decoder_inputs
        )
        data = residual_connection(decoder_inputs, data)
        data = self.norm_after_decoder_input_attention(data)
        decoder_inputs = self.drop_out_after_decoder_input_attention(data)

        encoder_outputs = self.decoder_encoder_attention(
            encoder_outputs,
            encoder_outputs,
            decoder_inputs,
        )
        data = residual_connection(encoder_outputs, data)
        data = self.norm_after_decoder_encoder_attention(data)
        data = self.drop_out_after_decoder_encoder_attention(data)

        mlp_outputs = self.mlp(data)
        data = residual_connection(data, mlp_outputs)
        data = self.norm_after_mlp(data)
        data = self.drop_out_after_mlp(data)

        return data


class EncoderAndDecoderLayers(nn.Module):
    def __init__(
        self,
        max_seq_length: int,
        embedding_size: int,
        layers: nn.ModuleList,
    ):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding_size = embedding_size
        self.layers = layers

    @classmethod
    def from_config(cls, config: TransformerConfig) -> "EncoderAndDecoderLayers":
        return cls(
            config.max_seq_length,
            config.embedding_size,
            EncoderAndDecoderLayer.create_block(config),
        )

    def forward(
        self, decoder_inputs: torch.Tensor, encoder_outputs: torch.Tensor
    ) -> torch.Tensor:
        embeddings = decoder_inputs
        for layer in self.layers:
            embeddings = layer(embeddings, encoder_outputs)

        return embeddings
