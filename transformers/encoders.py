from typing import TypeVar

import torch
from torch import nn

from attentions.models import BaseAttention
from attentions.models import VanillaAttentionForEncoder
from transformers.utils import create_drop_out_layer
from transformers.utils import create_mlp
from transformers.utils import create_norm_layer
from transformers.utils import residual_connection

TypeAttention = TypeVar("TypeAttention", bound=BaseAttention)


def create_attention_layer(
    source_seq_length: int,
    query_seq_length: int,
    embedding_size: int,
) -> VanillaAttentionForEncoder:
    return VanillaAttentionForEncoder(
        embedding_size=embedding_size,
        output_embedding_size=embedding_size,
        source_sequence_size=source_seq_length,
        query_sequence_size=query_seq_length,
    )


class Encoder(nn.Module):
    def __init__(
        self,
        attention: TypeAttention,
        mlp: nn.Module,
        norm_after_attention: nn.Module,
        norm_after_mlp: nn.Module,
        drop_out_after_attention: nn.Module,
        drop_out_after_mlp: nn.Module,
    ):
        super().__init__()
        self.attention = attention
        self.mlp = mlp
        self.norm_after_attention = norm_after_attention
        self.norm_after_mlp = norm_after_mlp
        self.drop_out_after_attention = drop_out_after_attention
        self.drop_out_after_mlp = drop_out_after_mlp

    @classmethod
    def create_from_config(
        cls,
        seq_length: int,
        embedding_size: int,
    ) -> "Encoder":
        attention = create_attention_layer(seq_length, seq_length, embedding_size)
        mlp = create_mlp(embedding_size, embedding_size)
        norm_after_attention = create_norm_layer(attention.embedding_size)
        norm_after_mlp = create_norm_layer(embedding_size)
        drop_out_after_attention = create_drop_out_layer()
        drop_out_after_mlp = create_drop_out_layer()
        return cls(
            attention,
            mlp,
            norm_after_attention,
            norm_after_mlp,
            drop_out_after_attention,
            drop_out_after_mlp,
        )

    @classmethod
    def create_encoder_block(
        cls,
        seq_length: int,
        embedding_size: int,
        num_of_encoder: int,
    ) -> nn.ModuleList:
        return nn.ModuleList(
            [cls.create_from_config(seq_length, embedding_size)] * num_of_encoder
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        data = self.attention(inputs, inputs, inputs)
        data = residual_connection(inputs, data)
        data = self.norm_after_attention(data)
        data = self.drop_out_after_attention(data)

        mlp_outputs = self.mlp(data)
        data = residual_connection(data, mlp_outputs)
        data = self.norm_after_mlp(data)
        data = self.drop_out_after_mlp(data)

        return data


class EncoderBlock(nn.Module):
    def __init__(self, max_seq_length: int, embedding_size: int, num_of_encoders: int):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.embedding_size = embedding_size
        self.num_of_encoders = num_of_encoders
        self.encoders = Encoder.create_encoder_block(
            max_seq_length,
            embedding_size,
            num_of_encoders,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = inputs
        for encoder in self.encoders:
            embeddings = encoder(inputs)

        return embeddings
