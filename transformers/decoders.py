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


def create_decoder_self_attention_layer(
    decoder_input_seq_length: int,
    embedding_size: int,
) -> VanillaAttentionForDecoder:
    return VanillaAttentionForDecoder(
        embedding_size=embedding_size,
        output_embedding_size=embedding_size,
        source_sequence_size=decoder_input_seq_length,
        query_sequence_size=decoder_input_seq_length,
    )


def create_decoder_encoder_attention_layer(
    encoder_input_seq_length: int,
    decoder_input_seq_length: int,
    embedding_size: int,
) -> VanillaAttentionForEncoder:
    return VanillaAttentionForEncoder(
        embedding_size=embedding_size,
        output_embedding_size=embedding_size,
        source_sequence_size=encoder_input_seq_length,
        query_sequence_size=decoder_input_seq_length,
    )


class Decoder(nn.Module):
    def __init__(
        self,
        decoder_self_attention: TypeAttention,
        decoder_encoder_attention: TypeAttention,
        mlp: nn.Module,
        norm_after_decoder_input_attention: nn.Module,
        norm_after_encoder_input_attention: nn.Module,
        norm_after_mlp: nn.Module,
        drop_out_after_decoder_input_attention: nn.Module,
        drop_out_after_encoder_input_attention: nn.Module,
        drop_out_after_mlp: nn.Module,
    ):
        super().__init__()
        self.decoder_self_attention = decoder_self_attention
        self.decoder_encoder_attention = decoder_encoder_attention
        self.mlp = mlp
        self.norm_after_decoder_input_attention = norm_after_decoder_input_attention
        self.norm_after_encoder_input_attention = norm_after_encoder_input_attention
        self.norm_after_mlp = norm_after_mlp
        self.drop_out_after_decoder_input_attention = (
            drop_out_after_decoder_input_attention
        )
        self.drop_out_after_encoder_input_attention = (
            drop_out_after_encoder_input_attention
        )
        self.drop_out_after_mlp = drop_out_after_mlp

    @classmethod
    def create_from_config(
        cls,
        source_seq_length: int,
        query_seq_length: int,
        embedding_size: int,
    ) -> "Decoder":
        decoder_input_attention = create_decoder_self_attention_layer(
            query_seq_length,
            embedding_size,
        )
        decoder_encoder_attention = create_decoder_encoder_attention_layer(
            source_seq_length,
            query_seq_length,
            embedding_size,
        )
        mlp = create_mlp(embedding_size, embedding_size)
        norm_after_decoder_input_attention = create_norm_layer(
            decoder_input_attention.embedding_size
        )
        norm_after_encoder_input_attention = create_norm_layer(
            decoder_encoder_attention.embedding_size
        )
        norm_after_mlp = create_norm_layer(embedding_size)

        drop_out_after_decoder_input_attention = create_drop_out_layer()
        drop_out_after_encoder_input_attention = create_drop_out_layer()
        drop_out_after_mlp = create_drop_out_layer()

        return cls(
            decoder_input_attention,
            decoder_encoder_attention,
            mlp,
            norm_after_decoder_input_attention,
            norm_after_encoder_input_attention,
            norm_after_mlp,
            drop_out_after_decoder_input_attention,
            drop_out_after_encoder_input_attention,
            drop_out_after_mlp,
        )

    def forward(
        self, decoder_inputs: torch.Tensor, encoder_inputs: torch.Tensor
    ) -> torch.Tensor:
        data = self.decoder_self_attention(
            decoder_inputs, decoder_inputs, decoder_inputs
        )
        data = residual_connection(decoder_inputs, data)
        data = self.norm_after_decoder_input_attention(data)
        decoder_inputs = self.drop_out_after_decoder_input_attention(data)

        encoder_inputs = self.decoder_encoder_attention(
            encoder_inputs,
            encoder_inputs,
            decoder_inputs,
        )
        data = residual_connection(encoder_inputs, data)
        data = self.norm_after_encoder_input_attention(data)
        data = self.drop_out_after_encoder_input_attention(data)

        mlp_outputs = self.mlp(data)
        data = residual_connection(data, mlp_outputs)
        data = self.norm_after_mlp(data)
        data = self.drop_out_after_mlp(data)

        return data
