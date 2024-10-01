from dataclasses import dataclass

import torch
from torch import nn

from tasks.common.models import EncoderAndDecoderTransformer
from tasks.common.models import EncoderTransformer
from tasks.common.models import TransformerConfig


@dataclass
class Seq2SeqConfig:
    source_seq_length: int
    target_seq_length: int
    embedding_size: int
    source_seq_num_of_vocabulary: int
    target_seq_num_of_vocabulary: int
    num_of_encoders: int
    num_of_decoders: int


class NextToken(nn.Module):
    def __init__(
        self,
        encoder_transformer: EncoderTransformer,
        hybrid_transformer: EncoderAndDecoderTransformer,
    ):
        super().__init__()
        self.encoder_transformer = encoder_transformer
        self.hybrid_transformer = hybrid_transformer

    @classmethod
    def from_config(cls, config: Seq2SeqConfig) -> "NextToken":
        encoder_config = TransformerConfig(
            target_seq_length=config.source_seq_length,
            source_seq_length=config.source_seq_length,
            hidden_embedding_size=config.embedding_size,
            output_embedding_size=config.embedding_size,
            num_of_vocabulary=config.source_seq_num_of_vocabulary,
            num_of_encoder_or_decoder_layers=config.num_of_encoders,
        )
        encoder_and_decoder_config = TransformerConfig(
            target_seq_length=config.target_seq_length,
            source_seq_length=config.source_seq_length,
            hidden_embedding_size=config.embedding_size,
            output_embedding_size=config.target_seq_num_of_vocabulary,
            num_of_vocabulary=config.target_seq_num_of_vocabulary,
            num_of_encoder_or_decoder_layers=config.num_of_decoders,
        )
        return cls(
            EncoderTransformer.from_config(encoder_config),
            EncoderAndDecoderTransformer.from_config(encoder_and_decoder_config),
        )

    @staticmethod
    def aggregate_hidden_states(hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states[-1, :]

    def forward(
        self, target_seq: torch.Tensor, source_seq: torch.Tensor
    ) -> torch.Tensor:
        encoder_output_of_target_seq = self.encoder_transformer(source_seq)
        hidden_states = self.hybrid_transformer(
            target_seq, encoder_output_of_target_seq
        )
        hidden_states = self.aggregate_hidden_states(hidden_states)
        outputs = nn.functional.softmax(hidden_states, dim=-1)
        return outputs.unsqueeze(0)
