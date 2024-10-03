from dataclasses import dataclass

import torch
from torch import nn

from tasks.common.models import EncoderTransformer
from tasks.common.models import TransformerConfig


@dataclass
class ZeroShotTextClsConfig:
    max_source_seq_length: int
    max_target_seq_length: int
    source_hidden_embedding_size: int
    target_hidden_embedding_size: int
    max_target_options: int
    num_of_shared_vocabularies: int
    num_of_encoder_layers: int


class ZeroShotTextClassifier(nn.Module):
    def __init__(
        self,
        source_encoder_transformer: EncoderTransformer,
        target_encoder_transformer: EncoderTransformer,
        mlp_source: nn.Linear,
        mlp_target: nn.Linear,
    ):
        super().__init__()
        self.source_encoder_transformer = source_encoder_transformer
        self.target_encoder_transformer = target_encoder_transformer
        self.mlp_source = mlp_source
        self.mlp_target = mlp_target

    @staticmethod
    def post_process_source_encoder_transformer_outputs(
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states[-1, :].unsqueeze(0)

    @staticmethod
    def post_process_target_encoder_transformer_outputs(
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return hidden_states.T

    @classmethod
    def from_config(cls, config: ZeroShotTextClsConfig) -> "ZeroShotTextClassifier":
        source_encoder_config = TransformerConfig(
            source_seq_length=config.max_source_seq_length,
            target_seq_length=config.max_source_seq_length,
            hidden_embedding_size=config.source_hidden_embedding_size,
            num_of_encoder_or_decoder_layers=config.num_of_encoder_layers,
            num_of_vocabulary=config.num_of_shared_vocabularies,
            output_embedding_size=config.source_hidden_embedding_size,
        )
        target_encoder_config = TransformerConfig(
            source_seq_length=config.max_target_seq_length,
            target_seq_length=config.max_target_seq_length,
            hidden_embedding_size=config.target_hidden_embedding_size,
            num_of_encoder_or_decoder_layers=config.num_of_encoder_layers,
            num_of_vocabulary=config.num_of_shared_vocabularies,
            output_embedding_size=config.target_hidden_embedding_size,
        )
        return cls(
            EncoderTransformer.from_config(source_encoder_config),
            EncoderTransformer.from_config(target_encoder_config),
            nn.Linear(
                config.source_hidden_embedding_size, config.target_hidden_embedding_size
            ),
            nn.Linear(config.max_target_seq_length, config.max_target_options),
        )

    def forward(
        self, source_seq: torch.Tensor, target_seq: torch.Tensor
    ) -> torch.Tensor:
        source_seq_hidden_states = self.source_encoder_transformer(source_seq)
        source_seq_hidden_states = self.post_process_source_encoder_transformer_outputs(
            source_seq_hidden_states
        )
        source_seq_hidden_states = self.mlp_source(source_seq_hidden_states)

        target_seq_hidden_states = self.target_encoder_transformer(target_seq)
        target_seq_hidden_states = self.post_process_target_encoder_transformer_outputs(
            target_seq_hidden_states
        )
        target_seq_hidden_states = self.mlp_target(target_seq_hidden_states)

        hidden_states = torch.einsum(
            "ab,bc->ac",
            source_seq_hidden_states,
            target_seq_hidden_states,
        )

        return nn.functional.softmax(hidden_states, dim=-1)
