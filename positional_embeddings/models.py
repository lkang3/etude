from typing import Tuple

import torch
import torch.nn as nn


def get_attention_scores_adjusted_by_relative_positions(
    positions_embeddings: torch.Tensor, query: torch.Tensor,
) -> torch.Tensor:
    """

    :param positions_embeddings: (query_size, key_size, embedding_size)
    :param query: (query_size, embedding_size)
    :return: (query_size, key_size)
    """
    # (embedding_size, query_size, key_size)
    rel_positions_embeddings = positions_embeddings.transpose(0, 2).transpose(1, 2)
    return torch.einsum("qd, dqk -> qk", query, rel_positions_embeddings)


def get_attention_scores_adjusted_by_relative_positions_bias(
    positions_embeddings: torch.Tensor,
    unscaled_attention_scores: torch.Tensor,
):
    return positions_embeddings + unscaled_attention_scores


class BasePositionalEmbedding(nn.Module):

    def __init__(self):
        super().__init__()
        self.position_embedding = self.get_embedding()

    def get_embedding(self) -> nn.Embedding:
        raise NotImplemented()

    def get_positions_embedding_idx(self) -> torch.Tensor:
        raise NotImplemented()

    def forward(self) -> torch.Tensor:
        positions = self.get_positions_embedding_idx()
        return self.position_embedding(positions)


class VanillaPositionalEmbedding(BasePositionalEmbedding):
    def __init__(self, max_sequence_length: int, embedding_size: int):
        self.max_sequence_length = max_sequence_length
        self.max_sequence_length = max_sequence_length
        self.embedding_size = embedding_size
        super().__init__()

    def get_embedding(self) -> nn.Embedding:
        return nn.Embedding(self.max_sequence_length, self.embedding_size)

    def get_positions_embedding_idx(self) -> torch.Tensor:
        return torch.arange(self.max_sequence_length)


class RelativePositionalEmbedding(BasePositionalEmbedding):
    def __init__(self, target_sequence_size: int, source_sequence_size: int, embedding_size: int):
        self.target_sequence_size = target_sequence_size
        self.source_sequence_size = source_sequence_size
        self.embedding_size = embedding_size
        super().__init__()

    def get_embedding(self) -> nn.Embedding:
        return nn.Embedding(
            self.target_sequence_size + self.source_sequence_size - 1,
            self.embedding_size,
        )

    @staticmethod
    def get_relative_positions(
        target_sequence_size: int, source_sequence_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param target_sequence_size:
        :param source_sequence_size:
        :return:
        raw relative positions:
        positive relative positions: Value is ready to use with the self.embeddings()
        """
        rel_positions_in_query_sequence = torch.arange(target_sequence_size)
        rel_positions_in_source_sequence = torch.arange(source_sequence_size)
        raw_rel_positions = (
            rel_positions_in_source_sequence.unsqueeze(0)
            - rel_positions_in_query_sequence.unsqueeze(-1)
        )
        positive_rel_positions = -torch.min(raw_rel_positions) + raw_rel_positions
        return raw_rel_positions, positive_rel_positions

    def get_positions_embedding_idx(self) -> torch.Tensor:
        _, rel_positions = self.get_relative_positions(
            self.target_sequence_size, self.source_sequence_size,
        )
        return rel_positions


class RelativeBiasPositionalEmbedding(RelativePositionalEmbedding):
    def __init__(
        self,
        target_sequence_size: int,
        source_sequence_size: int,
    ):
        self.target_sequence_size = target_sequence_size
        self.source_sequence_size = source_sequence_size
        self.embedding_size = 1
        super().__init__(self.target_sequence_size, self.source_sequence_size, self.embedding_size)

    def forward(self) -> torch.Tensor:
        outputs = super().forward()
        return outputs.squeeze(-1)
