from dataclasses import dataclass
from typing import List

import torch

from attentions.enum import AttentionDirection


@dataclass
class AttentionSchema:
    direction: AttentionDirection

    def create_attention_mask_based_on_direction(
        self,
        query_seq_size: int,
        source_seq_size: int,
    ) -> torch.Tensor:
        mask = torch.ones((query_seq_size, source_seq_size))
        if self.direction == AttentionDirection.LEFT:
            mask = torch.tril(mask)
            return torch.masked_fill(mask, mask == 0, torch.nan)
        if self.direction == AttentionDirection.RIGHT:
            mask = torch.triu(mask)
            return torch.masked_fill(mask, mask == 0, torch.nan)
        return mask

    def create_sliding_window_mask(
        self,
        query_seq_size: int,
        source_seq_size: int,
        window_size: int,
    ) -> torch.Tensor:
        attention_mask = torch.zeros(query_seq_size, source_seq_size)
        for query_token_idx in range(query_seq_size):
            chunk_start_idx = (
                query_token_idx - window_size
                if query_token_idx > window_size
                else min(0, query_token_idx)
            )
            chunk_end_idx = (
                query_token_idx + window_size + 1
                if query_token_idx + window_size + 1 < source_seq_size
                else min(source_seq_size, query_token_idx + window_size + 1)
            )
            attention_mask[query_token_idx, chunk_start_idx:chunk_end_idx] = 1
        attention_mask = torch.masked_fill(
            attention_mask, attention_mask == 0, torch.nan
        )
        if self.direction != AttentionDirection.BOTH:
            mask_with_direction = self.create_attention_mask_based_on_direction(
                query_seq_size,
                source_seq_size,
            )
            attention_mask = torch.multiply(attention_mask, mask_with_direction)

        return attention_mask


@dataclass
class AttentionScores:
    query_seq_token_id: int
    source_seq_token_ids: List[int]
    scores: torch.Tensor

    def __eq__(self, other: "AttentionScores") -> bool:  # type: ignore
        return (
            self.query_seq_token_id == other.query_seq_token_id
            and self.source_seq_token_ids == other.source_seq_token_ids
            and torch.all(torch.isclose(self.scores, other.scores, rtol=1e-03))
        )

    def __hash__(self) -> int:
        return hash(
            (self.query_seq_token_id, str(self.source_seq_token_ids), self.scores)
        )

    def apply(self, source_token_sequences: torch.Tensor) -> torch.Tensor:
        """
        :param source_token_sequences: [[num_of_tokens, embedding_size]]
        :return:

        Example:
        scores = torch.Tensor([0.1, 0.2, 0.7])
        token_sequences = torch.ones(3, 16)
        scores.unsequeeze(-1) * token_sequences
        """
        return torch.sum(
            self.scores.unsqueeze(-1)
            * source_token_sequences[self.source_seq_token_ids, :],
            dim=0,
        )
