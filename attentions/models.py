from typing import List

import torch
import numpy as np
import torch.nn as nn

from attentions.entity import AttentionSchema
from attentions.entity import AttentionScores
from attentions.enum import AttentionDirection


class Head(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        head_size: int,
        dropout: float = 0.01,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.key_weights = nn.Linear(embedding_size, head_size, bias=False)
        self.query_weights = nn.Linear(embedding_size, head_size, bias=False)
        self.value_weights = nn.Linear(embedding_size, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def scale_attention_scores(self, attention_scores: torch.Tensor) -> torch.Tensor:
        return attention_scores * self.embedding_size ** -0.5

    @staticmethod
    def mask_attention_scores(attention_scores: torch.Tensor) -> torch.Tensor:
        mask = torch.tril(torch.ones_like(attention_scores))
        return torch.masked_fill(attention_scores, mask, float("-inf"))


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        k = self.key_weights(inputs)
        q = self.query_weights(inputs)
        v = self.value_weights(inputs)
        attention_scores = q @ k.transpose(-2, -1)
        attention_scores = self.scale_attention_scores(attention_scores)
        attention_scores = self.mask_attention_scores(attention_scores)
        attention_scores = nn.functional.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        outputs = attention_scores @ v
        return outputs


class MulitHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        head_size: int,
        dropout: float = 0.01,
    ):
        super().__init__()
        num_of_heads = embedding_size / head_size
        self.heads = nn.ModuleList([Head(embedding_size, head_size) for _ in range(num_of_heads)])
        self.linear = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs =  torch.cat([head(inputs) for head in self.heads], dim=-1)
        out = self.dropout(self.linear(inputs))
        return out


def chunk(hidden_states, window_overlap):
    """convert into overlapping chunks. Chunk size = 2w, overlap = w"""
    chunk_size = [
        hidden_states.size(0), #bs
        torch.div(hidden_states.size(1), window_overlap, rounding_mode="trunc") - 1, #n_chunks
        window_overlap * 2,
        hidden_states.size(2),
    ]

    overlapping_chunks = torch.empty(chunk_size, device=hidden_states.device)
    for chunk in range(chunk_size[1]):
        overlapping_chunks[:, chunk, :, :] = hidden_states[
            :, chunk * window_overlap : chunk * window_overlap + 2 * window_overlap, :
        ]
    return overlapping_chunks


class BaseAttention(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        weight_embedding_size: int,
        source_sequence_size: int,
        query_sequence_size: int,
        add_bias: bool = True,
        attention_schema: AttentionSchema = AttentionSchema(AttentionDirection.BOTH),
    ):
        super().__init__()
        self.key_weights = nn.Linear(embedding_size, weight_embedding_size, bias=add_bias)
        self.query_weights = nn.Linear(embedding_size, weight_embedding_size, bias=add_bias)
        self.value_weights = nn.Linear(embedding_size, weight_embedding_size, bias=add_bias)
        self.embedding_size = embedding_size
        self.weight_embedding_size = weight_embedding_size
        self.source_sequence_size = source_sequence_size
        self.query_sequence_size = query_sequence_size
        self.attention_schema = attention_schema

    def validate_inputs(self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor):
        assert (
            key.shape[0] == value.shape[0] == self.source_sequence_size,
            (key.shape, value.shape, query.shape)
        )
        assert query.shape[0] == self.query_sequence_size, query.shape
        assert (
            key.shape[-1] ==  value.shape[-1] == query.shape[-1] == self.embedding_size,
            (key.shape, value.shape, query.shape)
        )

    def calculate_attention_scores(
        self, key: torch.Tensor, query: torch.Tensor,
    ) -> List[AttentionScores]:
        raise NotImplementedError()

    @staticmethod
    def apply_attention_scores(
        attention_scores: List[AttentionScores],
        value: torch.Tensor,
    ) -> torch.Tensor:
        query_sequence_size = len(attention_scores)
        embedding_size = value.shape[-1]
        outputs = torch.zeros((query_sequence_size, embedding_size))
        for attention_score in attention_scores:
            query_token_id = attention_score.query_seq_token_id
            outputs[query_token_id, :] = attention_score.apply(value)

        return outputs

    def forward(self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        self.validate_inputs(key, value, query)
        key = self.key_weights(key)
        value = self.value_weights(value)
        query = self.query_weights(query)

        attention_scores = self.calculate_attention_scores(key, query)
        return self.apply_attention_scores(attention_scores, value)


class VanillaAttentionForEncoder(BaseAttention):
    def __init__(
        self,
        embedding_size: int,
        weight_embedding_size: int,
        source_sequence_size: int,
        query_sequence_size: int,
        add_bias: bool = True,
    ):
        super().__init__(
            embedding_size,
            weight_embedding_size,
            source_sequence_size,
            query_sequence_size,
            add_bias,
            attention_schema=AttentionSchema(AttentionDirection.BOTH),
        )

    def calculate_attention_scores(
        self, key: torch.Tensor, query: torch.Tensor,
    ) -> List[AttentionScores]:
        raw_scores = query @ key.T
        scores = raw_scores * self.embedding_size ** -0.5
        scores = torch.softmax(scores, dim=-1)

        query_sequence_size, key_sequence_size = scores.shape
        return [
            AttentionScores(
                query_seq_token_id=query_token_id,
                source_seq_token_ids=list(np.arange(key_sequence_size)),
                scores=scores[query_token_id, :],
            )
            for query_token_id in range(query_sequence_size)
        ]


class VanillaAttentionForDecoder(BaseAttention):
    def __init__(
        self,
        embedding_size: int,
        weight_embedding_size: int,
        source_sequence_size: int,
        query_sequence_size: int,
        add_bias: bool = True,
    ):
        super().__init__(
            embedding_size,
            weight_embedding_size,
            source_sequence_size,
            query_sequence_size,
            add_bias,
            attention_schema=AttentionSchema(AttentionDirection.LEFT),
        )

    def calculate_attention_scores(
        self, key: torch.Tensor, query: torch.Tensor,
    ) -> List[AttentionScores]:
        raw_scores = query @ key.T
        scores = raw_scores * self.embedding_size ** -0.5
        scores = torch.softmax(scores, dim=-1)
        attention_mask = self.attention_schema.create_attention_mask_based_on_direction(
            self.query_sequence_size, self.source_sequence_size,
        )
        scores = torch.multiply(scores, attention_mask)
        _, key_token_ids = torch.where(torch.isfinite(scores))

        query_sequence_size, key_sequence_size = scores.shape
        return [
            AttentionScores(
                query_seq_token_id=query_token_id,
                source_seq_token_ids=key_token_ids[query_token_id].tolist(),
                scores=scores[query_token_id, key_token_ids],
            )
            for query_token_id in range(query_sequence_size)
        ]


class SlideWindowAttention(BaseAttention):
    def __init__(
        self,
        embedding_size: int,
        weight_embedding_size: int,
        source_sequence_size: int,
        query_sequence_size: int,
        window_size: int,
        add_bias: bool = True,
        attention_schema: AttentionSchema = AttentionSchema(AttentionDirection.BOTH),
    ):
        super().__init__(
            embedding_size,
            weight_embedding_size,
            source_sequence_size,
            query_sequence_size,
            add_bias,
            attention_schema,
        )
        self.window_size = window_size

    def calculate_attention_scores(
        self, key: torch.Tensor, query: torch.Tensor,
    ) -> List[AttentionScores]:

        attention_scores = []
        attention_mask = self.attention_schema.create_sliding_window_mask(
            self.query_sequence_size, self.source_sequence_size, self.window_size,
        )
        for token_idx in range(self.query_sequence_size):
            key_token_ids = torch.where(~torch.isnan(attention_mask[token_idx, :]))[0]
            chunk_start_idx = key_token_ids[0]
            chunk_end_idx = key_token_ids[-1]
            scores = query[token_idx, :] @ key[chunk_start_idx: chunk_end_idx + 1, :].T
            scores = scores * self.embedding_size ** -0.5
            scores = torch.softmax(scores, dim=-1)

            attention_scores.append(AttentionScores(token_idx, list(key_token_ids), scores))

        return attention_scores
