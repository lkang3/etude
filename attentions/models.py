from typing import List

import numpy as np
import torch
from torch import nn

from attentions.entity import AttentionSchema
from attentions.entity import AttentionScores
from attentions.enum import AttentionDirection


class BaseAttention(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        output_embedding_size: int,
        source_sequence_size: int,
        query_sequence_size: int,
        add_bias: bool = True,
        attention_schema: AttentionSchema = AttentionSchema(AttentionDirection.BOTH),
    ):
        super().__init__()
        self.key_weights = nn.Linear(
            embedding_size, output_embedding_size, bias=add_bias
        )
        self.query_weights = nn.Linear(
            embedding_size, output_embedding_size, bias=add_bias
        )
        self.value_weights = nn.Linear(
            embedding_size, output_embedding_size, bias=add_bias
        )
        self.embedding_size = embedding_size
        self.output_embedding_size = output_embedding_size
        self.source_sequence_size = source_sequence_size
        self.query_sequence_size = query_sequence_size
        self.attention_schema = attention_schema

    def validate_inputs(
        self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor
    ):
        assert key.shape[0] == value.shape[0] == self.source_sequence_size, (
            key.shape,
            value.shape,
            self.source_sequence_size,
        )
        assert query.shape[0] == self.query_sequence_size, query.shape
        assert (
            key.shape[-1] == value.shape[-1] == query.shape[-1] == self.embedding_size
        ), (key.shape, value.shape, query.shape, self.embedding_size)

    def calculate_attention_scores(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
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

    def forward(
        self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
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
        output_embedding_size: int,
        source_sequence_size: int,
        query_sequence_size: int,
        add_bias: bool = True,
    ):
        super().__init__(
            embedding_size,
            output_embedding_size,
            source_sequence_size,
            query_sequence_size,
            add_bias,
            attention_schema=AttentionSchema(AttentionDirection.BOTH),
        )

    def calculate_attention_scores(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
    ) -> List[AttentionScores]:
        raw_scores = query @ key.T
        scores = raw_scores * self.embedding_size**-0.5
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
        output_embedding_size: int,
        source_sequence_size: int,
        query_sequence_size: int,
        add_bias: bool = True,
    ):
        super().__init__(
            embedding_size,
            output_embedding_size,
            source_sequence_size,
            query_sequence_size,
            add_bias,
            attention_schema=AttentionSchema(AttentionDirection.LEFT),
        )

    def calculate_attention_scores(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
    ) -> List[AttentionScores]:
        raw_scores = query @ key.T
        scores = raw_scores * self.embedding_size**-0.5
        scores = torch.softmax(scores, dim=-1)
        attention_mask = self.attention_schema.create_attention_mask_based_on_direction(
            self.query_sequence_size,
            self.source_sequence_size,
        )
        scores = torch.multiply(scores, attention_mask)
        query_token_ids, key_token_ids = torch.where(torch.isfinite(scores))

        query_sequence_size, _ = scores.shape
        return [
            AttentionScores(
                query_seq_token_id=query_token_id,
                source_seq_token_ids=key_token_ids[
                    query_token_ids == query_token_id
                ].tolist(),
                scores=scores[
                    query_token_id, key_token_ids[query_token_ids == query_token_id]
                ],
            )
            for query_token_id in range(query_sequence_size)
        ]


class SlideWindowAttention(BaseAttention):
    def __init__(
        self,
        embedding_size: int,
        output_embedding_size: int,
        source_sequence_size: int,
        query_sequence_size: int,
        window_size: int,
        add_bias: bool = True,
        attention_schema: AttentionSchema = AttentionSchema(AttentionDirection.BOTH),
    ):
        super().__init__(
            embedding_size,
            output_embedding_size,
            source_sequence_size,
            query_sequence_size,
            add_bias,
            attention_schema,
        )
        self.window_size = window_size

    def calculate_attention_scores(
        self,
        key: torch.Tensor,
        query: torch.Tensor,
    ) -> List[AttentionScores]:

        attention_scores = []
        attention_mask = self.attention_schema.create_sliding_window_mask(
            self.query_sequence_size,
            self.source_sequence_size,
            self.window_size,
        )
        for token_idx in range(self.query_sequence_size):
            key_token_ids = torch.where(~torch.isnan(attention_mask[token_idx, :]))[0]
            chunk_start_idx = key_token_ids[0]
            chunk_end_idx = key_token_ids[-1]
            scores = query[token_idx, :] @ key[chunk_start_idx : chunk_end_idx + 1, :].T
            scores = scores * self.embedding_size**-0.5
            scores = torch.softmax(scores, dim=-1)

            attention_scores.append(
                AttentionScores(token_idx, list(key_token_ids), scores)
            )

        return attention_scores


class MultipleHeadAttention(nn.Module):
    def __init__(self, heads: List[BaseAttention]):
        super().__init__()
        self.heads = heads
        linear_layer_size = len(self.heads) * heads[0].output_embedding_size
        self.linear = nn.Linear(linear_layer_size, linear_layer_size)
        self.dropout = nn.Dropout()

    def forward(
        self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat([head(key, value, query) for head in self.heads], dim=-1)
        out = self.dropout(self.linear(inputs))
        return out
