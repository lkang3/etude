from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn


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


@dataclass
class AttentionScores:
    target_seq_token_id: int
    source_seq_token_ids: List[int]
    scores: torch.Tensor

    def __eq__(self, other: "AttentionScores") -> bool:
        return (
            self.target_seq_token_id == other.target_seq_token_id
            and self.source_seq_token_ids == other.source_seq_token_ids
            and torch.all(torch.isclose(self.scores, other.scores, rtol=1e-03))
        )

    def __hash__(self) -> int:
        return hash((self.target_seq_token_id, str(self.source_seq_token_ids), self.scores))

    def apply(self, source_sequence: torch.Tensor) -> torch.Tensor:
        pass


"""
SWA
window desing
    - Sliding window
    - Dilated window
    - Global window
    - dropout
encoder or decoder (masking upper triangular part of attention score matrix)

"""


def calculate_slide_window_attention_scores(
    k: torch.Tensor,
    q: torch.Tensor,
    window_size: int,
) -> List[AttentionScores]:
    assert k.shape[-1] == q.shape[-1]
    embedding_size = k.shape[-1]
    source_token_size = k.shape[-2]
    target_token_size = q.shape[-2]

    attention_scores = []
    for token_idx in range(target_token_size):
        chunk_start_idx = token_idx - window_size if token_idx > window_size else min(0, token_idx)
        chunk_end_idx = (
            token_idx + window_size + 1
            if token_idx + window_size + 1 < source_token_size
            else min(source_token_size, token_idx + window_size + 1)
        )
        scores = q[token_idx, :] @ k[chunk_start_idx: chunk_end_idx, :].T
        scores = scores * embedding_size ** -0.5
        scores = torch.softmax(scores, dim=-1)
        attention_scores.append(
            AttentionScores(
                token_idx,
                list(range(chunk_start_idx, chunk_start_idx + len(scores))),
                scores,
            )
        )

    return attention_scores


class SlideWindowAttention(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        weight_embedding_size: int,
        source_token_size: int,
        target_token_size: int,
        window_size: int,
    ):
        super().__init__()
        self.key_weights = nn.Linear(embedding_size, weight_embedding_size, bias=False)
        self.query_weights = nn.Linear(embedding_size, weight_embedding_size, bias=False)
        self.value_weights = nn.Linear(embedding_size, weight_embedding_size, bias=False)
        self.embedding_size = embedding_size,
        self.weight_embedding_size = weight_embedding_size
        self.source_token_size = source_token_size
        self.target_token_size = target_token_size
        self.window_size = window_size

    def _validate_inputs(self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor):
        assert (
            key.shape[0] == value.shape[0] == self.source_token_size,
            (key.shape, value.shape, query.shape)
        )
        assert query.shape[0] == self.target_token_size, query.shape
        assert (
            key.shape[-1] ==  value.shape[-1] == query.shape[-1] == self.embedding_size,
            (key.shape, value.shape, query.shape)
        )

    def forward(self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(key, value, query)
        key = self.key_weights(key)
        value = self.value_weights(value)
        query = self.query_weights(query)

        attention_scores = calculate_slide_window_attention_scores(key, query, self.window_size)

        outputs = torch.zeros((self.target_token_size, self.weight_embedding_size))
        for attention_score in attention_scores:
            query_token_id = attention_score.target_seq_token_id
            value_token_ids = attention_score.source_seq_token_ids
            scores = attention_score.scores
            outputs[query_token_id, :] = torch.mean(
                scores.unsqueeze(-1) * value[value_token_ids,:],
                dim=0,
            )

        return outputs