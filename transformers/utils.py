import torch
from torch import nn as nn


def create_norm_layer(embedding_size: int) -> nn.LayerNorm:
    return nn.LayerNorm(embedding_size)


def create_mlp(input_embedding_size: int, output_embedding_size: int) -> nn.Linear:
    return nn.Linear(input_embedding_size, output_embedding_size)


def create_drop_out_layer(drop_out_prob: float=0.1) -> nn.Dropout:
    return nn.Dropout(drop_out_prob)


def residual_connection(input_one: torch.Tensor, input_two: torch.Tensor) -> torch.Tensor:
    return input_one + input_two
