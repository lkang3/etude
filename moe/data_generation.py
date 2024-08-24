from typing import Tuple

import torch


torch.manual_seed(123)


def multiclass_data(
    num_per_class: int,
    embedding_size: int,
    num_of_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_of_samples = num_per_class * num_of_classes
    x = torch.rand(num_of_samples , embedding_size)
    y = torch.hstack(
        [torch.full((num_per_class,), class_label) for class_label in range(num_of_classes)]
    )
    for class_label in range(num_of_classes):
        mask = y==class_label
        x[mask] = x[mask] - class_label
    shuffled_idx = torch.randperm(num_of_samples)
    return x[shuffled_idx], y[shuffled_idx]
