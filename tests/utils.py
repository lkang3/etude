import torch


def is_tensor_equal(tensor_one: torch.Tensor, tensor_two: torch.Tensor) -> bool:
    return torch.all(torch.isclose(tensor_one, tensor_two, equal_nan=True))
