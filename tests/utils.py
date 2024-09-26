import torch


def is_tensor_equal(tensor_one: torch.Tensor, tensor_two: torch.Tensor) -> bool:
    return torch.all(torch.isclose(tensor_one, tensor_two, equal_nan=True))


def is_normalized(data: torch.Tensor) -> bool:
    return torch.all(torch.isclose(torch.sum(data, -1), torch.tensor(1.0))).item()


def create_sentence_with_tokens(
    num_of_vocabulary: int, max_seq_length: int
) -> torch.Tensor:
    return torch.randint(0, num_of_vocabulary - 1, (max_seq_length,))
