import torch


def is_scalar(data: torch.Tensor) -> bool:
    return data.numel() == 1 and not data.shape


def is_tensor_equal(tensor_one: torch.Tensor, tensor_two: torch.Tensor) -> bool:
    return torch.all(torch.isclose(tensor_one, tensor_two, equal_nan=True)).item()


def is_normalized(data: torch.Tensor) -> bool:
    return torch.all(torch.isclose(torch.sum(data, -1), torch.tensor(1.0))).item()


def is_sentence_token_valid(sentence: torch.Tensor, valid_tokens: torch.Tensor) -> bool:
    return all(token in valid_tokens for token in sentence)


def create_sentence_with_tokens(
    num_of_vocabulary: int, max_seq_length: int
) -> torch.Tensor:
    return torch.randint(0, num_of_vocabulary - 1, (max_seq_length,))
