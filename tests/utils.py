from dataclasses import fields
from unittest.mock import Mock

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


def create_multi_channel_image_data(num_of_channels: int, height: int, width: int) -> torch.Tensor:
    data = []
    for i in range(1, num_of_channels+1):
        image_layer = torch.arange(height * width).view(height, width) * i
        data.append(image_layer)

    return torch.stack(data).float()


def create_dataclass_mock(data_class_object: object) -> Mock():
    data_class_fields = [field for field in fields(data_class_object)]
    data_class_field_names = {field.name for field in data_class_fields}
    data_class_attributes = [attr for attr in dir(data_class_object) if attr not in data_class_field_names]
    spec = data_class_fields
    spec.extend(data_class_attributes)
    return Mock(sepc=spec)
