import torch


def padding_sequence(
    input_sequence: torch.Tensor,
    value_to_pad: int,
    max_sequence_size: int,
) -> torch.Tensor:
    input_sequence_size = len(input_sequence)
    if max_sequence_size == input_sequence_size:
        return input_sequence
    if max_sequence_size < input_sequence_size:
        msg = (
            f"input_sequence length: {input_sequence_size} > "
            f"max_sequence_size: {max_sequence_size}"
        )
        raise ValueError(msg)

    length_to_pad = max_sequence_size - input_sequence_size
    return torch.cat((input_sequence, torch.full((length_to_pad,), value_to_pad)))
