from typing import List

import torch
from torch import nn

from tasks.causal_language_modeling.models import NextToken


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


class NextTokenTrainer:
    def __init__(self, model: NextToken, max_output_sequence_length: int):
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()
        if max_output_sequence_length >= self.model.max_sequence_length:
            msg = (
                f"max_output_sequence_length: {max_output_sequence_length} >= "
                f"self.model.max_sequence_length {self.model.max_sequence_length}"
            )
            raise ValueError(msg)
        self.max_sequence_length = self.model.max_sequence_length
        self.max_output_sequence_length = max_output_sequence_length

    @staticmethod
    def is_end_token(token: int) -> bool:
        return token == 999

    @staticmethod
    def reach_end_of_sequence(
        current_position_in_sequence: int, sequence_length: int
    ) -> bool:
        return current_position_in_sequence == sequence_length

    @staticmethod
    def stop_iteration(
        current_position_in_sequence: int,
        token_to_evaluate: int,
        sequence_length: int,
    ) -> bool:
        return NextTokenTrainer.is_end_token(
            token_to_evaluate
        ) or NextTokenTrainer.reach_end_of_sequence(
            current_position_in_sequence, sequence_length
        )

    @staticmethod
    def prepare_model_inputs(
        input_sentence: torch.Tensor,
        predicted_tokens: torch.Tensor,
        max_sequence_length: int,
    ) -> torch.Tensor:
        model_inputs = torch.cat((input_sentence, predicted_tokens))
        return padding_sequence(model_inputs, 0, max_sequence_length)

    def train(
        self,
        input_sentence_with_tokens: torch.Tensor,
        target_sentence_with_tokens: torch.Tensor,
    ) -> torch.Tensor:
        target_sequence_length = len(target_sentence_with_tokens)
        predicted_token = -1
        current_position = 0
        total_prediction = 0
        total_loss = 0.0
        while not self.stop_iteration(
            current_position, predicted_token, target_sequence_length
        ):
            model_inputs = self.prepare_model_inputs(
                input_sentence_with_tokens,
                target_sentence_with_tokens[:current_position],
                self.max_sequence_length,
            )
            token_probabilities = self.model(model_inputs)
            predicted_token = torch.argmax(token_probabilities)
            loss = self.loss_func(
                token_probabilities,
                torch.tensor([target_sentence_with_tokens[current_position]]),
            )
            total_loss += loss.item()
            current_position += 1
            total_prediction += 1

        return torch.tensor(total_loss / total_prediction)

    @torch.no_grad()
    def inference(self, input_sentence_with_tokens: torch.Tensor) -> torch.Tensor:
        predicted_token = torch.tensor(-1)
        current_inference_position = 0
        predicted_tokens: List[int] = []

        while not self.stop_iteration(
            current_inference_position,
            predicted_token,
            self.max_output_sequence_length,
        ):
            model_inputs = self.prepare_model_inputs(
                input_sentence_with_tokens,
                torch.tensor(
                    predicted_tokens,
                    dtype=input_sentence_with_tokens.dtype,
                ),
                self.max_sequence_length,
            )
            token_probabilities = self.model(model_inputs)
            predicted_token = torch.argmax(token_probabilities)
            predicted_tokens.append(predicted_token.item())
            current_inference_position += 1

        return torch.tensor(predicted_tokens)
