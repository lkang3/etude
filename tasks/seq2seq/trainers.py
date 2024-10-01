from typing import List
from typing import Tuple

import torch
from torch import nn

from tasks.seq2seq.models import NextToken
from tasks.seq2seq.models import Seq2SeqConfig
from tasks.utils import padding_sequence


class NextTokenTrainer:
    def __init__(
        self, model: NextToken, max_source_seq_size: int, max_target_seq_size: int
    ):
        self.model = model
        self.loss_func = nn.CrossEntropyLoss()
        self.max_source_seq_size = max_source_seq_size
        self.max_target_seq_size = max_target_seq_size

    @classmethod
    def from_config(cls, config: Seq2SeqConfig) -> "NextTokenTrainer":
        return cls(
            NextToken.from_config(config),
            config.source_seq_length,
            config.target_seq_length,
        )

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
        source_sentence: torch.Tensor,
        target_sentence: torch.Tensor,
        max_source_seq_length: int,
        max_target_seq_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            padding_sequence(source_sentence, 0, max_source_seq_length),
            padding_sequence(target_sentence, 0, max_target_seq_length),
        )

    def validate_source_sentence(self, source_sentence: torch.Tensor) -> None:
        assert len(source_sentence) <= self.max_source_seq_size, (
            len(source_sentence),
            self.max_source_seq_size,
        )

    def validate_target_sentence(self, target_sentence: torch.Tensor) -> None:
        assert len(target_sentence) <= self.max_target_seq_size, (
            len(target_sentence),
            self.max_target_seq_size,
        )

    def train(
        self,
        input_sentence_with_tokens: torch.Tensor,
        target_sentence_with_tokens: torch.Tensor,
    ) -> torch.Tensor:
        self.validate_source_sentence(input_sentence_with_tokens)
        self.validate_target_sentence(target_sentence_with_tokens)
        target_sequence_length = len(target_sentence_with_tokens)
        predicted_token = -1
        total_prediction = 0
        total_loss = 0.0
        start_token = 0
        predicted_tokens: List[int] = [start_token]
        current_position = 1

        while not self.stop_iteration(
            current_position, predicted_token, target_sequence_length
        ):
            model_encoder_input, model_decoder_input = self.prepare_model_inputs(
                input_sentence_with_tokens,
                torch.tensor(
                    predicted_tokens,
                    dtype=input_sentence_with_tokens.dtype,
                ),
                self.max_source_seq_size,
                self.max_target_seq_size,
            )
            token_probabilities = self.model(model_decoder_input, model_encoder_input)
            token_id = torch.argmax(token_probabilities)
            predicted_tokens.append(token_id.item())
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
        start_token = 0
        predicted_tokens: List[int] = [start_token]
        current_inference_position = 1

        while not self.stop_iteration(
            current_inference_position,
            predicted_token,
            self.max_target_seq_size,
        ):
            model_encoder_input, model_decoder_input = self.prepare_model_inputs(
                input_sentence_with_tokens,
                torch.tensor(
                    predicted_tokens,
                    dtype=input_sentence_with_tokens.dtype,
                ),
                self.max_source_seq_size,
                self.max_target_seq_size,
            )
            token_probabilities = self.model(model_decoder_input, model_encoder_input)
            predicted_token = torch.argmax(token_probabilities)
            predicted_tokens.append(predicted_token.item())
            current_inference_position += 1

        return torch.tensor(predicted_tokens)
