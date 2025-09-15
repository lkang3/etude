import torch
from torch import nn

from tasks.common.distance import DistanceType
from tasks.masked_prediction.entities import MaskedPredictionWithSequenceInput
from tasks.masked_prediction.models import MaskedPredictionModel
from tests.utils import is_scalar

torch.manual_seed(123)


class DummyLanguageModelToPretrain(nn.Module):
    def __init__(self, num_of_tokens: int):
        super().__init__()
        self.tokenization_layer = nn.Embedding(num_of_tokens, 1)
        self.model = nn.Linear(1, 3)

    def forward(self, data: torch.Tensor):
        hidden_states = self.tokenization_layer(data)
        return self.model(hidden_states)


class TestMaskedPredictionWithSentence:
    def test_forward(self) -> None:
        max_seq_length = 11
        total_token_number = 14
        one_input_sequence = torch.randint(0, total_token_number - 2, (max_seq_length,))

        token_of_mask = total_token_number - 1  # the last token is reserved for mask
        model_to_pretrain = DummyLanguageModelToPretrain(
            num_of_tokens=total_token_number
        )
        model_input = MaskedPredictionWithSequenceInput(
            data=one_input_sequence,
            mask_value=token_of_mask,
            mask_percent=0.2,
        )
        model = MaskedPredictionModel(
            model_to_pretrain=model_to_pretrain,
            distance_type=DistanceType.L1,
        )
        distance = model(model_input)
        assert is_scalar(distance)
