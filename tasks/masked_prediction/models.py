import torch
from torch import nn

from tasks.common.distance import DistanceType
from tasks.masked_prediction.entities import TypeMaskedPredictionInput


class MaskedPredictionModel(nn.Module):
    def __init__(
        self,
        model_to_pretrain: nn.Module,
        distance_type: DistanceType,
    ):
        super().__init__()
        self.model_to_pretrain = model_to_pretrain
        self.distance_type = distance_type
        self.distance_layer: DistanceType = DistanceType.create_contrastive_layer(self.distance_type)

    def get_pretrain_model_outputs_with_masked_input_data(
        self, inputs: TypeMaskedPredictionInput
    ) -> torch.Tensor:
        return self.model_to_pretrain(inputs.get_masked_data())

    def get_pretrain_model_outputs_with_original_input_data(
        self, inputs: TypeMaskedPredictionInput
    ) -> torch.Tensor:
        return self.model_to_pretrain(inputs.get_data())

    def forward(self, inputs: TypeMaskedPredictionInput) -> torch.Tensor:
        model_outputs_with_masked_inputs = (
            self.get_pretrain_model_outputs_with_masked_input_data(inputs)
        )
        model_outputs_with_original_inputs = (
            self.get_pretrain_model_outputs_with_original_input_data(inputs)
        )
        actual_outputs = inputs.filter_data_with_mask_indices(
            model_outputs_with_masked_inputs
        )
        original_outputs = inputs.filter_data_with_mask_indices(
            model_outputs_with_original_inputs
        )
        return self.distance_layer(actual_outputs, original_outputs)
