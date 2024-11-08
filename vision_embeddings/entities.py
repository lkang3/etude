from dataclasses import dataclass
from typing import List

import torch


@dataclass
class KernelConfig:
    height: int
    width: int
    stride: int


@dataclass
class ImageSize:
    height: int
    width: int
    channels: int = 0


@dataclass
class ImageSelectOffset:
    start: int
    end: int


@dataclass
class DataPatchMetadata:
    patch_row_index: int
    patch_col_index: int
    data_subset_row_offsets: ImageSelectOffset
    data_subset_col_offsets: ImageSelectOffset
    is_padding: bool = False
    padding_value: torch.float = torch.tensor(0.0)

    @staticmethod
    def padding(data_patch_to_pad: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def get_data_patch(self, data: torch.Tensor) -> torch.Tensor:
        return data[
            self.data_subset_row_offsets.start: self.data_subset_row_offsets.end,
            self.data_subset_col_offsets.start: self.data_subset_col_offsets.end,
        ]

    def update_data_with_patch(
        self, data_to_update: torch.Tensor, data_patch: torch.Tensor,
    ) -> torch.Tensor:
        data_to_update[self.patch_row_index, self.patch_col_index] = data_patch
        return data_to_update


@dataclass
class ConvMetadata:
    output_size: ImageSize
    data_patch_metadata: List[DataPatchMetadata]

    def create_init_conv_output(self) -> torch.Tensor:
        return torch.rand(self.output_size.height, self.output_size.width)
