from typing import List

import torch
from torch import nn

from vision_embeddings.entities import KernelConfig, ImageSize, ImageSelectOffset, \
    DataPatchMetadata, ConvMetadata


class VisionEmbeddingWithNonOverlappingSquarePatch(nn.Module):
    def __init__(
        self,
        input_embedding_size: int,
        output_embedding_size: int,
        image_size: int,
        image_patch_size: int,
    ):
        super().__init__()
        self.input_embedding_size = input_embedding_size
        self.output_embedding_size = output_embedding_size
        self.image_size = image_size
        self.image_patch_size = image_patch_size
        self.num_of_patches = self.get_num_of_patches()
        self.patcher = nn.Unfold(
            kernel_size=(self.image_patch_size, self.image_patch_size),
            stride=(self.image_patch_size, self.image_patch_size),
        )
        self.mlp = nn.Linear(self.input_embedding_size, self.output_embedding_size)

    def get_num_of_patches(self) -> int:
        return (self.image_size // self.image_patch_size) ** 2

    @staticmethod
    def prioritize_patch_dimension(data) -> torch.Tensor:
        return data.permute(-1, -2)

    def separate_input_embedding_dimension(self, data) -> torch.Tensor:
        return data.view(
            self.num_of_patches,
            self.image_patch_size * self.image_patch_size,
            self.input_embedding_size,
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # size: (input_embedding_size * (image_patch_size * image_patch_size) , num_of_patches)
        patches = self.patcher(data)
        # size: (num_of_patches, input_embedding_size * (image_patch_size * image_patch_size))
        patches = self.prioritize_patch_dimension(patches)
        # size: (num_of_patches, image_patch_size * image_patch_size, input_embedding_size)
        patches = self.separate_input_embedding_dimension(patches)
        # return self.mlp(patches)
        return patches


class BasicConvKernel:
    def __init__(
        self,
        kernel_config: KernelConfig,
    ):
        self.kernel_config = kernel_config
        self.model = self.create_model_from_config(kernel_config)

    def create_model_from_config(self, kernel_config: KernelConfig) -> "BasicConvKernel":
        raise NotImplementedError()

    def apply_kernel(self, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class DummyConvKernel(BasicConvKernel):
    def create_model_from_config(self, kernel_config: KernelConfig) -> nn.Module:
        embedding_size = kernel_config.height * kernel_config.width
        return nn.Linear(embedding_size, embedding_size)

    def pre_conv_process(self, data: torch.Tensor) -> torch.Tensor:
        return data.flatten()

    def post_conv_process(self, data: torch.Tensor) -> torch.Tensor:
        return torch.sum(data)

    def apply_kernel(self, data: torch.Tensor) -> torch.Tensor:
        data = self.pre_conv_process(data)
        data = self.model(data)
        return self.post_conv_process(data)


class ConvOneXOneDimensionReducer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, add_bias: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.add_bias = add_bias
        self.model = self.create_dense_layer()

    def create_dense_layer(self) -> nn.Module:
        return nn.Linear(self.input_dim, self.output_dim, bias=self.add_bias)

    def get_num_of_trainable_params(self) -> int:
        num_of_weights = self.model.weight.numel()
        num_of_bias = self.model.bias.numel() if self.model.bias is not None else 0
        return num_of_weights + num_of_bias


class DummyConv(nn.Module):
    def __init__(self, kernel_config: KernelConfig):
        super().__init__()
        self.kernel_config = kernel_config
        self.kernel = DummyConvKernel(self.kernel_config)

    @staticmethod
    def get_apply_kernel_row_offsets(
        image_size: ImageSize, kernel_config: KernelConfig,
    ) -> List[ImageSelectOffset]:
        offsets = []
        kernel_stride = kernel_config.stride
        for start in range(0, image_size.height, kernel_stride):
            offsets.append(
                ImageSelectOffset(start, min(start + kernel_config.height, image_size.height)))
        return offsets

    @staticmethod
    def get_apply_kernel_col_offsets(
        image_size: ImageSize, kernel_config: KernelConfig,
    ) -> List[ImageSelectOffset]:
        offsets = []
        kernel_stride = kernel_config.stride
        for start in range(0, image_size.width, kernel_stride):
            offsets.append(
                ImageSelectOffset(start, min(start + kernel_config.width, image_size.width)))
        return offsets

    @staticmethod
    def get_conv_metadata(
        image_height: int,
        image_width: int,
        kernel_height: int,
        kernel_width: int,
        kernel_stride: int,
    ) -> ConvMetadata:
        image_size = ImageSize(image_height, image_width)
        kernel_config = KernelConfig(kernel_height, kernel_width, kernel_stride)
        kernel_apply_row_offsets = DummyConv.get_apply_kernel_row_offsets(image_size, kernel_config)
        kernel_apply_col_offsets = DummyConv.get_apply_kernel_col_offsets(image_size, kernel_config)
        apply_kernel_schemas = []
        for conv_row_index, row_offsets in enumerate(kernel_apply_row_offsets):
            for conv_col_index, col_offsets in enumerate(kernel_apply_col_offsets):
                apply_kernel_schemas.append(
                    DataPatchMetadata(conv_row_index, conv_col_index, row_offsets, col_offsets)
                )
        return ConvMetadata(
            ImageSize(len(kernel_apply_row_offsets), len(kernel_apply_col_offsets)),
            apply_kernel_schemas,
        )

    @staticmethod
    def run_conv(data: torch.Tensor, conv_metadata: ConvMetadata, conv_kernel: DummyConvKernel) -> torch.Tensor:
        output = conv_metadata.create_init_conv_output()
        for data_patch_metadata in conv_metadata.data_patch_metadata:
            data_patch = data_patch_metadata.get_data_patch(data)
            data_patch_metadata.update_data_with_patch(output, conv_kernel.apply_kernel(data_patch))
        return output

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        image_height = data.shape[0]
        image_width = data.shape[1]
        conv_schema = self.get_conv_metadata(
            image_height, image_width, self.kernel_config.height, self.kernel_config.width, self.kernel_config.stride
        )
        return self.run_conv(data, conv_schema, self.kernel)
