from unittest.mock import Mock

import pytest
import torch

from tests.utils import create_multi_channel_image_data
from tests.utils import is_scalar
from vision_embeddings.models import DummyConv
from vision_embeddings.models import DummyConvKernel
from vision_embeddings.models import ConvMetadata
from vision_embeddings.entities import KernelConfig, ImageSize, ImageSelectOffset
from vision_embeddings.models import \
    VisionEmbeddingWithNonOverlappingSquarePatch
from vision_embeddings.models import ConvOneXOneDimensionReducer


torch.manual_seed(123)


class TestVisionEmbeddingWithNonOverlappingSquarePatch:
    def test_forward(self) -> None:
        num_of_channels = 3
        image_size = 4
        input_data = create_multi_channel_image_data(num_of_channels, image_size, image_size)
        output_embedding_size = 17
        patch_size = 2
        model = VisionEmbeddingWithNonOverlappingSquarePatch(
            input_embedding_size=num_of_channels,
            output_embedding_size=output_embedding_size,
            image_size=image_size,
            image_patch_size=patch_size,
        )

        expected_num_of_patches = 4
        assert model.num_of_patches == expected_num_of_patches
        output_data = model(input_data)
        assert output_data.shape == (
            expected_num_of_patches,
            patch_size * patch_size,
            output_embedding_size,
        )


class TestDummyConvKernel:
    def test_forward(self) -> None:
        height = 2
        width = 3
        config = KernelConfig(height, width, Mock())
        conv = DummyConvKernel(config)

        data = torch.rand(height, width)
        output = conv.apply_kernel(data)
        assert is_scalar(output)


class TestDummyConv:
    def test_get_apply_kernel_row_offsets(self) -> None:
        image_size = ImageSize(4, 4)
        kernel_config = KernelConfig(2, 2, 1)
        kernel_offsets = DummyConv.get_apply_kernel_row_offsets(image_size, kernel_config)
        assert kernel_offsets == [
            ImageSelectOffset(0, 2),
            ImageSelectOffset(1, 3),
            ImageSelectOffset(2, 4),
            ImageSelectOffset(3, 4),
        ]

    def test_get_apply_kernel_col_offsets(self) -> None:
        image_size = ImageSize(4, 4)
        kernel_config = KernelConfig(2, 2, 1)
        kernel_offsets = DummyConv.get_apply_kernel_row_offsets(image_size, kernel_config)
        assert kernel_offsets == [
            ImageSelectOffset(0, 2),
            ImageSelectOffset(1, 3),
            ImageSelectOffset(2, 4),
            ImageSelectOffset(3, 4),
        ]

    def test_get_conv_schema(self) -> None:
        image_height = 4
        image_width = 4
        kernel_height = 2
        kernel_width = 2
        kernel_stride = 2
        conv_metadata = DummyConv.get_conv_metadata(
            image_height,
            image_width,
            kernel_height,
            kernel_width,
            kernel_stride,
        )

        assert isinstance(conv_metadata, ConvMetadata)
        assert conv_metadata.output_size == ImageSize(2, 2)
        assert len(conv_metadata.data_patch_metadata) == 4

    def test_forward(self) -> None:
        image_size = ImageSize(4, 4)
        kernel_config = KernelConfig(2, 2, 2)
        model = DummyConv(kernel_config)
        data = torch.rand(image_size.height, image_size.width)

        output = model(data)
        assert output.shape == (2, 2)


class TestConvOneXOneDimensionReducer:
    def test_get_num_of_trainable_params_with_bias(self) -> None:
        input_channels = 5
        output_channels = 6
        model = ConvOneXOneDimensionReducer(input_channels, output_channels, add_bias=True)

        assert (
            model.get_num_of_trainable_params()
            == input_channels * output_channels + output_channels
        )

    def test_get_num_of_trainable_params_without_bias(self) -> None:
        input_channels = 5
        output_channels = 6
        model = ConvOneXOneDimensionReducer(input_channels, output_channels, add_bias=False)

        assert model.get_num_of_trainable_params() == input_channels * output_channels
