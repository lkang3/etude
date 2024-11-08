import pytest
from typing import Iterator
from tests.utils import create_dataclass_mock
from unittest.mock import Mock
from unittest.mock import patch

import torch

from vision_embeddings.models import DummyConvKernel, ConvOneXOneDimensionReducer
from vision_embeddings.models import DummyConv
from vision_embeddings.entities import KernelConfig, ImageSize, DataPatchMetadata, ConvMetadata


class TestDummyConv:
    @pytest.fixture
    def mock_conv_metadata_class(self) -> Iterator[Mock]:
        with patch("vision_embeddings.models.ConvMetadata") as mock_func:
            yield mock_func

    @pytest.fixture
    def get_apply_kernel_row_offsets(self) -> Iterator[Mock]:
        with patch.object(DummyConv, "get_apply_kernel_row_offsets") as mock_func:
            yield mock_func

    @pytest.fixture
    def get_apply_kernel_col_offsets(self) -> Iterator[Mock]:
        with patch.object(DummyConv, "get_apply_kernel_col_offsets") as mock_func:
            yield mock_func

    @pytest.fixture
    def get_conv_metadata(self) -> Iterator[Mock]:
        with patch.object(DummyConv, "get_conv_metadata") as mock_func:
            yield mock_func

    @pytest.fixture
    def run_conv(self) -> Iterator[Mock]:
        with patch.object(DummyConv, "run_conv") as mock_func:
            yield mock_func

    @pytest.fixture
    def get_data_patch(self) -> Iterator[Mock]:
        with patch.object(DataPatchMetadata, "get_data_patch") as mock_func:
            yield mock_func

    @pytest.fixture
    def dummy_conv_kernel_apply_kernel(self) -> Iterator[Mock]:
        with patch.object(DummyConvKernel, "apply_kernel") as mock_func:
            yield mock_func

    def test_forward(
        self,
        get_conv_metadata: Mock,
        run_conv: Mock,
    ) -> None:
        height = 20
        width = 23
        data = torch.rand(height, width)
        kernel_height = 2
        kernel_width = 2
        kernel_stride = 1
        kernel_config = KernelConfig(kernel_height, kernel_width, kernel_stride)
        model = DummyConv(kernel_config)
        model(data)

        get_conv_metadata.assert_called_once_with(
            height, width, kernel_height, kernel_width, kernel_stride
        )
        run_conv.assert_called_once_with(data, get_conv_metadata.return_value, model.kernel)

    def test_get_conv_metadata(
        self,
        mock_conv_metadata_class: Mock,
        get_apply_kernel_row_offsets: Mock,
        get_apply_kernel_col_offsets: Mock,
    ) -> None:
        image_height = Mock()
        image_width = Mock()
        kernel_height = Mock()
        kernel_width = Mock()
        kernel_stride = Mock()
        kernel_row_offsets = Mock()
        get_apply_kernel_row_offsets.return_value = [kernel_row_offsets]
        kernel_col_offsets = Mock()
        get_apply_kernel_col_offsets.return_value = [kernel_col_offsets]
        DummyConv.get_conv_metadata(image_height, image_width, kernel_height, kernel_width, kernel_stride)

        image_size = ImageSize(image_height, image_width)
        kernel_config = KernelConfig(kernel_height, kernel_width, kernel_stride)
        get_apply_kernel_row_offsets.assert_called_once_with(image_size, kernel_config)
        get_apply_kernel_col_offsets.assert_called_once_with(image_size, kernel_config)
        mock_conv_metadata_class.assert_called_once_with(
            ImageSize(1, 1),
            [DataPatchMetadata(0, 0, kernel_row_offsets, kernel_col_offsets)]
        )

    def test_run_conv(self) -> None:
        data = Mock()
        data_patch_metadata = create_dataclass_mock(DataPatchMetadata)
        conv_metadata = create_dataclass_mock(ConvMetadata)
        conv_metadata.data_patch_metadata = [data_patch_metadata]
        kernel = Mock()
        DummyConv.run_conv(data, conv_metadata, kernel)

        conv_metadata.create_init_conv_output.assert_called_once_with()
        data_patch_metadata.get_data_patch.assert_called_once_with(data)
        data_patch = data_patch_metadata.get_data_patch.return_value
        kernel.apply_kernel.assert_called_once_with(data_patch)
        data_patch_metadata.update_data_with_patch.assert_called_once_with(
            conv_metadata.create_init_conv_output.return_value,
            kernel.apply_kernel.return_value,
        )


class TestConvOneXOneDimensionReducer:
    @pytest.fixture
    def mock_create_dense_layer(self) -> Iterator[Mock]:
        with patch.object(ConvOneXOneDimensionReducer, "create_dense_layer") as mock_func:
            yield mock_func

    def test_init(self, mock_create_dense_layer: Mock) -> None:
        ConvOneXOneDimensionReducer(Mock(), Mock())

        mock_create_dense_layer.assert_called_once_with()

    def test_forward(self) -> None:
        pass

"""
class Conv3D
    def create_channel_filter_metadata() -> List[ChannelFilerMetadata]
    def create_channel_filters(List[ChannelFilerMetadata]) -> List[ChannelFilter]
    def create_unified_channel_filters(KernelConfig) -> nn.Module
    
    def __init__()
        channel_filters: List[ChannelFilter]
    
    def apply_conv()
    
    def aggregate()
    
    def forward()
        create_channel_filter_metadata()
        

ChannelFilterMetadata
    channel_id
    KernelMetadata
    is_unified
    
    
create_channel_filter(ChannelFilterMetadata) -> ChannelFilter

ChannelFilter
    metadata: ChannelFilterMetadata
    model: nn.Model
    
ChannelFilters
    List[ChannelFilter]

UnifiedFilter
    



apply(ChannelKernels)
aggregate(ChannelKernels)
    

conv_of_ch_i(ch_i) -> aggregate

unified_conv(ch_0, ch_1, ..., ch_n)


apply_on_channels()

"""




