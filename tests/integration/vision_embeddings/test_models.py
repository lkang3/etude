import torch

from vision_embeddings.models import \
    VisionEmbeddingWithNonOverlappingSquarePatch

torch.manual_seed(123)


class TestVisionEmbeddingWithNonOverlappingSquarePatch:
    def test_forward(self) -> None:
        num_of_channels = 3
        image_size = 4
        input_data = torch.rand(num_of_channels, image_size, image_size)
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
