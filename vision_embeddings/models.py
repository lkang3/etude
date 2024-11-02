import torch
from torch import nn


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
