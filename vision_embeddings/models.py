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

    def restructure_with_patches(self, data) -> torch.Tensor:
        return data.view(
            self.num_of_patches,
            self.image_patch_size * self.image_patch_size,
            self.input_embedding_size,
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        patches = self.patcher(data)
        patches = self.restructure_with_patches(patches)
        return self.mlp(patches)
