import pytest
import torch

from positional_embeddings.models import get_attention_scores_adjusted_by_relative_positions
from positional_embeddings.models import get_attention_scores_adjusted_by_relative_positions_bias
from positional_embeddings.models import RelativePositionalEmbedding
from positional_embeddings.models import RelativeBiasPositionalEmbedding
from positional_embeddings.models import VanillaPositionalEmbedding
from tests.utils import is_tensor_equal


class TestVanillaPositionalEmbeddings:
    def test_forward(self) -> None:
        max_sequence_length = 5
        embedding_size = 7
        model = VanillaPositionalEmbedding(max_sequence_length, embedding_size)
        sequence_length = max_sequence_length
        output = model()

        assert output.shape == (sequence_length, embedding_size)


class TestRelativePositionalEmbeddings:
    def test_get_relative_positions(self) -> None:
        raw_rel_positions, positive_rel_positions = (
            RelativePositionalEmbedding.get_relative_positions(3, 4)
        )
        is_tensor_equal(
            raw_rel_positions,
            torch.tensor(
                [
                    [0, 1, 2, 3],
                    [-1, 0, 1, 2],
                    [-2, -1, 0, 1],
                 ]
            )
        )
        is_tensor_equal(
            positive_rel_positions,
            torch.tensor(
                [
                    [2, 3, 4, 5],
                    [1, 2, 3, 4],
                    [0, 1, 2, 3],
                ]
            )
        )

    def test_forward(self) -> None:
        target_sequence_size = 3
        source_sequence_length = 4
        embedding_size = 5
        model = RelativePositionalEmbedding(
            target_sequence_size, source_sequence_length, embedding_size,
        )
        output = model()

        assert model.position_embedding.weight.shape == (
            target_sequence_size + source_sequence_length - 1, embedding_size
        )
        assert output.shape == (target_sequence_size, source_sequence_length, embedding_size)


class TestRelativeBiasPositionalEmbedding:
    @pytest.fixture
    def target_sequence_size(self) -> int:
        return 2

    @pytest.fixture
    def source_sequence_size(self) -> int:
        return 3

    @pytest.fixture
    def embedding_size(self) -> int:
        return 4

    @pytest.fixture
    def model(
        self,
        target_sequence_size: int,
        source_sequence_size: int,
        embedding_size: int,
    ) -> RelativeBiasPositionalEmbedding:
        return RelativeBiasPositionalEmbedding(
            target_sequence_size, source_sequence_size, embedding_size,
        )

    def test_forward(self) -> None:
        target_sequence_size = 2
        source_sequence_length = 3
        model = RelativeBiasPositionalEmbedding(target_sequence_size, source_sequence_length)

        outputs = model.forward()
        assert outputs.shape == (target_sequence_size, source_sequence_length)


def test_get_adjusted_attention_scores() -> None:
    target_sequence_size = 2
    source_sequence_length = 3
    embedding_size = 4
    rel_positions_embeddings = torch.rand(
        target_sequence_size, source_sequence_length, embedding_size,
    )
    query = torch.rand(target_sequence_size, embedding_size)
    scores = get_attention_scores_adjusted_by_relative_positions(rel_positions_embeddings, query)

    assert scores.shape == (target_sequence_size, source_sequence_length)


def test_get_attention_scores_adjusted_by_relative_positions_bias() -> None:
    attention_score = torch.zeros(2, 3)
    rel_positions = torch.rand(2, 3)
    outputs = get_attention_scores_adjusted_by_relative_positions_bias(
        attention_score, rel_positions,
    )

    is_tensor_equal(outputs, rel_positions)
