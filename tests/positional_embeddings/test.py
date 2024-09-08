from typing import Iterator
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from positional_embeddings.models import RelativeBiasPositionalEmbedding
from positional_embeddings.models import RelativePositionalEmbedding
from positional_embeddings.models import RotaryPositionalEmbeddings
from positional_embeddings.models import SinusoidalPositionalEmbeddings
from positional_embeddings.models import VanillaPositionalEmbedding
from positional_embeddings.models import get_attention_scores_adjusted_by_relative_positions
from positional_embeddings.models import get_attention_scores_adjusted_by_relative_positions_bias
from tests.utils import is_tensor_equal


class TestVanillaPositionalEmbeddings:
    def test_forward(self) -> None:  # ignore: type
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
            ),
        )
        is_tensor_equal(
            positive_rel_positions,
            torch.tensor(
                [
                    [2, 3, 4, 5],
                    [1, 2, 3, 4],
                    [0, 1, 2, 3],
                ]
            ),
        )

    def test_forward(self) -> None:
        target_sequence_size = 3
        source_sequence_length = 4
        embedding_size = 5
        model = RelativePositionalEmbedding(
            target_sequence_size,
            source_sequence_length,
            embedding_size,
        )
        output = model()

        assert model.position_embedding.weight.shape == (
            target_sequence_size + source_sequence_length - 1,
            embedding_size,
        )
        assert output.shape == (
            target_sequence_size,
            source_sequence_length,
            embedding_size,
        )


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
    ) -> RelativeBiasPositionalEmbedding:
        return RelativeBiasPositionalEmbedding(
            target_sequence_size, source_sequence_size
        )

    def test_forward(self) -> None:
        target_sequence_size = 2
        source_sequence_length = 3
        model = RelativeBiasPositionalEmbedding(
            target_sequence_size, source_sequence_length
        )

        outputs = model.forward()
        assert outputs.shape == (target_sequence_size, source_sequence_length)


class TestRotaryPositionalEmbeddings:
    @pytest.fixture
    def model(self) -> RotaryPositionalEmbeddings:
        target_sequence_size = 2
        source_sequence_size = 3
        embedding_size = 4
        theta_base_value = 10000.0
        return RotaryPositionalEmbeddings(
            target_sequence_size,
            source_sequence_size,
            embedding_size,
            theta_base_value,
        )

    def test_position_embedding_size(self, model: RotaryPositionalEmbeddings) -> None:
        assert model.position_embedding_size == max(
            model.source_sequence_size, model.target_sequence_size
        )

    def test_get_embedding(self, model: RotaryPositionalEmbeddings) -> None:
        position_embedding = model.get_embedding()
        assert position_embedding.shape == (
            model.position_embedding_size,
            model.embedding_size,
            model.embedding_size,
        )
        assert position_embedding.requires_grad is False

    def test_forward(self, model: RotaryPositionalEmbeddings) -> None:
        output = model()
        assert output.shape == (
            model.position_embedding_size,
            model.embedding_size,
            model.embedding_size,
        )

    def test_apply_on_single_token(self, model: RotaryPositionalEmbeddings) -> None:
        embedding_size = 4
        token_order = 0
        input_token = torch.rand(embedding_size)
        rotated_input_token = model.apply_on_single_token(input_token, 0)

        assert embedding_size == model.embedding_size
        assert token_order <= model.position_embedding_size
        assert rotated_input_token.shape == input_token.shape

    def test_apply_on_tokens(self, model: RotaryPositionalEmbeddings) -> None:
        embedding_size = 4
        num_tokens = 3
        input_tokens = torch.rand(num_tokens, embedding_size)
        rotated_input_tokens = model.apply_on_tokens(input_tokens)

        assert embedding_size == model.embedding_size
        assert num_tokens <= model.position_embedding_size
        assert rotated_input_tokens.shape == input_tokens.shape


class TestSinusoidalPositionalEmbeddings:
    @pytest.fixture
    def mock_get_theta(self) -> Iterator[Mock]:
        with patch.object(SinusoidalPositionalEmbeddings, "get_theta") as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_even_embedding_indices(self) -> Iterator[Mock]:
        with patch.object(
            SinusoidalPositionalEmbeddings, "get_even_embedding_indices"
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_odd_embedding_indices(self) -> Iterator[Mock]:
        with patch.object(
            SinusoidalPositionalEmbeddings, "get_odd_embedding_indices"
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_positional_embeddings_at_even_embedding_positions(
        self,
    ) -> Iterator[Mock]:
        with patch.object(
            SinusoidalPositionalEmbeddings,
            "get_positional_embeddings_at_even_embedding_positions",
        ) as mock_func:
            yield mock_func

    @pytest.fixture
    def mock_get_positional_embeddings_at_odd_embedding_positions(
        self,
    ) -> Iterator[Mock]:
        with patch.object(
            SinusoidalPositionalEmbeddings,
            "get_positional_embeddings_at_odd_embedding_positions",
        ) as mock_func:
            yield mock_func

    def test_get_embeddings(
        self,
        mock_get_positional_embeddings_at_odd_embedding_positions: Mock,
        mock_get_positional_embeddings_at_even_embedding_positions: Mock,
        mock_get_odd_embedding_indices: Mock,
        mock_get_even_embedding_indices: Mock,
    ) -> None:
        max_sequence_size = 10
        embedding_size = 5
        odd_embedding_indices = torch.arange(1, embedding_size, 2)
        even_embedding_indices = torch.arange(0, embedding_size, 2)
        mock_get_odd_embedding_indices.return_value = odd_embedding_indices
        mock_get_even_embedding_indices.return_value = even_embedding_indices
        expected_embeddings_at_odd_embedding_positions = torch.ones(
            max_sequence_size,
            len(odd_embedding_indices),
        )
        mock_get_positional_embeddings_at_odd_embedding_positions.return_value = (
            expected_embeddings_at_odd_embedding_positions
        )
        expected_embeddings_at_even_embedding_positions = torch.ones(
            max_sequence_size,
            len(even_embedding_indices),
        )
        mock_get_positional_embeddings_at_even_embedding_positions.return_value = (
            expected_embeddings_at_even_embedding_positions
        )

        model = SinusoidalPositionalEmbeddings(max_sequence_size, embedding_size)
        positional_embeddings = model.position_embedding

        mock_get_odd_embedding_indices.assert_called_once()
        mock_get_even_embedding_indices.assert_called_once()
        mock_get_positional_embeddings_at_odd_embedding_positions.assert_called_once()
        mock_get_positional_embeddings_at_even_embedding_positions.assert_called_once()
        is_tensor_equal(
            positional_embeddings[:, odd_embedding_indices],
            expected_embeddings_at_odd_embedding_positions,
        )
        is_tensor_equal(
            positional_embeddings[:, even_embedding_indices],
            expected_embeddings_at_even_embedding_positions,
        )

    def test_forward(self) -> None:
        max_sequence_size = 10
        embedding_size = 5
        model = SinusoidalPositionalEmbeddings(max_sequence_size, embedding_size)

        positional_embeddings = model()

        assert positional_embeddings.shape == (max_sequence_size, embedding_size)
        assert positional_embeddings.requires_grad is False


def test_get_adjusted_attention_scores() -> None:
    target_sequence_size = 2
    source_sequence_length = 3
    embedding_size = 4
    rel_positions_embeddings = torch.rand(
        target_sequence_size,
        source_sequence_length,
        embedding_size,
    )
    query = torch.rand(target_sequence_size, embedding_size)
    scores = get_attention_scores_adjusted_by_relative_positions(
        rel_positions_embeddings, query
    )

    assert scores.shape == (target_sequence_size, source_sequence_length)


def test_get_attention_scores_adjusted_by_relative_positions_bias() -> None:
    attention_score = torch.zeros(2, 3)
    rel_positions = torch.rand(2, 3)
    outputs = get_attention_scores_adjusted_by_relative_positions_bias(
        attention_score,
        rel_positions,
    )

    is_tensor_equal(outputs, rel_positions)
