from typing import List

import pytest
import torch
from torch import nn

from tasks.contextual_learning.contextual_token_generator import ContextualModelOutput
from tasks.contextual_learning.contextual_token_generator import (
    NLPContextualModelConfig,
)
from tasks.contextual_learning.contextual_token_generator import NLPContextualModelType
from tasks.contextual_learning.models import ContextualLearningNLPModel


class DummyModelToPretrain(nn.Module):
    def forward(self, data: torch.Tensor):
        return data


class TestContextualLearningNLPModel:
    @pytest.fixture
    def dummy_npl_data(self) -> torch.Tensor:
        num_of_tokens = 5
        num_of_sentences_in_on_batch = 3
        num_of_tokens_in_one_sentence = 7
        embedding_size = 2

        return torch.randint(
            1,
            num_of_tokens,
            (
                num_of_sentences_in_on_batch,
                num_of_tokens_in_one_sentence,
                embedding_size,
            ),
        ).float()

    def test_cbow_style_learning(self, dummy_npl_data: torch.Tensor) -> None:
        embedding_size = dummy_npl_data.shape[-1]
        contextual_model_config = NLPContextualModelConfig(
            model_type=NLPContextualModelType.C_BOW,
            embedding_size=embedding_size,
            num_of_contextual_samples=5,
        )
        model_to_pretrain = DummyModelToPretrain()
        learning_model = ContextualLearningNLPModel(
            model_to_pretrain, contextual_model_config
        )

        model_outputs: List[ContextualModelOutput] = learning_model(dummy_npl_data)
        assert all(
            model_output.projected_target_data.shape
            == (contextual_model_config.num_of_target_samples, embedding_size)
            for model_output in model_outputs
        )
        assert all(
            model_output.expected_target_data.shape
            == (contextual_model_config.num_of_target_samples, embedding_size)
            for model_output in model_outputs
        )

    def test_skip_gram_style_learning(self, dummy_npl_data: torch.Tensor) -> None:
        embedding_size = dummy_npl_data.shape[-1]
        num_of_target_samples = 5
        contextual_model_config = NLPContextualModelConfig(
            model_type=NLPContextualModelType.SKIP_GRAM,
            embedding_size=embedding_size,
            num_of_target_samples=num_of_target_samples,
        )
        model_to_pretrain = DummyModelToPretrain()
        learning_model = ContextualLearningNLPModel(
            model_to_pretrain, contextual_model_config
        )

        model_outputs: List[ContextualModelOutput] = learning_model(dummy_npl_data)
        assert all(
            model_output.projected_target_data.shape
            == (num_of_target_samples, embedding_size)
            for model_output in model_outputs
        )
        assert all(
            model_output.expected_target_data.shape
            == (num_of_target_samples, embedding_size)
            for model_output in model_outputs
        )
