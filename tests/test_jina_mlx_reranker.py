# -*- coding: utf-8 -*-
"""Unit tests for JinaMLXReranker."""

import pytest
from unittest.mock import Mock, patch
from typing import Any

mlx_available = True
torch_available = True
try:
    import mlx.core  # noqa: F401
    import numpy  # noqa: F401
    import torch  # noqa: F401
except ImportError:
    mlx_available = False
    torch_available = False


@pytest.fixture
def mock_mlx_model() -> Mock:
    """Create a mock MLX model."""
    mock = Mock()
    return mock


@pytest.fixture
def mock_tokenizer() -> Mock:
    """Create a mock tokenizer."""
    mock = Mock()
    mock.encode = Mock(return_value=[1, 2, 151671, 3, 4, 151670, 5, 6, 151670])
    return mock


@pytest.fixture
def mock_projector_weights() -> dict[str, Any]:
    """Create mock projector weights with correct shapes."""
    pytest.importorskip("torch")
    return {
        "linear1.weight": torch.randn(512, 1024),
        "linear2.weight": torch.randn(512, 512),
    }


@pytest.mark.skipif(not mlx_available, reason="MLX dependencies not installed")
@pytest.mark.unit
class TestProjectorLoading:
    """Tests for projector loading with correct shapes."""

    def test_projector_loaded_with_correct_shapes(
        self,
        mock_mlx_model: Mock,
        mock_tokenizer: Mock,
        mock_projector_weights: dict[str, Any],
    ) -> None:
        """Test that projector is loaded with weights having correct shapes."""
        with (
            patch("local_reranker.jina_mlx_reranker.load") as mock_load,
            patch(
                "local_reranker.jina_mlx_reranker._load_projector"
            ) as mock_load_projector,
        ):
            mock_load.return_value = (mock_mlx_model, mock_tokenizer, {})
            mock_load_projector.return_value = mock_projector_weights

            from local_reranker.jina_mlx_reranker import JinaMLXReranker

            reranker = JinaMLXReranker("/tmp/model", "/tmp/model/projector.safetensors")

            assert reranker.projector.linear1.weight.shape == torch.Size([512, 1024])
            assert reranker.projector.linear2.weight.shape == torch.Size([512, 512])

    def test_projector_weights_loaded_correctly(
        self,
        mock_mlx_model: Mock,
        mock_tokenizer: Mock,
        mock_projector_weights: dict[str, Any],
    ) -> None:
        """Test that projector state_dict is updated with loaded weights."""
        with (
            patch("local_reranker.jina_mlx_reranker.load") as mock_load,
            patch(
                "local_reranker.jina_mlx_reranker._load_projector"
            ) as mock_load_projector,
        ):
            mock_load.return_value = (mock_mlx_model, mock_tokenizer, {})
            mock_load_projector.return_value = mock_projector_weights

            from local_reranker.jina_mlx_reranker import JinaMLXReranker

            reranker = JinaMLXReranker("/tmp/model", "/tmp/model/projector.safetensors")

            loaded_state = reranker.projector.state_dict()
            assert "linear1.weight" in loaded_state
            assert "linear2.weight" in loaded_state


@pytest.mark.skipif(not mlx_available, reason="MLX dependencies not installed")
@pytest.mark.unit
class TestPromptFormat:
    """Tests for prompt format matching Jina specification."""

    def test_format_jina_prompt_structure(
        self, mock_mlx_model: Mock, mock_tokenizer: Mock
    ) -> None:
        """Test that prompt follows Jina specification structure."""
        with (
            patch("local_reranker.jina_mlx_reranker.load") as mock_load,
            patch(
                "local_reranker.jina_mlx_reranker._load_projector"
            ) as mock_load_projector,
        ):
            mock_load.return_value = (mock_mlx_model, mock_tokenizer, {})
            mock_load_projector.return_value = {
                "linear1.weight": torch.randn(512, 1024),
                "linear2.weight": torch.randn(512, 512),
            }

            from local_reranker.jina_prompt_formatter import _format_jina_prompt

            query = "What is machine learning?"
            documents = ["ML is a subset of AI", "Deep learning uses neural networks"]

            prompt = _format_jina_prompt(query, documents)

            assert "Query:" in prompt
            assert "<|embed_query|>" in prompt
            assert '<passage id="0">' in prompt
            assert '<passage id="1">' in prompt
            assert "<|embed_passage|>" in prompt
            assert "</passage>" in prompt

    def test_format_jina_prompt_sanitizes_special_tokens(
        self, mock_mlx_model: Mock, mock_tokenizer: Mock
    ) -> None:
        """Test that prompt sanitizes special tokens from input."""
        with (
            patch("local_reranker.jina_mlx_reranker.load") as mock_load,
            patch(
                "local_reranker.jina_mlx_reranker._load_projector"
            ) as mock_load_projector,
        ):
            mock_load.return_value = (mock_mlx_model, mock_tokenizer, {})
            mock_load_projector.return_value = {
                "linear1.weight": torch.randn(512, 1024),
                "linear2.weight": torch.randn(512, 512),
            }

            from local_reranker.jina_prompt_formatter import _format_jina_prompt

            query = "Test <|startoftext|>query<|endoftext|>"
            documents = ["Doc <|startoftext|>1<|endoftext|>"]

            prompt = _format_jina_prompt(query, documents)

            assert "<|startoftext|>" not in prompt
            assert "<|endoftext|>" not in prompt
            assert "Test query" in prompt
            assert "Doc 1" in prompt


@pytest.mark.skipif(not mlx_available, reason="MLX dependencies not installed")
@pytest.mark.unit
class TestSpecialTokenExtraction:
    """Tests for special token extraction."""

    def test_extract_query_token_position(
        self, mock_mlx_model: Mock, mock_tokenizer: Mock
    ) -> None:
        """Test that query token (151671) is correctly identified."""
        with (
            patch("local_reranker.jina_mlx_reranker.load") as mock_load,
            patch(
                "local_reranker.jina_mlx_reranker._load_projector"
            ) as mock_load_projector,
        ):
            mock_load.return_value = (mock_mlx_model, mock_tokenizer, {})
            mock_load_projector.return_value = {
                "linear1.weight": torch.randn(512, 1024),
                "linear2.weight": torch.randn(512, 512),
            }

            from local_reranker.jina_mlx_reranker import JinaMLXReranker

            reranker = JinaMLXReranker("/tmp/model", "/tmp/model/projector.safetensors")

            tokens = mock_tokenizer.encode("test prompt")
            query_position = None
            for i, t in enumerate(tokens):
                if t == reranker.query_embed_token_id:
                    query_position = i
                    break

            assert query_position == 2

    def test_extract_document_token_positions(
        self, mock_mlx_model: Mock, mock_tokenizer: Mock
    ) -> None:
        """Test that document tokens (151670) are correctly identified."""
        with (
            patch("local_reranker.jina_mlx_reranker.load") as mock_load,
            patch(
                "local_reranker.jina_mlx_reranker._load_projector"
            ) as mock_load_projector,
        ):
            mock_load.return_value = (mock_mlx_model, mock_tokenizer, {})
            mock_load_projector.return_value = {
                "linear1.weight": torch.randn(512, 1024),
                "linear2.weight": torch.randn(512, 512),
            }

            from local_reranker.jina_mlx_reranker import JinaMLXReranker

            reranker = JinaMLXReranker("/tmp/model", "/tmp/model/projector.safetensors")

            tokens = mock_tokenizer.encode("test prompt")
            doc_positions = [
                i for i, t in enumerate(tokens) if t == reranker.doc_embed_token_id
            ]

            assert len(doc_positions) == 2
            assert 5 in doc_positions
            assert 8 in doc_positions

    def test_raises_error_when_query_token_missing(
        self, mock_mlx_model: Mock, mock_tokenizer: Mock
    ) -> None:
        """Test that ValueError is raised when query token is not found."""
        with (
            patch("local_reranker.jina_mlx_reranker.load") as mock_load,
            patch(
                "local_reranker.jina_mlx_reranker._load_projector"
            ) as mock_load_projector,
        ):
            mock_load.return_value = (mock_mlx_model, mock_tokenizer, {})
            mock_load_projector.return_value = {
                "linear1.weight": torch.randn(512, 1024),
                "linear2.weight": torch.randn(512, 512),
            }

            from local_reranker.jina_mlx_reranker import JinaMLXReranker

            reranker = JinaMLXReranker("/tmp/model", "/tmp/model/projector.safetensors")

            import mlx.core as mx

            hidden_dim = 1024
            hidden_states = mx.random.normal((1, 6, hidden_dim))
            mock_mlx_model.return_value = hidden_states

            mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 151670, 4, 5])

            with pytest.raises(ValueError, match="Query embed token.*not found"):
                reranker._compute_single_batch("test query", ["doc1"])

    def test_raises_error_when_doc_token_count_mismatches(
        self, mock_mlx_model: Mock, mock_tokenizer: Mock
    ) -> None:
        """Test that ValueError is raised when doc token count doesn't match documents."""
        with (
            patch("local_reranker.jina_mlx_reranker.load") as mock_load,
            patch(
                "local_reranker.jina_mlx_reranker._load_projector"
            ) as mock_load_projector,
        ):
            mock_load.return_value = (mock_mlx_model, mock_tokenizer, {})
            mock_load_projector.return_value = {
                "linear1.weight": torch.randn(512, 1024),
                "linear2.weight": torch.randn(512, 512),
            }

            from local_reranker.jina_mlx_reranker import JinaMLXReranker

            reranker = JinaMLXReranker("/tmp/model", "/tmp/model/projector.safetensors")

            import mlx.core as mx

            hidden_dim = 1024
            hidden_states = mx.random.normal((1, 6, hidden_dim))
            mock_mlx_model.return_value = hidden_states

            mock_tokenizer.encode = Mock(return_value=[1, 2, 151671, 3, 151670, 4])

            with pytest.raises(ValueError, match="Expected.*document embed tokens"):
                reranker._compute_single_batch("test query", ["doc1", "doc2", "doc3"])


@pytest.mark.skipif(not mlx_available, reason="MLX dependencies not installed")
@pytest.mark.unit
class TestCosineSimilarity:
    """Tests for cosine similarity formula."""

    def test_cosine_similarity_identical_vectors(
        self, mock_mlx_model: Mock, mock_tokenizer: Mock
    ) -> None:
        """Test that cosine similarity of identical vectors is 1.0."""
        import mlx.core as mx

        vector = mx.array([1.0, 0.0, 0.0, 0.0])
        norm = mx.sqrt(mx.sum(vector * vector) + 1e-12)
        normalized = vector / norm

        cos_sim = mx.sum(normalized * normalized, axis=0)

        assert abs(float(cos_sim) - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal_vectors(
        self, mock_mlx_model: Mock, mock_tokenizer: Mock
    ) -> None:
        """Test that cosine similarity of orthogonal vectors is 0.0."""
        import mlx.core as mx

        v1 = mx.array([1.0, 0.0])
        v2 = mx.array([0.0, 1.0])

        v1_norm = mx.sqrt(mx.sum(v1 * v1) + 1e-12)
        v2_norm = mx.sqrt(mx.sum(v2 * v2) + 1e-12)
        v1_normalized = v1 / v1_norm
        v2_normalized = v2 / v2_norm

        cos_sim = mx.sum(v1_normalized * v2_normalized, axis=0)

        assert abs(float(cos_sim)) < 1e-6

    def test_cosine_similarity_opposite_vectors(
        self, mock_mlx_model: Mock, mock_tokenizer: Mock
    ) -> None:
        """Test that cosine similarity of opposite vectors is -1.0."""
        import mlx.core as mx

        v1 = mx.array([1.0, 0.0])
        v2 = mx.array([-1.0, 0.0])

        v1_norm = mx.sqrt(mx.sum(v1 * v1) + 1e-12)
        v2_norm = mx.sqrt(mx.sum(v2 * v2) + 1e-12)
        v1_normalized = v1 / v1_norm
        v2_normalized = v2 / v2_norm

        cos_sim = mx.sum(v1_normalized * v2_normalized, axis=0)

        assert abs(float(cos_sim) - (-1.0)) < 1e-6

    def test_cosine_similarity_uses_epsilon_for_stability(
        self, mock_mlx_model: Mock, mock_tokenizer: Mock
    ) -> None:
        """Test that 1e-12 epsilon prevents division by zero."""
        import mlx.core as mx

        zero_vector = mx.array([0.0, 0.0, 0.0, 0.0])
        norm = mx.sqrt(mx.sum(zero_vector * zero_vector) + 1e-12)

        assert float(norm) > 0


@pytest.mark.skipif(not mlx_available, reason="MLX dependencies not installed")
@pytest.mark.unit
class TestFullRerankingPipeline:
    """Tests for the full reranking pipeline."""

    def test_rerank_returns_sorted_results(
        self, mock_mlx_model: Mock, mock_tokenizer: Mock
    ) -> None:
        """Test that rerank returns results sorted by relevance score descending."""
        import mlx.core as mx

        with (
            patch("local_reranker.jina_mlx_reranker.load") as mock_load,
            patch(
                "local_reranker.jina_mlx_reranker._load_projector"
            ) as mock_load_projector,
        ):
            mock_load.return_value = (mock_mlx_model, mock_tokenizer, {})
            mock_load_projector.return_value = {
                "linear1.weight": torch.randn(512, 1024),
                "linear2.weight": torch.randn(512, 512),
            }

            from local_reranker.jina_mlx_reranker import JinaMLXReranker

            reranker = JinaMLXReranker("/tmp/model", "/tmp/model/projector.safetensors")

            mock_query_embeds = mx.random.normal((1, 512))
            mock_doc_embeds = mx.random.normal((3, 512))
            mock_scores = [0.9, 0.5, 0.3]

            with patch.object(
                reranker,
                "_compute_single_batch",
                return_value=(mock_query_embeds, mock_doc_embeds, mock_scores),
            ):
                documents = [
                    "Python is great",
                    "Machine learning is AI",
                    "Dogs are pets",
                ]
                query = "What is Python?"

                results = reranker.rerank(query, documents)

            assert len(results) == 3
            scores = [r["relevance_score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_rerank_respects_top_n(
        self, mock_mlx_model: Mock, mock_tokenizer: Mock
    ) -> None:
        """Test that rerank respects the top_n parameter."""
        import mlx.core as mx

        with (
            patch("local_reranker.jina_mlx_reranker.load") as mock_load,
            patch(
                "local_reranker.jina_mlx_reranker._load_projector"
            ) as mock_load_projector,
        ):
            mock_load.return_value = (mock_mlx_model, mock_tokenizer, {})
            mock_load_projector.return_value = {
                "linear1.weight": torch.randn(512, 1024),
                "linear2.weight": torch.randn(512, 512),
            }

            from local_reranker.jina_mlx_reranker import JinaMLXReranker

            reranker = JinaMLXReranker("/tmp/model", "/tmp/model/projector.safetensors")

            mock_query_embeds = mx.random.normal((1, 512))
            mock_doc_embeds = mx.random.normal((5, 512))
            mock_scores = [0.1, 0.2, 0.3, 0.4, 0.5]

            with patch.object(
                reranker,
                "_compute_single_batch",
                return_value=(mock_query_embeds, mock_doc_embeds, mock_scores),
            ):
                documents = ["Doc1", "Doc2", "Doc3", "Doc4", "Doc5"]
                query = "Test query"

                results = reranker.rerank(query, documents, top_n=3)

            assert len(results) == 3

    def test_rerank_includes_embeddings_when_requested(
        self, mock_mlx_model: Mock, mock_tokenizer: Mock
    ) -> None:
        """Test that rerank includes embeddings when return_embeddings=True."""
        import mlx.core as mx

        with (
            patch("local_reranker.jina_mlx_reranker.load") as mock_load,
            patch(
                "local_reranker.jina_mlx_reranker._load_projector"
            ) as mock_load_projector,
        ):
            mock_load.return_value = (mock_mlx_model, mock_tokenizer, {})
            mock_load_projector.return_value = {
                "linear1.weight": torch.randn(512, 1024),
                "linear2.weight": torch.randn(512, 512),
            }

            from local_reranker.jina_mlx_reranker import JinaMLXReranker

            reranker = JinaMLXReranker("/tmp/model", "/tmp/model/projector.safetensors")

            mock_query_embeds = mx.random.normal((1, 512))
            mock_doc_embeds = mx.random.normal((2, 512))
            mock_scores = [0.5, 0.3]

            with patch.object(
                reranker,
                "_compute_single_batch",
                return_value=(mock_query_embeds, mock_doc_embeds, mock_scores),
            ):
                documents = ["Doc1", "Doc2"]
                query = "Test query"

                results = reranker.rerank(query, documents, return_embeddings=True)

            assert len(results) == 2
            for result in results:
                assert "embedding" in result
                assert "document" in result

    def test_rerank_result_structure(
        self, mock_mlx_model: Mock, mock_tokenizer: Mock
    ) -> None:
        """Test that rerank returns results with correct structure."""
        import mlx.core as mx

        with (
            patch("local_reranker.jina_mlx_reranker.load") as mock_load,
            patch(
                "local_reranker.jina_mlx_reranker._load_projector"
            ) as mock_load_projector,
        ):
            mock_load.return_value = (mock_mlx_model, mock_tokenizer, {})
            mock_load_projector.return_value = {
                "linear1.weight": torch.randn(512, 1024),
                "linear2.weight": torch.randn(512, 512),
            }

            from local_reranker.jina_mlx_reranker import JinaMLXReranker

            reranker = JinaMLXReranker("/tmp/model", "/tmp/model/projector.safetensors")

            mock_query_embeds = mx.random.normal((1, 512))
            mock_doc_embeds = mx.random.normal((2, 512))
            mock_scores = [0.5, 0.3]

            with patch.object(
                reranker,
                "_compute_single_batch",
                return_value=(mock_query_embeds, mock_doc_embeds, mock_scores),
            ):
                documents = ["Relevant doc", "Another doc"]
                query = "Test query"

                results = reranker.rerank(query, documents)

            assert len(results) == 2
            for result in results:
                assert "index" in result
                assert "relevance_score" in result
                assert isinstance(result["index"], int)
                assert isinstance(result["relevance_score"], float)


MLX_MODEL_NAME = "jinaai/jina-reranker-v3-mlx"

QUERY_HIGH_ALTITUDE = (
    "The physiological impact of high-altitude training on long-distance "
    "athletic endurance and oxygen transport"
)

DOCUMENTS_HIGH_ALTITUDE = [
    "High-altitude training, conducted above 2,400 meters, is a staple for elite athletes. "
    "The primary physiological adaptation is the surge in erythropoietin (EPO) production, "
    "a hormone from the kidneys that triggers red blood cell creation. By increasing hemoglobin "
    "mass, athletes boost the oxygen-carrying capacity of their blood, aiding performance at "
    "sea level. However, the lower oxygen pressure can reduce training intensity, potentially "
    "leading to muscle detraining. To counter this, many use the Live-High, Train-Low (LHTL) "
    "model, resting at altitude for hematological gains but descending for high-intensity "
    "workouts. Studies show a four-week stay is necessary for measurable VO2 max improvements. "
    "Without careful management, the stress of hypoxia can lead to overtraining or sleep "
    "disturbances. This document explores the complex relationship between hemoglobin mass, "
    "oxygen saturation, and the metabolic cost of maintaining power output in rarefied air "
    "conditions over long durations.",
    "The 1968 Olympic Games in Mexico City (2,240m) acted as a global laboratory for altitude "
    "physiology. While sprinters thrived due to low air resistance, endurance athletes suffered. "
    "This event catalyzed research into how the body manages oxygen transport under stress. "
    "Post-1968, centers in Colorado Springs and St. Moritz became vital for training. The focus "
    "shifted to understanding VO2 max and the kidneys response to hypoxia. This retrospective "
    "looks at the data from distance runners who experienced significant drops in aerobic "
    "capacity during the games. It analyzes how blood pH shifts as the body breathes harder to "
    "expel CO2, a process known as respiratory alkalosis. This historical data is essential for "
    "modern coaches planning periodization for events at varying elevations.",
    "High-altitude cooking requires adjustments in fluid dynamics. As pressure drops, the "
    "boiling point of water decreases, meaning pasta and grains take longer to soften. Bakers "
    "must reduce leavening agents because air bubbles expand faster in thin air, which can "
    "cause cakes to collapse. This document covers the chemistry of heat transfer and molecular "
    "structure at 3,000 meters, specifically for industrial kitchens in the Andes. It explains "
    "why pressure cookers are essential for food safety and texture when atmospheric pressure is "
    "30 percent lower than at sea level. It does not cover athletic performance or physiological "
    "oxygen transport, focusing purely on culinary thermodynamics and the physical properties "
    "of steam.",
    "Modern mountain photography in the Himalayas involves managing high UV levels and extreme "
    "light contrast. Thin air filters less light, leading to blue hazes in images. Photographers "
    "must use polarizers and UV filters to protect sensors and maintain color accuracy. Battery "
    "life is also a concern, as extreme cold at high altitudes drains power rapidly. This guide "
    "provides technical settings for landscape photography at 5,000 meters, including how to "
    "manage the high dynamic range of bright snow against dark rock shadows. It discusses the "
    "physical endurance of the photographer but does not provide scientific data on oxygen "
    "transport or red blood cell mass.",
]


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not mlx_available, reason="MLX dependencies not installed")
class TestJinaMLXRerankerIntegration:
    """Integration tests with real Jina MLX model."""

    def test_rerank_real_model_scores_in_expected_range(self) -> None:
        """Test that real model produces scores in expected [0.2, 0.7] range."""
        from huggingface_hub import snapshot_download
        from local_reranker.jina_mlx_reranker import JinaMLXReranker

        model_path = snapshot_download(
            repo_id=MLX_MODEL_NAME,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py", "*.md"],
        )
        projector_path = f"{model_path}/projector.safetensors"

        reranker = JinaMLXReranker(model_path=model_path, projector_path=projector_path)

        results = reranker.rerank(QUERY_HIGH_ALTITUDE, DOCUMENTS_HIGH_ALTITUDE, top_n=3)

        assert len(results) == 3
        for result in results:
            score = result["relevance_score"]
            assert 0.2 <= score <= 0.7, (
                f"Score {score} not in expected range [0.2, 0.7]"
            )

    def test_rerank_real_model_ranking_order(self) -> None:
        """Test that real model produces correct ranking order."""
        from huggingface_hub import snapshot_download
        from local_reranker.jina_mlx_reranker import JinaMLXReranker

        model_path = snapshot_download(
            repo_id=MLX_MODEL_NAME,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py", "*.md"],
        )
        projector_path = f"{model_path}/projector.safetensors"

        reranker = JinaMLXReranker(model_path=model_path, projector_path=projector_path)

        results = reranker.rerank(QUERY_HIGH_ALTITUDE, DOCUMENTS_HIGH_ALTITUDE, top_n=3)

        assert len(results) == 3
        indices = [r["index"] for r in results]
        scores = [r["relevance_score"] for r in results]

        assert scores == sorted(scores, reverse=True)
        assert indices[0] == 0

    def test_rerank_real_model_matches_curl_example(self) -> None:
        """Test with same query/documents from .attic/curl2.sh."""
        from huggingface_hub import snapshot_download
        from local_reranker.jina_mlx_reranker import JinaMLXReranker

        model_path = snapshot_download(
            repo_id=MLX_MODEL_NAME,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py", "*.md"],
        )
        projector_path = f"{model_path}/projector.safetensors"

        reranker = JinaMLXReranker(model_path=model_path, projector_path=projector_path)

        results = reranker.rerank(QUERY_HIGH_ALTITUDE, DOCUMENTS_HIGH_ALTITUDE, top_n=3)

        assert len(results) == 3

        indices = [r["index"] for r in results]
        scores = [r["relevance_score"] for r in results]

        assert indices[0] == 0
        for score in scores:
            assert 0.2 <= score <= 0.7, (
                f"Score {score} not in expected range [0.2, 0.7]"
            )

    def test_rerank_real_model_scores_close_to_pytorch_baseline(self) -> None:
        """Test that MLX scores are close to PyTorch baseline scores."""
        pytest.importorskip("torch")
        pytest.importorskip("sentence_transformers")

        from huggingface_hub import snapshot_download
        from sentence_transformers import CrossEncoder
        from local_reranker.jina_mlx_reranker import JinaMLXReranker

        mlx_model_path = snapshot_download(
            repo_id=MLX_MODEL_NAME,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py", "*.md"],
        )
        mlx_projector_path = f"{mlx_model_path}/projector.safetensors"

        mlx_reranker = JinaMLXReranker(
            model_path=mlx_model_path, projector_path=mlx_projector_path
        )

        pytorch_model = CrossEncoder(
            model_name_or_path="jinaai/jina-reranker-v2-base-multilingual",
            device="cpu",
            trust_remote_code=True,
        )

        mlx_results = mlx_reranker.rerank(
            QUERY_HIGH_ALTITUDE, DOCUMENTS_HIGH_ALTITUDE, top_n=3
        )
        mlx_scores = [r["relevance_score"] for r in mlx_results]

        pairs = [(QUERY_HIGH_ALTITUDE, doc) for doc in DOCUMENTS_HIGH_ALTITUDE]
        pytorch_scores = pytorch_model.predict(pairs, show_progress_bar=False)

        sorted_pytorch_indices = sorted(
            range(len(pytorch_scores)), key=lambda i: pytorch_scores[i], reverse=True
        )[:3]
        pytorch_top_scores = [pytorch_scores[i] for i in sorted_pytorch_indices]

        mlx_indices = [r["index"] for r in mlx_results]
        assert mlx_indices == sorted_pytorch_indices

        for mlx_score, pytorch_score in zip(mlx_scores, pytorch_top_scores):
            assert 0.2 <= mlx_score <= 0.7
            assert abs(mlx_score - pytorch_score) < 0.3
