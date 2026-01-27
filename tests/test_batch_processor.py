# -*- coding: utf-8 -*-
"""Tests for batch processor utility module."""

from typing import Optional
from unittest.mock import Mock, patch

from local_reranker.batch_processor import (
    DocumentTextExtractor,
    extract_document_text,
    create_rerank_result,
    create_placeholder_result,
    ModelPredictor,
    process_batches,
    BatchProcessor,
)
from local_reranker.models import RerankRequest, RerankResult, RerankDocument


class TestDocumentTextExtractor:
    """Test DocumentTextExtractor utility class."""

    def test_extract_from_string(self):
        """Test extracting text from string document."""
        text = "This is a test document"
        result = DocumentTextExtractor.extract(text)
        assert result == text

    def test_extract_from_empty_string(self):
        """Test extracting from empty string."""
        result = DocumentTextExtractor.extract("")
        assert result == ""

    def test_extract_from_whitespace_string(self):
        """Test extracting from whitespace-only string."""
        result = DocumentTextExtractor.extract("   ")
        assert result == ""

    def test_extract_from_dict_with_text_field(self):
        """Test extracting from dict with 'text' field."""
        doc = {"text": "Document content"}
        result = DocumentTextExtractor.extract(doc)
        assert result == "Document content"

    def test_extract_from_dict_with_content_field(self):
        """Test extracting from dict with 'content' field."""
        doc = {"content": "Document content"}
        result = DocumentTextExtractor.extract(doc)
        assert result == "Document content"

    def test_extract_from_dict_with_custom_field(self):
        """Test extracting from dict with custom field."""
        doc = {"body": "Document content"}
        result = DocumentTextExtractor.extract(doc, text_fields=["body"])
        assert result == "Document content"

    def test_extract_from_dict_without_text_fields(self):
        """Test extracting from dict without matching fields."""
        doc = {"id": 123, "metadata": {}}
        result = DocumentTextExtractor.extract(doc)
        assert result == ""

    def test_extract_from_dict_with_default(self):
        """Test extracting from dict with custom default."""
        doc = {"id": 123}
        result = DocumentTextExtractor.extract(doc, default="default_text")
        assert result == "default_text"

    def test_extract_from_invalid_type(self):
        """Test extracting from invalid type."""
        result = DocumentTextExtractor.extract(123)  # type: ignore[arg-type]
        assert result == ""

    def test_extract_from_dict_with_whitespace_text(self):
        """Test extracting from dict with whitespace text."""
        doc = {"text": "   "}
        result = DocumentTextExtractor.extract(doc)
        assert result == ""

    def test_extract_with_multiple_fields_first_match(self):
        """Test extracting with multiple fields, first matches."""
        doc = {"text": "first", "content": "second"}
        result = DocumentTextExtractor.extract(doc, text_fields=["text", "content"])
        assert result == "first"

    def test_extract_with_multiple_fields_second_match(self):
        """Test extracting with multiple fields, second matches."""
        doc = {"content": "second", "body": "third"}
        result = DocumentTextExtractor.extract(
            doc, text_fields=["text", "content", "body"]
        )
        assert result == "second"

    def test_extract_batch_from_strings(self):
        """Test extracting from batch of strings."""
        docs = ["doc1", "doc2", "doc3"]
        results = DocumentTextExtractor.extract_batch(docs)  # type: ignore[arg-type]
        assert results == docs

    def test_extract_batch_from_dicts(self):
        """Test extracting from batch of dicts."""
        docs = [{"text": "doc1"}, {"text": "doc2"}, {"text": "doc3"}]
        results = DocumentTextExtractor.extract_batch(docs)  # type: ignore[arg-type]
        assert results == ["doc1", "doc2", "doc3"]

    def test_extract_batch_from_mixed(self):
        """Test extracting from batch of mixed types."""
        docs = ["doc1", {"text": "doc2"}, {"content": "doc3"}]
        results = DocumentTextExtractor.extract_batch(docs)
        assert results == ["doc1", "doc2", "doc3"]

    def test_extract_batch_skip_invalid(self):
        """Test extracting from batch with invalid docs (skip)."""
        docs = ["doc1", {"id": 123}, "doc3"]
        results = DocumentTextExtractor.extract_batch(docs, skip_invalid=True)
        assert results == ["doc1", "doc3"]

    def test_extract_batch_keep_invalid(self):
        """Test extracting from batch with invalid docs (keep)."""
        docs = ["doc1", {"id": 123}, "doc3"]
        results = DocumentTextExtractor.extract_batch(
            docs, skip_invalid=False, default=""
        )
        assert len(results) == 3
        assert results[0] == "doc1"
        assert results[1] == ""
        assert results[2] == "doc3"


class TestExtractDocumentText:
    """Test backward-compatible extract_document_text function."""

    def test_extract_from_string(self):
        """Test extracting from string."""
        text = "Test document"
        result = extract_document_text(text)
        assert result == text

    def test_extract_from_dict_with_text_key(self):
        """Test extracting from dict with 'text' key."""
        doc = {"text": "Test document"}
        result = extract_document_text(doc)
        assert result == "Test document"

    def test_extract_from_dict_without_text_key(self):
        """Test extracting from dict without 'text' key."""
        doc = {"content": "Test document"}
        result = extract_document_text(doc)
        assert result == ""

    def test_extract_from_invalid_type(self):
        """Test extracting from invalid type."""
        result = extract_document_text(123)  # type: ignore[arg-type]
        assert result == ""


class TestCreateRerankResult:
    """Test create_rerank_result function."""

    def test_create_valid_result(self):
        """Test creating a valid rerank result."""
        result_dict = {"index": 0, "relevance_score": 0.95}
        batch_docs = ["Document 1"]
        result = create_rerank_result(result_dict, 0, False, batch_docs)
        assert result is not None
        assert result.index == 0
        assert result.relevance_score == 0.95
        assert result.document is None

    def test_create_result_with_document(self):
        """Test creating a result with document content."""
        result_dict = {"index": 0, "relevance_score": 0.95}
        batch_docs = ["Document 1"]
        result = create_rerank_result(result_dict, 0, True, batch_docs)
        assert result is not None
        assert result.index == 0
        assert result.relevance_score == 0.95
        assert result.document is not None
        assert result.document.text == "Document 1"

    def test_create_result_with_invalid_index(self):
        """Test creating a result with invalid index."""
        result_dict = {"index": 999, "relevance_score": 0.95}
        batch_docs = ["Document 1"]
        result = create_rerank_result(result_dict, 0, False, batch_docs)
        assert result is None

    def test_create_result_with_negative_index(self):
        """Test creating a result with negative index."""
        result_dict = {"index": -1, "relevance_score": 0.95}
        batch_docs = ["Document 1"]
        result = create_rerank_result(result_dict, 0, False, batch_docs)
        assert result is None

    def test_create_result_missing_index_key(self):
        """Test creating a result with missing index key."""
        result_dict = {"relevance_score": 0.95}
        batch_docs = ["Document 1"]
        result = create_rerank_result(result_dict, 0, False, batch_docs)
        assert result is None

    def test_create_result_missing_score_key(self):
        """Test creating a result with missing score key."""
        result_dict = {"index": 0}
        batch_docs = ["Document 1"]
        result = create_rerank_result(result_dict, 0, False, batch_docs)
        assert result is None

    def test_create_result_with_invalid_score(self):
        """Test creating a result with invalid score."""
        result_dict = {"index": 0, "relevance_score": "invalid"}
        batch_docs = ["Document 1"]
        result = create_rerank_result(result_dict, 0, False, batch_docs)
        assert result is None

    def test_create_result_with_empty_document(self):
        """Test creating a result with empty document text."""
        result_dict = {"index": 0, "relevance_score": 0.95}
        batch_docs = [""]
        result = create_rerank_result(result_dict, 0, True, batch_docs)
        assert result is not None
        assert result.index == 0
        assert result.relevance_score == 0.95
        assert result.document is None


class TestCreatePlaceholderResult:
    """Test create_placeholder_result function."""

    def test_create_placeholder_without_document(self):
        """Test creating a placeholder without document."""
        batch_docs = ["Document 1"]
        result = create_placeholder_result(0, 5, False, batch_docs)
        assert result.index == 5
        assert result.relevance_score == 0.0
        assert result.document is None

    def test_create_placeholder_with_document(self):
        """Test creating a placeholder with document."""
        batch_docs = ["Document 1"]
        result = create_placeholder_result(0, 5, True, batch_docs)
        assert result.index == 5
        assert result.relevance_score == 0.0
        assert result.document is not None
        assert result.document.text == "Document 1"

    def test_create_placeholder_out_of_range(self):
        """Test creating a placeholder with out-of-range index."""
        batch_docs = ["Document 1"]
        result = create_placeholder_result(999, 5, True, batch_docs)
        assert result.index == 5
        assert result.relevance_score == 0.0
        assert result.document is None


class TestProcessBatches:
    """Test process_batches function."""

    @patch("local_reranker.batch_processor.BatchManager")
    @patch("local_reranker.batch_processor.ResultAggregator")
    def test_process_batches_empty_documents(
        self, mock_aggregator_class, mock_batch_manager_class
    ):
        """Test processing with empty documents."""
        mock_batch_manager = Mock()
        mock_batch_manager.create_batches.return_value = ([], [])
        mock_batch_manager_class.return_value = mock_batch_manager

        request = RerankRequest(query="test", documents=[])

        results = process_batches(
            request=request,
            batch_manager=mock_batch_manager,
            model_predictor=Mock(),
            backend_name="Test",
        )

        assert results == []

    @patch("local_reranker.batch_processor.BatchManager")
    @patch("local_reranker.batch_processor.ResultAggregator")
    def test_process_batches_no_batches_created(
        self, mock_aggregator_class, mock_batch_manager_class
    ):
        """Test processing when no batches are created."""
        mock_batch_manager = Mock()
        mock_batch_manager.create_batches.return_value = ([], [])
        mock_batch_manager_class.return_value = mock_batch_manager

        request = RerankRequest(query="test", documents=["doc1", "doc2"])

        results = process_batches(
            request=request,
            batch_manager=mock_batch_manager,
            model_predictor=Mock(),
            backend_name="Test",
        )

        assert results == []

    @patch("local_reranker.batch_processor.BatchManager")
    @patch("local_reranker.batch_processor.ResultAggregator")
    def test_process_batches_success(
        self, mock_aggregator_class, mock_batch_manager_class
    ):
        """Test successful batch processing."""
        mock_batch_manager = Mock()
        mock_batch_manager.create_batches.return_value = (
            [["doc1", "doc2"]],
            [[0, 1]],
        )
        mock_batch_manager_class.return_value = mock_batch_manager

        mock_aggregator = Mock()
        mock_aggregator.get_sorted_results.return_value = [
            Mock(index=0, relevance_score=0.9),
            Mock(index=1, relevance_score=0.7),
        ]
        mock_aggregator.get_batch_statistics.return_value = {
            "total_batches": 1,
            "completion_rate": 1.0,
        }
        mock_aggregator_class.return_value = mock_aggregator

        def mock_predictor(
            query: str, documents: list, return_documents: Optional[bool]
        ):
            return [
                {"index": i, "relevance_score": 0.9 - i * 0.2}
                for i in range(len(documents))
            ]

        request = RerankRequest(query="test", documents=["doc1", "doc2"])

        results = process_batches(
            request=request,
            batch_manager=mock_batch_manager,
            model_predictor=mock_predictor,
            backend_name="Test",
        )

        assert len(results) == 2
        mock_aggregator.add_batch_results.assert_called_once()
        mock_aggregator.set_total_document_count.assert_called_once()
        mock_aggregator.get_sorted_results.assert_called_once()

    @patch("local_reranker.batch_processor.BatchManager")
    @patch("local_reranker.batch_processor.ResultAggregator")
    def test_process_batches_with_top_n(
        self, mock_aggregator_class, mock_batch_manager_class
    ):
        """Test batch processing with top_n limit."""
        mock_batch_manager = Mock()
        mock_batch_manager.create_batches.return_value = (
            [["doc1", "doc2"]],
            [[0, 1]],
        )
        mock_batch_manager_class.return_value = mock_batch_manager

        mock_aggregator = Mock()
        mock_aggregator.get_sorted_results.return_value = [
            Mock(index=0, relevance_score=0.9),
        ]
        mock_aggregator.get_batch_statistics.return_value = {
            "total_batches": 1,
            "completion_rate": 1.0,
        }
        mock_aggregator_class.return_value = mock_aggregator

        def mock_predictor(
            query: str, documents: list, return_documents: Optional[bool]
        ):
            return [
                {"index": i, "relevance_score": 0.9 - i * 0.2}
                for i in range(len(documents))
            ]

        request = RerankRequest(query="test", documents=["doc1", "doc2"], top_n=1)

        results = process_batches(
            request=request,
            batch_manager=mock_batch_manager,
            model_predictor=mock_predictor,
            backend_name="Test",
        )

        assert len(results) == 1
        mock_aggregator.get_sorted_results.assert_called_once_with(1)

    @patch("local_reranker.batch_processor.BatchManager")
    @patch("local_reranker.batch_processor.ResultAggregator")
    def test_process_batches_result_count_mismatch(
        self, mock_aggregator_class, mock_batch_manager_class
    ):
        """Test batch processing with result count mismatch."""
        mock_batch_manager = Mock()
        mock_batch_manager.create_batches.return_value = (
            [["doc1", "doc2"]],
            [[0, 1]],
        )
        mock_batch_manager_class.return_value = mock_batch_manager

        mock_aggregator = Mock()
        mock_aggregator_class.return_value = mock_aggregator

        def mock_predictor(
            query: str, documents: list, return_documents: Optional[bool]
        ):
            return [{"index": 0, "relevance_score": 0.9}]

        request = RerankRequest(query="test", documents=["doc1", "doc2"])

        results = process_batches(
            request=request,
            batch_manager=mock_batch_manager,
            model_predictor=mock_predictor,
            backend_name="Test",
        )

        assert results == []

    @patch("local_reranker.batch_processor.BatchManager")
    @patch("local_reranker.batch_processor.ResultAggregator")
    def test_process_batches_with_count_non_empty(
        self, mock_aggregator_class, mock_batch_manager_class
    ):
        """Test batch processing with count_non_empty flag."""
        mock_batch_manager = Mock()
        mock_batch_manager.create_batches.return_value = (
            [["doc1", "doc2"]],
            [[0, 1]],
        )
        mock_batch_manager_class.return_value = mock_batch_manager

        mock_aggregator = Mock()
        mock_aggregator.get_sorted_results.return_value = [
            Mock(index=0, relevance_score=0.9),
            Mock(index=1, relevance_score=0.7),
        ]
        mock_aggregator.get_batch_statistics.return_value = {
            "total_batches": 1,
            "completion_rate": 1.0,
        }
        mock_aggregator_class.return_value = mock_aggregator

        def mock_predictor(
            query: str, documents: list, return_documents: Optional[bool]
        ):
            return [
                {"index": i, "relevance_score": 0.9 - i * 0.2}
                for i in range(len(documents))
            ]

        request = RerankRequest(query="test", documents=["doc1", "doc2"])

        results = process_batches(
            request=request,
            batch_manager=mock_batch_manager,
            model_predictor=mock_predictor,
            backend_name="Test",
            count_non_empty=True,
        )

        assert len(results) == 2
        mock_aggregator.set_total_document_count.assert_called_once_with(2)

    @patch("local_reranker.batch_processor.BatchManager")
    @patch("local_reranker.batch_processor.ResultAggregator")
    def test_process_batches_count_non_empty_with_empty_docs(
        self, mock_aggregator_class, mock_batch_manager_class
    ):
        """Test batch processing with count_non_empty and empty documents."""
        mock_batch_manager = Mock()
        mock_batch_manager.create_batches.return_value = (
            [["doc1", "", "doc3"]],
            [[0, 1, 2]],
        )
        mock_batch_manager_class.return_value = mock_batch_manager

        mock_aggregator = Mock()
        mock_aggregator.get_sorted_results.return_value = []
        mock_aggregator.get_batch_statistics.return_value = {
            "total_batches": 1,
            "completion_rate": 0.0,
        }
        mock_aggregator_class.return_value = mock_aggregator

        def mock_predictor(
            query: str, documents: list, return_documents: Optional[bool]
        ):
            return []

        request = RerankRequest(query="test", documents=["doc1", "", "doc3"])

        process_batches(
            request=request,
            batch_manager=mock_batch_manager,
            model_predictor=mock_predictor,
            backend_name="Test",
            count_non_empty=True,
        )

        mock_aggregator.set_total_document_count.assert_called_once_with(2)

    @patch("local_reranker.batch_processor.BatchManager")
    @patch("local_reranker.batch_processor.ResultAggregator")
    def test_process_batches_multiple_batches(
        self, mock_aggregator_class, mock_batch_manager_class
    ):
        """Test processing multiple batches."""
        mock_batch_manager = Mock()
        mock_batch_manager.create_batches.return_value = (
            [["doc1", "doc2"], ["doc3", "doc4"]],
            [[0, 1], [2, 3]],
        )
        mock_batch_manager_class.return_value = mock_batch_manager

        mock_aggregator = Mock()
        mock_aggregator.get_sorted_results.return_value = [
            Mock(index=0, relevance_score=0.9),
            Mock(index=2, relevance_score=0.8),
            Mock(index=1, relevance_score=0.7),
            Mock(index=3, relevance_score=0.6),
        ]
        mock_aggregator.get_batch_statistics.return_value = {
            "total_batches": 2,
            "completion_rate": 1.0,
        }
        mock_aggregator_class.return_value = mock_aggregator

        call_count = [0]

        def mock_predictor(
            query: str, documents: list, return_documents: Optional[bool]
        ):
            call_idx = call_count[0]
            call_count[0] += 1
            return [
                {"index": i, "relevance_score": 0.9 - call_idx * 0.1 - i * 0.05}
                for i in range(len(documents))
            ]

        request = RerankRequest(
            query="test", documents=["doc1", "doc2", "doc3", "doc4"]
        )

        results = process_batches(
            request=request,
            batch_manager=mock_batch_manager,
            model_predictor=mock_predictor,
            backend_name="Test",
        )

        assert len(results) == 4
        assert call_count[0] == 2
        assert mock_aggregator.add_batch_results.call_count == 2

    @patch("local_reranker.batch_processor.BatchManager")
    @patch("local_reranker.batch_processor.ResultAggregator")
    def test_process_batches_empty_results_from_predictor(
        self, mock_aggregator_class, mock_batch_manager_class
    ):
        """Test batch processing when predictor returns empty results."""
        mock_batch_manager = Mock()
        mock_batch_manager.create_batches.return_value = (
            [["doc1", "doc2"]],
            [[0, 1]],
        )
        mock_batch_manager_class.return_value = mock_batch_manager

        mock_aggregator = Mock()
        mock_aggregator.get_sorted_results.return_value = []
        mock_aggregator.get_batch_statistics.return_value = {
            "total_batches": 1,
            "completion_rate": 0.0,
        }
        mock_aggregator_class.return_value = mock_aggregator

        def mock_predictor(
            query: str, documents: list, return_documents: Optional[bool]
        ):
            return []

        request = RerankRequest(query="test", documents=["doc1", "doc2"])

        results = process_batches(
            request=request,
            batch_manager=mock_batch_manager,
            model_predictor=mock_predictor,
            backend_name="Test",
        )

        assert len(results) == 0
        mock_aggregator.add_batch_results.assert_not_called()

    @patch("local_reranker.batch_processor.BatchManager")
    @patch("local_reranker.batch_processor.ResultAggregator")
    def test_process_batches_with_return_documents(
        self, mock_aggregator_class, mock_batch_manager_class
    ):
        """Test batch processing with return_documents=True."""
        mock_batch_manager = Mock()
        mock_batch_manager.create_batches.return_value = (
            [["doc1", "doc2"]],
            [[0, 1]],
        )
        mock_batch_manager_class.return_value = mock_batch_manager

        mock_aggregator = Mock()
        mock_aggregator.get_sorted_results.return_value = [
            Mock(index=0, relevance_score=0.9),
            Mock(index=1, relevance_score=0.7),
        ]
        mock_aggregator.get_batch_statistics.return_value = {
            "total_batches": 1,
            "completion_rate": 1.0,
        }
        mock_aggregator_class.return_value = mock_aggregator

        def mock_predictor(
            query: str, documents: list, return_documents: Optional[bool]
        ):
            assert return_documents is True
            return [
                {"index": i, "relevance_score": 0.9 - i * 0.2}
                for i in range(len(documents))
            ]

        request = RerankRequest(
            query="test", documents=["doc1", "doc2"], return_documents=True
        )

        results = process_batches(
            request=request,
            batch_manager=mock_batch_manager,
            model_predictor=mock_predictor,
            backend_name="Test",
        )

        assert len(results) == 2


class TestModelPredictorProtocol:
    """Test ModelPredictor protocol."""

    def test_predictor_is_protocol(self):
        """Test that ModelPredictor is a Protocol."""
        assert hasattr(ModelPredictor, "__protocol_attrs__")

    def test_predictor_function_complies(self):
        """Test that a function complies with ModelPredictor protocol."""

        def my_predictor(query: str, documents: list, return_documents: bool) -> list:
            return [{"index": i, "relevance_score": 0.5} for i in range(len(documents))]

        request = RerankRequest(query="test", documents=["doc1"])
        results = my_predictor(
            query=request.query, documents=request.documents, return_documents=False
        )
        assert len(results) == 1


class TestBatchProcessor:
    """Test BatchProcessor utility class."""

    def test_process_batched_results_empty(self):
        """Test processing empty batch results."""
        results = BatchProcessor.process_batched_results([])
        assert results == []

    def test_process_batched_results_single_batch(self):
        """Test processing single batch of results."""
        batch = [
            RerankResult(document=None, index=0, relevance_score=0.9),
            RerankResult(document=None, index=1, relevance_score=0.7),
        ]
        results = BatchProcessor.process_batched_results([batch])
        assert len(results) == 2
        assert results[0].index == 0
        assert results[0].relevance_score == 0.9
        assert results[1].index == 1
        assert results[1].relevance_score == 0.7

    def test_process_batched_results_multiple_batches(self):
        """Test processing multiple batches of results."""
        batch1 = [
            RerankResult(document=None, index=0, relevance_score=0.9),
            RerankResult(document=None, index=1, relevance_score=0.7),
        ]
        batch2 = [
            RerankResult(document=None, index=2, relevance_score=0.8),
            RerankResult(document=None, index=3, relevance_score=0.6),
        ]
        results = BatchProcessor.process_batched_results([batch1, batch2])
        assert len(results) == 4
        assert results[0].index == 0
        assert results[0].relevance_score == 0.9
        assert results[1].index == 2
        assert results[1].relevance_score == 0.8
        assert results[2].index == 1
        assert results[2].relevance_score == 0.7
        assert results[3].index == 3
        assert results[3].relevance_score == 0.6

    def test_process_batched_results_with_top_n(self):
        """Test processing batch results with top_n limit."""
        batch = [
            RerankResult(document=None, index=0, relevance_score=0.9),
            RerankResult(document=None, index=1, relevance_score=0.7),
            RerankResult(document=None, index=2, relevance_score=0.5),
        ]
        results = BatchProcessor.process_batched_results([batch], top_n=2)
        assert len(results) == 2
        assert results[0].relevance_score == 0.9
        assert results[1].relevance_score == 0.7

    def test_process_batched_results_without_top_n(self):
        """Test processing batch results without top_n limit."""
        batch = [
            RerankResult(document=None, index=0, relevance_score=0.9),
            RerankResult(document=None, index=1, relevance_score=0.7),
        ]
        results = BatchProcessor.process_batched_results([batch], top_n=None)
        assert len(results) == 2

    def test_process_batched_results_with_empty_batches(self):
        """Test processing batch results with empty batches."""
        batch1 = [
            RerankResult(document=None, index=0, relevance_score=0.9),
        ]
        batch2 = []
        batch3 = [
            RerankResult(document=None, index=1, relevance_score=0.7),
        ]
        results = BatchProcessor.process_batched_results([batch1, batch2, batch3])
        assert len(results) == 2
        assert results[0].relevance_score == 0.9
        assert results[1].relevance_score == 0.7

    def test_process_batched_results_variable_batch_sizes(self):
        """Test processing batches with different sizes."""
        batch1 = [
            RerankResult(document=None, index=0, relevance_score=0.9),
        ]
        batch2 = [
            RerankResult(document=None, index=1, relevance_score=0.8),
            RerankResult(document=None, index=2, relevance_score=0.7),
        ]
        batch3 = [
            RerankResult(document=None, index=3, relevance_score=0.6),
            RerankResult(document=None, index=4, relevance_score=0.5),
            RerankResult(document=None, index=5, relevance_score=0.4),
        ]
        results = BatchProcessor.process_batched_results([batch1, batch2, batch3])
        assert len(results) == 6
        assert results[0].relevance_score == 0.9
        assert results[-1].relevance_score == 0.4

    def test_process_batched_results_with_documents(self):
        """Test processing batch results with document content."""
        batch = [
            RerankResult(
                document=RerankDocument(text="doc1"), index=0, relevance_score=0.9
            ),
            RerankResult(
                document=RerankDocument(text="doc2"), index=1, relevance_score=0.7
            ),
        ]
        results = BatchProcessor.process_batched_results([batch])
        assert len(results) == 2
        assert results[0].document is not None
        assert results[0].document.text == "doc1"
        assert results[1].document is not None
        assert results[1].document.text == "doc2"

    def test_process_batched_results_mixed_documents(self):
        """Test processing batch results with mixed document presence."""
        batch = [
            RerankResult(
                document=RerankDocument(text="doc1"), index=0, relevance_score=0.9
            ),
            RerankResult(document=None, index=1, relevance_score=0.8),
            RerankResult(
                document=RerankDocument(text="doc3"), index=2, relevance_score=0.7
            ),
        ]
        results = BatchProcessor.process_batched_results([batch])
        assert len(results) == 3
        assert results[0].document is not None
        assert results[1].document is None
        assert results[2].document is not None

    def test_process_batched_results_negative_scores(self):
        """Test processing batch results with negative scores."""
        batch = [
            RerankResult(document=None, index=0, relevance_score=0.5),
            RerankResult(document=None, index=1, relevance_score=-0.2),
            RerankResult(document=None, index=2, relevance_score=0.3),
        ]
        results = BatchProcessor.process_batched_results([batch])
        assert len(results) == 3
        assert results[0].relevance_score == 0.5
        assert results[1].relevance_score == 0.3
        assert results[2].relevance_score == -0.2

    def test_process_batched_results_zero_scores(self):
        """Test processing batch results with zero scores."""
        batch = [
            RerankResult(document=None, index=0, relevance_score=0.0),
            RerankResult(document=None, index=1, relevance_score=0.0),
        ]
        results = BatchProcessor.process_batched_results([batch])
        assert len(results) == 2
        assert all(r.relevance_score == 0.0 for r in results)

    def test_process_batched_results_equal_scores(self):
        """Test processing batch results with equal scores (stable sort)."""
        batch = [
            RerankResult(document=None, index=0, relevance_score=0.5),
            RerankResult(document=None, index=1, relevance_score=0.5),
            RerankResult(document=None, index=2, relevance_score=0.5),
        ]
        results = BatchProcessor.process_batched_results([batch])
        assert len(results) == 3
        assert all(r.relevance_score == 0.5 for r in results)

    def test_process_batched_results_top_n_greater_than_total(self):
        """Test processing with top_n greater than total results."""
        batch = [
            RerankResult(document=None, index=0, relevance_score=0.9),
            RerankResult(document=None, index=1, relevance_score=0.7),
        ]
        results = BatchProcessor.process_batched_results([batch], top_n=10)
        assert len(results) == 2

    def test_process_batched_results_top_n_zero(self):
        """Test processing with top_n=0."""
        batch = [
            RerankResult(document=None, index=0, relevance_score=0.9),
        ]
        results = BatchProcessor.process_batched_results([batch], top_n=0)
        assert len(results) == 0

    def test_process_batched_results_large_number_of_batches(self):
        """Test processing many small batches."""
        batches = [
            [
                RerankResult(document=None, index=i, relevance_score=0.9 - i * 0.1),
            ]
            for i in range(10)
        ]
        results = BatchProcessor.process_batched_results(batches)
        assert len(results) == 10
        assert results[0].relevance_score == 0.9
        assert results[-1].relevance_score == 0.0
