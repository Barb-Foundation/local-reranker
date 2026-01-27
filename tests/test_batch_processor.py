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
)
from local_reranker.models import RerankRequest


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
