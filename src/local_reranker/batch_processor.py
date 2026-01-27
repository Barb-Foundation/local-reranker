# -*- coding: utf-8 -*-
"""Shared batch processing logic for reranker implementations."""

from typing import List, Dict, Any, Optional, Union
from typing_extensions import Protocol
import logging

from .models import RerankRequest, RerankResult, RerankDocument
from .batch_manager import BatchManager
from .result_aggregator import ResultAggregator

logger = logging.getLogger(__name__)


class ModelPredictor(Protocol):
    """Protocol for model prediction functions."""

    def __call__(
        self, query: str, documents: List[str], return_documents: Optional[bool]
    ) -> List[Dict[str, Any]]:
        """Predict scores for documents given a query.

        Args:
            query: The query text
            documents: List of document texts
            return_documents: Whether to include document content in results

        Returns:
            List of result dictionaries with keys: index, relevance_score, document (optional)
        """
        ...


class DocumentTextExtractor:
    """Utility for extracting text from various document formats."""

    COMMON_TEXT_FIELDS = ["text", "content", "document", "body", "message"]

    @classmethod
    def extract(
        cls,
        doc: Union[str, Dict[str, Any]],
        text_fields: Optional[List[str]] = None,
        default: Optional[str] = "",
    ) -> Optional[str]:
        """Extract text from a document (string or dict).

        Args:
            doc: Document as string or dict
            text_fields: List of field names to try (defaults to common fields)
            default: Default value if no text found

        Returns:
            Document text as string, or default if not found
        """
        if isinstance(doc, str):
            return doc.strip() if doc.strip() else default

        if not isinstance(doc, dict):
            return default

        fields_to_try = text_fields or cls.COMMON_TEXT_FIELDS

        for field in fields_to_try:
            if field in doc:
                text = doc[field]
                if isinstance(text, str) and text.strip():
                    return text.strip()

        return default

    @classmethod
    def extract_batch(
        cls,
        documents: List[Union[str, Dict[str, Any]]],
        text_fields: Optional[List[str]] = None,
        default: Optional[str] = "",
        skip_invalid: bool = True,
    ) -> List[str]:
        """Extract text from a batch of documents.

        Args:
            documents: List of documents (strings or dicts)
            text_fields: List of field names to try
            default: Default value for invalid documents
            skip_invalid: If True, skip invalid documents; if False, use default

        Returns:
            List of extracted document texts
        """
        results = []
        for i, doc in enumerate(documents):
            text = cls.extract(doc, text_fields, default)
            if text or not skip_invalid:
                if text is not None:
                    results.append(text)
                else:
                    logger.debug(f"Skipping invalid document at index {i}")
            else:
                logger.debug(f"Skipping invalid document at index {i}")
        return results


def extract_document_text(doc: Union[str, Dict[str, Any]]) -> str:
    """Extract text from a document (string or dict).

    Args:
        doc: Document as string or dict with 'text' key

    Returns:
        Document text as string
    """
    result = DocumentTextExtractor.extract(doc, text_fields=["text"], default="")
    return result or ""


def create_rerank_result(
    result_dict: Dict[str, Any],
    original_idx: int,
    return_documents: Optional[bool],
    batch_docs: List[str],
) -> Optional[RerankResult]:
    """Create a RerankResult from a model result dictionary.

    Args:
        result_dict: Result dictionary from model
        original_idx: Original document index
        return_documents: Whether to include document content
        batch_docs: List of documents in the current batch

    Returns:
        RerankResult or None if invalid
    """
    try:
        relevance_score = float(result_dict["relevance_score"])
        batch_relative_index = int(result_dict["index"])

        if batch_relative_index < 0 or batch_relative_index >= len(batch_docs):
            logger.warning(
                f"Invalid batch-relative index {batch_relative_index} in batch result"
            )
            return None

        document = None
        if return_documents:
            doc_text = batch_docs[batch_relative_index]
            if doc_text:
                document = RerankDocument(text=doc_text)

        return RerankResult(
            document=document,
            index=original_idx,
            relevance_score=relevance_score,
        )

    except (KeyError, ValueError, TypeError) as e:
        logger.error(f"Error creating rerank result: {e}")
        return None


def create_placeholder_result(
    batch_relative_index: int,
    original_idx: int,
    return_documents: Optional[bool],
    batch_docs: List[str],
) -> RerankResult:
    """Create a placeholder result for failed processing.

    Args:
        batch_relative_index: Index within the batch
        original_idx: Original document index
        return_documents: Whether to include document content
        batch_docs: List of documents in the current batch

    Returns:
        RerankResult with neutral score
    """
    document = None
    if return_documents and batch_relative_index < len(batch_docs):
        doc_text = batch_docs[batch_relative_index]
        if doc_text:
            document = RerankDocument(text=doc_text)

    return RerankResult(
        document=document,
        index=original_idx,
        relevance_score=0.0,
    )


def process_batches(
    request: RerankRequest,
    batch_manager: BatchManager,
    model_predictor: ModelPredictor,
    backend_name: str,
    count_non_empty: bool = False,
) -> List[RerankResult]:
    """Process documents in batches using the provided model predictor.

    Args:
        request: Rerank request containing query, documents, and options
        batch_manager: Batch manager for creating batches
        model_predictor: Function that processes a batch of documents
        backend_name: Name of the backend for logging (e.g., "PyTorch", "MLX")
        count_non_empty: If True, count only non-empty documents

    Returns:
        List of rerank results, sorted by relevance score (descending)
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"[{backend_name}] Starting batched rerank request")
        logger.debug(f"[{backend_name}] Query: {request.query}")
        logger.debug(f"[{backend_name}] Number of documents: {len(request.documents)}")
        logger.debug(f"[{backend_name}] Batch size: {batch_manager.batch_size}")

    if not request.documents:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[{backend_name}] No documents to rerank")
        return []

    batches, batch_indices = batch_manager.create_batches(request)

    if not batches:
        logger.warning(f"[{backend_name}] No batches created")
        return []

    aggregator = ResultAggregator()

    if count_non_empty:
        non_empty_count = sum(
            1 for doc in request.documents if doc and extract_document_text(doc)
        )
        aggregator.set_total_document_count(non_empty_count)
    else:
        aggregator.set_total_document_count(len(request.documents))

    for batch_idx, (batch_docs, original_indices) in enumerate(
        zip(batches, batch_indices)
    ):
        logger.info(
            f"[{backend_name}] Processing batch {batch_idx + 1}/{len(batches)}: "
            f"{len(batch_docs)} documents"
        )

        try:
            batch_results = model_predictor(
                query=request.query,
                documents=batch_docs,
                return_documents=request.return_documents or False,
            )

            if not batch_results:
                logger.warning(f"[{backend_name}] No results for batch {batch_idx + 1}")
                continue

            if len(batch_results) != len(batch_docs):
                logger.error(
                    f"[{backend_name}] Result count mismatch in batch {batch_idx + 1}: "
                    f"expected {len(batch_docs)} results, got {len(batch_results)}"
                )
                raise ValueError(
                    f"Model returned {len(batch_results)} results for {len(batch_docs)} documents"
                )

            formatted_results = []
            for batch_relative_index, result_dict in enumerate(batch_results):
                result = create_rerank_result(
                    result_dict=result_dict,
                    original_idx=original_indices[batch_relative_index]
                    if batch_relative_index < len(original_indices)
                    else original_indices[-1],
                    return_documents=request.return_documents or False,
                    batch_docs=batch_docs,
                )
                if result:
                    formatted_results.append(result)

            aggregator.add_batch_results(formatted_results, original_indices)
            logger.info(
                f"[{backend_name}] Batch {batch_idx + 1} completed: "
                f"{len(formatted_results)} results"
            )

        except ValueError as value_error:
            logger.error(
                f"[{backend_name}] Batch {batch_idx + 1} validation failed: {value_error}"
            )
            return []
        except Exception as batch_error:
            logger.error(
                f"[{backend_name}] Batch {batch_idx + 1} failed: {batch_error}"
            )
            raise batch_error

    final_results = aggregator.get_sorted_results(request.top_n)

    stats = aggregator.get_batch_statistics()
    logger.info(
        f"[{backend_name}] Batched reranking completed: {len(final_results)} final results, "
        f"{stats.get('total_batches', 0)} batches, "
        f"completion_rate={stats.get('completion_rate', 0):.2%}"
    )

    return final_results
