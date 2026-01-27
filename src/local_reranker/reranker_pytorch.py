# -*- coding: utf-8 -*-
"""PyTorch implementation of the reranker protocol."""

import logging
from typing import List, Union, Dict, Any, Optional, Tuple, AsyncGenerator
from typing_extensions import override
import asyncio
import time

import torch
from sentence_transformers import CrossEncoder
from .reranker import Reranker as RerankerProtocol
from .models import RerankRequest, RerankResult, RerankDocument, ProgressUpdate
from .batch_manager import BatchManager
from .result_aggregator import ResultAggregator

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- Default Model ---
# Default model is now managed in config.py
# This constant is kept for backward compatibility
DEFAULT_MODEL_NAME = "jinaai/jina-reranker-v2-base-multilingual"


class Reranker(RerankerProtocol):
    """PyTorch implementation of the reranker protocol using CrossEncoder."""

    @override
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_concurrent_batches: Optional[int] = None,
    ):
        """
        Initializes the Reranker with batching support.

        Args:
            model_name: The name of the CrossEncoder model to load
                        (e.g., 'jinaai/jina-reranker-v2-base-multilingual').
            device: The device to run the model on ('cpu', 'cuda', 'mps').
                    If None, attempts to auto-detect.
            batch_size: Number of documents per batch (auto-detected if None).
            max_concurrent_batches: Maximum concurrent batches for processing.
        """
        self.model_name = model_name
        self.device = device or self._get_best_device()  # Auto-detect if not specified
        self.batch_manager = BatchManager(
            batch_size=batch_size, max_concurrent_batches=max_concurrent_batches
        )

        logger.info(
            f"Initializing Reranker with model '{self.model_name}' on device '{self.device}' "
            f"with batching (batch_size={self.batch_manager.batch_size})"
        )

        try:
            # Basic CrossEncoder initialization
            logger.info(
                f"Attempting basic CrossEncoder initialization for '{self.model_name}'..."
            )
            self.model = CrossEncoder(
                model_name_or_path=self.model_name,  # Renamed argument
                device=self.device,
                trust_remote_code=True,  # Add back based on ValueError
            )
            logger.info(
                f"Successfully loaded model '{self.model_name}' to device '{self.device}'."
            )
        except Exception as e:
            logger.error(
                f"Failed to load model '{self.model_name}': {e}", exc_info=True
            )
            # Fallback to CPU if specific device fails? Or just raise?
            # For now, let's raise to make the issue clear.
            raise RuntimeError(
                f"Could not load model '{self.model_name}'. Ensure it's installed or accessible."
            ) from e

    def _get_best_device(self) -> str:
        """Auto-detects the best available device."""
        if torch.cuda.is_available():
            logger.info("CUDA detected. Using GPU.")
            return "cuda"
        # Check for Apple Silicon MPS (requires PyTorch >= 1.12)
        elif torch.backends.mps.is_available():
            if torch.backends.mps.is_built():  # Extra check for older PyTorch versions
                logger.info("MPS detected. Using Apple Silicon GPU.")
                return "mps"
            else:
                logger.warning("MPS available but not built. Falling back to CPU.")
                return "cpu"
        else:
            logger.info("No GPU detected (CUDA or MPS). Using CPU.")
            return "cpu"

    def _prepare_input_pairs(
        self, query: str, documents: List[Union[str, Dict[str, Any]]]
    ) -> Tuple[List[Tuple[str, str]], List[int]]:
        """Prepares query-document pairs for the CrossEncoder model."""
        pairs = []
        original_indices = []
        for i, doc in enumerate(documents):
            doc_text = (
                doc if isinstance(doc, str) else doc.get("text", "")
            )  # Handle string or dict
            if doc_text:  # Avoid empty documents
                pairs.append((query, doc_text))
                original_indices.append(i)
            else:
                logger.warning(f"Skipping empty document at index {i}.")
        return pairs, original_indices

    @override
    def rerank(self, request: RerankRequest) -> List[RerankResult]:
        """
        Reranks documents based on the request using batching.

        Args:
            request: The rerank request containing query, documents, and options.

        Returns:
            A list of rerank results, sorted by relevance score (descending).
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[PyTorch] Starting batched rerank request")
            logger.debug(f"[PyTorch] Query: {request.query}")
            logger.debug(f"[PyTorch] Number of documents: {len(request.documents)}")
            logger.debug(f"[PyTorch] Batch size: {self.batch_manager.batch_size}")

        if not request.documents:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("[PyTorch] No documents to rerank")
            return []

        # Create batches using batch manager
        batches, batch_indices = self.batch_manager.create_batches(request)

        if not batches:
            logger.warning("[PyTorch] No batches created")
            return []

        # Initialize result aggregator
        aggregator = ResultAggregator()
        # Count only non-empty documents
        non_empty_count = sum(
            1
            for doc in request.documents
            if doc and (doc if isinstance(doc, str) else doc.get("text", ""))
        )
        aggregator.set_total_document_count(non_empty_count)

        # Process each batch
        for batch_idx, (batch_docs, original_indices) in enumerate(
            zip(batches, batch_indices)
        ):
            logger.info(
                f"[PyTorch] Processing batch {batch_idx + 1}/{len(batches)}: "
                f"{len(batch_docs)} documents"
            )

            try:
                # Prepare sentence pairs for this batch
                sentence_pairs, batch_original_indices = self._prepare_input_pairs(
                    request.query, batch_docs
                )

                if not sentence_pairs:
                    logger.warning(
                        f"[PyTorch] No valid pairs for batch {batch_idx + 1}"
                    )
                    continue

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"[PyTorch] Computing scores for {len(sentence_pairs)} pairs in batch {batch_idx + 1}..."
                    )

                # Get scores for this batch
                batch_scores = self.model.predict(
                    sentence_pairs, show_progress_bar=False
                )

                # Validate that model returned correct number of scores
                if len(batch_scores) != len(sentence_pairs):
                    logger.error(
                        f"[PyTorch] Model returned {len(batch_scores)} scores for "
                        f"{len(sentence_pairs)} pairs. This indicates a model error."
                    )
                    # Return empty list instead of raising exception for model errors
                    return []

                # Convert batch scores to our format
                batch_results = []
                for batch_relative_index, score in zip(
                    batch_original_indices, batch_scores
                ):
                    if batch_relative_index < 0 or batch_relative_index >= len(
                        original_indices
                    ):
                        logger.warning(
                            f"[PyTorch] Invalid batch-relative index {batch_relative_index} in batch result"
                        )
                        continue
                    original_idx = original_indices[batch_relative_index]

                    doc_content = None
                    if request.return_documents:
                        original_doc = batch_docs[
                            batch_original_indices[batch_relative_index]
                        ]
                        doc_text = (
                            original_doc
                            if isinstance(original_doc, str)
                            else original_doc.get("text", "")
                        )
                        doc_content = RerankDocument(text=doc_text)

                    batch_result = RerankResult(
                        document=doc_content,
                        index=original_idx,
                        relevance_score=float(score),
                    )
                    batch_results.append(batch_result)

                # Add batch results to aggregator
                aggregator.add_batch_results(batch_results, original_indices)

                logger.info(
                    f"[PyTorch] Batch {batch_idx + 1} completed: "
                    f"{len(batch_results)} results"
                )

            except Exception as batch_error:
                logger.error(f"[PyTorch] Batch {batch_idx + 1} failed: {batch_error}")
                # Re-raise to propagate model errors (like mismatched scores)
                raise batch_error

        # Get final sorted results
        # Note: We filter out empty documents during processing, so use get_sorted_results
        # instead of get_complete_results which would add placeholders for missing docs
        final_results = aggregator.get_sorted_results(request.top_n)

        # Log final statistics
        stats = aggregator.get_batch_statistics()
        logger.info(
            f"[PyTorch] Batched reranking completed: {len(final_results)} final results, "
            f"{stats.get('total_batches', 0)} batches, "
            f"completion_rate={stats.get('completion_rate', 0):.2%}"
        )

        return final_results

    async def rerank_async(
        self, request: RerankRequest
    ) -> AsyncGenerator[ProgressUpdate, None]:
        """Async reranking with progress updates for PyTorch backend.

        Args:
            request: The rerank request containing query, documents, and options.

        Yields:
            ProgressUpdate objects with current processing status.
        """
        start_time = time.time()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[PyTorch] Starting async batched rerank request")
            logger.debug(f"[PyTorch] Query: {request.query}")
            logger.debug(f"[PyTorch] Number of documents: {len(request.documents)}")
            logger.debug(f"[PyTorch] Batch size: {self.batch_manager.batch_size}")

        if not request.documents:
            logger.warning("[PyTorch] No documents to rerank")
            return

        # Create batches using batch manager
        batches, batch_indices = self.batch_manager.create_batches(request)
        total_batches = len(batches)

        if not batches:
            logger.warning("[PyTorch] No batches created")
            return

        # Initialize result aggregator
        aggregator = ResultAggregator()
        # Count only non-empty documents
        non_empty_count = sum(
            1
            for doc in request.documents
            if doc and (doc if isinstance(doc, str) else doc.get("text", ""))
        )
        aggregator.set_total_document_count(non_empty_count)

        # Process each batch with progress updates
        for batch_idx, (batch_docs, original_indices) in enumerate(
            zip(batches, batch_indices)
        ):
            batch_start_time = time.time()

            logger.info(
                f"[PyTorch] Processing async batch {batch_idx + 1}/{total_batches}: "
                f"{len(batch_docs)} documents"
            )

            # Yield progress update before processing batch
            elapsed_time = time.time() - start_time
            documents_processed = batch_idx * self.batch_manager.batch_size
            yield ProgressUpdate(
                current_batch=batch_idx + 1,
                total_batches=total_batches,
                documents_processed=documents_processed,
                total_documents=len(request.documents),
                current_batch_results=0,
                elapsed_time=elapsed_time,
            )

            try:
                # Prepare sentence pairs for this batch
                sentence_pairs, batch_original_indices = self._prepare_input_pairs(
                    request.query, batch_docs
                )

                if not sentence_pairs:
                    logger.warning(
                        f"[PyTorch] No valid pairs for async batch {batch_idx + 1}"
                    )
                    continue

                # Process batch asynchronously
                batch_scores = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.predict(sentence_pairs, show_progress_bar=False),
                )

                # Convert batch scores to our format
                batch_results = []
                for batch_relative_index, score in zip(
                    batch_original_indices, batch_scores
                ):
                    if batch_relative_index < 0 or batch_relative_index >= len(
                        original_indices
                    ):
                        logger.warning(
                            f"[PyTorch] Invalid batch-relative index {batch_relative_index} in async batch result"
                        )
                        continue
                    original_idx = original_indices[batch_relative_index]

                    doc_content = None
                    if request.return_documents:
                        original_doc = batch_docs[
                            batch_original_indices[batch_relative_index]
                        ]
                        doc_text = (
                            original_doc
                            if isinstance(original_doc, str)
                            else original_doc.get("text", "")
                        )
                        doc_content = RerankDocument(text=doc_text)

                    batch_result = RerankResult(
                        document=doc_content,
                        index=original_idx,
                        relevance_score=float(score),
                    )
                    batch_results.append(batch_result)

                # Add batch results to aggregator
                aggregator.add_batch_results(batch_results, original_indices)

                # Yield progress update after completing batch
                elapsed_time = time.time() - start_time
                documents_processed = (batch_idx + 1) * self.batch_manager.batch_size
                documents_processed = min(documents_processed, len(request.documents))

                yield ProgressUpdate(
                    current_batch=batch_idx + 1,
                    total_batches=total_batches,
                    documents_processed=documents_processed,
                    total_documents=len(request.documents),
                    current_batch_results=len(batch_results),
                    elapsed_time=elapsed_time,
                )

                batch_time = time.time() - batch_start_time
                logger.info(
                    f"[PyTorch] Async batch {batch_idx + 1} completed in {batch_time:.2f}s: "
                    f"{len(batch_results)} results"
                )

            except Exception as batch_error:
                logger.error(
                    f"[PyTorch] Async batch {batch_idx + 1} failed: {batch_error}"
                )
                # Re-raise to propagate model errors
                raise batch_error

                # Yield progress update for failed batch
                elapsed_time = time.time() - start_time
                documents_processed = (batch_idx + 1) * self.batch_manager.batch_size
                documents_processed = min(documents_processed, len(request.documents))

                yield ProgressUpdate(
                    current_batch=batch_idx + 1,
                    total_batches=total_batches,
                    documents_processed=documents_processed,
                    total_documents=len(request.documents),
                    current_batch_results=len(placeholder_results),
                    elapsed_time=elapsed_time,
                )

                logger.warning(
                    f"[PyTorch] Added {len(placeholder_results)} placeholder results for failed async batch"
                )

        # Final progress update
        elapsed_time = time.time() - start_time
        final_results = aggregator.get_sorted_results(request.top_n)

        yield ProgressUpdate(
            current_batch=total_batches,
            total_batches=total_batches,
            documents_processed=len(request.documents),
            total_documents=len(request.documents),
            current_batch_results=len(final_results),
            elapsed_time=elapsed_time,
        )

        # Log final statistics
        stats = aggregator.get_batch_statistics()
        logger.info(
            f"[PyTorch] Async batched reranking completed in {elapsed_time:.2f}s: "
            f"{len(final_results)} final results, "
            f"{stats.get('total_batches', 0)} batches, "
            f"completion_rate={stats.get('completion_rate', 0):.2%}"
        )

    async def rerank_async_final(self, request: RerankRequest) -> List[RerankResult]:
        """Async reranking that returns final results for PyTorch backend.

        Args:
            request: The rerank request containing query, documents, and options.

        Returns:
            A list of rerank results, sorted by relevance score (descending).
        """
        # Use the async generator but collect final results
        aggregator = ResultAggregator()
        # Count only non-empty documents
        non_empty_count = sum(
            1
            for doc in request.documents
            if doc and (doc if isinstance(doc, str) else doc.get("text", ""))
        )
        aggregator.set_total_document_count(non_empty_count)

        # Create batches using batch manager
        batches, batch_indices = self.batch_manager.create_batches(request)

        if not batches:
            return []

        # Process all batches concurrently
        async def process_single_batch(batch_idx, batch_docs, original_indices):
            try:
                # Prepare sentence pairs for this batch
                sentence_pairs, batch_original_indices = self._prepare_input_pairs(
                    request.query, batch_docs
                )

                if not sentence_pairs:
                    logger.warning(
                        f"[PyTorch] No valid pairs for concurrent batch {batch_idx + 1}"
                    )
                    return [], original_indices

                batch_scores = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.predict(sentence_pairs, show_progress_bar=False),
                )

                # Convert batch scores to our format
                batch_results = []
                for batch_relative_index, score in zip(
                    batch_original_indices, batch_scores
                ):
                    if batch_relative_index < 0 or batch_relative_index >= len(
                        original_indices
                    ):
                        logger.warning(
                            f"[PyTorch] Invalid batch-relative index {batch_relative_index} in concurrent batch result"
                        )
                        continue
                    original_idx = original_indices[batch_relative_index]

                    doc_content = None
                    if request.return_documents:
                        original_doc = batch_docs[
                            batch_original_indices[batch_relative_index]
                        ]
                        doc_text = (
                            original_doc
                            if isinstance(original_doc, str)
                            else original_doc.get("text", "")
                        )
                        doc_content = RerankDocument(text=doc_text)

                    batch_result = RerankResult(
                        document=doc_content,
                        index=original_idx,
                        relevance_score=float(score),
                    )
                    batch_results.append(batch_result)

                return batch_results, original_indices

            except Exception as batch_error:
                logger.error(
                    f"[PyTorch] Async batch {batch_idx + 1} failed: {batch_error}"
                )
                # Re-raise to propagate model errors
                raise batch_error

                return placeholder_results, original_indices

        # Process batches concurrently (limited by max_concurrent_batches)
        semaphore = asyncio.Semaphore(self.batch_manager.max_concurrent_batches)

        async def process_with_semaphore(batch_idx, batch_docs, original_indices):
            async with semaphore:
                return await process_single_batch(
                    batch_idx, batch_docs, original_indices
                )

        # Create tasks for all batches
        tasks = [
            process_with_semaphore(batch_idx, batch_docs, original_indices)
            for batch_idx, (batch_docs, original_indices) in enumerate(
                zip(batches, batch_indices)
            )
        ]

        # Wait for all batches to complete
        batch_results_list = await asyncio.gather(*tasks)

        # Aggregate all results
        for formatted_results, original_indices in batch_results_list:
            aggregator.add_batch_results(formatted_results, original_indices)

        # Get final sorted results
        # Note: We filter out empty documents during processing, so use get_sorted_results
        # instead of get_complete_results which would add placeholders for missing docs
        final_results = aggregator.get_sorted_results(request.top_n)

        # Log final statistics
        stats = aggregator.get_batch_statistics()
        logger.info(
            f"[PyTorch] Concurrent async reranking completed: {len(final_results)} final results, "
            f"{stats.get('total_batches', 0)} batches, "
            f"completion_rate={stats.get('completion_rate', 0):.2%}"
        )

        return final_results
