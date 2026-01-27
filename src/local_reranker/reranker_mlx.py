# -*- coding: utf-8 -*-
"""MLX-based reranker implementation for Apple Silicon optimization."""

from typing import List, Optional, AsyncGenerator
import logging
import os
import sys
import importlib.util
import asyncio
import time

from .reranker import Reranker as RerankerProtocol
from .models import RerankRequest, RerankResult, RerankDocument, ProgressUpdate
from .batch_manager import BatchManager
from .result_aggregator import ResultAggregator
from .mlx_cross_encoder import MLXCrossEncoderReranker

logger = logging.getLogger(__name__)


class Reranker(RerankerProtocol):
    """MLX implementation of reranker protocol for Apple Silicon optimization."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v3-mlx",
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_concurrent_batches: Optional[int] = None,
    ):
        """Initialize MLX reranker with batching support.

        Args:
            model_name: The name of the MLX model to load.
            device: The device to run the model on. MLX auto-detects Apple Silicon GPU/CPU.
                    This parameter is kept for protocol compatibility but ignored.
            batch_size: Number of documents per batch (auto-detected if None).
            max_concurrent_batches: Maximum concurrent batches for processing.

        Raises:
            ImportError: If MLX dependencies are not installed.
            RuntimeError: If model loading fails.
        """
        self.model_name = model_name
        self.device = device  # Ignored for MLX, kept for compatibility

        # Initialize batch manager
        self.batch_manager = BatchManager(
            batch_size=batch_size, max_concurrent_batches=max_concurrent_batches
        )

        try:
            # Import MLX dependencies
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm import load
            import numpy as np
            from safetensors import safe_open

            # Download and prepare model files
            model_path = self._prepare_model_files(model_name)

            # Load MLX reranker implementation
            self.model = self._load_mlx_reranker(model_path)
            logger.info(f"Successfully loaded MLX reranker with batching: {model_name}")

        except ImportError as e:
            raise ImportError(
                "MLX dependencies not found. Install with: pip install mlx mlx-lm safetensors"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load MLX model '{model_name}': {e}") from e

    def _prepare_model_files(self, model_name: str) -> str:
        """Prepare model files by downloading from HuggingFace if needed."""
        try:
            from huggingface_hub import snapshot_download

            # Download model to cache directory
            model_path = snapshot_download(
                repo_id=model_name,
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.py"],
            )
            return model_path

        except ImportError:
            raise ImportError(
                "huggingface-hub not found. Install with: pip install huggingface-hub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download model files: {e}") from e

    def _load_mlx_reranker(self, model_path: str):
        """Load MLX reranker, falling back to the internal cross-encoder as needed."""
        rerank_file = os.path.join(model_path, "rerank.py")
        projector_path = os.path.join(model_path, "projector.safetensors")

        if os.path.exists(rerank_file):
            try:
                reranker = self._load_repo_reranker(
                    rerank_file=rerank_file,
                    model_path=model_path,
                    projector_path=projector_path,
                )
                logger.info(
                    "[MLX] Using repo-provided MLXReranker implementation from %s",
                    rerank_file,
                )
                return reranker
            except Exception as repo_error:  # pragma: no cover - log path
                logger.warning(
                    "[MLX] Failed to load repo-provided reranker at %s: %s."
                    " Falling back to internal cross-encoder.",
                    rerank_file,
                    repo_error,
                )
        else:
            logger.info(
                "[MLX] rerank.py not found in %s. Using internal cross-encoder fallback.",
                model_path,
            )

        return MLXCrossEncoderReranker(
            model_path=model_path,
            projector_path=projector_path,
        )

    def _load_repo_reranker(
        self, rerank_file: str, model_path: str, projector_path: str
    ):
        """Load a repo-provided MLXReranker implementation from rerank.py."""
        spec = importlib.util.spec_from_file_location("mlx_reranker", rerank_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec from {rerank_file}")

        mlx_reranker_module = importlib.util.module_from_spec(spec)
        sys.modules["mlx_reranker"] = mlx_reranker_module
        spec.loader.exec_module(mlx_reranker_module)

        MLXRerankerImpl = mlx_reranker_module.MLXReranker
        return MLXRerankerImpl(
            model_path=model_path,
            projector_path=projector_path,
        )

    def _fallback_batch_processing(
        self, request: RerankRequest, documents: List[str]
    ) -> List[dict]:
        """
        Fallback processing method that handles large document sets by splitting into smaller batches.

        Args:
            request: Original rerank request
            documents: List of document strings

        Returns:
            List of raw result dictionaries from model
        """
        logger.info("[MLX] Starting fallback batch processing")

        # Use smaller batch size for fallback
        fallback_batch_size = 8  # Conservative batch size
        all_results = []

        for batch_start in range(0, len(documents), fallback_batch_size):
            batch_end = min(batch_start + fallback_batch_size, len(documents))
            batch_docs = documents[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))

            logger.info(
                f"[MLX] Processing fallback batch {batch_start // fallback_batch_size + 1}: "
                f"documents {batch_start}-{batch_end - 1} ({len(batch_docs)} docs)"
            )

            try:
                # Process this smaller batch
                batch_results = self.model.rerank(
                    query=request.query,
                    documents=batch_docs,
                    top_n=None,  # Get all results from this batch
                    return_embeddings=request.return_documents,
                )

                # Adjust indices to match original document positions
                for result in batch_results:
                    result["index"] = batch_indices[result["index"]]

                all_results.extend(batch_results)
                logger.info(
                    f"[MLX] Fallback batch completed: {len(batch_results)} results"
                )

            except Exception as batch_error:
                logger.error(
                    f"[MLX] Fallback batch {batch_start // fallback_batch_size + 1} failed: {batch_error}"
                )
                # Add placeholder results for failed batch to maintain document count
                for i, original_idx in enumerate(batch_indices):
                    placeholder_result = {
                        "document": batch_docs[i] if request.return_documents else None,
                        "relevance_score": 0.0,  # Neutral score for failed processing
                        "index": original_idx,
                        "embedding": None,
                    }
                    all_results.append(placeholder_result)
                logger.warning(
                    f"[MLX] Added {len(batch_indices)} placeholder results for failed batch"
                )

        logger.info(
            f"[MLX] Fallback processing completed: {len(all_results)} total results"
        )
        return all_results

    def rerank(self, request: RerankRequest) -> List[RerankResult]:
        """Rerank documents using the MLX backend with batching support.

        Args:
            request: The rerank request containing query, documents, and options.

        Returns:
            A list of rerank results, sorted by relevance score (descending).

        Raises:
            ValueError: If the request is invalid.
            RuntimeError: If reranking fails.
        """
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[MLX] Starting batched rerank request")
            logger.debug(f"[MLX] Query: {request.query}")
            logger.debug(f"[MLX] Number of documents: {len(request.documents)}")
            logger.debug(f"[MLX] Batch size: {self.batch_manager.batch_size}")

        try:
            # Validate inputs
            if not request.documents:
                logger.warning("[MLX] No documents to process")
                return []

            if not request.query or not request.query.strip():
                logger.warning("[MLX] Empty query provided")
                return []

            # Create batches using batch manager
            batches, batch_indices = self.batch_manager.create_batches(request)

            if not batches:
                logger.warning("[MLX] No batches created")
                return []

            # Initialize result aggregator
            aggregator = ResultAggregator()
            aggregator.set_total_document_count(len(request.documents))

            # Process each batch
            for batch_idx, (batch_docs, original_indices) in enumerate(
                zip(batches, batch_indices)
            ):
                logger.info(
                    f"[MLX] Processing batch {batch_idx + 1}/{len(batches)}: "
                    f"{len(batch_docs)} documents"
                )

                try:
                    # Call MLX model for this batch
                    batch_results = self.model.rerank(
                        query=request.query,
                        documents=batch_docs,
                        top_n=None,  # Get all results from this batch
                        return_embeddings=request.return_documents,
                    )

                    # Convert batch results to our format
                    formatted_results = []
                    for result in batch_results:
                        try:
                            # Handle document text based on return_documents flag
                            document = None
                            if request.return_documents and "document" in result:
                                doc_text = result["document"]
                                if doc_text:  # Ensure document text is not empty
                                    document = RerankDocument(text=doc_text)

                            # Extract relevance score and ensure it's a float
                            relevance_score = float(result["relevance_score"])

                            # Map batch-relative index to original document index
                            batch_relative_index = int(result["index"])
                            if batch_relative_index < 0 or batch_relative_index >= len(
                                original_indices
                            ):
                                logger.warning(
                                    f"[MLX] Invalid batch-relative index {batch_relative_index} in batch result"
                                )
                                continue
                            index = original_indices[batch_relative_index]

                            # Validate index range
                            if index < 0 or index >= len(request.documents):
                                logger.warning(
                                    f"[MLX] Invalid original index {index} in batch result"
                                )
                                continue

                            rerank_result = RerankResult(
                                document=document,
                                index=index,
                                relevance_score=relevance_score,
                            )
                            formatted_results.append(rerank_result)

                        except (KeyError, ValueError, TypeError) as result_error:
                            logger.error(
                                f"[MLX] Error processing batch result: {result_error}"
                            )
                            continue

                    # Add batch results to aggregator
                    aggregator.add_batch_results(formatted_results, original_indices)
                    logger.info(
                        f"[MLX] Batch {batch_idx + 1} completed: "
                        f"{len(formatted_results)} results"
                    )

                except Exception as batch_error:
                    logger.error(f"[MLX] Batch {batch_idx + 1} failed: {batch_error}")

                    # Add placeholder results for failed batch
                    placeholder_results = []
                    for i, original_idx in enumerate(original_indices):
                        placeholder_result = RerankResult(
                            document=RerankDocument(text=batch_docs[i])
                            if request.return_documents
                            else None,
                            index=original_idx,
                            relevance_score=0.0,  # Neutral score for failed processing
                        )
                        placeholder_results.append(placeholder_result)

                    aggregator.add_batch_results(placeholder_results, original_indices)
                    logger.warning(
                        f"[MLX] Added {len(placeholder_results)} placeholder results for failed batch"
                    )

            # Get final sorted results
            final_results = aggregator.get_complete_results(
                request.documents, request.return_documents, request.top_n
            )

            # Log final statistics
            stats = aggregator.get_batch_statistics()
            logger.info(
                f"[MLX] Batched reranking completed: {len(final_results)} final results, "
                f"{stats.get('total_batches', 0)} batches, "
                f"completion_rate={stats.get('completion_rate', 0):.2%}"
            )

            return final_results

        except ValueError as e:
            logger.error(f"[MLX] Validation error: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"[MLX] Runtime error: {e}")
            raise
        except Exception as e:
            logger.error(
                f"[MLX] Unexpected error during batched reranking: {e}", exc_info=True
            )
            raise RuntimeError(f"MLX batched reranking failed: {e}") from e

    async def rerank_async(
        self, request: RerankRequest
    ) -> AsyncGenerator[ProgressUpdate, None]:
        """Async reranking with progress updates for MLX backend.

        Args:
            request: The rerank request containing query, documents, and options.

        Yields:
            ProgressUpdate objects with current processing status.
        """
        start_time = time.time()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[MLX] Starting async batched rerank request")
            logger.debug(f"[MLX] Query: {request.query}")
            logger.debug(f"[MLX] Number of documents: {len(request.documents)}")
            logger.debug(f"[MLX] Batch size: {self.batch_manager.batch_size}")

        try:
            # Validate inputs
            if not request.documents:
                logger.warning("[MLX] No documents to process")
                return

            if not request.query or not request.query.strip():
                logger.warning("[MLX] Empty query provided")
                return

            # Create batches using batch manager
            batches, batch_indices = self.batch_manager.create_batches(request)
            total_batches = len(batches)

            if not batches:
                logger.warning("[MLX] No batches created")
                return

            # Initialize result aggregator
            aggregator = ResultAggregator()
            aggregator.set_total_document_count(len(request.documents))

            # Process each batch with progress updates
            for batch_idx, (batch_docs, original_indices) in enumerate(
                zip(batches, batch_indices)
            ):
                batch_start_time = time.time()

                logger.info(
                    f"[MLX] Processing async batch {batch_idx + 1}/{total_batches}: "
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
                    # Process batch asynchronously (simulate with asyncio.sleep if needed)
                    batch_results = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: self.model.rerank(
                            query=request.query,
                            documents=batch_docs,
                            top_n=None,
                            return_embeddings=request.return_documents,
                        ),
                    )

                    # Convert batch results to our format
                    formatted_results = []
                    for result in batch_results:
                        try:
                            # Handle document text based on return_documents flag
                            document = None
                            if request.return_documents and "document" in result:
                                doc_text = result["document"]
                                if doc_text:  # Ensure document text is not empty
                                    document = RerankDocument(text=doc_text)

                            # Extract relevance score and ensure it's a float
                            relevance_score = float(result["relevance_score"])

                            # Map batch-relative index to original document index
                            batch_relative_index = int(result["index"])
                            if batch_relative_index < 0 or batch_relative_index >= len(
                                original_indices
                            ):
                                logger.warning(
                                    f"[MLX] Invalid batch-relative index {batch_relative_index} in async batch result"
                                )
                                continue
                            index = original_indices[batch_relative_index]

                            # Validate index range
                            if index < 0 or index >= len(request.documents):
                                logger.warning(
                                    f"[MLX] Invalid original index {index} in async batch result"
                                )
                                continue

                            rerank_result = RerankResult(
                                document=document,
                                index=index,
                                relevance_score=relevance_score,
                            )
                            formatted_results.append(rerank_result)

                        except (KeyError, ValueError, TypeError) as result_error:
                            logger.error(
                                f"[MLX] Error processing batch result: {result_error}"
                            )
                            continue

                    # Add batch results to aggregator
                    aggregator.add_batch_results(formatted_results, original_indices)

                    # Yield progress update after completing batch
                    elapsed_time = time.time() - start_time
                    documents_processed = (
                        batch_idx + 1
                    ) * self.batch_manager.batch_size
                    documents_processed = min(
                        documents_processed, len(request.documents)
                    )

                    yield ProgressUpdate(
                        current_batch=batch_idx + 1,
                        total_batches=total_batches,
                        documents_processed=documents_processed,
                        total_documents=len(request.documents),
                        current_batch_results=len(formatted_results),
                        elapsed_time=elapsed_time,
                    )

                    batch_time = time.time() - batch_start_time
                    logger.info(
                        f"[MLX] Async batch {batch_idx + 1} completed in {batch_time:.2f}s: "
                        f"{len(formatted_results)} results"
                    )

                except Exception as batch_error:
                    logger.error(
                        f"[MLX] Async batch {batch_idx + 1} failed: {batch_error}"
                    )

                    # Add placeholder results for failed batch
                    placeholder_results = []
                    for i, original_idx in enumerate(original_indices):
                        placeholder_result = RerankResult(
                            document=RerankDocument(text=batch_docs[i])
                            if request.return_documents
                            else None,
                            index=original_idx,
                            relevance_score=0.0,  # Neutral score for failed processing
                        )
                        placeholder_results.append(placeholder_result)

                    aggregator.add_batch_results(placeholder_results, original_indices)

                    # Yield progress update for failed batch
                    elapsed_time = time.time() - start_time
                    documents_processed = (
                        batch_idx + 1
                    ) * self.batch_manager.batch_size
                    documents_processed = min(
                        documents_processed, len(request.documents)
                    )

                    yield ProgressUpdate(
                        current_batch=batch_idx + 1,
                        total_batches=total_batches,
                        documents_processed=documents_processed,
                        total_documents=len(request.documents),
                        current_batch_results=len(placeholder_results),
                        elapsed_time=elapsed_time,
                    )

                    logger.warning(
                        f"[MLX] Added {len(placeholder_results)} placeholder results for failed async batch"
                    )

            # Final progress update
            elapsed_time = time.time() - start_time
            final_results = aggregator.get_complete_results(
                request.documents, request.return_documents, request.top_n
            )

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
                f"[MLX] Async batched reranking completed in {elapsed_time:.2f}s: "
                f"{len(final_results)} final results, "
                f"{stats.get('total_batches', 0)} batches, "
                f"completion_rate={stats.get('completion_rate', 0):.2%}"
            )

        except ValueError as e:
            logger.error(f"[MLX] Async validation error: {e}")
            raise
        except RuntimeError as e:
            logger.error(f"[MLX] Async runtime error: {e}")
            raise
        except Exception as e:
            logger.error(
                f"[MLX] Unexpected error during async batched reranking: {e}",
                exc_info=True,
            )
            raise RuntimeError(f"MLX async batched reranking failed: {e}") from e

    async def rerank_async_final(self, request: RerankRequest) -> List[RerankResult]:
        """Async reranking that returns final results for MLX backend.

        Args:
            request: The rerank request containing query, documents, and options.

        Returns:
            A list of rerank results, sorted by relevance score (descending).
        """
        # Use the async generator but collect final results
        aggregator = ResultAggregator()
        aggregator.set_total_document_count(len(request.documents))

        # Create batches using batch manager
        batches, batch_indices = self.batch_manager.create_batches(request)

        if not batches:
            return []

        # Process all batches concurrently
        async def process_single_batch(batch_idx, batch_docs, original_indices):
            try:
                batch_results = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.model.rerank(
                        query=request.query,
                        documents=batch_docs,
                        top_n=None,
                        return_embeddings=request.return_documents,
                    ),
                )

                # Convert batch results to our format
                formatted_results = []
                for result in batch_results:
                    try:
                        document = None
                        if request.return_documents and "document" in result:
                            doc_text = result["document"]
                            if doc_text:
                                document = RerankDocument(text=doc_text)

                        relevance_score = float(result["relevance_score"])

                        # Map batch-relative index to original document index
                        batch_relative_index = int(result["index"])
                        if batch_relative_index < 0 or batch_relative_index >= len(
                            original_indices
                        ):
                            logger.warning(
                                f"[MLX] Invalid batch-relative index {batch_relative_index} in concurrent batch result"
                            )
                            continue
                        index = original_indices[batch_relative_index]

                        if index < 0 or index >= len(request.documents):
                            continue

                        rerank_result = RerankResult(
                            document=document,
                            index=index,
                            relevance_score=relevance_score,
                        )
                        formatted_results.append(rerank_result)

                    except (KeyError, ValueError, TypeError):
                        continue

                return formatted_results, original_indices

            except Exception as batch_error:
                logger.error(f"[MLX] Async batch {batch_idx + 1} failed: {batch_error}")

                # Add placeholder results for failed batch
                placeholder_results = []
                for i, original_idx in enumerate(original_indices):
                    placeholder_result = RerankResult(
                        document=RerankDocument(text=batch_docs[i])
                        if request.return_documents
                        else None,
                        index=original_idx,
                        relevance_score=0.0,
                    )
                    placeholder_results.append(placeholder_result)

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
        final_results = aggregator.get_complete_results(
            request.documents, request.return_documents, request.top_n
        )

        # Log final statistics
        stats = aggregator.get_batch_statistics()
        logger.info(
            f"[MLX] Concurrent async reranking completed: {len(final_results)} final results, "
            f"{stats.get('total_batches', 0)} batches, "
            f"completion_rate={stats.get('completion_rate', 0):.2%}"
        )

        return final_results
