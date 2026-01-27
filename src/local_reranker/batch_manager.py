# -*- coding: utf-8 -*-
"""Batch manager for efficient document processing in reranking operations."""

from typing import List, Tuple, Optional
import logging
import os
import psutil

from .models import RerankRequest

logger = logging.getLogger(__name__)


class BatchManager:
    """Manages document batching for optimal reranking performance."""

    def __init__(
        self,
        batch_size: Optional[int] = None,
        max_concurrent_batches: Optional[int] = None,
        memory_limit_mb: Optional[int] = None,
    ):
        """
        Initialize batch manager with configuration.

        Args:
            batch_size: Number of documents per batch (auto-detected if None)
            max_concurrent_batches: Maximum concurrent batches (default: 3)
            memory_limit_mb: Memory limit in MB for dynamic sizing
        """
        # Configuration from environment or defaults
        self.batch_size = (
            batch_size if batch_size is not None else self._get_env_batch_size()
        )
        self.max_concurrent_batches = (
            max_concurrent_batches or self._get_env_max_concurrent()
        )
        self.memory_limit_mb = memory_limit_mb or self._get_env_memory_limit()

        # Auto-adjust batch size based on system resources
        if batch_size is None:
            self.batch_size = self._calculate_optimal_batch_size()

        logger.info(
            f"BatchManager initialized: batch_size={self.batch_size}, "
            f"max_concurrent={self.max_concurrent_batches}, "
            f"memory_limit={self.memory_limit_mb}MB"
        )

    def _get_env_batch_size(self) -> int:
        """Get batch size from environment variable."""
        try:
            return int(os.getenv("RERANKER_BATCH_SIZE", "12"))
        except ValueError:
            logger.warning("Invalid RERANKER_BATCH_SIZE, using default 12")
            return 12

    def _get_env_max_concurrent(self) -> int:
        """Get max concurrent batches from environment variable."""
        try:
            return int(os.getenv("RERANKER_MAX_CONCURRENT_BATCHES", "3"))
        except ValueError:
            logger.warning("Invalid RERANKER_MAX_CONCURRENT_BATCHES, using default 3")
            return 3

    def _get_env_memory_limit(self) -> int:
        """Get memory limit from environment variable."""
        try:
            return int(os.getenv("RERANKER_MEMORY_LIMIT_MB", "1024"))
        except ValueError:
            logger.warning("Invalid RERANKER_MEMORY_LIMIT_MB, using default 1024")
            return 1024

    def _calculate_optimal_batch_size(self) -> int:
        """
        Calculate optimal batch size based on available memory and system resources.

        Returns:
            Optimal batch size for current system
        """
        try:
            # Get available memory
            memory_info = psutil.virtual_memory()
            available_memory_gb = memory_info.available / (1024**3)  # Convert to GB

            # Base batch size calculation
            # Assume ~50MB per batch for typical documents (conservative estimate)
            memory_based_batch = max(4, min(20, int(available_memory_gb * 2)))

            # Consider CPU cores for concurrent processing
            cpu_cores = psutil.cpu_count() or 1
            cpu_based_batch = max(4, min(16, cpu_cores * 2))

            # Use the more conservative of the two calculations
            optimal_batch = min(memory_based_batch, cpu_based_batch)

            logger.info(
                f"Auto-calculated batch size: {optimal_batch} "
                f"(memory_based={memory_based_batch}, cpu_based={cpu_based_batch}, "
                f"available_memory={available_memory_gb:.1f}GB)"
            )

            return optimal_batch

        except Exception as e:
            logger.warning(
                f"Failed to auto-calculate batch size: {e}, using default 12"
            )
            return 12

    def create_batches(
        self, request: RerankRequest
    ) -> Tuple[List[List[str]], List[List[int]]]:
        """
        Create document batches for processing.

        Args:
            request: Rerank request containing documents

        Returns:
            Tuple of (batches, batch_indices) where:
            - batches: List of document batches
            - batch_indices: List of original document indices for each batch
        """
        if not request.documents:
            logger.warning("BatchManager: No documents to batch")
            return [], []

        # Extract document texts
        documents = []
        for i, doc in enumerate(request.documents):
            if isinstance(doc, str):
                documents.append(doc)
            elif isinstance(doc, dict) and "text" in doc:
                documents.append(doc["text"])
            else:
                logger.warning(f"BatchManager: Skipping invalid document at index {i}")
                continue

        if not documents:
            logger.warning("BatchManager: No valid documents after processing")
            return [], []

        # Create batches
        batches = []
        batch_indices = []

        for i in range(0, len(documents), self.batch_size):
            end_idx = min(i + self.batch_size, len(documents))
            batch_docs = documents[i:end_idx]
            batch_idx = list(range(i, end_idx))

            batches.append(batch_docs)
            batch_indices.append(batch_idx)

            if logger.isEnabledFor(logging.DEBUG):
                total_chars = sum(len(doc) for doc in batch_docs)
                logger.debug(
                    f"BatchManager: Created batch {len(batches)}: "
                    f"{len(batch_docs)} docs, {total_chars} chars, "
                    f"indices {batch_idx[0]}-{batch_idx[-1]}"
                )

        logger.info(
            f"BatchManager: Created {len(batches)} batches from {len(documents)} documents "
            f"(batch_size={self.batch_size})"
        )

        return batches, batch_indices

    def estimate_processing_time(
        self, document_count: int, avg_doc_length: int = 1000
    ) -> float:
        """
        Estimate processing time based on document count and characteristics.

        Args:
            document_count: Number of documents to process
            avg_doc_length: Average document length in characters

        Returns:
            Estimated processing time in seconds
        """
        # Base processing time per document (conservative estimates)
        base_time_per_doc = 0.1  # 100ms per document base

        # Adjust for document length (longer docs take more time)
        length_factor = max(1.0, avg_doc_length / 1000.0)

        # Adjust for batching efficiency
        batch_count = (document_count + self.batch_size - 1) // self.batch_size
        batch_efficiency = 1.0 / (1.0 + batch_count * 0.1)  # Diminishing returns

        # Calculate total time
        total_time = (
            document_count * base_time_per_doc * length_factor * batch_efficiency
        )

        logger.debug(
            f"BatchManager: Estimated processing time for {document_count} docs: "
            f"{total_time:.2f}s (batch_count={batch_count}, efficiency={batch_efficiency:.2f})"
        )

        return total_time

    def should_use_fallback(self, document_count: int, total_chars: int) -> bool:
        """
        Determine if fallback processing should be used based on document characteristics.

        Args:
            document_count: Number of documents
            total_chars: Total characters in all documents

        Returns:
            True if fallback processing recommended
        """
        # Use fallback if:
        # 1. Very large document count
        if document_count > 50:
            return True

        # 2. Very long total content
        if total_chars > 100000:  # 100K characters
            return True

        # 3. High average document length
        avg_length = total_chars / document_count if document_count > 0 else 0
        if avg_length > 5000:  # 5K characters average
            return True

        return False

    def get_status(self) -> dict:
        """
        Get current batch manager status and statistics.

        Returns:
            Dictionary with manager status
        """
        try:
            memory_info = psutil.virtual_memory()
            return {
                "batch_size": self.batch_size,
                "max_concurrent_batches": self.max_concurrent_batches,
                "memory_limit_mb": self.memory_limit_mb,
                "available_memory_gb": memory_info.available / (1024**3),
                "total_memory_gb": memory_info.total / (1024**3),
                "memory_usage_percent": memory_info.percent,
            }
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {
                "batch_size": self.batch_size,
                "max_concurrent_batches": self.max_concurrent_batches,
                "memory_limit_mb": self.memory_limit_mb,
                "error": str(e),
            }
