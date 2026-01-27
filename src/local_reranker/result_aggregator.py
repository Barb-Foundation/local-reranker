# -*- coding: utf-8 -*-
"""Result aggregation and ordering system for batched reranking."""

from typing import Any, List, Optional
import logging

from .models import RerankResult, RerankDocument

logger = logging.getLogger(__name__)


class ResultAggregator:
    """Manages result aggregation and ordering across multiple batches."""

    def __init__(self):
        """Initialize result aggregator."""
        self.results: List[RerankResult] = []
        self.batch_results: List[List[RerankResult]] = []
        self.total_documents: int = 0

    def add_batch_results(
        self,
        batch_results: List[RerankResult],
        batch_original_indices: Optional[List[int]] = None,
    ) -> None:
        """
        Add results from a single batch to the aggregator.

        Args:
            batch_results: Results from processing a batch
            batch_original_indices: Original document indices for this batch
        """
        if not batch_results:
            logger.warning("ResultAggregator: Empty batch results received")
            return

        # Store batch information
        self.batch_results.append(batch_results)
        if batch_original_indices is not None:
            logger.debug(
                "ResultAggregator: Added batch %d with %d results, indices %s",
                len(self.batch_results),
                len(batch_results),
                batch_original_indices,
            )
        else:
            logger.debug(
                "ResultAggregator: Added batch %d with %d results",
                len(self.batch_results),
                len(batch_results),
            )

    def set_total_document_count(self, count: int) -> None:
        """Set the total number of documents being processed."""
        self.total_documents = count
        logger.debug(f"ResultAggregator: Total document count set to {count}")

    def get_sorted_results(self, top_n: Optional[int] = None) -> List[RerankResult]:
        """
        Get all results sorted by relevance score while maintaining original document indices.

        Args:
            top_n: Optional limit on number of results to return

        Returns:
            List of RerankResult objects sorted by relevance score (descending)
        """
        if not self.batch_results:
            logger.warning("ResultAggregator: No batch results to aggregate")
            return []

        # Flatten all batch results with their original indices
        all_results = []
        for batch_idx, batch_result_list in enumerate(self.batch_results):
            for result_idx, result in enumerate(batch_result_list):
                # Preserve the index provided by the reranker result
                ordered_result = RerankResult(
                    document=result.document,
                    index=result.index,
                    relevance_score=result.relevance_score,
                )
                all_results.append(ordered_result)

                logger.debug(
                    "ResultAggregator: Batch %d, Result %d: index=%s, score=%.4f",
                    batch_idx,
                    result_idx,
                    result.index,
                    result.relevance_score,
                )

        logger.info(
            f"ResultAggregator: Aggregated {len(all_results)} results from "
            f"{len(self.batch_results)} batches"
        )

        # Sort by relevance score (descending)
        sorted_results = sorted(
            all_results, key=lambda x: x.relevance_score, reverse=True
        )

        # Apply top_n limit if specified
        if top_n is not None:
            sorted_results = sorted_results[:top_n]

        # Validate we have results for all original documents
        if len(all_results) != self.total_documents:
            logger.warning(
                f"ResultAggregator: Document count mismatch. Expected {self.total_documents}, "
                f"got {len(all_results)}. Missing documents may indicate processing failures."
            )

        # Log top results for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("ResultAggregator: Top 10 sorted results:")
            for i, result in enumerate(sorted_results[:10]):
                doc_preview = ""
                if result.document:
                    doc_preview = result.document.text[:30].replace("\n", " ")
                logger.debug(
                    f"  {i + 1}: score={result.relevance_score:.4f}, "
                    f"index={result.index}, preview='{doc_preview}...'"
                )

        return sorted_results

    def get_batch_statistics(self) -> dict[str, Any]:
        """
        Get statistics about batch processing.

        Returns:
            Dictionary with batch processing statistics
        """
        if not self.batch_results:
            return {"error": "No batches processed"}

        total_results = sum(len(batch) for batch in self.batch_results)
        avg_results_per_batch = total_results / len(self.batch_results)

        stats = {
            "total_batches": len(self.batch_results),
            "total_results": total_results,
            "avg_results_per_batch": avg_results_per_batch,
            "total_documents": self.total_documents,
            "completion_rate": total_results / self.total_documents
            if self.total_documents > 0
            else 0,
        }

        # Add score statistics
        all_scores = [
            result.relevance_score for batch in self.batch_results for result in batch
        ]
        if all_scores:
            stats.update(
                {
                    "max_score": max(all_scores),
                    "min_score": min(all_scores),
                    "avg_score": sum(all_scores) / len(all_scores),
                }
            )

        return stats

    def validate_ordering(self) -> bool:
        """
        Validate that original document ordering is preserved in final results.

        Returns:
            True if ordering appears valid, False otherwise
        """
        if not self.batch_results:
            return False

        # Check for duplicate indices
        all_indices = []
        for batch_result_list in self.batch_results:
            for result in batch_result_list:
                if result.index in all_indices:
                    logger.error(
                        f"ResultAggregator: Duplicate index {result.index} found"
                    )
                    return False
                all_indices.append(result.index)

        # Check for missing indices (0 to total_documents-1)
        expected_indices = set(range(self.total_documents))
        actual_indices = set(all_indices)

        missing_indices = expected_indices - actual_indices
        if missing_indices:
            logger.error(
                f"ResultAggregator: Missing indices {sorted(missing_indices)}. "
                f"Expected 0-{self.total_documents - 1}"
            )
            return False

        # Check for out-of-range indices
        for idx in all_indices:
            if idx < 0 or idx >= self.total_documents:
                logger.error(f"ResultAggregator: Out-of-range index {idx}")
                return False

        logger.debug(
            f"ResultAggregator: Ordering validation passed for {len(all_indices)} results"
        )
        return True

    def create_missing_results(
        self, documents: List[Any], return_documents: bool = False
    ) -> List[RerankResult]:
        """
        Create placeholder results for missing documents to maintain complete document set.

        Args:
            documents: Original document list
            return_documents: Whether to include document content in results

        Returns:
            List of RerankResult objects for missing documents
        """
        if not self.batch_results or not documents:
            return []

        # Get all processed indices
        processed_indices = set()
        for batch_result_list in self.batch_results:
            for result in batch_result_list:
                processed_indices.add(result.index)

        # Find missing indices
        all_expected_indices = set(range(len(documents)))
        missing_indices = all_expected_indices - processed_indices

        if not missing_indices:
            return []

        logger.warning(
            f"ResultAggregator: Creating {len(missing_indices)} placeholder results "
            f"for missing indices {sorted(missing_indices)}"
        )

        missing_results = []
        for idx in sorted(missing_indices):
            # Create placeholder result
            document = None
            if return_documents and idx < len(documents):
                doc = documents[idx]
                if isinstance(doc, str):
                    document = RerankDocument(text=doc)
                elif isinstance(doc, dict) and "text" in doc:
                    document = RerankDocument(text=doc["text"])

            placeholder_result = RerankResult(
                document=document,
                index=idx,
                relevance_score=0.0,  # Neutral score for missing documents
            )
            missing_results.append(placeholder_result)

        return missing_results

    def get_complete_results(
        self,
        documents: List[Any],
        return_documents: bool = False,
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """
        Get complete results including placeholders for missing documents.

        Args:
            documents: Original document list
            return_documents: Whether to include document content
            top_n: Optional limit on number of results

        Returns:
            Complete list of results with placeholders for missing documents
        """
        # Get sorted results
        sorted_results = self.get_sorted_results(top_n)

        # Add missing results if any
        missing_results = self.create_missing_results(documents, return_documents)
        if missing_results:
            all_results = sorted_results + missing_results

            # Re-sort to include missing results (they have score 0.0, so will be at end)
            all_results = sorted(
                all_results, key=lambda x: x.relevance_score, reverse=True
            )

            # Apply top_n limit again if needed
            if top_n is not None:
                all_results = all_results[:top_n]

            logger.info(
                f"ResultAggregator: Final results: {len(all_results)} total "
                f"({len(sorted_results)} processed + {len(missing_results)} missing)"
            )

            return all_results

        return sorted_results
