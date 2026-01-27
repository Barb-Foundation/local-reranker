# -*- coding: utf-8 -*-
"""Protocol-based interface for reranker implementations."""

from typing import Protocol, List, Optional, AsyncGenerator, runtime_checkable

from .models import RerankRequest, RerankResult, ProgressUpdate


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranker implementations to enable multiple AI framework backends."""

    def __init__(self, model_name: str = ..., device: Optional[str] = None) -> None:
        """Initialize the reranker with a model name and device.

        Args:
            model_name: The name of the model to load.
            device: The device to run the model on ('cpu', 'cuda', 'mps').
                   If None, attempts to auto-detect.
        """
        ...

    def rerank(self, request: RerankRequest) -> List[RerankResult]:
        """Reranks documents based on the request.

        Args:
            request: The rerank request containing query, documents, and options.

        Returns:
            A list of rerank results, sorted by relevance score (descending).
        """
        ...

    def rerank_async(
        self, request: RerankRequest
    ) -> AsyncGenerator[ProgressUpdate, None]:
        """Async reranking with progress updates.

        Args:
            request: The rerank request containing query, documents, and options.

        Yields:
            ProgressUpdate objects with current processing status.
        """
        ...

    async def rerank_async_final(self, request: RerankRequest) -> List[RerankResult]:
        """Async reranking that returns final results.

        Args:
            request: The rerank request containing query, documents, and options.

        Returns:
            A list of rerank results, sorted by relevance score (descending).
        """
        ...
