# -*- coding: utf-8 -*-
"""Protocol-based interface for reranker implementations."""

from typing import Protocol, List, Optional, runtime_checkable

from .models import RerankRequest, RerankResult


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranker implementations to enable multiple AI framework backends."""

    def __init__(
        self,
        model_name: str = ...,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """Initialize the reranker with a model name and device.

        Args:
            model_name: The name of the model to load.
            device: The device to run the model on ('cpu', 'cuda', 'mps').
                   If None, attempts to auto-detect.
            batch_size: Number of documents per batch (auto-detected if None).
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
