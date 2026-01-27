# -*- coding: utf-8 -*-
"""Pydantic models for API request/response validation."""

from typing import List, Optional, Union, Dict, Any, AsyncGenerator
from pydantic import BaseModel, Field


class RerankRequest(BaseModel):
    """Request model for the /v1/rerank endpoint."""

    model: Optional[str] = Field(
        default=None,
        description="The model to use for reranking. If not provided, uses the default model.",
    )
    query: str = Field(..., description="The query text.")
    documents: List[Union[str, Dict[str, Any]]] = Field(
        ...,
        description="A list of documents to rerank. Can be strings or dictionaries.",
    )
    top_n: Optional[int] = Field(
        default=None,
        description="The number of highest-scoring documents to return. If None, returns all documents sorted by score.",
    )
    return_documents: Optional[bool] = Field(
        default=False,
        description="Whether to return the document content in the response.",
    )  # Defaulting to False, but PRD implies True. Let's check Jina docs later.


class RerankDocument(BaseModel):
    """Model for a document within the RerankResponse results."""

    text: str


class RerankResult(BaseModel):
    """Model for a single reranking result."""

    document: Optional[RerankDocument] = Field(
        default=None,
        description="The document content, returned if return_documents is True.",
    )
    index: int = Field(
        ..., description="The original index of the document in the input list."
    )
    relevance_score: float = Field(
        ..., description="The relevance score computed by the model."
    )


class RerankResponse(BaseModel):
    """Response model for the /v1/rerank endpoint."""

    id: Optional[str] = Field(
        default=None, description="A unique identifier for the request."
    )  # Often added by the framework
    results: List[RerankResult] = Field(..., description="List of reranked results.")
    # Potentially add usage stats here later, matching Jina's API if applicable


class ProgressUpdate(BaseModel):
    """Model for progress updates during async reranking."""

    current_batch: int = Field(..., description="Current batch being processed.")
    total_batches: int = Field(..., description="Total number of batches.")
    documents_processed: int = Field(
        ..., description="Number of documents processed so far."
    )
    total_documents: int = Field(
        ..., description="Total number of documents to process."
    )
    current_batch_results: int = Field(
        ..., description="Number of results in current batch."
    )
    elapsed_time: float = Field(..., description="Elapsed time in seconds.")
