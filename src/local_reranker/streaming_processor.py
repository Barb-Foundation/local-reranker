# -*- coding: utf-8 -*-
"""Streaming document processor for memory-efficient processing."""

import logging
from typing import Iterator, List, Union, Dict, Any, Optional, cast
from pathlib import Path
import json
import csv

from .models import RerankRequest
from .batch_processor import DocumentTextExtractor

logger = logging.getLogger(__name__)


class StreamingDocumentProcessor:
    """Memory-efficient document processor that handles large document sets."""

    def __init__(self, chunk_size: int = 1000):
        """Initialize streaming processor.

        Args:
            chunk_size: Number of documents to load into memory at once.
        """
        self.chunk_size = chunk_size

    def stream_documents_from_list(
        self, documents: List[Union[str, Dict[str, Any]]]
    ) -> Iterator[List[Union[str, Dict[str, Any]]]]:
        """Stream documents from a list in chunks.

        Args:
            documents: List of documents to stream.

        Yields:
            Chunks of documents.
        """
        for i in range(0, len(documents), self.chunk_size):
            yield documents[i : i + self.chunk_size]

    def stream_documents_from_file(
        self, file_path: str, document_field: str = "text"
    ) -> Iterator[List[str]]:
        """Stream documents from a JSON or CSV file.

        Args:
            file_path: Path to the file containing documents.
            document_field: Field name containing document text (for JSON).

        Yields:
            Chunks of document strings.
        """
        path_obj = Path(file_path)

        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path_obj}")

        if path_obj.suffix.lower() == ".json":
            yield from self._stream_from_json(path_obj, document_field)
        elif path_obj.suffix.lower() == ".csv":
            yield from self._stream_from_csv(path_obj)
        else:
            raise ValueError(f"Unsupported file format: {path_obj.suffix}")

    def _stream_from_json(
        self, file_path: Path, document_field: str
    ) -> Iterator[List[str]]:
        """Stream documents from JSON file."""
        documents = []

        with open(file_path, "r", encoding="utf-8") as f:
            if file_path.stat().st_size > 100 * 1024 * 1024:  # 100MB+
                # For large files, stream line by line
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        doc_text = self._extract_text_from_dict(data, document_field)
                        if doc_text:
                            documents.append(doc_text)

                            if len(documents) >= self.chunk_size:
                                yield documents
                                documents = []
                    except json.JSONDecodeError:
                        continue

                if documents:
                    yield documents
            else:
                # For smaller files, load all at once
                try:
                    data = json.load(f)

                    if isinstance(data, list):
                        for item in data:
                            doc_text = self._extract_text_from_dict(
                                item, document_field
                            )
                            if doc_text:
                                documents.append(doc_text)

                                if len(documents) >= self.chunk_size:
                                    yield documents
                                    documents = []
                    elif isinstance(data, dict):
                        # Handle nested structure
                        for key, value in data.items():
                            if isinstance(value, list):
                                for item in value:
                                    doc_text = self._extract_text_from_dict(
                                        item, document_field
                                    )
                                    if doc_text:
                                        documents.append(doc_text)

                                        if len(documents) >= self.chunk_size:
                                            yield documents
                                            documents = []

                    if documents:
                        yield documents

                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON file: {e}")

    def _stream_from_csv(self, file_path: Path) -> Iterator[List[str]]:
        """Stream documents from CSV file."""
        documents = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)

            # Skip header if present
            try:
                header = next(reader)
                # Try to identify text column
                text_col = 0
                for i, col in enumerate(header):
                    if col.lower() in ["text", "content", "document", "body"]:
                        text_col = i
                        break
            except StopIteration:
                text_col = 0

            for row in reader:
                if text_col < len(row):
                    doc_text = row[text_col].strip()
                    if doc_text:
                        documents.append(doc_text)

                        if len(documents) >= self.chunk_size:
                            yield documents
                            documents = []

            if documents:
                yield documents

    def _extract_text_from_dict(
        self, data: Dict[str, Any], document_field: str
    ) -> Optional[str]:
        """Extract text from dictionary document."""
        text = DocumentTextExtractor.extract(data, text_fields=[document_field])
        if text:
            return text

        text = DocumentTextExtractor.extract(data)
        return text

    def create_streaming_request(
        self,
        query: str,
        document_source: Union[List[Union[str, Dict[str, Any]]], str],
        return_documents: bool = False,
        top_n: Optional[int] = None,
        document_field: str = "text",
    ) -> Iterator[RerankRequest]:
        """Create streaming rerank requests.

        Args:
            query: Query text.
            document_source: Either a list of documents or path to a file.
            return_documents: Whether to return documents in results.
            top_n: Number of top results to return.
            document_field: Field name for document text (when loading from files).

        Yields:
            RerankRequest objects for each chunk.
        """
        if isinstance(document_source, list):
            for chunk in self.stream_documents_from_list(document_source):
                yield RerankRequest(
                    query=query,
                    documents=chunk,
                    return_documents=return_documents,
                    top_n=top_n,
                )
        elif isinstance(document_source, str):
            for chunk in self.stream_documents_from_file(
                document_source, document_field
            ):
                yield RerankRequest(
                    query=query,
                    documents=cast(List[Union[str, Dict[str, Any]]], chunk),
                    return_documents=return_documents,
                    top_n=top_n,
                )
        else:
            raise ValueError("document_source must be a list or file path string")

    def estimate_memory_usage(
        self, num_documents: int, avg_doc_length: int = 1000
    ) -> Dict[str, Union[float, int]]:
        """Estimate memory usage for processing documents.

        Args:
            num_documents: Number of documents to process.
            avg_doc_length: Average document length in characters.

        Returns:
            Estimated memory usage in MB.
        """
        # Rough estimation: each character ~1 byte, plus overhead
        chunk_memory = (self.chunk_size * avg_doc_length * 2) / (1024 * 1024)  # MB
        total_memory = (num_documents * avg_doc_length * 2) / (1024 * 1024)  # MB

        return {
            "chunk_memory_mb": chunk_memory,
            "total_memory_mb": total_memory,
            "recommended_chunk_size": min(
                self.chunk_size, int(512 * 1024 * 1024 / (avg_doc_length * 2))
            ),  # Target 512MB per chunk
        }
