# -*- coding: utf-8 -*-
"""Text processing utilities for handling long documents."""

from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Handles text truncation and chunking for long documents."""
    
    def __init__(
        self,
        max_length: int = 8192,
        truncation_strategy: str = "head",
        chunk_overlap: int = 100,
        enable_chunking: bool = False
    ):
        """Initialize text processor.
        
        Args:
            max_length: Maximum characters per document
            truncation_strategy: How to truncate ("head", "tail", "middle")
            chunk_overlap: Overlap characters when chunking
            enable_chunking: Whether to chunk long documents
        """
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.chunk_overlap = chunk_overlap
        self.enable_chunking = enable_chunking
        
        if truncation_strategy not in ["head", "tail", "middle"]:
            raise ValueError(f"Invalid truncation strategy: {truncation_strategy}")
    
    def process_document(self, text: str) -> List[str]:
        """Process a single document, returning chunks or truncated text.
        
        Args:
            text: The document text to process
            
        Returns:
            List of processed text chunks (usually 1 item unless chunking enabled)
        """
        if len(text) <= self.max_length:
            return [text]
        
        if self.enable_chunking:
            return self._chunk_text(text)
        else:
            return [self._truncate_text(text)]
    
    def _truncate_text(self, text: str) -> str:
        """Truncate text using the configured strategy."""
        if len(text) <= self.max_length:
            return text
        
        if self.truncation_strategy == "head":
            return text[:self.max_length]
        elif self.truncation_strategy == "tail":
            return text[-self.max_length:]
        elif self.truncation_strategy == "middle":
            # Take from beginning and end, with preference to beginning
            half_length = self.max_length // 2
            return text[:half_length] + text[-half_length:]
        else:
            # Default to head
            return text[:self.max_length]
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text into overlapping segments."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.max_length
            
            if end >= len(text):
                # Last chunk - take remaining text
                chunks.append(text[start:])
                break
            
            # Try to break at word boundary
            chunk = text[start:end]
            last_space = chunk.rfind(' ')
            
            if last_space > self.max_length * 0.8:  # If we found a good break point
                chunk = chunk[:last_space]
                end = start + last_space
            
            chunks.append(chunk)
            
            # Move start position with overlap
            start = max(end - self.chunk_overlap, start + 1)
        
        logger.debug(f"Chunked text of length {len(text)} into {len(chunks)} chunks")
        return chunks
    
    def process_documents(self, documents: List[str]) -> Tuple[List[str], List[int]]:
        """Process multiple documents, tracking original indices.
        
        Args:
            documents: List of document texts
            
        Returns:
            Tuple of (processed_documents, original_indices)
        """
        processed = []
        indices = []
        
        for i, doc in enumerate(documents):
            chunks = self.process_document(doc)
            processed.extend(chunks)
            indices.extend([i] * len(chunks))
        
        return processed, indices