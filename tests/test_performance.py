# -*- coding: utf-8 -*-
"""Performance tests for long document handling."""

import time
import pytest
from local_reranker.reranker_pytorch import Reranker as PyTorchReranker
from local_reranker.text_processing import TextProcessor
from local_reranker.models import RerankRequest


class TestTextProcessing:
    """Test text processing utilities."""
    
    def test_truncation_head(self):
        """Test head truncation strategy."""
        processor = TextProcessor(max_length=100, truncation_strategy="head")
        text = "a" * 200
        result = processor.process_document(text)
        assert len(result) == 1
        assert len(result[0]) == 100
        assert result[0] == "a" * 100
    
    def test_truncation_tail(self):
        """Test tail truncation strategy."""
        processor = TextProcessor(max_length=100, truncation_strategy="tail")
        text = "a" * 200
        result = processor.process_document(text)
        assert len(result) == 1
        assert len(result[0]) == 100
        assert result[0] == "a" * 100
    
    def test_truncation_middle(self):
        """Test middle truncation strategy."""
        processor = TextProcessor(max_length=100, truncation_strategy="middle")
        text = "a" * 200
        result = processor.process_document(text)
        assert len(result) == 1
        assert len(result[0]) == 100
        # Should be 50 chars from start + 50 chars from end
        assert result[0] == "a" * 50 + "a" * 50
    
    def test_chunking(self):
        """Test document chunking."""
        processor = TextProcessor(max_length=100, enable_chunking=True, chunk_overlap=20)
        text = "a" * 250
        result = processor.process_document(text)
        assert len(result) == 3  # 250 chars with 100 max and 20 overlap = 3 chunks
        assert all(len(chunk) <= 100 for chunk in result)
    
    def test_short_document_passthrough(self):
        """Test that short documents pass through unchanged."""
        processor = TextProcessor(max_length=1000)
        text = "short text"
        result = processor.process_document(text)
        assert len(result) == 1
        assert result[0] == text


@pytest.mark.slow
class TestPerformance:
    """Performance tests for reranking with long documents."""
    
    @pytest.fixture
    def short_documents(self):
        """Fixture providing short documents."""
        return [
            "The Eiffel Tower is located in Paris.",
            "Paris is the capital city of France.",
            "The Louvre Museum is in Paris.",
            "France is a country in Western Europe.",
        ]
    
    @pytest.fixture
    def long_documents(self):
        """Fixture providing long documents (5000-10000 chars)."""
        base_text = "This is a sample document that contains repetitive text to simulate long passages. " * 50
        return [
            base_text * 2,  # ~5000 chars
            base_text * 3,  # ~7500 chars  
            base_text * 4,  # ~10000 chars
            base_text * 2.5,  # ~6250 chars
        ]
    
    def test_short_document_performance(self, short_documents):
        """Test performance with short documents (baseline)."""
        request = RerankRequest(
            query="What is the capital of France?",
            documents=short_documents,
            top_n=2
        )
        
        # Test with text processor (no truncation needed)
        processor = TextProcessor(max_length=8192)
        start_time = time.time()
        processed_docs, _ = processor.process_documents(short_documents)
        processing_time = time.time() - start_time
        
        assert len(processed_docs) == len(short_documents)
        assert processing_time < 0.01  # Should be very fast
    
    def test_long_document_truncation_performance(self, long_documents):
        """Test performance improvement with truncation."""
        request = RerankRequest(
            query="What is this document about?",
            documents=long_documents,
            max_length=1000,
            truncation_strategy="head"
        )
        
        processor = TextProcessor(max_length=1000, truncation_strategy="head")
        start_time = time.time()
        processed_docs, _ = processor.process_documents(long_documents)
        processing_time = time.time() - start_time
        
        assert len(processed_docs) == len(long_documents)
        assert all(len(doc) <= 1000 for doc in processed_docs)
        assert processing_time < 0.01  # Should be very fast
    
    def test_chunking_performance(self, long_documents):
        """Test performance with chunking enabled."""
        processor = TextProcessor(max_length=2000, enable_chunking=True, chunk_overlap=200)
        start_time = time.time()
        processed_docs, original_indices = processor.process_documents(long_documents)
        processing_time = time.time() - start_time
        
        assert len(processed_docs) > len(long_documents)  # Should create more chunks
        assert all(len(doc) <= 2000 for doc in processed_docs)
        assert processing_time < 0.05  # Should still be fast
    
    @pytest.mark.integration
    def test_reranker_with_long_documents(self, long_documents):
        """Test actual reranker performance with long documents."""
        # This test requires the model to be loaded and may be slow
        pytest.skip("Skipping integration test - requires model loading")
        
        # Test without text processing (should be slow)
        request_no_processing = RerankRequest(
            query="What is this document about?",
            documents=long_documents,
            top_n=2
        )
        
        # Test with text processing (should be faster)
        request_with_processing = RerankRequest(
            query="What is this document about?", 
            documents=long_documents,
            top_n=2,
            max_length=1000,
            truncation_strategy="head"
        )
        
        # Compare timing (would need actual reranker instance)
        # This is a placeholder for the actual performance comparison
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])