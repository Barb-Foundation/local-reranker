# -*- coding: utf-8 -*-
"""Tokenization optimization with caching for improved performance."""

import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from threading import RLock
import pickle
import os

try:
    import torch
    from transformers import AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    AutoTokenizer = None

logger = logging.getLogger(__name__)


@dataclass
class TokenizationResult:
    """Result of tokenization operation."""

    input_ids: List[int]
    attention_mask: List[int]
    token_count: int
    processing_time: float
    cache_hit: bool = False


@dataclass
class CacheStats:
    """Tokenization cache statistics."""

    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    cache_size: int = 0
    total_time_saved: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests

    @property
    def miss_rate(self) -> float:
        """Cache miss rate (0.0 to 1.0)."""
        return 1.0 - self.hit_rate


class TokenizationCache:
    """Thread-safe tokenization cache with LRU eviction."""

    def __init__(
        self,
        max_size: int = 10000,
        max_memory_mb: float = 512.0,
        persist_to_disk: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """Initialize tokenization cache.

        Args:
            max_size: Maximum number of cached entries.
            max_memory_mb: Maximum memory usage in MB.
            persist_to_disk: Whether to persist cache to disk.
            cache_dir: Directory for disk cache (auto-generated if None).
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.persist_to_disk = persist_to_disk

        # In-memory cache
        self._cache: Dict[str, TokenizationResult] = {}
        self._access_order: List[str] = []
        self._lock = RLock()
        self._current_memory_bytes = 0

        # Statistics
        self._stats = CacheStats()

        # Disk cache setup
        if persist_to_disk:
            self.cache_dir = cache_dir or os.path.expanduser(
                "~/.cache/reranker_tokenizer"
            )
            os.makedirs(self.cache_dir, exist_ok=True)
            self._disk_cache_file = os.path.join(self.cache_dir, "tokenizer_cache.pkl")
            self._load_disk_cache()

        logger.info(
            f"Tokenization cache initialized: max_size={max_size}, "
            f"max_memory={max_memory_mb}MB, persist={persist_to_disk}"
        )

    def get(self, text: str, tokenizer_name: str) -> Optional[TokenizationResult]:
        """Get cached tokenization result.

        Args:
            text: Text to tokenize.
            tokenizer_name: Name of tokenizer used.

        Returns:
            Cached result or None if not found.
        """
        key = self._make_key(text, tokenizer_name)

        with self._lock:
            if key in self._cache:
                # Move to end of access order (LRU)
                self._access_order.remove(key)
                self._access_order.append(key)

                result = self._cache[key]
                result.cache_hit = True

                self._stats.hits += 1
                self._stats.total_requests += 1

                return result

            self._stats.misses += 1
            self._stats.total_requests += 1

            return None

    def put(
        self,
        text: str,
        tokenizer_name: str,
        input_ids: List[int],
        attention_mask: List[int],
        processing_time: float,
    ):
        """Cache tokenization result.

        Args:
            text: Original text.
            tokenizer_name: Name of tokenizer.
            input_ids: Token IDs.
            attention_mask: Attention mask.
            processing_time: Time taken for tokenization.
        """
        key = self._make_key(text, tokenizer_name)

        # Estimate memory usage
        result_size = self._estimate_result_size(input_ids, attention_mask)

        with self._lock:
            # Evict if necessary
            while (
                len(self._cache) >= self.max_size
                or self._current_memory_bytes + result_size > self.max_memory_bytes
            ):
                if not self._evict_oldest():
                    break

            # Add new entry
            result = TokenizationResult(
                input_ids=input_ids.copy(),
                attention_mask=attention_mask.copy(),
                token_count=len(input_ids),
                processing_time=processing_time,
                cache_hit=False,
            )

            self._cache[key] = result
            self._access_order.append(key)
            self._current_memory_bytes += result_size

            # Update stats
            self._stats.cache_size = len(self._cache)

            # Persist to disk if enabled
            if self.persist_to_disk:
                self._save_to_disk()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                total_requests=self._stats.total_requests,
                cache_size=self._stats.cache_size,
                total_time_saved=self._stats.total_time_saved,
            )

    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._current_memory_bytes = 0
            self._stats = CacheStats()

            if self.persist_to_disk and os.path.exists(self._disk_cache_file):
                os.remove(self._disk_cache_file)

        logger.info("Tokenization cache cleared")

    def _make_key(self, text: str, tokenizer_name: str) -> str:
        """Create cache key from text and tokenizer name."""
        # Use SHA-256 for consistent, collision-resistant keys
        content = f"{tokenizer_name}:{text}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _estimate_result_size(
        self, input_ids: List[int], attention_mask: List[int]
    ) -> int:
        """Estimate memory usage of tokenization result."""
        # Rough estimation: each int = 28 bytes (Python overhead), plus overhead
        return (len(input_ids) + len(attention_mask)) * 28 + 200

    def _evict_oldest(self) -> bool:
        """Evict oldest entry from cache."""
        if not self._access_order:
            return False

        oldest_key = self._access_order.pop(0)
        if oldest_key in self._cache:
            result = self._cache.pop(oldest_key)
            self._current_memory_bytes -= self._estimate_result_size(
                result.input_ids, result.attention_mask
            )
            return True

        return False

    def _load_disk_cache(self):
        """Load cache from disk."""
        if not os.path.exists(self._disk_cache_file):
            return

        try:
            with open(self._disk_cache_file, "rb") as f:
                data = pickle.load(f)

            with self._lock:
                self._cache = data.get("cache", {})
                self._access_order = data.get("access_order", [])
                self._current_memory_bytes = sum(
                    self._estimate_result_size(r.input_ids, r.attention_mask)
                    for r in self._cache.values()
                )
                self._stats.cache_size = len(self._cache)

            logger.info(f"Loaded {len(self._cache)} entries from disk cache")

        except Exception as e:
            logger.warning(f"Failed to load disk cache: {e}")

    def _save_to_disk(self):
        """Save cache to disk periodically."""
        if hasattr(self, "_last_save_time"):
            if time.time() - self._last_save_time < 60:  # Save at most once per minute
                return

        try:
            data = {
                "cache": self._cache,
                "access_order": self._access_order,
                "timestamp": time.time(),
            }

            with open(self._disk_cache_file, "wb") as f:
                pickle.dump(data, f)

            self._last_save_time = time.time()

        except Exception as e:
            logger.warning(f"Failed to save disk cache: {e}")


class OptimizedTokenizer:
    """Tokenizer with caching and performance optimization."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache: Optional[TokenizationCache] = None,
        max_length: int = 512,
        truncation: bool = True,
    ):
        """Initialize optimized tokenizer.

        Args:
            model_name: Name of tokenizer model.
            cache: Tokenization cache instance.
            max_length: Maximum sequence length.
            truncation: Whether to truncate sequences.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.truncation = truncation
        self.cache = cache or TokenizationCache()

        # Initialize tokenizer
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library not available")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info(f"Loaded tokenizer: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer {model_name}: {e}")
            raise

        # Performance tracking
        self._total_tokenizations = 0
        self._total_time = 0.0

    def tokenize(
        self, texts: Union[str, List[str]], return_tensors: str = "pt"
    ) -> Union[TokenizationResult, List[TokenizationResult]]:
        """Tokenize text(s) with caching.

        Args:
            texts: Single text or list of texts to tokenize.
            return_tensors: Format of returned tensors.

        Returns:
            Tokenization result(s).
        """
        single_text = isinstance(texts, str)
        texts_list = [texts] if single_text else texts

        results = []

        for text in texts_list:
            start_time = time.time()

            # Check cache first
            cached_result = self.cache.get(text, self.model_name)
            if cached_result:
                results.append(cached_result)
                continue

            # Tokenize
            try:
                encoded = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=self.truncation,
                    padding=False,  # Handle padding separately if needed
                    return_tensors=None,  # Get as Python lists
                )

                processing_time = time.time() - start_time

                # Cache the result
                self.cache.put(
                    text=text,
                    tokenizer_name=self.model_name,
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    processing_time=processing_time,
                )

                result = TokenizationResult(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    token_count=len(encoded["input_ids"]),
                    processing_time=processing_time,
                    cache_hit=False,
                )

                results.append(result)

                # Update stats
                self._total_tokenizations += 1
                self._total_time += processing_time

            except Exception as e:
                logger.error(f"Tokenization failed for text: {e}")
                # Return empty result on failure
                results.append(
                    TokenizationResult(
                        input_ids=[],
                        attention_mask=[],
                        token_count=0,
                        processing_time=time.time() - start_time,
                        cache_hit=False,
                    )
                )

        return results[0] if single_text else results

    def tokenize_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> List[TokenizationResult]:
        """Tokenize a batch of texts efficiently.

        Args:
            texts: List of texts to tokenize.
            batch_size: Batch size for processing.

        Returns:
            List of tokenization results.
        """
        all_results = []

        # Process in batches to manage memory
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_results = self.tokenize(batch_texts)
            all_results.extend(batch_results)

        return all_results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get tokenizer performance statistics."""
        cache_stats = self.cache.get_stats()

        avg_tokenization_time = (
            self._total_time / self._total_tokenizations
            if self._total_tokenizations > 0
            else 0
        )

        return {
            "tokenizer": {
                "model_name": self.model_name,
                "max_length": self.max_length,
                "total_tokenizations": self._total_tokenizations,
                "total_time": self._total_time,
                "avg_time_per_tokenization": avg_tokenization_time,
            },
            "cache": {
                "hit_rate": cache_stats.hit_rate,
                "miss_rate": cache_stats.miss_rate,
                "total_requests": cache_stats.total_requests,
                "cache_size": cache_stats.cache_size,
                "hits": cache_stats.hits,
                "misses": cache_stats.misses,
            },
        }

    def clear_cache(self):
        """Clear tokenizer cache."""
        self.cache.clear()
        self._total_tokenizations = 0
        self._total_time = 0.0


# Global tokenizer cache instance
_global_tokenizer_cache: Optional[TokenizationCache] = None


def get_global_tokenizer_cache() -> TokenizationCache:
    """Get or create global tokenizer cache."""
    global _global_tokenizer_cache
    if _global_tokenizer_cache is None:
        _global_tokenizer_cache = TokenizationCache(
            max_size=5000, max_memory_mb=256.0, persist_to_disk=True
        )
    return _global_tokenizer_cache


def create_optimized_tokenizer(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_global_cache: bool = True,
) -> OptimizedTokenizer:
    """Create optimized tokenizer with caching.

    Args:
        model_name: Name of tokenizer model.
        use_global_cache: Whether to use global cache.

    Returns:
        Optimized tokenizer instance.
    """
    cache = get_global_tokenizer_cache() if use_global_cache else None
    return OptimizedTokenizer(model_name=model_name, cache=cache)
