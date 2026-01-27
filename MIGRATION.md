# Migration Guide: Batch Processing Refactoring

This guide documents the breaking changes introduced by the batch processing refactoring. These changes simplify the codebase, remove unused complexity, and improve maintainability.

## Overview

The batch processing refactoring (v2.0) removed asynchronous processing capabilities that were not used by the API or tests, and extracted common batch processing logic into a shared module. This reduces code duplication by ~60-70% across both reranker backends.

**Affected Versions**: Users upgrading from v1.x to v2.0

## Breaking Changes

### 1. Removed Async Methods from Reranker Protocol

**What Changed**:
- `rerank_async()` method removed from `Reranker` protocol
- `rerank_async_final()` method removed from `Reranker` protocol
- Async implementations removed from both `reranker_pytorch.py` and `reranker_mlx.py`

**Impact**:
- Direct library users who were calling async methods will need to use the synchronous `rerank()` method instead
- The `/v1/rerank` API endpoint continues to work identically (it only ever used the sync method)

**Migration**:

```python
# Before (v1.x)
results = await reranker.rerank_async(request)
final_results = await reranker.rerank_async_final(request)

# After (v2.0)
results = reranker.rerank(request)
```

### 2. Removed ProgressUpdate Model

**What Changed**:
- `ProgressUpdate` model removed from `models.py`
- This model was not used by the API or tests

**Impact**:
- Any code importing or using `ProgressUpdate` will break
- This affects only direct library users, not API consumers

**Migration**:
- Remove any imports of `ProgressUpdate`
- If you need progress tracking, you'll need to implement your own solution

### 3. Removed max_concurrent_batches from BatchManager

**What Changed**:
- `max_concurrent_batches` parameter removed from `BatchManager.__init__()`
- `_get_env_max_concurrent()` method removed
- Related concurrent batch processing logic removed

**Impact**:
- Code that explicitly sets `max_concurrent_batches` will raise a `TypeError`
- Batch processing now uses sequential processing with GPU vectorization only

**Migration**:

```python
# Before (v1.x)
batch_manager = BatchManager(
    batch_size=12,
    max_concurrent_batches=4
)

# After (v2.0)
batch_manager = BatchManager(
    batch_size=12
)
# Batches are processed sequentially with GPU vectorization
```

### 4. Removed Concurrent Batch Processing

**What Changed**:
- Concurrent batch processing removed from both backends
- ResultAggregator simplified (no concurrent results)
- BatchManager no longer tracks concurrent batches

**Impact**:
- No performance impact for single-GPU setups (GPU vectorization maintained)
- Slight simplification in logging and status reporting
- `BatchManager.get_status()` no longer returns concurrent batch info

**Migration**:
- No code changes required if using the standard `rerank()` method
- Monitor logs for sequential batch processing instead of concurrent processing

## New Features

### BatchProcessor Module

A new shared module `batch_processor.py` has been added with common batch processing logic:

```python
from local_reranker.batch_processor import BatchProcessor, process_batches

# Use shared batch processing
results = process_batches(
    request=request,
    batch_manager=batch_manager,
    model_predictor=my_predictor,
    backend_name="MyBackend"
)
```

**Key Components**:
- `process_batches()`: Main batch processing orchestration
- `DocumentTextExtractor`: Utility for extracting text from various document formats
- `BatchProcessor`: Utility class for processing batched results

### Improved Document Text Extraction

The new `DocumentTextExtractor` class provides robust text extraction:

```python
from local_reranker.batch_processor import DocumentTextExtractor

# Extract text from string
text = DocumentTextExtractor.extract("Hello World")

# Extract text from dict with custom fields
text = DocumentTextExtractor.extract(
    {"content": "Hello", "title": "World"},
    text_fields=["content", "title", "text"]
)
```

## API Compatibility

### Endpoints

The `/v1/rerank` API endpoint continues to work identically:

```bash
curl -X POST "http://localhost:8010/v1/rerank" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "jina-reranker-v2-base-multilingual",
           "query": "What are the benefits of using FastAPI?",
           "documents": ["FastAPI is fast", "Django is popular"],
           "top_n": 3,
           "return_documents": true
         }'
```

**No changes required** for API consumers.

### Request/Response Models

The following models remain unchanged:
- `RerankRequest`
- `RerankResponse`
- `RerankResult`
- `RerankDocument`

## Performance Characteristics

### GPU Vectorization Maintained

The refactoring maintains GPU vectorization benefits:

- Documents still processed in batches (not one-by-one)
- Optimal batch size calculation based on GPU memory unchanged
- Matrix operations remain parallelized at the GPU level

### Single-GPU Optimization

The concurrent batch processing that was removed provided no performance benefit for single-GPU setups:

- GPU vectorization already processes entire batches in parallel
- Sequential batch processing with GPU acceleration is optimal
- Memory usage more predictable without concurrent batches

**Performance Impact**: No regression in throughput for single-GPU configurations.

## Testing

If you have tests that use the removed async methods:

```python
# Before (v1.x)
@pytest.mark.asyncio
async def test_async_rerank():
    results = await reranker.rerank_async(request)
    assert len(results) > 0

# After (v2.0)
def test_sync_rerank():
    results = reranker.rerank(request)
    assert len(results) > 0
```

## Migration Checklist

- [ ] Update code that calls `rerank_async()` to use `rerank()`
- [ ] Remove imports of `ProgressUpdate` model
- [ ] Remove `max_concurrent_batches` parameter from `BatchManager` initialization
- [ ] Update tests that use async methods to use sync methods
- [ ] Remove environment variables related to concurrent batch processing
- [ ] Review and update custom backends that implement the `Reranker` protocol
- [ ] Verify your tests pass with `uv run pytest`

## Rollback Instructions

If you need to rollback to v1.x:

```bash
# Checkout the version before the refactoring
git checkout <commit_hash_before_refactoring>

# Reinstall
uv pip install -e ".[dev]"
```

## Support

If you encounter issues migrating:

1. Check the [README.md](README.md) for usage examples
2. Review the [PRD](docs/prd-batch-refactoring.md) for detailed design decisions
3. Check the test suite in `tests/` for examples of the new API
4. Open an issue on GitHub with details about your migration problem

## Summary of Changes

| Component | v1.x | v2.0 | Status |
|-----------|-----|------|--------|
| `rerank()` | ✓ | ✓ | Unchanged |
| `rerank_async()` | ✓ | ✗ | Removed |
| `rerank_async_final()` | ✓ | ✗ | Removed |
| `ProgressUpdate` | ✓ | ✗ | Removed |
| `max_concurrent_batches` | ✓ | ✗ | Removed |
| Concurrent batch processing | ✓ | ✗ | Removed |
| GPU vectorization | ✓ | ✓ | Maintained |
| BatchProcessor module | ✗ | ✓ | Added |
| DocumentTextExtractor | ✗ | ✓ | Added |
| API endpoint | ✓ | ✓ | Compatible |

## Next Steps

1. Update your code to use the new sync-only API
2. Run your test suite with `uv run pytest`
3. Verify performance meets your requirements
4. Consider using the new `BatchProcessor` for custom implementations
