# PRD: Batch Processing Refactoring

## Introduction/Overview

This PRD outlines a comprehensive refactoring of batch processing logic in `reranker_pytorch.py` and `reranker_mlx.py`. The current implementation contains approximately 500 lines of duplicated code, complex async batch processing that provides no performance benefit for single-GPU setups, and unused async methods.

**Goal**: Simplify codebase by removing unused complexity, eliminating duplication, and creating a shared batch processing module. This refactoring will improve code maintainability, readability, and reduce technical debt, making it easier to onboard new developers.

**Impact**: This is a breaking change that removes unused async methods. The refactoring reduces PyTorch backend from 567 lines to ~150-180 lines (65-73% reduction) and MLX backend from 723 lines to ~180-220 lines (65-75% reduction).

---

## Goals

1. **Code Complexity Reduction**: Reduce total lines of code by ~60-70% across both reranker backends
2. **Eliminate Duplication**: Extract common batch processing logic into a shared module
3. **Remove Unused Complexity**: Delete async methods (`rerank_async`, `rerank_async_final`) that are not used by API or tests
4. **Simplify Processing**: Maintain GPU vectorization benefits while removing misleading concurrent batch processing
5. **Test Coverage**: Add unit tests for new batch processor module
6. **Documentation**: Clearly document all breaking changes in README and migration guide
7. **Performance Validation**: Benchmark before and after to ensure no regression in GPU vectorization performance

---

## User Stories

### As a Developer Maintaining the Codebase
- **Given**: I need to understand how to reranker processes batches of documents
- **When**: I read the code in `reranker_pytorch.py` or `reranker_mlx.py`
- **Then**: I should see clear, simple logic that's easy to follow, not 500+ lines of duplicated complex batching code

### As a New Developer Onboarding
- **Given**: I'm learning how to reranker works and need to add a new backend
- **When**: I examine the existing implementations
- **Then**: I should see a shared batch processing module that I can reuse, minimal code in each backend, and clear separation of concerns

### As a Developer Running Tests
- **Given**: I execute the test suite after refactoring
- **When**: Tests run
- **Then**: All existing tests must pass, and new unit tests for the batch processor should provide good coverage

### As a Developer Using the API
- **Given**: I'm using the `/v1/rerank` endpoint
- **When**: I send a rerank request with 100 documents
- **Then**: The request should process with the same or better performance (GPU vectorization maintained), and results should be identical to before refactoring

---

## Functional Requirements

### Phase 1: Remove Unused Async Complexity

1. **FR-1.1**: Remove `rerank_async()` method definition from `reranker.py` protocol (lines 34-45)
2. **FR-1.2**: Remove `rerank_async_final()` method definition from `reranker.py` protocol (lines 47-56)
3. **FR-1.3**: Remove `rerank_async()` implementation from `reranker_pytorch.py` (lines 256-439)
4. **FR-1.4**: Remove `rerank_async_final()` implementation from `reranker_pytorch.py` (lines 441-566)
5. **FR-1.5**: Remove `rerank_async()` implementation from `reranker_mlx.py` (lines 372-596)
6. **FR-1.6**: Remove `rerank_async_final()` implementation from `reranker_mlx.py` (lines 598-722)
7. **FR-1.7**: Remove `ProgressUpdate` model from `models.py` (lines 61-75) or remove unused imports if keeping for future use
8. **FR-1.8**: Remove async-related imports that are no longer needed after deletions

### Phase 2: Simplify Core Processing

9. **FR-2.1**: Simplify `rerank()` method in `reranker_pytorch.py` to use sequential batch processing only
10. **FR-2.2**: Simplify `rerank()` method in `reranker_mlx.py` to use sequential batch processing only
11. **FR-2.3**: Create `_prepare_inputs()` helper method in `reranker_pytorch.py` for backend-specific input preparation (creates query-document pairs)
12. **FR-2.4**: Create `_prepare_inputs()` helper method in `reranker_mlx.py` for backend-specific input preparation (returns documents as-is)
13. **FR-2.5**: Create `_run_inference()` helper method in `reranker_pytorch.py` for model prediction calls
14. **FR-2.6**: Create `_run_inference()` helper method in `reranker_mlx.py` for MLX reranking calls
15. **FR-2.7**: Create `_convert_to_results()` helper method in both backends to convert model outputs to `RerankResult` objects
16. **FR-2.8**: Remove `ResultAggregator` usage from both backends
17. **FR-2.9**: Replace `ResultAggregator` with simple Python list operations (extend, sort, slice)
18. **FR-2.10**: Mark `ResultAggregator` class as deprecated with deprecation warning or remove entirely

### Phase 3: Extract Commonality

19. **FR-3.1**: Create new file `src/local_reranker/batch_processor.py`
20. **FR-3.2**: Implement `BatchProcessor` class with shared batch processing logic
21. **FR-3.3**: Implement `process_batched_rerank()` static method in `BatchProcessor` that orchestrates sequential batch processing with GPU vectorization
22. **FR-3.4**: Implement `_convert_scores_to_results()` static method in `BatchProcessor` for result conversion
23. **FR-3.5**: Refactor `rerank()` in `reranker_pytorch.py` to use `BatchProcessor.process_batched_rerank()`
24. **FR-3.6**: Refactor `rerank()` in `reranker_mlx.py` to use `BatchProcessor.process_batched_rerank()`

### Phase 4: Simplify BatchManager

25. **FR-4.1**: Remove `max_concurrent_batches` parameter from `BatchManager.__init__()`
26. **FR-4.2**: Remove `_get_env_max_concurrent()` method from `BatchManager`
27. **FR-4.3**: Remove `max_concurrent_batches` from `get_status()` return dictionary
28. **FR-4.4**: Keep optimal batch size calculation logic (GPU memory-based)
29. **FR-4.5**: Keep simple batch creation logic
30. **FR-4.6**: Keep memory management methods if they provide value

### Phase 5: Testing and Documentation

31. **FR-5.1**: Add unit tests for `BatchProcessor` class in `tests/test_batch_processor.py`
32. **FR-5.2**: Add tests for `process_batched_rerank()` method with various batch sizes
33. **FR-5.3**: Add tests for `_convert_scores_to_results()` method
34. **FR-5.4**: Ensure all existing tests pass after refactoring: `test_reranker.py`, `test_mlx_reranker.py`, `test_integration.py`
35. **FR-5.5**: Update `README.md` to remove mentions of async support if present
36. **FR-5.6**: Create migration guide documenting breaking changes for existing users
37. **FR-5.7**: Run performance benchmarks before and after refactoring
38. **FR-5.8**: Document benchmark results to validate no regression in GPU vectorization performance

---

## Non-Goals (Out of Scope)

1. **NG-1**: This refactoring will NOT implement new features or functionality. The goal is simplification, not enhancement.
2. **NG-2**: This refactoring will NOT optimize for multi-GPU setups. Single-GPU GPU vectorization is maintained.
3. **NG-3**: This refactoring will NOT implement streaming or incremental results. The API returns complete results only.
4. **NG-4**: This refactoring will NOT implement new async patterns. All async methods are being removed.
5. **NG-5**: This refactoring will NOT maintain backward compatibility for unused async methods. Breaking changes are acceptable and documented.
6. **NG-6**: This refactoring will NOT add performance optimizations beyond simplification. Performance validation ensures no regression.
7. **NG-7**: This refactoring will NOT integrate with external services or APIs beyond what already exists.

---

## Design Considerations

### Code Organization

- The new `BatchProcessor` should be a utility class with static methods, not an abstract base class
- Backend-specific logic (input preparation, inference calls) stays in respective backend files
- Common logic (batch iteration, result aggregation, sorting) moves to shared module
- Both backends will use the same simple pattern: create batches → process sequentially → sort results

### Maintainability

- Each method should have a single responsibility
- Helper methods should be well-named and documented with docstrings
- The main `rerank()` method should read like a high-level summary of the process
- Type hints should be used throughout for clarity

### Error Handling

- Maintain existing error handling patterns
- Empty documents should be filtered before processing
- Model errors should propagate up to API layer
- Logging should be preserved for debugging

---

## Technical Considerations

### GPU Vectorization (Must Preserve)

- Batch processing must continue to leverage GPU parallel matrix operations
- Optimal batch size calculation based on memory must remain
- Documents should still be processed in batches (not one-by-one)
- Performance should be equivalent or better than before refactoring

### Dependencies

- Existing dependencies remain unchanged: `torch`, `sentence-transformers`, `mlx`, `mlx-lm`
- No new dependencies are required
- `BatchManager` class will continue to be used by both backends
- `ResultAggregator` will be deprecated but not immediately removed (use deprecation warning)

### API Compatibility

- The `/v1/rerank` endpoint (`api.py`) will continue to work identically
- `RerankRequest` and `RerankResponse` models are unchanged
- Only the protocol interface changes (async methods removed)
- This is a breaking change only for direct library users who call async methods

### Testing Framework

- Use `pytest` for all tests
- Use `pytest-mock` for mocking model calls
- Tests should be fast (unit tests) and comprehensive
- Integration tests should verify end-to-end behavior

---

## Open Questions

1. **ResultAggregator Deprecation**: Should we use a deprecation warning for one version before removal, or remove immediately given no external dependencies?
   - *Context*: `ResultAggregator` adds complexity but may be used internally. No external usage found in codebase.

2. **ProgressUpdate Model**: Should we keep `ProgressUpdate` model for potential future use, or remove entirely?
   - *Context*: It's currently unused, but might be useful if we add progress updates later.

3. **Benchmark Baseline**: Do existing benchmarks exist that we should use for comparison, or should we establish a new baseline?
   - *Context*: Need to validate that GPU vectorization performance is maintained.

4. **BatchManager Methods**: Should we keep memory monitoring methods (`get_status()`, `should_use_fallback()`) or simplify further?
   - *Context*: These provide useful debugging info but add complexity.

5. **Documentation Format**: Should breaking changes be documented in a separate MIGRATION.md or added to README.md?
   - *Context*: Need clear communication for any external users of the library.

---

## Success Criteria

1. **Code Reduction**: PyTorch backend reduced to ≤180 lines, MLX backend reduced to ≤220 lines
2. **Test Coverage**: All existing tests pass, new batch processor has ≥80% test coverage
3. **Performance**: GPU vectorization throughput equivalent or better than before (validated with benchmarks)
4. **Complexity**: Cyclomatic complexity reduced by ≥50% in both backends
5. **Duplication**: Zero lines of duplicated batch processing logic between backends
6. **Documentation**: README updated, breaking changes documented, migration guide created
7. **Timeline**: Complete within 2 weeks (urgent priority)
