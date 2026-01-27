[PRD]
# PRD: Batch Processing Refactoring

## Overview
Comprehensive refactoring of batch processing logic in `reranker_pytorch.py` and `reranker_mlx.py`. The current implementation contains approximately 500 lines of duplicated code, complex async batch processing that provides no performance benefit for single-GPU setups, and unused async methods.

**Goal**: Simplify codebase by removing unused complexity, eliminating duplication, and creating a shared batch processing module. This refactoring will improve code maintainability, readability, and reduce technical debt.

**Impact**: PyTorch backend reduced from 567 to ~150-180 lines (65-73% reduction), MLX backend reduced from 723 to ~180-220 lines (65-75% reduction).

## Goals
- Reduce total lines of code by ~60-70% across both reranker backends
- Eliminate duplication by extracting common batch processing logic into a shared module
- Remove unused async methods (`rerank_async`, `rerank_async_final`) that are not used by API or tests
- Maintain GPU vectorization benefits while removing misleading concurrent batch processing
- Standardize behavior between backends (filter empty documents instead of placeholders)
- Add comprehensive unit tests for new batch processor module
- Document all breaking changes in README and migration guide
- Validate performance to ensure no regression in GPU vectorization

## Quality Gates

These commands must pass for every user story:
- `uv run pytest` - Run all tests
- `uv run ruff check .` - Linting
- `uv run pyright` - Type checking

Code must pass human review before merging.

## User Stories

### US-001: Remove async methods from protocol
**Description:** As a developer, I want to remove unused async methods from the protocol to simplify the interface.

**Acceptance Criteria:**
- [ ] Remove `rerank_async()` method definition from `src/local_reranker/reranker.py` protocol
- [ ] Remove `rerank_async_final()` method definition from `src/local_reranker/reranker.py` protocol
- [ ] Remove `AsyncGenerator` import from typing imports if no longer needed
- [ ] Remove `ProgressUpdate` from imports if no longer needed
- [ ] Protocol file reduced to ~35 lines

### US-002: Remove async implementations from PyTorch backend
**Description:** As a developer, I want to remove the unused async implementations from the PyTorch backend.

**Acceptance Criteria:**
- [ ] Remove `rerank_async()` method from `src/local_reranker/reranker_pytorch.py`
- [ ] Remove `rerank_async_final()` method from `src/local_reranker/reranker_pytorch.py`
- [ ] Remove async-related imports (`AsyncGenerator`, `asyncio`, `time` if only used for async)
- [ ] Remove any async-only helper methods
- [ ] All existing tests still pass
- [ ] File reduced to ~250 lines

### US-003: Remove async implementations from MLX backend
**Description:** As a developer, I want to remove the unused async implementations from the MLX backend.

**Acceptance Criteria:**
- [ ] Remove `rerank_async()` method from `src/local_reranker/reranker_mlx.py`
- [ ] Remove `rerank_async_final()` method from `src/local_reranker/reranker_mlx.py`
- [ ] Remove async-related imports (`AsyncGenerator`, `asyncio`, `time` if only used for async)
- [ ] Remove any async-only helper methods
- [ ] All existing tests still pass
- [ ] File reduced to ~370 lines

### US-004: Remove async-related models and imports
**Description:** As a developer, I want to clean up unused models and imports across the codebase.

**Acceptance Criteria:**
- [ ] Remove `ProgressUpdate` model from `src/local_reranker/models.py` or remove unused imports if keeping for future use
- [ ] Remove `ProgressUpdate` imports from `reranker.py`, `reranker_pytorch.py`, and `reranker_mlx.py`
- [ ] Update any remaining references to async types in type hints
- [ ] All existing tests still pass

### US-005: Simplify BatchManager
**Description:** As a developer, I want to remove the unused concurrent batch processing parameters from BatchManager.

**Acceptance Criteria:**
- [ ] Remove `max_concurrent_batches` parameter from `BatchManager.__init__()` in `src/local_reranker/batch_manager.py`
- [ ] Remove `_get_env_max_concurrent()` method from `BatchManager`
- [ ] Remove `max_concurrent_batches` from `get_status()` return dictionary
- [ ] Remove `RERANKER_MAX_CONCURRENT_BATCHES` environment variable handling
- [ ] Keep batch size calculation and memory management logic
- [ ] Update initialization in both reranker backends to remove `max_concurrent_batches` parameter
- [ ] All existing tests still pass

### US-006: Create BatchProcessor utility module
**Description:** As a developer, I want to create a shared batch processing utility to eliminate code duplication.

**Acceptance Criteria:**
- [ ] Create new file `src/local_reranker/batch_processor.py`
- [ ] Implement `BatchProcessor` class with static methods only (no instantiation, no `__init__`)
- [ ] Implement `process_batched_results()` static method that flattens, sorts, and filters batch results
- [ ] Implement `_convert_model_output_to_results()` static method for converting model outputs to `RerankResult` objects
- [ ] Handle both string and dict document formats in result conversion
- [ ] Filter empty documents (skip them, no placeholders)
- [ ] Include comprehensive docstrings for all methods
- [ ] Add type hints throughout
- [ ] Module should be ~50-80 lines of code (keep it simple)
- [ ] Example implementation pattern:
  ```python
  class BatchProcessor:
      @staticmethod
      def process_batched_results(batch_results, top_n=None):
          all_results = []
          for batch in batch_results:
              all_results.extend(batch)
          all_results.sort(key=lambda x: x.relevance_score, reverse=True)
          return all_results[:top_n] if top_n else all_results
  ```

### US-007: Refactor PyTorch backend to use BatchProcessor
**Description:** As a developer, I want to simplify the PyTorch backend by using the shared BatchProcessor.

**Acceptance Criteria:**
- [ ] Remove `ResultAggregator` import and usage from `src/local_reranker/reranker_pytorch.py`
- [ ] Refactor `rerank()` method to use simple list operations for result aggregation
- [ ] Create `_prepare_inputs()` helper method that creates query-document pairs
- [ ] Create `_run_inference()` helper method that calls model.predict()
- [ ] Create `_convert_batch_to_results()` helper method that converts model output to `RerankResult` objects
- [ ] Filter empty documents before batch creation
- [ ] Use `BatchProcessor.process_batched_results()` for final result sorting and top_n filtering
- [ ] File reduced to ~150-180 lines
- [ ] All existing tests pass
- [ ] Performance equivalent or better than before
- [ ] Example refactored pattern:
  ```python
  def rerank(self, request: RerankRequest) -> List[RerankResult]:
      batches, batch_indices = self.batch_manager.create_batches(request)
      all_batch_results = []
      for batch_docs, original_indices in zip(batches, batch_indices):
          inputs = self._prepare_inputs(request.query, batch_docs)
          scores = self._run_inference(inputs)
          batch_results = self._convert_batch_to_results(scores, original_indices, request.return_documents)
          all_batch_results.append(batch_results)
      return BatchProcessor.process_batched_results(all_batch_results, request.top_n)
  ```

### US-008: Refactor MLX backend to use BatchProcessor
**Description:** As a developer, I want to simplify the MLX backend by using the shared BatchProcessor.

**Acceptance Criteria:**
- [ ] Remove `ResultAggregator` import and usage from `src/local_reranker/reranker_mlx.py`
- [ ] Remove placeholder creation logic for failed documents (standardize on filtering)
- [ ] Refactor `rerank()` method to use simple list operations for result aggregation
- [ ] Create `_prepare_inputs()` helper method that returns documents as-is
- [ ] Create `_run_inference()` helper method that calls model.rerank()
- [ ] Create `_convert_batch_to_results()` helper method that converts model output to `RerankResult` objects
- [ ] Filter empty documents before batch creation
- [ ] Use `BatchProcessor.process_batched_results()` for final result sorting and top_n filtering
- [ ] File reduced to ~180-220 lines
- [ ] All existing tests pass
- [ ] Performance equivalent or better than before
- [ ] Example refactored pattern:
  ```python
  def rerank(self, request: RerankRequest) -> List[RerankResult]:
      batches, batch_indices = self.batch_manager.create_batches(request)
      all_batch_results = []
      for batch_docs, original_indices in zip(batches, batch_indices):
          inputs = self._prepare_inputs(batch_docs)
          raw_results = self._run_inference(request.query, inputs, request.return_documents)
          batch_results = self._convert_batch_to_results(raw_results, original_indices)
          all_batch_results.append(batch_results)
      return BatchProcessor.process_batched_results(all_batch_results, request.top_n)
   ```
 
### US-009: Add BatchProcessor unit tests
**Description:** As a developer, I want comprehensive unit tests for the new BatchProcessor module.

**Acceptance Criteria:**
- [ ] Create `tests/test_batch_processor.py` with unit tests for `BatchProcessor` class
- [ ] Add tests for `process_batched_rerank()` with various batch sizes (1, 10, 100 documents)
- [ ] Add tests for `_convert_scores_to_results()` method
- [ ] Add tests for edge cases (empty documents, single document, large document counts)
- [ ] Use mocked model calls for fast unit tests
- [ ] Achieve 80%+ test coverage for BatchProcessor
- [ ] All tests pass with `uv run pytest`

### US-010: Update documentation
**Description:** As a developer, I want to update documentation to reflect the breaking changes.

**Acceptance Criteria:**
- [ ] Update `README.md` to remove mentions of async support if present
- [ ] Create `MIGRATION.md` documenting breaking changes for existing users
- [ ] Document removed methods: `rerank_async()`, `rerank_async_final()`
- [ ] Document removed parameters: `max_concurrent_batches`
- [ ] Document new `BatchProcessor` module
- [ ] Update code examples to show simplified usage

### US-011: Run performance benchmarks
**Description:** As a developer, I want to validate that refactoring did not cause performance regression.

**Acceptance Criteria:**
- [ ] Create benchmark script to measure GPU vectorization throughput
- [ ] Run benchmarks with 100, 500, 1000 documents before refactoring (baseline)
- [ ] Run same benchmarks after refactoring (current)
- [ ] Document results showing no regression in GPU vectorization performance
- [ ] Document results showing maintained or improved batch processing speed
- [ ] Add benchmark results to PR description

## Functional Requirements

### Phase 1: Remove Unused Async Complexity
1. **FR-1.1**: Remove `rerank_async()` method definition from `reranker.py` protocol
2. **FR-1.2**: Remove `rerank_async_final()` method definition from `reranker.py` protocol
3. **FR-1.3**: Remove `rerank_async()` implementation from `reranker_pytorch.py`
4. **FR-1.4**: Remove `rerank_async_final()` implementation from `reranker_pytorch.py`
5. **FR-1.5**: Remove `rerank_async()` implementation from `reranker_mlx.py`
6. **FR-1.6**: Remove `rerank_async_final()` implementation from `reranker_mlx.py`
7. **FR-1.7**: Remove `ProgressUpdate` model from `models.py` or remove unused imports if keeping for future use
8. **FR-1.8**: Remove async-related imports that are no longer needed after deletions

### Phase 2: Simplify Core Processing
9. **FR-2.1**: Simplify `rerank()` method in `reranker_pytorch.py` to use sequential batch processing only
10. **FR-2.2**: Simplify `rerank()` method in `reranker_mlx.py` to use sequential batch processing only
11. **FR-2.3**: Create `_prepare_inputs()` helper method in `reranker_pytorch.py` for backend-specific input preparation (creates query-document pairs)
12. **FR-2.4**: Create `_prepare_inputs()` helper method in `reranker_mlx.py` for backend-specific input preparation (returns documents as-is)
13. **FR-2.5**: Create `_run_inference()` helper method in `reranker_pytorch.py` for model prediction calls
14. **FR-2.6**: Create `_run_inference()` helper method in `reranker_mlx.py` for MLX reranking calls
15. **FR-2.7**: Create `_convert_batch_to_results()` helper method in both backends to convert model outputs to `RerankResult` objects
16. **FR-2.8**: Remove `ResultAggregator` import and usage from both backends
17. **FR-2.9**: Replace `ResultAggregator` with simple Python list operations (extend, sort, slice)
18. **FR-2.10**: Standardize on filtering empty documents instead of creating placeholder results

### Phase 3: Extract Commonality
19. **FR-3.1**: Create new file `src/local_reranker/batch_processor.py`
20. **FR-3.2**: Implement `BatchProcessor` utility class with static methods only (no instantiation needed)
21. **FR-3.3**: Implement `process_batched_results()` static method that flattens, sorts, and applies top_n limit
22. **FR-3.4**: Implement `_convert_model_output_to_results()` static method for converting model outputs to `RerankResult` objects
23. **FR-3.5**: Refactor `rerank()` in `reranker_pytorch.py` to use `BatchProcessor.process_batched_results()`
24. **FR-3.6**: Refactor `rerank()` in `reranker_mlx.py` to use `BatchProcessor.process_batched_results()`

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

## Non-Goals (Out of Scope)

1. **NG-1**: This refactoring will NOT implement new features or functionality. The goal is simplification, not enhancement.
2. **NG-2**: This refactoring will NOT optimize for multi-GPU setups. Single-GPU GPU vectorization is maintained.
3. **NG-3**: This refactoring will NOT implement streaming or incremental results. The API returns complete results only.
4. **NG-4**: This refactoring will NOT implement new async patterns. All async methods are being removed.
5. **NG-5**: This refactoring will NOT maintain backward compatibility for unused async methods. Breaking changes are acceptable and documented.
6. **NG-6**: This refactoring will NOT add performance optimizations beyond simplification. Performance validation ensures no regression.
7. **NG-7**: This refactoring will NOT integrate with external services or APIs beyond what already exists.

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
- `ResultAggregator` will be removed entirely in this PR (not deprecated)

### API Compatibility
- The `/v1/rerank` endpoint (`api.py`) will continue to work identically
- `RerankRequest` and `RerankResponse` models are unchanged
- Only the protocol interface changes (async methods removed)
- This is a breaking change only for direct library users who call async methods

### Code Organization
- The new `BatchProcessor` should be a utility class with static methods only, not an abstract base class
- **Implementation hint**: Use `@staticmethod` decorators, no `__init__` method needed
- Backend-specific logic (input preparation, inference calls) stays in respective backend files
- Common logic (batch iteration, result aggregation, sorting) moves to shared module
- Both backends will use the same simple pattern: create batches → process sequentially → sort results

### BatchProcessor Implementation
- **Simple utility approach**: Keep BatchProcessor to ~50-80 lines of code
- **process_batched_results()**: Three operations only - flatten, sort, slice
  ```python
  all_results = []
  for batch in batch_results:
      all_results.extend(batch)
  all_results.sort(key=lambda x: x.relevance_score, reverse=True)
  return all_results[:top_n] if top_n else all_results
  ```
- **_convert_model_output_to_results()**: Handle both string and dict document formats
  ```python
  doc_text = doc if isinstance(doc, str) else doc.get("text", "")
  return RerankResult(document=RerankDocument(text=doc_text) if return_docs else None, ...)
  ```
- No placeholders for failed documents - filter them out instead
- No ResultAggregator dependency - use simple list operations

### ResultAggregator Removal
- Remove all imports and usage of `ResultAggregator` from both backends
- Do not use deprecation warnings - remove entirely in this PR
- If statistics are needed, add simple logging in backends directly
- If validation is needed, add inline assertions or logging

### Error Handling
- Maintain existing error handling patterns
- Empty documents should be filtered before processing (not turned into placeholders)
- Model errors should propagate up to API layer
- Logging should be preserved for debugging
- Batch failures should raise exceptions (not return placeholder results)

## Success Metrics

1. **Code Reduction**: PyTorch backend reduced to ≤180 lines, MLX backend reduced to ≤220 lines
2. **Test Coverage**: All existing tests pass, new batch processor has ≥80% test coverage
3. **Performance**: GPU vectorization throughput equivalent or better than before (validated with benchmarks)
4. **Complexity**: Cyclomatic complexity reduced by ≥50% in both backends
5. **Duplication**: Zero lines of duplicated batch processing logic between backends
6. **Documentation**: README updated, breaking changes documented, migration guide created
7. **Timeline**: Complete within 2 weeks (urgent priority)

## Open Questions

1. **Benchmark Baseline**: Do existing benchmarks exist that we should use for comparison, or should we establish a new baseline?
    - *Context*: Need to validate that GPU vectorization performance is maintained.

2. **BatchManager Methods**: Should we keep memory monitoring methods (`get_status()`, `should_use_fallback()`) or simplify further?
    - *Context*: These provide useful debugging info but add complexity.

3. **Documentation Format**: Should breaking changes be documented in a separate MIGRATION.md or added to README.md?
    - *Context*: Need clear communication for any external users of the library.
[/PRD]
