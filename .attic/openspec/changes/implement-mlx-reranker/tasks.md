# Tasks: Implement MLX Reranker

## Implementation Tasks

### 1. Add MLX Dependencies
- [x] Add `mlx` and `mlx-lm` to `pyproject.toml` dependencies
- [x] Update `pyproject.toml` with MLX-specific dependency notes
- [x] Test dependency installation on Apple Silicon

### 2. Create MLX Reranker Implementation
- [x] Create `src/local_reranker/reranker_mlx.py` with MLX reranker class
- [x] Implement the `Reranker` protocol with MLX backend
- [x] Add MLXReranker import from rerank module with ImportError handling
- [x] Implement MLX model initialization with model_path parameter
- [x] Add support for custom model_path and projector_path parameters
- [x] Add MLX-specific error handling and logging
- [x] Implement request format conversion to MLX parameters (query, documents, top_n, return_embeddings)
- [x] Implement result format conversion from MLX dict to RerankResult objects
- [x] Handle document text extraction from MLX results
- [x] Support embedding return when return_embeddings is True
- [x] Add device compatibility handling (MLX auto-detects Apple Silicon)

### 3. Update Configuration System
- [x] Add `"mlx"` to `RERANKER_DEFAULTS` in `config.py`
- [x] Set default model to `"jinaai/jina-reranker-v3-mlx"`
- [x] Update `get_available_rerankers()` to include MLX option
- [x] Add MLX description and Apple Silicon optimization notes

### 4. Update API Integration
- [x] Modify `api.py` lifespan function to handle MLX reranker type
- [x] Add conditional import for MLX dependencies
- [x] Implement MLX reranker instantiation with error handling
- [x] Add MLX-specific error messages and fallbacks

### 5. Update CLI Support
- [x] Modify `cli.py` to support MLX reranker selection
- [x] Add MLX to available reranker types in help text
- [x] Add MLX validation and error handling
- [x] Update CLI documentation for MLX option

### 6. Create Comprehensive Tests
- [x] Add MLX reranker tests to `tests/test_reranker.py`
- [x] Create protocol compliance tests for MLX implementation
- [x] Add mocked MLXReranker class tests for CI/CD compatibility
- [x] Test MLX initialization with ImportError handling
- [x] Test MLX model loading with custom paths
- [x] Test request format conversion to MLX parameters
- [x] Test result format conversion from MLX dict to RerankResult
- [x] Test embedding handling when return_embeddings is True
- [x] Test document text extraction from MLX results
- [x] Test relevance_score conversion to float
- [x] Test index preservation from MLX results
- [x] Test multilingual content handling
- [x] Test top_n filtering works correctly
- [x] Add API integration tests with MLX backend
- [x] Add CLI integration tests for MLX option

### 7. Update Documentation
- [x] Update `README.md` with MLX installation instructions
- [x] Document MLX backend usage and configuration
- [x] Add Apple Silicon optimization notes
- [x] Update environment variable documentation
- [x] Add troubleshooting section for MLX-specific issues

### 8. Validation and Testing
- [x] Run full test suite with MLX implementation
- [x] Test API endpoints with MLX backend
- [x] Test CLI functionality with MLX option
- [x] Validate configuration loading and validation
- [x] Performance testing on Apple Silicon (if available)
- [x] Cross-platform compatibility testing

## Dependencies and Ordering

### Sequential Dependencies
1. **Dependencies** must be added before implementation
2. **MLX Implementation** must be created before configuration updates
3. **Configuration** must be updated before API/CLI integration
4. **Tests** should be written after implementation but before final validation

### Parallelizable Work
- Documentation updates can be done alongside implementation
- API and CLI integration can be developed in parallel
- Different test categories can be written simultaneously

## Validation Criteria

### Functional Requirements
- [x] MLX reranker implements protocol correctly
- [x] API endpoints work with MLX backend
- [x] CLI supports MLX reranker selection
- [x] Configuration properly handles MLX settings
- [x] Error handling works for all MLX-specific scenarios

### Quality Requirements
- [x] All tests pass (unit, integration, CLI)
- [x] Code follows project conventions and style guidelines
- [x] Documentation is complete and accurate
- [x] No regression in existing PyTorch functionality
- [x] Performance improvement on Apple Silicon devices

### Platform Requirements
- [x] Works correctly on Apple Silicon (M1/M2/M3)
- [x] Gracefully handles MLX unavailability on other platforms
- [x] Provides clear error messages and installation guidance
- [x] Maintains backward compatibility with existing configurations