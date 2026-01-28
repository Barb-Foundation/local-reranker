[PRD]
# PRD: Adopt Jina MLX Reranker

## Overview

Replace current MLX reranker implementation that attempts to load `rerank.py` from model files with a proper Jina-style MLX reranker implementation directly integrated into codebase. The current implementation has a fallback mechanism that produces incorrect scores (0.01 to -0.07 instead of expected 0.2 to 0.7) due to mismatched projector architecture and random weight initialization.

## Jina Reference Implementation

The Jina reference implementation can be studied in `./docs/jina-reranker/`. This directory contains the official Jina MLX reranker implementation that should be ported to our codebase:

### Directory Contents

```
./docs/jina-reranker/
├── added_tokens.json              # Special tokens configuration
├── config.json                    # Model configuration
├── generation_config.json          # Generation parameters
├── merges.txt                    # Byte-pair encoding merges
├── model.safetensors             # Model weights
├── model.safetensors.index.json   # Model index
├── projector.safetensors         # Projector MLP weights (512, 1024) and (512, 512)
├── requirements.txt               # Python dependencies
├── rerank.py                    # Main Jina MLX reranker implementation (250 lines)
├── special_tokens_map.json       # Special token mappings
├── test_examples.py              # Example usage
├── tokenizer.json               # Tokenizer configuration
├── tokenizer_config.json         # Tokenizer settings
└── vocab.json                   # Vocabulary
```

### Key Implementation Details

The reference implementation (`rerank.py`) provides:
- `MLPProjector` class with fixed architecture (1024→512→512, ReLU, no bias)
- `load_projector` function for weight loading from safetensors
- `format_docs_prompts_func` for prompt formatting with special tokens
- `MLXReranker` class with `_compute_single_batch` and `rerank` methods
- Cosine similarity without pre-normalization
- Hard-coded special token IDs: doc_embed_token_id=151670, query_embed_token_id=151671

## Goals

- Fix MLX reranker scores to match expected range (0.2 to 0.7)
- Port Jina reference implementation from `./docs/jina-reranker/rerank.py` to our codebase
- Remove dependency on external rerank.py files from model repositories
- Improve MLX reranker reliability with explicit error messages
- Maintain correct ranking behavior matching PyTorch baseline
- Separate unit tests from integration tests for faster development

## Quality Gates

These commands must pass for every user story:
- `uv run ruff check src/` - Linting
- `uv run pyright src/` - Type checking
- `uv run pytest tests/ -m "not (integration or slow)"` - Run unit tests only (skip integration and slow tests)

Integration tests are run explicitly with:
- `uv run pytest tests/ -m integration` - Run integration tests only

## User Stories

### US-001: Create JinaMLPProjector class
**Description:** As a developer, I want a fixed-architecture MLP projector (1024→512→512) so that Jina projector weights can be loaded correctly.

**Acceptance Criteria:**
- [ ] Create JinaMLPProjector class with fixed dimensions (1024→512→512)
- [ ] Use ReLU activation (not GELU)
- [ ] No bias terms on either layer
- [ ] Implement __call__ method for forward pass

### US-002: Create projector weight loading function
**Description:** As a developer, I want a safe function to load projector weights so that invalid configurations are caught early.

**Acceptance Criteria:**
- [ ] Implement _load_projector function that loads from projector.safetensors
- [ ] Verify linear1.weight shape is (512, 1024)
- [ ] Verify linear2.weight shape is (512, 512)
- [ ] Raise FileNotFoundError if projector file doesn't exist
- [ ] Raise ValueError with explicit message if shapes don't match

### US-003: Create prompt formatting helper functions
**Description:** As a developer, I want helper functions to format Jina prompts so that special tokens are placed correctly.

**Acceptance Criteria:**
- [ ] Implement _sanitize_input function to remove special tokens from text
- [ ] Implement _format_jina_prompt function with complete prompt structure
- [ ] Include fixed system message for reranking tasks
- [ ] Format documents with `<passage id="{i}">` tags
- [ ] Place embed tokens immediately after document/query text

### US-004: Create JinaMLXReranker class with initialization
**Description:** As a developer, I want a JinaMLXReranker class that loads the model and projector so that reranking can be performed.

**Acceptance Criteria:**
- [ ] Implement __init__ method accepting model_path and projector_path
- [ ] Load MLX model and tokenizer using mlx_lm.load
- [ ] Load projector using _load_projector function
- [ ] Define special tokens: doc_embed_token_id=151670, query_embed_token_id=151671
- [ ] Raise RuntimeError if model or projector loading fails

### US-005: Implement core batch computation method
**Description:** As a developer, I want a _compute_single_batch method that extracts embeddings and computes cosine similarity so that documents can be reranked.

**Acceptance Criteria:**
- [ ] Format prompt using _format_jina_prompt
- [ ] Tokenize prompt and get hidden states from model
- [ ] Find positions of special tokens (151670, 151671)
- [ ] Extract embeddings at special token positions
- [ ] Project embeddings through MLP
- [ ] Compute cosine similarity WITHOUT pre-normalization: (A·B) / (||A|| * ||B||)
- [ ] Return query_embeds, doc_embeds, scores
- [ ] Raise ValueError if special tokens not found

### US-006: Implement public rerank method
**Description:** As a developer, I want a rerank method that returns ranked documents so that it can be used via the Reranker protocol.

**Acceptance Criteria:**
- [ ] Implement rerank method accepting query, documents, top_n, return_embeddings
- [ ] Process all documents in single batch via _compute_single_batch
- [ ] Sort results by relevance score (descending)
- [ ] Return list of dicts with: document, relevance_score, index, embedding (if requested)
- [ ] Support top_n parameter to limit results

### US-007: Update reranker_mlx.py to use JinaMLXReranker
**Description:** As a developer, I want reranker_mlx.py to use the new JinaMLXReranker so that rerank.py loading is removed.

**Acceptance Criteria:**
- [ ] Add import: from .jina_mlx_reranker import JinaMLXReranker
- [ ] Remove import: importlib.util
- [ ] Remove _load_repo_reranker method completely
- [ ] Replace _load_mlx_reranker to directly instantiate JinaMLXReranker
- [ ] Remove all rerank.py file existence checks
- [ ] Raise explicit RuntimeError on failure instead of falling back

### US-008: Update module exports
**Description:** As a developer, I want JinaMLXReranker exported from __init__.py so that it can be imported by users.

**Acceptance Criteria:**
- [ ] Check src/local_reranker/__init__.py for MLXCrossEncoderReranker export
- [ ] Add JinaMLXReranker to imports and __all__ list

### US-009: Remove outdated MLX reranker tests
**Description:** As a developer, I want to remove tests for functionality that no longer exists so that the test suite is clean.

**Acceptance Criteria:**
- [ ] Remove test_load_mlx_reranker_uses_fallback_when_file_missing
- [ ] Remove test_load_mlx_reranker_falls_back_on_repo_error
- [ ] Remove test_load_mlx_reranker_prefers_repo_impl
- [ ] Keep test_mlx_reranker_implements_protocol
- [ ] Update test_initialization_runtime_error to mock JinaMLXReranker

### US-010: Add unit tests for JinaMLXReranker
**Description:** As a developer, I want unit tests for JinaMLXReranker so that its components work correctly in isolation.

**Acceptance Criteria:**
- [ ] Create tests/test_jina_mlx_reranker.py
- [ ] Add test for projector loading with correct shapes
- [ ] Add test for prompt format matching Jina specification
- [ ] Add test for special token extraction
- [ ] Add test for cosine similarity formula
- [ ] Add test for full reranking pipeline
- [ ] Mark all tests with @pytest.mark.unit
- [ ] Mark slow tests with @pytest.mark.slow (if any)

### US-011: Add integration test with real model
**Description:** As a developer, I want an integration test with the actual Jina MLX model so that scores match expected range.

**Acceptance Criteria:**
- [ ] Create integration test in tests/test_jina_mlx_reranker.py
- [ ] Mark test with @pytest.mark.integration
- [ ] Mark test with @pytest.mark.slow
- [ ] Load real jinaai/jina-reranker-v3-mlx model
- [ ] Test with same query/documents from .attic/curl2.sh
- [ ] Verify scores are in range [0.2, 0.7] (not [0.01, -0.07])
- [ ] Verify ranking matches [0, 3, 9] indices
- [ ] Verify scores close to PyTorch baseline

## Functional Requirements

- FR-1: JinaMLXReranker must load projector.safetensors with exact shapes (512, 1024) and (512, 512)
- FR-2: Special tokens must be hard-coded: doc_embed_token_id=151670, query_embed_token_id=151671
- FR-3: Cosine similarity must be computed without pre-normalization: (A·B) / (||A|| * ||B||)
- FR-4: Prompt format must match Jina specification with fixed system message
- FR-5: Missing projector file must raise explicit FileNotFoundError
- FR-6: Invalid projector shapes must raise explicit ValueError with expected vs actual
- FR-7: No attempt to load rerank.py from model files
- FR-8: No fallback to MLXCrossEncoderReranker for Jina models
- FR-9: Unit tests must run with `pytest -m "not (integration or slow)"`
- FR-10: Integration tests must run with `pytest -m integration`
- FR-11: Slow tests must be marked with @pytest.mark.slow

## Non-Goals (Out of Scope)

- Supporting non-Jina MLX models in this implementation
- Making special token IDs configurable
- Adding fallback paths for alternative MLX models
- Removing MLXCrossEncoderReranker class (kept for future use with other models)
- Auto-detecting system theme or environment
- Modifying PyTorch reranker implementation

## Technical Considerations

- JinaMLXReranker is specific to Jina reranker v3-mlx and will not work with other models
- Projector dimensions are fixed (1024→512→512) as this is part of the trained model
- MLXCrossEncoderReranker remains in codebase for potential use with non-Jina models
- Unit tests should use mocks to avoid loading real models
- Integration tests require downloading jinaai/jina-reranker-v3-mlx (~2GB)
- Projector uses ReLU activation, not GELU (critical difference from internal fallback)
- Score range [0.2, 0.7] indicates proper projector loading, [0.01, -0.07] indicates random weights

## Success Metrics

- MLX reranker produces scores in expected range [0.2, 0.7]
- Ranking matches PyTorch baseline ([0, 3, 9] indices)
- No attempt to load rerank.py from model files
- All unit tests pass with `pytest -m "not (integration or slow)"`
- Code passes linter and type checker
- Explicit error messages for missing/invalid projector files

## Open Questions

- Should we add pytest configuration to separate unit/integration/slow tests by default?
- Should integration tests be optional (require --run-integration flag)?
[/PRD]