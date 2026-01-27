## 1. Design and Spec
- [x] 1.1 Review existing MLX reranker change (`implement-mlx-reranker`) and current MLX backend behavior.
- [x] 1.2 Finalize internal MLX cross-encoder reranker responsibilities (prompting, projector, similarity) and fallback behavior.
- [x] 1.3 Validate this change with `openspec validate add-mlx-generic-reranker --strict`.

## 2. Implementation
- [x] 2.1 Add a new internal MLX reranker module (e.g. `mlx_cross_encoder.py`) that wraps `mlx_lm.load` and exposes `rerank(query, documents, top_n, return_embeddings)`.
- [x] 2.2 Implement projector handling based on model config and optional `projector.safetensors` file.
- [x] 2.3 Implement prompt formatting and hidden-state extraction using special marker tokens and tokenizer-derived token IDs.
- [x] 2.4 Update `reranker_mlx.Reranker` to prefer repo-provided `rerank.py` when present, and otherwise fall back to the internal MLX reranker.
- [x] 2.5 Improve MLX loader logging and error messages to surface which implementation path is used and why.

## 3. Testing and Validation
- [x] 3.1 Add unit tests for the internal MLX reranker wrapper (mocking `mlx_lm` and tokenizer) to verify prompt construction, embedding extraction, and ranking behavior.
- [x] 3.2 Add or update integration tests for the MLX backend to cover both `rerank.py`-present and `rerank.py`-absent model scenarios.
- [x] 3.3 Manually verify MLX backend behavior with `jinaai/jina-reranker-v3-mlx` and a generic MLX model (e.g. `mku64/Qwen3-Reranker-0.6B-mlx-8Bit`).
- [x] 3.4 Update documentation (README/CLI help) to describe MLX fallback behavior and supported model layouts.
