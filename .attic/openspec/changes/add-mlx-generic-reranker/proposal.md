# Change: Add internal MLX generic reranker fallback

## Why
The current MLX backend assumes that every MLX-compatible Hugging Face model repository ships a `rerank.py` script that exposes an `MLXReranker` implementation. This works for `jinaai/jina-reranker-v3-mlx`, but fails for other MLX models such as `mku64/Qwen3-Reranker-0.6B-mlx-8Bit` which only provide a base Qwen3 LM and tokenizer. As a result, users see a hard startup failure (`rerank.py not found`) even though the model weights are valid and could be used for reranking.

## What Changes
- Introduce an internal MLX cross-encoder reranker wrapper, built on top of `mlx_lm`, that can turn a base LM into a reranker using prompt markers, hidden-state extraction, a small projector, and cosine similarity scoring.
- Update the MLX backend loader so that:
  - When a model repo includes `rerank.py`, it continues to use the repo-provided `MLXReranker` implementation (current behavior for Jina's model).
  - When `rerank.py` is missing, it falls back to the internal MLX reranker wrapper instead of failing at startup.
- Define clear behavior for projector weights (`projector.safetensors`) and config-driven dimensions so the internal reranker remains robust across MLX models.
- Improve logging and error messages around MLX model loading to make it clear when the built-in fallback is used vs when the repo-specific implementation is used.

## Impact
- Affected specs: `mlx-reranker` (add requirements for internal fallback and model compatibility when `rerank.py` is absent).
- Affected code (later implementation):
  - `src/local_reranker/reranker_mlx.py` (model loading and fallback selection).
  - New internal MLX reranker module (e.g. `src/local_reranker/mlx_cross_encoder.py`).
  - Optional documentation updates for MLX backend usage and model compatibility.
- No breaking API changes; the external `/rerank` API and CLI interface remain compatible. The change only broadens the set of MLX models that can be used successfully.
