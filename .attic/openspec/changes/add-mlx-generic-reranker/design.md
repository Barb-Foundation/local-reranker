## Context
The MLX backend currently expects every MLX-compatible Hugging Face model repository to ship a `rerank.py` module exposing an `MLXReranker` implementation (as done by `jinaai/jina-reranker-v3-mlx`). This assumption fails for other MLX models that only provide a base LM and tokenizer (for example `mku64/Qwen3-Reranker-0.6B-mlx-8Bit`), causing startup errors even though the underlying weights are usable.

The goal of this change is to introduce an internal MLX reranker wrapper that can turn a generic MLX LM into a cross-encoder reranker, and to wire it into the existing MLX backend as a fallback when `rerank.py` is not present.

## Goals / Non-Goals
- Goals:
  - Provide a built-in MLX reranker implementation that works for MLX-converted LMs without requiring a repo-specific `rerank.py`.
  - Preserve existing behavior for repos that do provide `rerank.py` (e.g. Jina's model).
  - Clearly define how projector weights and model config are used to derive embedding dimensions.
  - Improve error messages and logging for MLX model loading paths.
- Non-Goals:
  - Redesign the external `/rerank` API or CLI interface.
  - Implement advanced optimization strategies beyond basic batching already present in the MLX backend.
  - Guarantee identical ranking quality to repo-specific rerankers; the internal wrapper is a reasonable default, not a replacement for specialized training.

## Decisions
- Decision: Introduce an internal `MLXCrossEncoderReranker` (exact name TBD) that wraps `mlx_lm.load(model_path)` and exposes a `rerank(query, documents, top_n, return_embeddings)` method returning a list of result dicts compatible with the current MLX backend.
  - Rationale: Align with the existing `MLXReranker` contract from `rerank.py` while decoupling from model-specific scripts.

- Decision: Implement prompt formatting and embedding extraction using special marker tokens defined in the project, and derive their token IDs via the tokenizer instead of hard-coding numeric IDs.
  - Rationale: Avoid coupling to specific tokenizer vocabularies and make the implementation more portable across MLX models.

- Decision: Use a small MLX-based projector (e.g. two linear layers with a non-linearity) to map hidden states at marker positions into a fixed embedding space, with the input dimension taken from `config["hidden_size"]` and the output dimension configurable (default 512).
  - Rationale: Preserve the core idea of projecting model hidden states into a reranker-specific embedding space while allowing per-model flexibility.

- Decision: Update `_load_mlx_reranker` in `reranker_mlx.py` to:
  - Prefer a repo-provided `rerank.py` when present (current behavior).
  - Instantiate the internal `MLXCrossEncoderReranker` when `rerank.py` is absent, using the same `model_path` and optional `projector.safetensors`.
  - Log which path is used so operators can understand runtime behavior.
  - Rationale: Maintain backward compatibility while broadening supported MLX model layouts.

- Decision: Define projector-handling behavior explicitly:
  - If `projector.safetensors` is present and compatible with the configured projector, load those weights.
  - If `projector.safetensors` is missing or incompatible, either:
    - Fail with a clear error message, or
    - Fall back to using raw hidden states (no projector) as embeddings.
  - The exact choice will be configurable or clearly documented in the implementation stage.

## Risks / Trade-offs
- Risk: Ranking quality from the internal wrapper may be lower than a dedicated reranker model like Jina's.
  - Mitigation: Document that the internal MLX reranker is a generic fallback; for best quality, users should prefer curated reranker repos that ship dedicated logic.

- Risk: Performance overhead due to prompt length and cross-encoding behavior could impact latency for large document sets.
  - Mitigation: Reuse existing batching mechanisms (`BatchManager`) and keep the initial implementation simple. Optimize only if real workloads warrant it.

- Risk: Misconfiguration around projector dimensions or missing projector weights could lead to confusing runtime errors.
  - Mitigation: Centralize projector handling, validate dimensions early, and surface clear error messages in logs and exceptions.

## Migration Plan
- Add the internal MLX reranker module and wire it into the MLX backend loader.
- Ensure all tests continue to pass for existing MLX scenarios using `jinaai/jina-reranker-v3-mlx`.
- Add tests for the new fallback path using a model repo that does not provide `rerank.py`.
- Update MLX backend documentation to describe the fallback behavior and how to choose between specialized and generic MLX models.

## Open Questions
- Should the absence of `projector.safetensors` be treated as an error by default, or should the system automatically fall back to using raw hidden states as embeddings?
- Do we want configuration flags (environment variables or settings) to control the internal MLX reranker embedding dimension and projector behavior, or is a fixed 512-dimensional embedding sufficient for now?
- Should we expose any metrics (e.g. which MLX path was used, approximate embedding computation time) via logging or future observability hooks?
