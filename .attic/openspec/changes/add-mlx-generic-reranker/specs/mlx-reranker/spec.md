## ADDED Requirements
### Requirement: MLX Generic Reranker Fallback
The system SHALL provide an internal MLX-based reranker implementation that can operate when a model repository does not ship a `rerank.py` module.

#### Scenario: MLX fallback without rerank.py
- **WHEN** the MLX backend initializes with a model repository that does not contain `rerank.py`
- **THEN** it SHALL instantiate an internal MLX reranker wrapper built on top of `mlx_lm` using the downloaded model files
- **AND** it SHALL avoid failing API startup solely due to the absence of `rerank.py`
- **AND** it SHALL log that the internal MLX fallback reranker has been selected for this model.

#### Scenario: MLX preference for repo-provided reranker
- **WHEN** the MLX backend initializes with a model repository that contains `rerank.py` exposing a compatible `MLXReranker` implementation
- **THEN** it SHALL prefer the repo-provided `MLXReranker` for reranking
- **AND** it SHALL only use the internal MLX fallback when `rerank.py` is not present or cannot be loaded.

### Requirement: MLX Generic Reranker Behavior
The internal MLX reranker wrapper SHALL expose a rerank interface compatible with the existing MLX backend expectations.

#### Scenario: MLX internal rerank interface
- **WHEN** the internal MLX reranker is used
- **THEN** it SHALL accept `query`, `documents`, optional `top_n`, and `return_embeddings` parameters
- **AND** it SHALL return a list of result dictionaries containing `document`, `relevance_score`, and `index`
- **AND** it SHALL include an `embedding` field when `return_embeddings` is True, and omit or null it when `return_embeddings` is False.

#### Scenario: MLX internal rerank scoring
- **WHEN** the internal MLX reranker processes a request
- **THEN** it SHALL construct a prompt that encodes the query and all documents in a single sequence using clearly delimited marker tokens
- **AND** it SHALL run the MLX model to obtain hidden states
- **AND** it SHALL derive a query embedding and document embeddings from hidden states at marker positions using a small projector or directly from hidden states
- **AND** it SHALL compute relevance scores via similarity between the query embedding and each document embedding and sort documents by decreasing relevance score.

### Requirement: MLX Fallback Diagnostics
The system SHALL provide clear diagnostics about which MLX reranker implementation path is used for a given model.

#### Scenario: MLX fallback logging
- **WHEN** the MLX backend selects the internal fallback reranker because `rerank.py` is missing or unusable
- **THEN** it SHALL log a message indicating that the internal MLX fallback reranker is being used for the configured model
- **AND** it SHALL include the model name or path in the log message
- **AND** it SHALL include the reason for falling back when `rerank.py` is present but cannot be loaded.

#### Scenario: MLX repo reranker logging
- **WHEN** the MLX backend successfully loads a repo-provided `rerank.py` and `MLXReranker`
- **THEN** it SHALL log a message indicating that the repo-specific MLX reranker implementation is in use for the configured model.
