# Change: Refactor Reranker to Pluggable Architecture

## Why
The current reranker implementation is tightly coupled to PyTorch/sentence-transformers, making it difficult to support other AI frameworks like MLX. A pluggable architecture will enable multiple backend implementations while maintaining API compatibility.

## What Changes
- Extract abstract reranker interface using `typing.Protocol`
- Rename `reranker.py` to `reranker_pytorch.py` 
- Create new `reranker.py` with the protocol definition
- Update PyTorch implementation to explicitly implement the protocol with `@override`
- Update imports in `api.py` to use the new structure
- **BREAKING**: Import path changes for reranker class

## Impact
- Affected specs: reranker capability
- Affected code: `src/local_reranker/reranker.py`, `src/local_reranker/api.py`
- New files: `src/local_reranker/reranker_pytorch.py`