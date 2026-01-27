## Context

The current reranker implementation is tightly coupled to PyTorch and sentence-transformers, creating a barrier to supporting other AI frameworks like MLX. The API and CLI currently depend directly on the concrete `Reranker` class, making it impossible to swap implementations without changing import paths and type hints.

## Goals / Non-Goals

**Goals:**
- Enable pluggable AI framework backends (PyTorch, MLX, etc.)
- Maintain full API compatibility for existing clients
- Provide type-safe interface through Python protocols
- Support all current functionality without breaking changes
- Enable easy addition of new reranker implementations

**Non-Goals:**
- Change the external API contract
- Modify the request/response format
- Add new functionality beyond pluggability
- Support runtime framework switching (design-time choice only)

## Decisions

### Protocol-Based Interface Design
**Decision**: Use `typing.Protocol` to define the reranker interface rather than abstract base classes.

**Rationale**: 
- Protocols provide structural typing - any class with the right methods implements the interface
- No inheritance required, keeping implementations clean
- Better IDE support and type checking with modern Python
- Allows existing classes to be retrofitted without modification

**Protocol Methods** (based on API usage analysis):
```python
class RerankerProtocol(Protocol):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None) -> None: ...
    def compute_scores(self, query: str, documents: List[Union[str, Dict[str, Any]]]) -> List[Tuple[int, float]]: ...
```

### Implementation Strategy
**Decision**: Rename existing implementation to `reranker_pytorch.py` and create new `reranker.py` with protocol definition.

**Rationale**:
- Preserves existing PyTorch implementation unchanged
- Clear separation of interface from implementation
- Follows Python convention of protocols in separate modules
- Enables future implementations (MLX, ONNX, etc.)

### API Integration
**Decision**: Update API to use protocol type in dependency injection.

**Rationale**:
- Maintains FastAPI's type safety and dependency injection
- Allows any protocol-compliant implementation
- No changes to endpoint signatures or behavior
- Preserves existing error handling and logging

## Risks / Trade-offs

### Risks
- **Import path changes**: Existing imports may break (mitigated by clear migration path)
- **Type checking complexity**: Protocol compliance requires careful implementation
- **Testing overhead**: Need to test protocol compliance separately

### Trade-offs
- **Complexity vs Flexibility**: Added protocol complexity for framework flexibility
- **Performance**: Minimal overhead from protocol type checking (compile-time only)
- **Maintenance**: More files to maintain but clearer separation of concerns

## Migration Plan

### Phase 1: Protocol Definition
1. Extract protocol interface from current `Reranker` class
2. Define protocol in new `reranker.py` with proper type hints
3. Ensure protocol covers all methods used by API and CLI

### Phase 2: Implementation Refactor
1. Rename `reranker.py` to `reranker_pytorch.py`
2. Add `@override` decorators to protocol methods
3. Verify protocol compliance through static analysis

### Phase 3: API Integration
1. Update imports in `api.py` to use protocol type
2. Modify dependency injection to accept protocol type
3. Update type hints throughout the codebase

### Phase 4: Testing & Validation
1. Create protocol compliance tests
2. Verify API functionality unchanged
3. Test with both string and dict document formats
4. Validate error handling and edge cases

## Protocol Interface Specification

Based on analysis of `api.py:95` and current usage patterns:

```python
from typing import Protocol, List, Union, Dict, Any, Tuple, Optional

class RerankerProtocol(Protocol):
    """Protocol for reranker implementations supporting multiple AI frameworks."""
    
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, device: Optional[str] = None) -> None:
        """Initialize the reranker with specified model and device."""
        ...
    
    def compute_scores(self, query: str, documents: List[Union[str, Dict[str, Any]]]) -> List[Tuple[int, float]]:
        """Compute relevance scores for query-document pairs.
        
        Args:
            query: Search query string
            documents: List of documents (strings or dicts with 'text' key)
            
        Returns:
            List of (original_index, score) tuples for ranking
        """
        ...
```

**Key Design Decisions:**
- Protocol matches exactly what the API uses (`api.py:95-98`)
- Supports both string and dict document formats (current behavior)
- Returns indexed scores to maintain original document ordering
- Device parameter optional for auto-detection (current behavior)
- Model name parameter with default for flexibility

## Open Questions

- Should device auto-detection be part of protocol or implementation detail? (Current: implementation detail)
- Should protocol include model validation methods? (Current: no, keep simple)
- How to handle framework-specific configuration? (Current: constructor parameters only)