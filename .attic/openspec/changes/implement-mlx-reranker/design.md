# Design: MLX Reranker Implementation

## Architecture Overview

The MLX reranker implementation follows the existing pluggable architecture pattern established in the project. It implements the `Reranker` protocol while leveraging MLX-specific optimizations for Apple Silicon.

## Component Design

### 1. MLX Reranker Class (`reranker_mlx.py`)
```python
from rerank import MLXReranker as MLXRerankerImpl
from typing import List, Optional
from .reranker import Reranker as RerankerProtocol
from .models import RerankRequest, RerankResult, RerankDocument

class Reranker(RerankerProtocol):
    """MLX implementation of the reranker protocol."""
    
    def __init__(self, model_name: str = "jinaai/jina-reranker-v3-mlx", device: Optional[str] = None):
        # MLX automatically handles Apple Silicon GPU/CPU selection
        # Device parameter kept for protocol compatibility but ignored
        try:
            self.model = MLXRerankerImpl(model_path=model_name)
            self.model_name = model_name
            logger.info(f"Successfully loaded MLX reranker: {model_name}")
        except ImportError as e:
            raise ImportError(
                "MLX dependencies not found. Install with: pip install mlx mlx-lm"
            ) from e
        
    def rerank(self, request: RerankRequest) -> List[RerankResult]:
        # Convert our RerankRequest to MLX format
        mlx_results = self.model.rerank(
            query=request.query,
            documents=request.documents,
            top_n=request.top_n,
            return_embeddings=request.return_documents
        )
        
        # Convert MLX results to our RerankResult format
        return [
            RerankResult(
                document=RerankDocument(text=result['document']) if request.return_documents else None,
                index=result['index'],
                relevance_score=result['relevance_score']
            )
            for result in mlx_results
        ]
```

### MLX Integration Examples

Based on the Hugging Face `jinaai/jina-reranker-v3-mlx` model, here are the key integration patterns:

#### Basic MLX Usage
```python
from rerank import MLXReranker

# Initialize the reranker
reranker = MLXReranker()

# Your query and documents
query = "What are the health benefits of green tea?"
documents = [
    "Green tea contains antioxidants called catechins that may help reduce inflammation and protect cells from damage.",
    "El precio del café ha aumentado un 20% este año debido a problemas en la cadena de suministro.",
    "Studies show that drinking green tea regularly can improve brain function and boost metabolism.",
    "Basketball is one of the most popular sports in United States.",
    "绿茶富含儿茶素等抗氧化剂，可以降低心脏病风险，还有助于控制体重。",
    "Le thé vert est riche en antioxydants et peut améliorer la fonction cérébrale.",
]

# Rerank documents
results = reranker.rerank(query, documents)

# Results are sorted by relevance score (highest first)
for result in results:
    print(f"Score: {result['relevance_score']:.4f}")
    print(f"Document: {result['document'][:100]}...")
    print()
```

#### MLX API Reference
```python
reranker.rerank(
    query: str,                      # Search query
    documents: List[str],            # Documents to rank
    top_n: Optional[int] = None,     # Return only top N (default: all)
    return_embeddings: bool = False, # Include doc embeddings (default: False)
)
```

**Returns:** List of dicts with keys:
- `document`: Original document text
- `relevance_score`: Float score (higher = more relevant)
- `index`: Position in input documents list
- `embedding`: Document embedding (if `return_embeddings=True`)

#### Advanced MLX Usage
```python
# Get only top 3 results
top_results = reranker.rerank(query, documents, top_n=3)

# Get embeddings for further processing
results_with_embeddings = reranker.rerank(query, documents, return_embeddings=True)
for result in results_with_embeddings:
    embedding = result['embedding']  # numpy array of shape (512,)
    # Use embedding for downstream tasks...
```

#### Custom Model Path
```python
# If model files are in a different location
reranker = MLXReranker(
    model_path="/path/to/model",
    projector_path="/path/to/projector.safetensors"
)
```

### 2. Configuration Integration
- Add `"mlx"` to `RERANKER_DEFAULTS` with default model
- Update `get_available_rerankers()` to include MLX option
- MLX-specific environment variables when needed

### 3. API Integration
- Modify `api.py` lifespan function to handle MLX type
- Conditional import pattern for MLX dependencies
- Error handling for MLX unavailability

### 4. CLI Integration
- Add MLX to available reranker types
- Update help text and validation

## Technical Considerations

### MLX-Specific Behavior
1. **Device Management**: MLX automatically uses Apple Silicon GPU when available
2. **Model Loading**: Uses HuggingFace model hub with MLX-specific format
3. **Memory Efficiency**: Lower memory footprint than PyTorch equivalents
4. **Performance**: Optimized matrix operations for Apple Silicon

### Error Handling Strategy
1. **Import Fallback**: Graceful handling when MLX is not installed
2. **Model Loading**: Clear error messages for model download failures
3. **Platform Detection**: Inform users about Apple Silicon optimization

### Testing Strategy
1. **Mock Dependencies**: Mock MLX libraries for CI/CD environments
2. **Protocol Compliance**: Verify MLX implementation follows protocol
3. **Integration Testing**: Test API endpoints with MLX backend
4. **Performance Testing**: Benchmark against PyTorch on Apple Silicon

## Implementation Flow

### Initialization
```python
def __init__(self, model_name: str = "jinaai/jina-reranker-v3-mlx", device: Optional[str] = None):
    try:
        from rerank import MLXReranker as MLXRerankerImpl
        self.model = MLXRerankerImpl(model_path=model_name)
        self.model_name = model_name
        logger.info(f"Successfully loaded MLX reranker: {model_name}")
    except ImportError as e:
        raise ImportError(
            "MLX dependencies not found. Install with: pip install mlx mlx-lm"
        ) from e
```

### Reranking Process
```python
def rerank(self, request: RerankRequest) -> List[RerankResult]:
    # Convert our RerankRequest to MLX format
    mlx_results = self.model.rerank(
        query=request.query,
        documents=request.documents,
        top_n=request.top_n,
        return_embeddings=request.return_documents
    )
    
    # Convert MLX results to our RerankResult format
    return [
        RerankResult(
            document=RerankDocument(text=result['document']) if request.return_documents else None,
            index=result['index'],
            relevance_score=result['relevance_score']
        )
        for result in mlx_results
    ]
```

### MLX Model Loading Example
```python
# Standard initialization (uses default model path)
reranker = MLXRerankerImpl()

# Custom model path
reranker = MLXRerankerImpl(
    model_path="/path/to/model",
    projector_path="/path/to/projector.safetensors"
)
```

### Error Scenarios
1. MLX not installed → ImportError with helpful message
2. Model not found → RuntimeError with download suggestions
3. Invalid input → ValueError following existing patterns

## Dependencies and Trade-offs

### New Dependencies
- `mlx`: Core MLX framework
- `mlx-lm`: MLX language model utilities
- `huggingface-hub`: For model downloads (already included)

### Trade-offs
1. **Platform Specificity**: MLX only works on Apple Silicon
   - Mitigation: Keep PyTorch as default for cross-platform
2. **Dependency Bloat**: Adding new ML framework
   - Mitigation: Make MLX optional, import only when needed
3. **Testing Complexity**: Need to mock MLX for CI
   - Mitigation: Comprehensive mocking strategy

## Future Extensibility
The implementation maintains the pluggable architecture, making it easy to add:
- Additional MLX models
- Other Apple Silicon frameworks
- Framework-specific optimizations

This design ensures the MLX reranker integrates seamlessly with the existing codebase while providing Apple Silicon users with optimized performance.