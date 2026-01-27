# Project Context

## Purpose
A lightweight, local reranker API implementation compatible with Cohere's reranking service. The project enables privacy-focused document reranking by running models entirely on local infrastructure with pluggable AI framework support.

## Tech Stack
- **Python 3.12+** with modern syntax and type hints
- **FastAPI** for async web API framework
- **Pluggable AI frameworks** (PyTorch, MLX, or other modern local AI model frameworks)
- **Uvicorn** for ASGI server
- **Pydantic** for data validation and serialization
- **pytest** for unit and integration testing
- **uv** for dependency management and package distribution

## Project Conventions

### Code Style
- Follow PEP 8 style guide with strict type hints
- Use dataclasses for data containers
- Prefer pathlib over os.path for file operations
- Use explicit exception handling with specific exception types
- Keep functions small and focused on single tasks
- Use docstrings for all public modules, functions, classes, and methods
- Use list/dict/set comprehensions for concise, readable code
- Prefer composition over inheritance

### Architecture Patterns
- **FastAPI lifespan management** for model loading/cleanup
- **Async/await patterns** for concurrent request handling
- **Pluggable model framework interface** for supporting multiple AI backends
- **Service layer pattern** separating API endpoints from business logic
- **Model caching** to avoid repeated model loading
- **Pydantic models** for request/response validation

### Testing Strategy
- **pytest** for all testing with mock objects for external dependencies
- **Unit tests** for individual modules (reranker, device management, models)
- **Integration tests** for full API endpoint testing
- **httpx** for async HTTP client testing
- **Performance benchmarks** for different hardware configurations

### Git Workflow
- **Feature branches** for new development
- **Semantic versioning** for releases (currently 0.1.1)
- **Conventional commits** for clear commit messages
- **Main branch protection** for stable releases

## Domain Context
- **Reranking**: ML technique to reorder documents by relevance to a query
- **Cross-Encoder models**: Neural networks that compare query-document pairs
- **Cohere API compatibility**: API structure matching `/v1/rerank` endpoint
- **RAG applications**: Retrieval-Augmented Generation systems
- **Pluggable backends**: Support for multiple AI frameworks (PyTorch, MLX, etc.)
- **Model**: Configurable reranker models via framework plugins

## Important Constraints
- **Privacy-first**: All processing must occur locally, no external API calls
- **Performance target**: <500ms response time for typical document sets
- **Memory efficiency**: Must work on Mac M1 with 16GB RAM
- **API compatibility**: Must match Cohere's request/response format exactly
- **Framework flexibility**: Must support multiple AI backends through pluggable interface

## External Dependencies
- **Model repositories**: For downloading reranker models (framework-dependent)
- **AI framework libraries**: PyTorch, MLX, or other modern local AI frameworks
- **FastAPI/Uvicorn**: For web server functionality
- **No external APIs**: Completely self-contained operation
