# CLI Module Design

## Architecture Overview

The CLI module is designed to provide a clean separation between the command-line interface and the FastAPI application logic. This follows the single responsibility principle and makes the codebase more maintainable and testable.

## Component Structure

### CLI Module (`src/local_reranker/cli.py`)
- **Purpose**: Handle command-line argument parsing and server startup
- **Dependencies**: uvicorn (for server runtime), argparse (for CLI parsing)
- **Interface**: Provides `main()` function as entry point and `run_server()` for programmatic use

### API Module (`src/local_reranker/api.py`)
- **Purpose**: Define FastAPI application, endpoints, and business logic
- **Dependencies**: FastAPI, reranker components
- **Interface**: Exposes `app` instance for external server startup

## Design Decisions

### 1. Separation of Concerns
- CLI code moved from `api.py` to dedicated `cli.py` module
- API module focuses solely on web application logic
- CLI module handles user interaction and server configuration

### 2. Argument Parsing Strategy
- Uses Python's built-in `argparse` for compatibility and simplicity
- Provides sensible defaults matching development environment
- Includes validation for port numbers and log levels
- Supports development features like `--reload` flag

### 3. Server Startup Pattern
- `run_server()` function allows programmatic usage
- `main()` function handles CLI-specific concerns
- Uvicorn configuration passed through cleanly
- No hardcoded server parameters

### 4. Error Handling Approach
- Leverages uvicorn's built-in error handling
- Provides clear error messages for invalid arguments
- Uses logging for startup information

## Testing Strategy

### Unit Testing
- Mock `uvicorn.run` to avoid actual server startup
- Test argument parsing with various input combinations
- Validate parameter passing to server startup function

### Integration Testing
- Test CLI script execution via subprocess
- Verify server responds to health checks when started via CLI
- Test error scenarios and graceful shutdown

### Code Quality
- Static type checking with mypy
- Linting with ruff for PEP 8 compliance
- Comprehensive docstring coverage

## Trade-offs Considered

### Simplicity vs. Features
- Chose argparse over click for minimal dependencies
- Limited CLI options to essential server configuration
- Avoided complex configuration file support

### Testing Complexity
- Mocking uvicorn requires careful test design
- Integration tests need proper cleanup
- Subprocess testing adds complexity but provides real-world validation

## Future Extensibility

The current design allows for future enhancements:
- Additional CLI arguments for model configuration
- Environment variable support for configuration
- Multiple server startup modes (development, production)
- Configuration file support if needed

## Dependencies and Imports

```python
# CLI module dependencies
import argparse  # Built-in, no external dependency
import logging   # Built-in, for startup messages
import uvicorn   # External, for ASGI server

# API module access
# Imports "local_reranker.api:app" as string for uvicorn
# No direct import needed, avoiding circular dependencies
```

This design ensures clean module boundaries while maintaining functionality and testability.