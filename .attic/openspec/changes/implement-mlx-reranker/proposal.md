# Proposal: Implement MLX-based Reranker

## Summary
Add MLX framework support to the existing pluggable reranker system, enabling optimized performance for Apple Silicon devices while maintaining full API compatibility and protocol compliance.

## Problem Statement
The current implementation only supports PyTorch-based rerankers, which may not provide optimal performance on Apple Silicon (M1/M2/M3) devices. MLX is Apple's machine learning framework specifically designed for these chips, offering better memory efficiency and performance.

## Proposed Solution
Implement a new MLX-based reranker that:
1. Uses the `jinaai/jina-reranker-v3-mlx` model as default
2. Follows the existing `Reranker` protocol for seamless integration
3. Provides Apple Silicon optimization while maintaining cross-platform compatibility
4. Includes comprehensive testing coverage

## Scope
### In Scope
- New MLX reranker implementation (`reranker_mlx.py`)
- Configuration updates to support MLX backend
- API integration for MLX reranker selection
- CLI support for MLX option
- Comprehensive unit and integration tests
- Documentation updates

### Out of Scope
- Changes to existing PyTorch implementation
- API endpoint modifications (maintains compatibility)
- Core protocol changes (uses existing interface)

## Benefits
1. **Performance**: Optimized for Apple Silicon with lower memory footprint
2. **Choice**: Users can select optimal backend for their hardware
3. **Compatibility**: Maintains existing API and protocol structure
4. **Future-proof**: Extensible architecture for additional frameworks

## Risks and Mitigations
- **MLX Dependency**: Add conditional imports and graceful fallbacks
- **Testing Complexity**: Mock MLX dependencies for CI environments
- **Platform Specificity**: Ensure MLX is optional, PyTorch remains default

## Success Criteria
- MLX reranker implements the protocol correctly
- API endpoints work with MLX backend
- Tests pass on all platforms (with mocked MLX when unavailable)
- Performance improvement on Apple Silicon devices
- Documentation is complete and accurate