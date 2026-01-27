# Performance Optimization Proposal

## Summary
Optimize the local reranker to handle high-volume requests (90+ documents with 6,000+ characters) efficiently while maintaining document integrity and API compatibility.

## Problem Statement
The current reranker implementation has critical performance bottlenecks when processing large document sets:
- **533-second processing time** for 90 documents (6,000+ chars each)
- **Missing documents in results** (90 input → 19 output)
- **Memory inefficiency** with synchronous processing of all documents
- **No batching strategy** causing memory pressure and timeouts

## Root Cause Analysis
1. **Synchronous Processing**: All documents processed simultaneously without batching
2. **Memory Duplication**: Documents duplicated as query-document pairs in memory
3. **No Async Processing**: FastAPI endpoint is async but core reranking blocks completely
4. **Missing Error Handling**: Documents may be silently dropped during processing
5. **Unused Optimizations**: Text processing utilities exist but aren't integrated

## Proposed Solution
Implement a multi-phase performance optimization strategy:

### Phase 1: Critical Bug Fixes
- Investigate and fix the 90→19 document loss issue
- Ensure proper document return in API responses
- Add comprehensive error handling and logging

### Phase 2: Strategic Batching
- Implement configurable document batching (8-16 docs per batch)
- Maintain result ordering across batches
- Add memory-efficient batch processing

### Phase 3: Async Processing
- Make core reranking operations asynchronous
- Enable concurrent batch processing
- Add progress tracking for long-running requests

### Phase 4: Memory Optimization
- Implement streaming document processing
- Add memory usage monitoring
- Optimize tokenization and model inference

## Constraints
- **No chunking**: Documents must be evaluated in their entirety
- **API compatibility**: Must maintain Cohere API compatibility
- **Privacy-first**: All processing must remain local
- **Framework flexibility**: Support both PyTorch and MLX backends

## Success Criteria
- **Processing time**: Reduce from 533s to under 30s for 90 documents
- **Document integrity**: 100% of input documents returned in results
- **Memory efficiency**: Peak memory usage under 1GB for typical workloads
- **API compatibility**: Maintain existing request/response format
- **Error handling**: Comprehensive logging and graceful failure handling

## Implementation Approach
1. **Investigate current MLX model behavior** to understand document loss
2. **Implement batching system** with configurable batch sizes
3. **Add async processing** for non-blocking operations
4. **Integrate performance monitoring** and optimization
5. **Add comprehensive testing** for performance and correctness

## Risk Mitigation
- **Backward compatibility**: Maintain existing API interface
- **Incremental rollout**: Phase-based implementation with testing at each stage
- **Performance monitoring**: Add metrics to track improvements
- **Fallback mechanisms**: Ensure graceful degradation if optimizations fail