# Performance Optimization Design

## Architecture Overview

The performance optimization introduces a layered approach to handle high-volume reranking requests efficiently while maintaining document integrity and API compatibility.

## System Architecture

### Current Architecture Issues
```
Request → API (async) → Reranker (sync) → Model (all docs at once)
         ↓
    533s blocking, memory pressure, document loss
```

### Proposed Architecture
```
Request → API (async) → Batch Manager → Async Reranker → Model (batches)
         ↓                    ↓
    Progress tracking    Memory-efficient processing
         ↓                    ↓
    Streaming results   Configurable batch sizes
```

## Core Components

### 1. Batch Manager
**Purpose**: Split large document sets into optimal batch sizes
- **Configurable batch sizes**: 8-16 documents per batch (hardware-dependent)
- **Memory-aware batching**: Adjust batch size based on available memory
- **Result ordering**: Maintain original document order across batches
- **Error isolation**: Failures in one batch don't affect others

### 2. Async Reranker Interface
**Purpose**: Make core reranking operations non-blocking
- **Async batch processing**: Process multiple batches concurrently
- **Progress tracking**: Real-time progress updates for long requests
- **Cancellation support**: Allow request cancellation during processing
- **Resource management**: Proper cleanup of async resources

### 3. Memory Optimization Layer
**Purpose**: Reduce memory footprint during processing
- **Streaming document processing**: Load documents lazily
- **Tokenization optimization**: Cache and reuse tokenization results
- **Memory monitoring**: Track and limit peak memory usage
- **Garbage collection**: Proactive cleanup of processed batches

### 4. Performance Monitoring
**Purpose**: Track and optimize performance metrics
- **Timing metrics**: Per-document and per-batch processing times
- **Memory metrics**: Peak and average memory usage
- **Error tracking**: Document processing failures and recovery
- **Throughput monitoring**: Documents per second processing rate

## Data Flow

### Request Processing Flow
1. **Request Validation**: Validate input parameters and document format
2. **Batch Planning**: Calculate optimal batch size and create batch plan
3. **Async Processing**: Process batches concurrently with progress tracking
4. **Result Aggregation**: Combine batch results maintaining original order
5. **Response Generation**: Create API response with performance metadata

### Batch Processing Flow
1. **Document Preparation**: Load and validate documents for current batch
2. **Model Inference**: Run reranking on batch documents
3. **Result Processing**: Convert model output to standard format
4. **Memory Cleanup**: Release batch-specific resources
5. **Progress Update**: Report batch completion to progress tracker

## Configuration Strategy

### Environment Variables
```bash
RERANKER_BATCH_SIZE=12              # Documents per batch
RERANKER_MAX_CONCURRENT_BATCHES=3   # Concurrent batch processing
RERANKER_MEMORY_LIMIT_MB=1024       # Memory usage limit
RERANKER_PROGRESS_TRACKING=true     # Enable progress updates
RERANKER_TIMEOUT_SECONDS=300        # Request timeout
```

### Dynamic Configuration
- **Hardware detection**: Auto-adjust batch size based on available memory
- **Model-specific tuning**: Different batch sizes for different models
- **Load-based adaptation**: Reduce batch size under high load

## Error Handling Strategy

### Document-Level Errors
- **Isolation**: Failed documents don't stop batch processing
- **Logging**: Comprehensive error logging with document context
- **Fallback**: Return original order for failed documents with score 0.0

### Batch-Level Errors
- **Retry mechanism**: Automatic retry with smaller batch size
- **Graceful degradation**: Continue with remaining batches
- **Partial results**: Return successfully processed results

### System-Level Errors
- **Timeout handling**: Configurable timeouts with partial result return
- **Memory errors**: Automatic batch size reduction
- **Model errors**: Fallback to CPU processing if GPU fails

## Performance Targets

### Processing Time Goals
- **90 documents (6,000 chars)**: Under 30 seconds total
- **Single batch (12 docs)**: Under 4 seconds per batch
- **Progress updates**: Every 2 seconds during processing

### Memory Usage Goals
- **Peak memory**: Under 1GB for typical workloads
- **Per-batch memory**: Under 200MB per concurrent batch
- **Memory growth**: Linear with batch count, not document count

### Throughput Goals
- **Documents per second**: 3+ documents per second average
- **Concurrent requests**: Support 5+ concurrent reranking requests
- **Resource utilization**: 80%+ GPU/CPU utilization during processing

## Implementation Phases

### Phase 1: Investigation & Bug Fixes
- Debug MLX model document loss (90→19 results)
- Fix document return issues in API responses
- Add comprehensive error handling and logging

### Phase 2: Batching Implementation
- Implement batch manager with configurable sizes
- Add result ordering across batches
- Integrate with existing reranker implementations

### Phase 3: Async Processing
- Convert synchronous reranking to async
- Add concurrent batch processing
- Implement progress tracking system

### Phase 4: Memory Optimization
- Add streaming document processing
- Implement memory monitoring and limits
- Optimize tokenization and model inference

### Phase 5: Performance Monitoring
- Add comprehensive metrics collection
- Implement performance tuning
- Add performance regression tests

## Testing Strategy

### Performance Tests
- **Load testing**: 90+ document scenarios
- **Memory testing**: Peak usage under various loads
- **Concurrency testing**: Multiple simultaneous requests
- **Stress testing**: System limits and failure modes

### Correctness Tests
- **Result ordering**: Verify original document order maintained
- **Score accuracy**: Compare batched vs non-batched results
- **Error handling**: Verify graceful failure handling
- **API compatibility**: Ensure backward compatibility

### Integration Tests
- **Backend compatibility**: Test with PyTorch and MLX
- **Hardware testing**: Various memory and CPU configurations
- **Configuration testing**: Different batch sizes and settings