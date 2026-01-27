## ADDED Requirements
### Requirement: MLX Reranker Implementation
The system SHALL provide an MLX-based reranker implementation for Apple Silicon optimization.

#### Scenario: MLX reranker class creation
- **WHEN** implementing the MLX reranker
- **THEN** a new class `Reranker` SHALL be created in `reranker_mlx.py`
- **AND** it SHALL implement the existing `Reranker` protocol
- **AND** it SHALL use `jinaai/jina-reranker-v3-mlx` as the default model

#### Scenario: MLX model initialization
- **WHEN** initializing the MLX reranker
- **THEN** it SHALL import MLXReranker from the rerank module
- **AND** it SHALL load the MLX reranker model using MLXReranker(model_path=model_name)
- **AND** it SHALL support custom model_path and projector_path parameters
- **AND** it SHALL handle ImportError when MLX dependencies are not available
- **AND** it SHALL provide clear installation instructions in error messages
- **AND** it SHALL log successful model loading with model name
- **AND** it SHALL handle model loading failures with appropriate error messages

#### Scenario: MLX reranking process
- **WHEN** processing a rerank request with MLX
- **THEN** it SHALL convert the request format to MLX-compatible format
- **AND** it SHALL call the MLX model's rerank method with parameters: query, documents, top_n, return_embeddings
- **AND** it SHALL convert MLX result dict format to RerankResult objects
- **AND** it SHALL handle document text extraction from MLX results
- **AND** it SHALL maintain the same sorting and filtering behavior as PyTorch
- **AND** it SHALL support embedding return when requested

#### Scenario: MLX result format conversion
- **WHEN** converting MLX results to RerankResult format
- **THEN** it SHALL extract 'document' text for document content when return_documents is True
- **AND** it SHALL extract 'relevance_score' as float for relevance_score
- **AND** it SHALL extract 'index' for original document position
- **AND** it SHALL handle 'embedding' array when return_embeddings is requested
- **AND** it SHALL create RerankDocument objects when document content is included

## ADDED Requirements
### Requirement: MLX Configuration Support
The system SHALL support MLX backend selection through configuration.

#### Scenario: MLX backend configuration
- **WHEN** configuring the reranker type
- **THEN** `"mlx"` SHALL be added to available reranker types
- **AND** the default model SHALL be `"jinaai/jina-reranker-v3-mlx"`
- **AND** configuration SHALL support MLX-specific settings when needed

#### Scenario: MLX availability detection
- **WHEN** selecting MLX as the reranker type
- **THEN** the system SHALL check for MLX library availability
- **AND** it SHALL provide clear error messages if MLX is not installed
- **AND** it SHALL suggest installation instructions for Apple Silicon users

## ADDED Requirements
### Requirement: MLX API Integration
The system SHALL integrate the MLX reranker with the existing API infrastructure.

#### Scenario: API MLX backend selection
- **WHEN** the API starts with MLX reranker type
- **THEN** the lifespan function SHALL conditionally import MLX dependencies
- **AND** it SHALL instantiate the MLX reranker with the configured model
- **AND** it SHALL handle MLX-specific initialization errors gracefully

#### Scenario: MLX request handling
- **WHEN** processing API requests with MLX backend
- **THEN** the API SHALL use the MLX reranker instance
- **AND** it SHALL maintain the same request/response format as other backends
- **AND** it SHALL provide consistent error handling across all backends

## ADDED Requirements
### Requirement: MLX CLI Support
The system SHALL support MLX reranker selection through the CLI interface.

#### Scenario: CLI MLX option
- **WHEN** users select a reranker type via CLI
- **THEN** `"mlx"` SHALL be available as a reranker option
- **AND** help text SHALL describe MLX as optimized for Apple Silicon
- **AND** validation SHALL accept MLX as a valid reranker type

#### Scenario: CLI MLX initialization
- **WHEN** starting the CLI with MLX reranker
- **THEN** the CLI SHALL initialize the MLX reranker with appropriate settings
- **AND** it SHALL handle MLX import errors with user-friendly messages
- **AND** it SHALL provide fallback suggestions when MLX is unavailable

## ADDED Requirements
### Requirement: MLX Testing Coverage
The system SHALL provide comprehensive testing for the MLX reranker implementation.

#### Scenario: MLX protocol compliance testing
- **WHEN** testing the MLX reranker
- **THEN** tests SHALL verify it implements the Reranker protocol correctly
- **AND** tests SHALL validate method signatures match the protocol
- **AND** tests SHALL use mocked MLX dependencies for cross-platform testing

#### Scenario: MLX functionality testing
- **WHEN** testing MLX reranker functionality
- **THEN** tests SHALL mock the MLXReranker class from rerank module
- **AND** tests SHALL verify request format conversion to MLX parameters
- **AND** tests SHALL verify result format conversion from MLX dict to RerankResult
- **AND** tests SHALL verify top_n filtering works correctly
- **AND** tests SHALL verify embedding handling when return_embeddings is True
- **AND** tests SHALL verify document text extraction from MLX results
- **AND** tests SHALL verify relevance_score conversion to float
- **AND** tests SHALL verify index preservation from MLX results

#### Scenario: MLX error handling testing
- **WHEN** testing MLX error conditions
- **THEN** tests SHALL verify handling of missing MLX dependencies with ImportError
- **AND** tests SHALL verify helpful error messages suggesting pip install mlx mlx-lm
- **AND** tests SHALL verify handling of model loading failures
- **AND** tests SHALL verify handling of invalid input data
- **AND** tests SHALL verify appropriate error messages are provided
- **AND** tests SHALL verify graceful fallback when MLX is unavailable on non-Apple Silicon platforms

#### Scenario: MLX integration testing
- **WHEN** testing MLX integration with API
- **THEN** tests SHALL verify API endpoints work with MLX backend
- **AND** tests SHALL verify request/response handling
- **AND** tests SHALL verify configuration loading
- **AND** tests SHALL use mocked MLX for CI/CD compatibility

#### Scenario: MLX CLI integration testing
- **WHEN** testing CLI with MLX backend
- **THEN** tests SHALL verify CLI accepts MLX as reranker type
- **AND** tests SHALL verify CLI initialization with MLX
- **AND** tests SHALL verify CLI error handling for MLX issues
- **AND** tests SHALL use mocked MLX dependencies

## ADDED Requirements
### Requirement: MLX API Compatibility
The system SHALL provide seamless integration with the MLX reranker API while maintaining protocol compliance.

#### Scenario: MLX API parameter mapping
- **WHEN** calling the MLX reranker
- **THEN** it SHALL map query parameter directly to MLX query
- **AND** it SHALL map documents list directly to MLX documents
- **AND** it SHALL map top_n parameter directly to MLX top_n
- **AND** it SHALL map return_documents to MLX return_embeddings
- **AND** it SHALL handle optional parameters with proper defaults

#### Scenario: MLX multilingual support
- **WHEN** processing multilingual content
- **THEN** the MLX reranker SHALL handle multiple languages natively
- **AND** it SHALL support English, Spanish, Chinese, French, and other languages
- **AND** it SHALL maintain relevance scoring across different languages
- **AND** it SHALL provide consistent ranking behavior for multilingual queries

#### Scenario: MLX embedding support
- **WHEN** return_embeddings is requested
- **THEN** the MLX reranker SHALL return numpy arrays of shape (512,)
- **AND** it SHALL include embeddings in the result conversion process
- **AND** it SHALL handle embedding storage in RerankResult when appropriate
- **AND** it SHALL maintain backward compatibility when embeddings are not requested

## MODIFIED Requirements
### Requirement: Dependency Management
The system SHALL include MLX dependencies for Apple Silicon optimization.

#### Scenario: MLX dependency addition
- **WHEN** installing project dependencies
- **THEN** `mlx` and `mlx-lm` SHALL be added to pyproject.toml
- **AND** dependencies SHALL be optional or platform-specific when possible
- **AND** installation instructions SHALL include MLX setup guidance

#### Scenario: Cross-platform compatibility
- **WHEN** installing on non-Apple Silicon platforms
- **THEN** the system SHALL gracefully handle MLX unavailability
- **AND** PyTorch SHALL remain the default backend
- **AND** users SHALL be informed about MLX platform requirements