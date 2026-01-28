# Implementation Plan: Adopt Jina MLX Reranker

## Overview

Replace functionality that attempts to load `rerank.py` from model files with a proper Jina-style MLX reranker implementation directly integrated into the codebase.

## Problem Statement

The current MLX reranker implementation attempts to:
1. Load `rerank.py` from model files if it exists
2. Fall back to `MLXCrossEncoderReranker` if loading fails

This causes issues:
- The fallback creates a projector with mismatched architecture (`hidden_size → hidden_size → 512`)
- Jina's projector is fixed at `1024 → 512 → 512`
- Weight loading fails silently, resulting in random projector initialization
- Random embeddings produce near-zero cosine similarity scores (~0.01 to -0.07)

### Current Symptoms
- MLX scores: 0.018, 0.012, -0.078 (ranking correct but scores very low)
- PyTorch scores: 0.664, 0.451, 0.258 (same ranking, proper score range)
- Score discrepancy due to random projector weights vs. properly trained weights

### Root Cause
The internal `MLXCrossEncoderReranker` fallback:
- Creates projector with dimensions based on model's `hidden_size`
- Attempts to load Jina's projector weights (`linear1.weight: [512, 1024]`)
- Shapes don't match if `hidden_size ≠ 1024`
- Falls back to random initialization
- Produces meaningless embeddings with near-zero similarity

---

## Solution Overview

1. Create new `jina_mlx_reranker.py` with Jina-style implementation
2. Update `reranker_mlx.py` to directly instantiate Jina reranker (no `rerank.py` loading)
3. Remove `_load_repo_reranker` method
4. Update tests to reflect new architecture
5. Keep `MLXCrossEncoderReranker` for potential future use with other models

---

## Detailed Changes Required

### 1. Create New File: `src/local_reranker/jina_mlx_reranker.py`

**Purpose**: Implement Jina-style reranker following the reference implementation exactly.

**File Structure** (~250 lines)

#### a. Imports (lines 1-10)
```python
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import numpy as np
import os
from typing import Optional, List, Dict, Tuple
from safetensors import safe_open
```

#### b. `JinaMLPProjector` class (lines ~15-40)
```python
class JinaMLPProjector(nn.Module):
    """MLP projector to project hidden states to embedding space.

    Fixed architecture: 1024 → 512 → 512 (Jina reranker v3 spec).
    Uses ReLU activation and no bias terms to match reference implementation.
    """

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.linear2 = nn.Linear(512, 512, bias=False)

    def __call__(self, x):
        x = self.linear1(x)
        x = nn.relu(x)
        x = self.linear2(x)
        return x
```

**Key Specifications**:
- Fixed input dimension: 1024 (matches Jina model hidden_size)
- Hidden dimension: 512
- Output dimension: 512
- No bias terms on either layer
- ReLU activation (not GELU - important difference from internal fallback)

#### c. `_load_projector` function (lines ~45-70)
```python
def _load_projector(projector_path: str) -> JinaMLPProjector:
    """Load projector weights from safetensors file.

    Args:
        projector_path: Path to projector.safetensors file

    Returns:
        JinaMLPProjector instance with loaded weights

    Raises:
        FileNotFoundError: If projector file doesn't exist
        RuntimeError: If weight loading fails or shapes are incorrect
    """
    if not os.path.exists(projector_path):
        raise FileNotFoundError(f"Projector file not found: {projector_path}")

    projector = JinaMLPProjector()

    try:
        with safe_open(projector_path, framework="numpy") as f:
            w1 = f.get_tensor("linear1.weight")
            w2 = f.get_tensor("linear2.weight")

            # Verify expected shapes for Jina projector
            if w1.shape != (512, 1024):
                raise ValueError(
                    f"Unexpected linear1.weight shape: {w1.shape}, expected (512, 1024)"
                )
            if w2.shape != (512, 512):
                raise ValueError(
                    f"Unexpected linear2.weight shape: {w2.shape}, expected (512, 512)"
                )

            projector.linear1.weight = mx.array(w1)
            projector.linear2.weight = mx.array(w2)
    except Exception as e:
        raise RuntimeError(f"Failed to load projector weights: {e}") from e

    return projector
```

#### d. `_sanitize_input` helper function (lines ~75-85)
```python
def _sanitize_input(text: str, special_tokens: Dict[str, str]) -> str:
    """Remove special tokens from input text.

    Args:
        text: Input text to sanitize
        special_tokens: Dictionary of special token strings to remove

    Returns:
        Sanitized text with special tokens removed
    """
    for token in special_tokens.values():
        text = text.replace(token, "")
    return text
```

#### e. `_format_jina_prompt` helper function (lines ~90-130)
```python
def _format_jina_prompt(
    query: str,
    docs: List[str],
    special_tokens: Dict[str, str],
    instruction: Optional[str] = None,
) -> str:
    """Format query and documents into Jina prompt format.

    The prompt structure is critical for correct embedding extraction.
    Special tokens are placed at specific positions to mark query and document boundaries.

    Args:
        query: Search query string
        docs: List of document strings
        special_tokens: Dictionary containing doc_embed_token and query_embed_token
        instruction: Optional instruction for reranking

    Returns:
        Formatted prompt string ready for tokenization
    """
    query = _sanitize_input(query, special_tokens)
    docs = [_sanitize_input(doc, special_tokens) for doc in docs]

    # System message - fixed for all Jina reranker prompts
    prefix = (
        "<|im_start|>system\n"
        "You are a search relevance expert who can determine a ranking of the passages "
        "based on how relevant they are to the query. "
        "If the query is a question, how relevant a passage is depends on how well it answers the question. "
        "If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. "
        "If an instruction is provided, you should follow the instruction when determining the ranking."
        "<|im_end|>\n<|im_start|>user\n"
    )

    # Assistant prefix - disables thinking for faster inference
    suffix = "<|im_end|>\n<|im_start|>assistant\n\n"

    doc_emb_token = special_tokens["doc_embed_token"]
    query_emb_token = special_tokens["query_embed_token"]

    # Build prompt content
    prompt = (
        f"I will provide you with {len(docs)} passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {query}\n"
    )

    if instruction:
        prompt += f'<instruct>\n{instruction}\n</instruct>\n'

    # Format each document with its ID and embed token
    doc_prompts = [
        f'<passage id="{i}">\n{doc}{doc_emb_token}\n</passage>'
        for i, doc in enumerate(docs)
    ]
    prompt += "\n".join(doc_prompts) + "\n"

    # Format query section
    prompt += f"<query>\n{query}{query_emb_token}\n</query>"

    return prefix + prompt + suffix
```

**Prompt Format Details**:
```
<|im_start|>system
You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. If the query is a question, how relevant a passage is depends on how well it answers the question. If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. If an instruction is provided, you should follow the instruction when determining the ranking.
<|im_end|>
<|im_start|>user
I will provide you with N passages, each indicated by a numerical identifier. Rank the passages based on their relevance to query: {query}

[optional instruction section]
<instruct>
{instruction}
</instruct>

<passage id="0">
doc0<|embed_token|>
</passage>
<passage id="1">
doc1<|embed_token|>
</passage>
...
<query>
query<|rerank_token|>
</query><|im_end|>
<|im_start|>assistant

<|EOT|>
```

#### f. `JinaMLXReranker` class (lines ~135-250)

**Constructor** (`__init__`, lines ~145-170):
```python
class JinaMLXReranker:
    """MLX-based implementation of Jina reranker v3.

    This implementation follows the Jina reference exactly:
    - Uses special tokens for embedding extraction
    - Fixed 1024→512→512 projector architecture
    - Cosine similarity without pre-normalization
    """

    def __init__(self, model_path: str, projector_path: str):
        """Initialize Jina MLX reranker.

        Args:
            model_path: Path to MLX-converted model directory
            projector_path: Path to projector.safetensors file

        Raises:
            FileNotFoundError: If projector file doesn't exist
            RuntimeError: If model or projector loading fails
        """
        # Load MLX model and tokenizer
        self.model, self.tokenizer = load(model_path)
        self.model.eval()

        # Load projector with pre-trained weights
        self.projector = _load_projector(projector_path)

        # Special tokens for embedding extraction
        self.special_tokens = {
            "query_embed_token": "<|rerank_token|>",
            "doc_embed_token": "<|embed_token|>"
        }

        # Token IDs for special tokens (hard-coded for Jina v3)
        self.doc_embed_token_id = 151670
        self.query_embed_token_id = 151671
```

**Core Method** (`_compute_single_batch`, lines ~175-220):
```python
    def _compute_single_batch(
        self,
        query: str,
        docs: List[str],
        instruction: Optional[str] = None,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """Compute embeddings for a single batch of documents.

        This is the core inference method that:
        1. Formats the prompt with special tokens
        2. Tokenizes and runs the model
        3. Extracts embeddings at special token positions
        4. Projects embeddings through MLP
        5. Computes cosine similarity scores

        Args:
            query: Search query string
            docs: List of document strings
            instruction: Optional instruction for reranking

        Returns:
            query_embeds: Query embeddings after projection [1, 512]
            doc_embeds: Document embeddings after projection [num_docs, 512]
            scores: Cosine similarity scores [num_docs]

        Raises:
            ValueError: If special tokens not found in tokenized prompt
        """
        # Step 1: Format prompt
        prompt = _format_jina_prompt(
            query,
            docs,
            special_tokens=self.special_tokens,
            instruction=instruction,
        )

        # Step 2: Tokenize using MLX tokenizer
        input_ids = self.tokenizer.encode(prompt)

        # Step 3: Get hidden states from model
        # Shape: [1, seq_len, hidden_size] where hidden_size = 1024
        hidden_states = self.model.model([input_ids])

        # Remove batch dimension
        # Shape: [seq_len, hidden_size]
        hidden_states = hidden_states[0]

        # Convert input_ids to numpy for efficient indexing
        input_ids_np = np.array(input_ids)

        # Step 4: Find positions of special tokens
        query_embed_positions = np.where(input_ids_np == self.query_embed_token_id)[0]
        doc_embed_positions = np.where(input_ids_np == self.doc_embed_token_id)[0]

        # Step 5: Extract embeddings at special token positions
        if len(query_embed_positions) == 0:
            raise ValueError("Query embed token (151671) not found in input prompt")
        if len(doc_embed_positions) == 0:
            raise ValueError("Document embed tokens (151670) not found in input prompt")

        # Extract single token embedding for query
        query_pos = int(query_embed_positions[0])
        query_hidden = mx.expand_dims(hidden_states[query_pos], axis=0)  # [1, hidden_size]

        # Extract single token embeddings for all documents
        doc_hidden = mx.stack(
            [hidden_states[int(pos)] for pos in doc_embed_positions]
        )  # [num_docs, hidden_size]

        # Step 6: Project embeddings through MLP
        query_embeds = self.projector(query_hidden)  # [1, 512]
        doc_embeds = self.projector(doc_hidden)  # [num_docs, 512]

        # Reshape for consistency with reference implementation
        query_embeds = mx.expand_dims(query_embeds, axis=0)  # [1, 1, 512]
        doc_embeds = mx.expand_dims(doc_embeds, axis=0)  # [1, num_docs, 512]

        # Step 7: Compute cosine similarity scores
        # IMPORTANT: Do NOT pre-normalize embeddings. Compute cosine similarity directly.
        query_expanded = mx.broadcast_to(query_embeds, doc_embeds.shape)  # [1, num_docs, 512]

        # Cosine similarity formula: (A · B) / (||A|| * ||B||)
        scores = mx.sum(doc_embeds * query_expanded, axis=-1) / (
            mx.sqrt(mx.sum(doc_embeds * doc_embeds, axis=-1)) *
            mx.sqrt(mx.sum(query_expanded * query_expanded, axis=-1))
        )  # [1, num_docs]

        return query_embeds, doc_embeds, scores
```

**Public Method** (`rerank`, lines ~225-250):
```python
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> List[dict]:
        """Rerank documents by relevance to a query.

        Args:
            query: Search query string
            documents: List of document strings to rank
            top_n: Return only top N results (default: all)
            return_embeddings: Include embeddings in output (default: False)

        Returns:
            List of dictionaries with keys:
                - document: Original document text
                - relevance_score: Similarity score (higher = more relevant)
                - index: Position in input documents list
                - embedding: Doc embedding if return_embeddings=True, else None
        """
        # Process all documents at once
        query_embeds, doc_embeds, scores = self._compute_single_batch(
            query, documents, instruction=None
        )

        # Convert to numpy for easier manipulation
        doc_embeds_np = np.array(doc_embeds[0])  # [num_docs, 512]
        scores_np = np.array(scores[0])  # [num_docs]

        # Sort by relevance score (descending)
        scores_argsort = np.argsort(scores_np)[::-1]

        # Determine top_n
        if top_n is None:
            top_n = len(documents)
        else:
            top_n = min(top_n, len(documents))

        # Build results list
        return [
            {
                'document': documents[scores_argsort[i]],
                'relevance_score': float(scores_np[scores_argsort[i]]),
                'index': int(scores_argsort[i]),
                'embedding': doc_embeds_np[scores_argsort[i]].tolist() if return_embeddings else None,
            }
            for i in range(top_n)
        ]
```

---

### 2. Update File: `src/local_reranker/reranker_mlx.py`

#### a. Remove imports (lines 8, 14)

**Remove**:
```python
import importlib.util  # Line 8
from .mlx_cross_encoder import MLXCrossEncoderReranker  # Line 14
```

**Add**:
```python
from .jina_mlx_reranker import JinaMLXReranker  # Add this import
```

#### b. Delete `_load_repo_reranker` method (lines 114-127)

**Completely remove** this entire method:
```python
def _load_repo_reranker(
    self, rerank_file: str, model_path: str, projector_path: str
):
    """Load a repo-provided MLXReranker implementation from rerank.py."""
    # ... entire method ...
```

**Reason**: No longer loading `rerank.py` from model files.

#### c. Replace `_load_mlx_reranker` method (lines 79-112)

**Current implementation** (lines 79-112):
```python
def _load_mlx_reranker(self, model_path: str):
    """Load MLX reranker, falling back to the internal cross-encoder as needed."""
    rerank_file = os.path.join(model_path, "rerank.py")
    projector_path = os.path.join(model_path, "projector.safetensors")

    if os.path.exists(rerank_file):
        try:
            reranker = self._load_repo_reranker(
                rerank_file=rerank_file,
                model_path=model_path,
                projector_path=projector_path,
            )
            logger.info(
                "[MLX] Using repo-provided MLXReranker implementation from %s",
                rerank_file,
            )
            return reranker
        except Exception as repo_error:
            logger.warning(
                "[MLX] Failed to load repo-provided reranker at %s: %s. "
                "Falling back to internal cross-encoder.",
                rerank_file,
                repo_error,
            )
    else:
        logger.info(
            "[MLX] rerank.py not found in %s. Using internal cross-encoder fallback.",
            model_path,
        )

    return MLXCrossEncoderReranker(
        model_path=model_path,
        projector_path=projector_path,
    )
```

**New implementation**:
```python
def _load_mlx_reranker(self, model_path: str):
    """Load Jina-style MLX reranker.

    Directly instantiates JinaMLXReranker without attempting to load
    rerank.py from model files. This ensures consistent behavior and
    proper projector weight loading.

    Args:
        model_path: Path to model directory from huggingface_hub

    Returns:
        JinaMLXReranker instance

    Raises:
        RuntimeError: If reranker loading fails
    """
    projector_path = os.path.join(model_path, "projector.safetensors")

    logger.info(f"[MLX] Loading Jina-style reranker from {model_path}")

    try:
        reranker = JinaMLXReranker(
            model_path=model_path,
            projector_path=projector_path,
        )
        logger.info("[MLX] Successfully loaded Jina MLX reranker")
        return reranker
    except Exception as e:
        raise RuntimeError(f"Failed to load Jina MLX reranker: {e}") from e
```

**Changes summary**:
- Removed `rerank_file` logic entirely (no longer checking for it)
- Removed try-except fallback pattern
- Directly instantiate `JinaMLXReranker`
- Explicit error raising instead of silent fallback
- Clear logging for debugging

#### d. Check for remaining references

Run:
```bash
grep -n "MLXCrossEncoderReranker" src/local_reranker/reranker_mlx.py
```

If no remaining references, the import removal in step 2a was successful.

---

### 3. Update Tests

#### a. File: `tests/test_mlx_reranker.py`

**Remove tests** (lines 46-100):

Delete these entire test methods:
1. `test_load_mlx_reranker_uses_fallback_when_file_missing` (lines 46-60)
2. `test_load_mlx_reranker_falls_back_on_repo_error` (lines 62-81)
3. `test_load_mlx_reranker_prefers_repo_impl` (lines 83-100)

**Reason**: These tests are testing functionality that no longer exists (fallback loading).

**Keep test**:
- `test_mlx_reranker_implements_protocol` (lines 31-36) - protocol compliance check
- `test_initialization_runtime_error` (lines 38-44) - should still work with new error handling

**Update tests**:

1. Update `test_initialization_runtime_error` to mock `JinaMLXReranker` instead:
```python
@patch("huggingface_hub.snapshot_download")
@patch("local_reranker.reranker_mlx.JinaMLXReranker")
def test_initialization_runtime_error(self, mock_jina_reranker, mock_snapshot_download):
    """Test handling of runtime errors during initialization."""
    mock_jina_reranker.side_effect = Exception("Loading failed")

    with pytest.raises(RuntimeError, match="Failed to load Jina MLX reranker"):
        MLXReranker()
```

**Add new tests** (optional but recommended):

```python
@patch("huggingface_hub.snapshot_download")
@patch("local_reranker.reranker_mlx.JinaMLXReranker")
def test_loads_jina_reranker_successfully(self, mock_jina_reranker, mock_snapshot_download):
    """Test that Jina reranker loads successfully when model is available."""
    mock_instance = Mock()
    mock_jina_reranker.return_value = mock_instance
    mock_snapshot_download.return_value = "/fake/model/path"

    reranker = MLXReranker()

    mock_jina_reranker.assert_called_once_with(
        model_path="/fake/model/path",
        projector_path="/fake/model/path/projector.safetensors",
    )
    assert reranker.model is mock_instance

@patch("huggingface_hub.snapshot_download")
@patch("local_reranker.reranker_mlx.JinaMLXReranker")
def test_missing_projector_raises_error(self, mock_jina_reranker, mock_snapshot_download):
    """Test that missing projector file raises explicit error."""
    mock_jina_reranker.side_effect = FileNotFoundError("projector.safetensors not found")
    mock_snapshot_download.return_value = "/fake/model/path"

    with pytest.raises(RuntimeError, match="Failed to load Jina MLX reranker"):
        MLXReranker()
```

#### b. File: `tests/test_mlx_cross_encoder.py`

**Status**: No changes required

**Reason**: These tests directly test the `MLXCrossEncoderReranker` class, which remains in the codebase for potential future use with non-Jina models. The tests are independent of the changes to `reranker_mlx.py`.

---

### 4. File: `src/local_reranker/mlx_cross_encoder.py`

**Status**: No changes required

**Reason**: Keep the `MLXCrossEncoderReranker` class intact. It may be useful for:
- Future models that aren't Jina rerankers
- Alternative implementations
- Reference for understanding cross-encoder patterns

The class is well-tested and documented. Removing it would delete useful code that could be repurposed later.

---

### 5. Check and Update `__init__.py` (if needed)

**Check**: Open `src/local_reranker/__init__.py` and look for any exports of `MLXCrossEncoderReranker`.

**If exported**:
```python
from .mlx_cross_encoder import MLXCrossEncoderReranker
```

**Action**: Add `JinaMLXReranker` to exports:
```python
from .mlx_cross_encoder import MLXCrossEncoderReranker
from .jina_mlx_reranker import JinaMLXReranker  # Add this line

__all__ = ["Reranker", "MLXCrossEncoderReranker", "JinaMLXReranker"]
```

**If not exported**: No changes needed.

---

## Implementation Details & Considerations

### Special Tokens

Jina reranker v3 uses two special tokens for embedding extraction:

| Token | String | Token ID | Placement |
|-------|--------|-----------|-----------|
| Document embed | `<|embed_token\|>` | 151670 | After each document text |
| Query embed | `<|rerank_token\|>` | 151671 | After query text |

**Important Notes**:
- These token IDs are **hard-coded** for Jina v3 models
- They cannot be changed without retraining the model
- The reference implementation uses these exact IDs
- Tokens are defined in `special_tokens_map.json` in the model directory

**Current tokenizer verification**:
```bash
# From our earlier check
{
  "additional_special_tokens": [
    "<|score_token|>",
    "<|embed_token|>",
    "<|rerank_token|>"
  ]
}
```

### Prompt Format

The prompt structure is **critical** for correct embedding extraction. Special tokens must be placed at exact positions.

**Complete prompt structure**:
```
<|im_start|>system
You are a search relevance expert who can determine a ranking of the passages based on how relevant they are to the query. If the query is a question, how relevant a passage is depends on how well it answers the question. If not, try to analyze the intent of the query and assess how well each passage satisfies the intent. If an instruction is provided, you should follow the instruction when determining the ranking.
<|im_end|>
<|im_start|>user
I will provide you with {N} passages, each indicated by a numerical identifier. Rank the passages based on their relevance to query: {query}

[instruct]
{instruction}
</instruct>

<passage id="0">
{doc0}<|embed_token|>
</passage>
<passage id="1">
{doc1}<|embed_token|>
</passage>
...
<passage id="{N-1}">
{doc_{N-1}}<|embed_token|>
</passage>
<query>
{query}<|rerank_token|>
</query><|im_end|>
<|im_start|>assistant

<|EOT|>
```

**Key requirements**:
- System message is **fixed** for all reranking tasks
- Each passage gets a unique `id="{i}"` attribute
- Special tokens appear immediately after document/query text
- Assistant section ends with `<|EOT|>` token
- **No thinking** (assistant starts with empty output)

### Projector Architecture

**Fixed dimensions** (cannot be changed):
```
Input: 1024  (Jina model hidden_size)
Hidden: 512
Output: 512
```

**Architecture details**:
```python
linear1: Linear(1024 -> 512, bias=False)
activation: ReLU
linear2: Linear(512 -> 512, bias=False)
```

**Weight tensor shapes**:
- `linear1.weight`: [512, 1024] (out_features, in_features)
- `linear2.weight`: [512, 512] (out_features, in_features)

**Why fixed dimensions?**
- The projector is trained for the specific model architecture
- Changing dimensions requires retraining from scratch
- Jina provides pre-trained projector with these exact dimensions
- Hard-coded dimensions match reference implementation exactly

### Cosine Similarity Calculation

**Critical difference from internal fallback**:

❌ **Internal fallback (wrong for Jina)**:
```python
def _prepare_embedding(self, vector):
    projected = self.projector(vector)
    norm = mx.sqrt(mx.sum(projected * projected) + 1e-12)
    return projected / norm  # ← PRE-NORMALIZES

def _compute_scores(self, query_embedding, doc_embeddings):
    # Since embeddings are pre-normalized, dot product = cosine similarity
    sims = mx.matmul(stacked_docs, query)
```

✅ **Jina reference (correct)**:
```python
def _compute_scores(self, query_embeds, doc_embeds):
    # Compute cosine similarity WITHOUT pre-normalization
    scores = mx.sum(doc_embeds * query_expanded, axis=-1) / (
        mx.sqrt(mx.sum(doc_embeds * doc_embeds, axis=-1)) *
        mx.sqrt(mx.sum(query_expanded * query_expanded, axis=-1))
    )
```

**Why the difference matters**:
1. **Numerical precision**: Computing `(A·B) / (||A|| * ||B||)` in one operation is more accurate
2. **Semantic correctness**: Pre-normalization may introduce rounding errors
3. **Reference compliance**: Jina implementation does it this way
4. **Score range**: Proper formula produces scores in expected range [0.2, 0.7]

**Mathematical equivalence (theoretically)**:
```
cosine(A, B) = (A · B) / (||A|| * ||B||)
               = (A/||A||) · (B/||B||)
```

But numerical precision differs between approaches.

### Error Handling Strategy

**New approach**: Explicit errors, no silent fallbacks

**Examples**:

1. **Missing projector file**:
```python
raise FileNotFoundError(f"Projector file not found: {projector_path}")
```

2. **Wrong weight shapes**:
```python
raise ValueError(
    f"Unexpected linear1.weight shape: {w1.shape}, expected (512, 1024)"
)
```

3. **Special tokens not found**:
```python
if len(query_embed_positions) == 0:
    raise ValueError("Query embed token (151671) not found in input prompt")
```

4. **Model loading failure**:
```python
raise RuntimeError(f"Failed to load Jina MLX reranker: {e}")
```

**Benefits**:
- Clear error messages for debugging
- No hidden issues from fallback path
- Forces users to fix actual problems
- Matches software engineering best practices

### Configuration Hard-coding

**What is hard-coded**:

| Setting | Value | Reason |
|---------|--------|--------|
| Special token IDs | 151670, 151671 | Part of Jina model training |
| Projector dimensions | 1024→512→512 | Trained for specific architecture |
| System message | Fixed text | Standard for all reranking tasks |
| Activation function | ReLU (not GELU) | Matches Jina training |

**What is configurable**:
- Model path (via `model_name` parameter)
- Batch size (via `batch_size` parameter)
- Number of results (via `top_n` parameter)
- Return embeddings flag (via `return_embeddings` parameter)

**Trade-off**: Less flexible vs. guaranteed correctness

---

## Testing Strategy

### Unit Tests

#### Test 1: Projector Loads Correctly
```python
def test_projector_loads_weights():
    """Test that projector loads weights with correct shapes."""
    # Mock safetensors.load to return fake weights
    # Verify linear1.weight and linear2.weight are set
    # Verify shapes are (512, 1024) and (512, 512)
```

#### Test 2: Prompt Formatting
```python
def test_format_jina_prompt():
    """Test prompt format matches Jina specification."""
    query = "test query"
    docs = ["doc1", "doc2"]

    prompt = _format_jina_prompt(query, docs, special_tokens={...})

    assert "<|im_start|>system" in prompt
    assert "<|passage id=\"0\">" in prompt
    assert "doc1<|embed_token|>" in prompt
    assert "<query>" in prompt
    assert "test query<|rerank_token|>" in prompt
```

#### Test 3: Special Token Detection
```python
def test_special_tokens_found():
    """Test that special tokens are correctly located in tokenized prompt."""
    # Create fake prompt with known token positions
    # Verify query_embed_positions and doc_embed_positions are correct
```

#### Test 4: Embedding Extraction
```python
def test_embeddings_extracted_at_correct_positions():
    """Test embeddings are extracted at special token positions."""
    # Mock hidden states
    # Verify extraction happens at correct indices
```

#### Test 5: Cosine Similarity Calculation
```python
def test_cosine_similarity_formula():
    """Test cosine similarity matches expected formula."""
    # Create fake embeddings
    # Compute similarity
    # Verify: (A·B) / (||A|| * ||B||)
```

### Integration Tests

#### Test 1: Full Reranking Flow
```python
def test_full_reranking():
    """Test complete reranking pipeline."""
    query = "What is machine learning?"
    documents = [ ... ]  # Multiple documents

    results = reranker.rerank(query, documents)

    assert len(results) == len(documents)
    assert all('relevance_score' in r for r in results)
    assert all('index' in r for r in results)
    assert scores_sorted_descending([r['relevance_score'] for r in results])
```

#### Test 2: Score Range
```python
def test_score_range():
    """Test scores are in expected range."""
    query = "test query"
    documents = [ ... ]

    results = reranker.rerank(query, documents)
    scores = [r['relevance_score'] for r in results]

    # Scores should be in reasonable range for relevant docs
    assert all(0 <= s <= 1 for s in scores)
```

#### Test 3: Top-N Functionality
```python
def test_top_n_returns_correct_count():
    """Test top_n parameter works correctly."""
    query = "test query"
    documents = [ ... ]  # 10 documents

    results = reranker.rerank(query, documents, top_n=3)

    assert len(results) == 3
    assert sorted_by_score_descending(results)
```

#### Test 4: Embedding Return
```python
def test_return_embeddings():
    """Test embeddings are returned when requested."""
    query = "test query"
    documents = ["doc1", "doc2"]

    results = reranker.rerank(query, documents, return_embeddings=True)

    assert all('embedding' in r for r in results)
    assert all(len(r['embedding']) == 512 for r in results)
```

### Regression Tests

#### Test 1: Existing Test Suite Passes
```bash
# Run all existing tests
uv run pytest tests/

# Run MLX-specific tests
uv run pytest tests/test_mlx_reranker.py
uv run pytest tests/test_mlx_cross_encoder.py
```

#### Test 2: No Breaking Changes to API
```python
# Verify Reranker protocol is still implemented
def test_protocol_compliance():
    reranker = Reranker()

    # Must implement rerank method
    assert hasattr(reranker, 'rerank')
    # Must accept RerankRequest
    assert callable(reranker.rerank)
```

#### Test 3: Backward Compatibility
```python
# Test existing usage patterns still work
def test_existing_usage():
    request = RerankRequest(
        query="test",
        documents=["doc1", "doc2"],
        top_n=5,
    )

    results = reranker.rerank(request)

    # Results format unchanged
    assert isinstance(results, list)
    assert all(isinstance(r, RerankResult) for r in results)
```

### Real-World Test (Critical)

Run the exact curl request from `.attic/curl2.sh`:

```bash
# Test MLX implementation
curl http://localhost:8010/v1/rerank -H "Content-Type: application/json" -d '{
  "model": "jinaai/jina-reranker-v3-mlx",
  "query": "The physiological impact of high-altitude training on long-distance athletic endurance and oxygen transport",
  "top_n": 3,
  "documents": [ ... ]  # Same documents as in curl2.sh
}'
```

**Expected results**:
- **Scores**: In range [0.2, 0.7] (NOT [0.01, -0.07])
- **Ranking**: [0, 3, 9] (correct indices)
- **Consistency**: Close to PyTorch baseline scores

---

## Risk Assessment

### Low Risk Items

1. **Creating new file**: Doesn't affect existing code
2. **Adding import**: Safe operation
3. **Removing unused method**: No external dependencies
4. **Removing outdated tests**: Only tests removed functionality

### Medium Risk Items

1. **No fallback path**: If Jina reranker fails, entire reranking fails
   - **Mitigation**: Clear error messages help debugging
   - **Mitigation**: Can add configuration option for alternative models later

2. **Hard-coded special tokens**: Won't work if token IDs change
   - **Mitigation**: Jina v3 has fixed token IDs
   - **Mitigation**: Can add configuration if needed in future

3. **Fixed projector architecture**: Only works with Jina reranker v3-mlx
   - **Mitigation**: This is the target model
   - **Mitigation**: Keep `MLXCrossEncoderReranker` for other models

4. **Changing error handling behavior**: Existing code may expect fallbacks
   - **Mitigation**: Error messages are explicit
   - **Mitigation**: Fallback was already unreliable

### High Risk Items

**None identified** if implementation follows specification.

---

## Success Criteria

Implementation is successful when:

1. ✅ **MLX reranker produces scores in expected range**
   - Scores: 0.2 to 0.7 (not 0.01 to -0.07)
   - Verified with curl test from `.attic/curl2.sh`

2. ✅ **Ranking remains correct**
   - Top 3 indices: [0, 3, 9]
   - Same ranking as PyTorch baseline

3. ✅ **No attempt to load `rerank.py`**
   - `_load_repo_reranker` method removed
   - No `importlib.util` usage
   - No file existence checks for `rerank.py`

4. ✅ **All tests pass**
   - Unit tests for `JinaMLXReranker`
   - Updated tests in `test_mlx_reranker.py`
   - Existing test suite passes

5. ✅ **Code is clean and documented**
   - Docstrings on all public methods
   - Type hints everywhere
   - Clear comments for complex logic
   - No dead code

6. ✅ **Clear error messages**
   - Missing projector: Explicit FileNotFoundError
   - Wrong weights: Explicit ValueError with shapes
   - Token not found: Explicit ValueError with token ID
   - Model load failure: Explicit RuntimeError with details

7. ✅ **Projector weights loaded correctly**
   - `linear1.weight` shape: [512, 1024]
   - `linear2.weight` shape: [512, 512]
   - No random initialization warnings

8. ✅ **Prompt format matches Jina specification**
   - System message present and correct
   - Special tokens at correct positions
   - Document IDs properly formatted
   - Query section properly formatted

---

## Questions & Decisions

### Q1: Should special token IDs be configurable?

**Answer**: No, hard-code them for now.

**Reasoning**:
- Jina v3 has fixed token IDs (151670, 151671)
- These are part of the trained model
- Changing them would require retraining
- Hard-coding matches reference implementation
- Can make configurable in future if needed

**Alternative**: If flexibility is needed later:
```python
def __init__(self, model_path: str, projector_path: str,
             doc_token_id: int = 151670,
             query_token_id: int = 151671):
```

### Q2: Should we keep `MLXCrossEncoderReranker`?

**Answer**: Yes, keep it in codebase.

**Reasoning**:
- Well-tested and documented
- May be useful for other models (e.g., different embedding models)
- Good reference for future implementations
- No downside to keeping it
- Can be used if user wants alternative reranker

**Action**: No changes to `mlx_cross_encoder.py`

### Q3: Is it acceptable to have no fallback path?

**Answer**: Yes, explicit errors are better than silent fallbacks.

**Reasoning**:
- Silent fallbacks hide real problems
- Random projector weights produce meaningless results
- Users should know when things fail
- Clear error messages guide debugging
- Fallback was already producing bad results

**Alternative**: If we want to support multiple backends:
```python
def __init__(self, model_name: str = "jinaai/jina-reranker-v3-mlx",
             fallback_model: Optional[str] = None):
    # Try primary, then fallback if specified
```

**Decision**: Keep simple for now. Add if needed.

### Q4: Should we add comprehensive unit tests for `JinaMLXReranker`?

**Answer**: Yes, add unit tests beyond integration test.

**Recommended tests**:
1. `test_projector_loads_correct_weights`
2. `test_prompt_format_matches_spec`
3. `test_special_tokens_extracted_correctly`
4. `test_cosine_similarity_formula`
5. `test_full_reranking_pipeline`

**Reasoning**:
- Isolates components for testing
- Catches bugs early
- Documents expected behavior
- Helps future developers understand code
- Low cost, high value

**Implementation**: Add to `tests/test_jina_mlx_reranker.py` (new file)

---

## Implementation Checklist

### Phase 1: Create New Implementation
- [ ] Create `src/local_reranker/jina_mlx_reranker.py`
- [ ] Implement `JinaMLPProjector` class
- [ ] Implement `_load_projector` function
- [ ] Implement `_sanitize_input` function
- [ ] Implement `_format_jina_prompt` function
- [ ] Implement `JinaMLXReranker.__init__` method
- [ ] Implement `JinaMLXReranker._compute_single_batch` method
- [ ] Implement `JinaMLXReranker.rerank` method
- [ ] Add comprehensive docstrings

### Phase 2: Update Existing Code
- [ ] Update imports in `reranker_mlx.py`
- [ ] Remove `_load_repo_reranker` method from `reranker_mlx.py`
- [ ] Replace `_load_mlx_reranker` implementation in `reranker_mlx.py`
- [ ] Update `__init__.py` exports if needed
- [ ] Remove outdated tests from `test_mlx_reranker.py`
- [ ] Update remaining tests in `test_mlx_reranker.py`
- [ ] Add new tests for `JinaMLXReranker` (optional but recommended)

### Phase 3: Verify & Test
- [ ] Run linter: `uv run ruff check src/`
- [ ] Run type checker: `uv run pyright src/`
- [ ] Run unit tests: `uv run pytest tests/test_mlx_reranker.py`
- [ ] Run integration test: `uv run pytest tests/test_mlx_cross_encoder.py`
- [ ] Run full test suite: `uv run pytest tests/`
- [ ] Test with curl request from `.attic/curl2.sh`
- [ ] Verify score range: [0.2, 0.7] (not [0.01, -0.07])
- [ ] Verify ranking: [0, 3, 9]
- [ ] Compare with PyTorch baseline

### Phase 4: Documentation
- [ ] Update this implementation plan with any lessons learned
- [ ] Update README.md if needed
- [ ] Update any relevant API documentation
- [ ] Add comments for future maintainers

---

## Estimated Timeline

| Phase | Tasks | Estimated Time |
|-------|-------|---------------|
| Phase 1: Create Implementation | 8 tasks | 2-3 hours |
| Phase 2: Update Existing Code | 6 tasks | 1-2 hours |
| Phase 3: Verify & Test | 9 tasks | 2-3 hours |
| Phase 4: Documentation | 4 tasks | 1 hour |
| **Total** | **27 tasks** | **6-9 hours** |

---

## Post-Implementation Tasks

After implementation is complete and tested:

1. **Monitor for issues**: Watch for any unexpected behaviors in production
2. **Gather feedback**: Check if users report issues with specific models
3. **Performance comparison**: Compare MLX vs PyTorch performance metrics
4. **Consider optimizations**: Look for opportunities to improve speed
5. **Plan enhancements**: Based on usage patterns, plan future improvements

---

## Appendix: Reference Files

### Jina Reference Implementation
**Location**: `~/.cache/huggingface/hub/models--jinaai--jina-reranker-v3-mlx/snapshots/1d19fe38ae4e6658221479747c1152d6136dd6ab/rerank.py`

**Key sections to reference**:
- `MLPProjector` class (lines 9-20)
- `load_projector` function (lines 23-34)
- `format_docs_prompts_func` function (lines 44-82)
- `MLXReranker._compute_single_batch` method (lines 108-176)
- `MLXReranker.rerank` method (lines 178-228)

### Current Implementation Issues
**Location**: `src/local_reranker/mlx_cross_encoder.py`

**Problems**:
- Line 51-60: `SimpleProjector` uses GELU instead of ReLU
- Line 62-99: `load_from_tensors` tries to load bias (Jina has no bias)
- Line 258-261: `_prepare_embedding` pre-normalizes (Jina doesn't)
- Line 302-322: `_initialize_projector` creates wrong architecture

### Test Files
**Current**: `tests/test_mlx_reranker.py`
**Status**: Tests outdated functionality (fallback loading)
**Action**: Remove outdated tests, add new tests for Jina reranker

---

## Contact & Support

For questions or issues during implementation:

1. Review this plan first
2. Check Jina reference implementation
3. Run tests for immediate feedback
4. Check logs for error details
5. Consult codebase for similar patterns

**Remember**: The goal is to match Jina reference implementation exactly. Any deviation should be intentional and documented.
