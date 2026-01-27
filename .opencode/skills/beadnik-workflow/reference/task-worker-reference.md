# Task Worker Reference

Complete guide for the task-worker subagent that executes individual beads tasks.

## Overview

Task workers are hidden subagents spawned by the orchestrator to execute ONE beads task completely.

## Key Characteristics

- **Hidden**: Not shown in autocomplete, invoked programmatically
- **Isolated**: Fresh context for each task
- **Self-contained**: Implements, tests, and documents
- **Accountable**: Must pass quality gates

## Lifecycle

### 1. Receive Context

Orchestrator provides:
```
Task: bd-xxx
Title: Remove async method from protocol
Description: Remove rerank_async() from reranker.py (lines 34-45)...
Acceptance Criteria:
- [ ] Method removed
- [ ] Imports cleaned
PRD File: spec/prd-batch-refactoring.md
PRD Section: Phase 1: Remove Unused Async Complexity
Codebase Patterns: [from progress.txt]
```

### 2. Read PRD

**CRITICAL:** Task worker MUST read the PRD to understand full context.

```bash
# Read the PRD file
cat spec/prd-batch-refactoring.md

# Focus on the relevant section
grep -A 50 "Phase 1: Remove Unused Async Complexity" spec/prd-batch-refactoring.md
```

### 3. Read Current State

```bash
# Read task details
br show bd-xxx

# Read relevant files
cat src/local_reranker/reranker.py
```

### 4. Implement Task

Follow acceptance criteria exactly:
- Make minimal, focused changes
- Follow Codebase Patterns
- Match existing code style

### 5. Run Quality Checks

**MUST run these commands:**

```bash
# Linting
ruff check .

# Type checking
mypy .

# Testing (if tests exist)
pytest
```

**All checks MUST pass.** If they fail, fix issues or report failure.

### 6. Document Work

#### Update progress.txt

Append to end of file:

```markdown
### [Date/Time] - bd-xxx - [Task Title]
**Status:** ✅ Complete

**Files Modified:**
- src/file.py: specific change description
- src/other.py: another change

**Implementation Summary:**
Brief description of what was done and why.

**New Patterns Discovered:**
- Pattern name: Description of reusable pattern
- API convention: How to use X in this codebase

**Gotchas Encountered:**
- Issue: Description of problem encountered
- Resolution: How it was solved

**Quality Checks:**
- ruff: ✅
- mypy: ✅
- pytest: ✅ (or N/A if no tests)
```

#### Update AGENTS.md

If you discovered reusable patterns:

1. **Identify directories modified**
   ```bash
   git diff --name-only
   ```

2. **Find relevant AGENTS.md**
   - Check each modified directory
   - Check parent directories

3. **Add patterns**
   ```markdown
   ## Patterns from bd-xxx
   
   ### [Pattern Name]
   [Description of the pattern]
   
   Example:
   ```python
   # Show example usage
   ```
   
   ### Gotcha: [Issue Name]
   [Description of issue and solution]
   ```

### 7. Return Status

Return simple JSON to orchestrator:

**Success:**
```json
{"status": "success", "task_id": "bd-xxx", "files_changed": 3}
```

**Failure:**
```json
{"status": "failed", "task_id": "bd-xxx", "reason": "ruff check failed with import errors"}
```

## Configuration

Agent configuration in `.opencode/agents/task-worker.md`:

```yaml
---
description: "Executes a single beads task completely including implementation, testing, and documentation updates"
mode: subagent
model: anthropic/claude-sonnet-4-20250514
temperature: 0.1
maxSteps: 50
tools:
  write: true
  edit: true
  bash: true
  read: true
permissions:
  bash:
    "pytest *": allow
    "ruff *": allow
    "mypy *": allow
    "br *": allow
---
```

Key settings:
- **mode: subagent** - Can only be invoked by other agents
- **hidden: true** - Not shown in user autocomplete
- **maxSteps: 50** - Limits iterations to control costs
- **temperature: 0.1** - Very focused, deterministic

## Critical Rules

### DO:
- ✅ Work on ONE task only
- ✅ Pass all quality checks
- ✅ Write to progress.txt
- ✅ Update AGENTS.md with patterns
- ✅ Be specific in descriptions
- ✅ Follow existing code patterns

### DON'T:
- ❌ Work on multiple tasks
- ❌ Skip quality checks
- ❌ Return without documenting
- ❌ Make sweeping changes
- ❌ Ignore Codebase Patterns
- ❌ Commit (orchestrator handles this)

## Pattern Recognition

### What to Capture in AGENTS.md

**Good patterns:**
- API usage conventions
- File organization patterns
- Testing approaches
- Error handling patterns
- Configuration requirements

**Example:**
```markdown
## Patterns from bd-1f6

### Protocol Method Removal
When removing methods from protocols:
1. Remove method signature
2. Remove from __all__ if present
3. Check imports in dependent files
4. Run ruff to catch unused imports

### Gotcha: Async Import Cleanup
AsyncGenerator imports often have multiple uses.
Search carefully before removing:
grep -r "AsyncGenerator" src/
```

### What NOT to Capture

- Story-specific details
- One-time debugging steps
- Information already in progress.txt
- Obvious conventions

## Quality Check Examples

### ruff check .

**Pass:**
```
All checks passed!
```

**Fail:**
```
src/local_reranker/reranker.py:10:1: F401 imported but unused
```

**Fix:**
```bash
# Remove unused import
# Or add noqa comment if intentional
```

### mypy .

**Pass:**
```
Success: no issues found in 12 source files
```

**Fail:**
```
src/local_reranker/reranker.py:45: error: Argument 1 to "rerank" has incompatible type
```

**Fix:**
```python
# Add proper type hints
# Or fix type mismatch
```

### pytest

**Pass:**
```
================ 15 passed, 2 skipped in 0.45s ================
```

**Fail:**
```
FAILED tests/test_reranker.py::test_rerank - AssertionError
```

**Fix:**
```python
# Fix implementation
# Or update test if behavior changed
```

## Error Handling

### When Checks Fail

1. **Read error carefully**
2. **Fix root cause** - Don't suppress
3. **Re-run checks** - Verify fix
4. **Document in progress.txt** - Note any gotchas

### When Unsure

If you encounter an issue you can't resolve:

1. **Document what you tried**
2. **Return failure status**
3. **Include details in reason field**

Example:
```json
{"status": "failed", "task_id": "bd-xxx", "reason": "mypy fails with complex type error in external library. Tried: adding stubs, ignoring line, casting."}
```

## Time Management

With `maxSteps: 50`, you have limited iterations:

- **Steps 1-5**: Read and understand
- **Steps 6-30**: Implement
- **Steps 31-45**: Quality checks and fixes
- **Steps 46-50**: Documentation and return

**If running low on steps:**
- Prioritize quality checks
- Return partial completion
- Document what's done

## Success Criteria

A task worker session is successful when:

1. ✅ Task fully implemented
2. ✅ All quality checks pass
3. ✅ progress.txt updated
4. ✅ AGENTS.md updated (if patterns found)
5. ✅ Status returned to orchestrator

## Best Practices

1. **Read before writing** - Understand existing code
2. **Make minimal changes** - Don't refactor unnecessarily
3. **Test as you go** - Don't wait until the end
4. **Document immediately** - While context is fresh
5. **Be specific** - "Changed X to Y" not "Fixed stuff"

## Example Session

```
[Task Worker bd-1f6 Started]

Step 1: Reading task context...
Task: Remove async method from reranker.py protocol
Acceptance: Remove lines 34-45, clean imports

Step 2: Reading current file...
cat src/local_reranker/reranker.py
[Shows current code]

Step 3: Implementing...
edit reranker.py
- Remove rerank_async() method
- Remove rerank_async_final() method
- Remove AsyncGenerator import

Step 4: Quality checks...
ruff check . ✅
mypy . ✅
pytest ✅

Step 5: Documenting...
Append to progress.txt:
### 2026-01-27 - bd-1f6 - Remove async method
**Status:** ✅ Complete
**Files:** reranker.py (removed 22 lines)
**Patterns:** Protocol cleanup checklist
**Gotchas:** None

Step 6: Return status...
{"status": "success", "task_id": "bd-1f6", "files_changed": 1}

[Task Worker bd-1f6 Complete]
```
