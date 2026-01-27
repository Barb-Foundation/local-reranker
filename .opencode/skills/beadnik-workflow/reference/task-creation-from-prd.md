# Task Creation from PRD

Detailed guide for converting Product Requirements Documents (PRDs) into beads task hierarchies.

## Overview

Before using the orchestrator, you need beads tasks. This guide covers converting PRDs into executable task hierarchies.

## Task Hierarchy

Beads uses three primary types for PRD-based work:

### 1. Epic (Level 1)
**Type:** `epic`  
**Scope:** Large initiatives spanning multiple phases  
**Example:** "Batch Processing Refactoring"  
**Parent:** None (top-level container)

```bash
br create --title="Batch Processing Refactoring" --type=epic --priority=2
```

### 2. Feature (Level 2)
**Type:** `feature`  
**Scope:** Major phases or deliverables from PRD sections  
**Example:** "Phase 1: Remove Unused Async Complexity"  
**Parent:** Epic (if exists)  
**Dependencies:** Use `blocks` for sequential phases

```bash
br create --title="Phase 1: Remove Async" --type=feature --priority=2
br dep add bd-<phase2> bd-<phase1>  # Phase 2 blocked by Phase 1
```

### 3. Task (Level 3)
**Type:** `task`  
**Scope:** Individual implementation work items  
**Example:** "Remove rerank_async() from protocol"  
**Parent:** Feature (parent-child dependency)

```bash
br create --title="Remove async method" --type=task --priority=1
```

## Task Size: Critical Rule

**Each task must be completable in ONE agent iteration (one context window).**

Agents spawn fresh sessions with no memory. If a task is too big, the agent runs out of context.

### Right-sized tasks:
- Remove a specific method from one file
- Create a single helper function
- Add a unit test for one function
- Update one configuration parameter
- Refactor one function to use a new pattern

### Too big (split these):
- "Refactor entire backend" → Split into: remove methods, create helpers, update callers
- "Add authentication" → Split into: schema, middleware, login UI, session handling
- "Update all tests" → Split into one task per test file

**Rule of thumb:** If you cannot describe the change in 2-3 sentences, it is too big.

## Task Ordering: Dependencies First

Order tasks so earlier ones don't depend on later ones.

**Correct order:**
1. Remove unused code (protocol changes)
2. Remove unused code (implementations)
3. Create new shared components
4. Refactor existing code to use new components
5. Add tests

**Wrong order:**
1. Refactor to use new component (component doesn't exist yet)
2. Create the new component

## Priority Guidelines

- **P0 (0):** Critical bug fixes, security issues
- **P1 (1):** Implementation work, core features (default for tasks)
- **P2 (2):** Epic/feature containers, user stories, metrics tracking
- **P3 (3):** Nice-to-have improvements
- **P4 (4):** Backlog items, future considerations

## Dependency Types

### parent-child
Groups tasks under their parent feature/epic.

```bash
# Child depends on parent
br dep add bd-<child-task> bd-<parent-feature>
```

### blocks
Sequential dependencies - task cannot start until blocked task completes.

```bash
# Phase 2 cannot start until Phase 1 is done
br dep add bd-<phase2> bd-<phase1>
```

## Acceptance Criteria: Must Be Verifiable

Each task should have clear, verifiable completion criteria.

### Good criteria:
- "Remove rerank_async() method from reranker.py (lines 34-45)"
- "Line count reduced from 567 to ≤180 lines"
- "All existing tests pass: pytest tests/test_reranker.py"
- "Run ruff and mypy with no errors"

### Bad criteria:
- "Works correctly"
- "Code is clean"
- "Good performance"

Always include where applicable:
```
- Run linter: ruff check .
- Run type checker: mypy .
- Run tests: pytest
```

## Conversion Rules

1. **Each PRD section becomes a feature** (Phase 1, Phase 2, etc.)
2. **Each functional requirement becomes a task**
3. **User stories become tasks** with descriptive titles (P2 priority)
4. **Sequential features use `blocks` dependencies**
5. **All child tasks use `parent-child` dependencies**
6. **Set priority based on work type** (P1 for work, P2 for containers/stories)

## Example Conversion

**Input PRD Section:**
```markdown
### Phase 1: Remove Unused Async Complexity

1. **FR-1.1**: Remove `rerank_async()` from `reranker.py` (lines 34-45)
2. **FR-1.2**: Remove `rerank_async_final()` from `reranker.py` (lines 47-56)
3. **FR-1.3**: Remove `rerank_async()` from `reranker_pytorch.py` (lines 256-439)
```

**Commands to execute:**

```bash
# Step 1: Create epic
br create --title="Batch Processing Refactoring" --type=epic --priority=2
# → bd-15z

# Step 2: Create Phase 1 feature
br create --title="Phase 1: Remove Unused Async Complexity" \
  --type=feature --priority=2
# → bd-3sn

# Step 3: Set parent relationship
br dep add bd-3sn bd-15z

# Step 4: Create implementation tasks
br create --title="Remove async method from protocol" \
  --type=task --priority=1 \
  --description="Remove rerank_async() from reranker.py protocol (lines 34-45).
  
  Acceptance Criteria:
  - [ ] Method removed from protocol
  - [ ] Imports cleaned up
  - [ ] Run: ruff check src/local_reranker/reranker.py"
# → bd-1f6

br dep add bd-1f6 bd-3sn

br create --title="Remove async_final from protocol" \
  --type=task --priority=1 \
  --description="Remove rerank_async_final() from reranker.py protocol (lines 47-56)."
# → bd-xxx

br dep add bd-xxx bd-3sn

br create --title="Remove async methods from PyTorch backend" \
  --type=task --priority=1 \
  --description="Remove rerank_async() (lines 256-439) from reranker_pytorch.py.
  
  Acceptance:
  - Method completely removed
  - ~180 lines deleted
  - Run: ruff check ."
# → bd-3nr

br dep add bd-3nr bd-3sn

# Step 5: Verify
br ready  # Should show unblocked tasks
```

## Common Patterns

### Sequential Phases
```bash
br dep add bd-phase2 bd-phase1
br dep add bd-phase3 bd-phase2
```

### Multiple Backends (Parallel)
```bash
br create --title="Remove async from PyTorch" --type=task --priority=1
br create --title="Remove async from MLX" --type=task --priority=1
# No dependency - can be done in parallel
```

### Implementation then Tests
```bash
br create --title="Implement feature X" --type=task --priority=1
# → bd-impl

br create --title="Add tests for feature X" --type=task --priority=1
# → bd-tests

br dep add bd-tests bd-impl
```

### User Stories
```bash
br create --title="Clear and simple batch processing" \
  --type=task --priority=2 \
  --description="As a developer, I should see clear, simple batch processing logic."
```

## Checklist Before Creating

- [ ] Epic created as top-level container (if large initiative)
- [ ] Features created for each major phase/section
- [ ] Sequential features have `blocks` dependencies
- [ ] Tasks are small enough (completable in one iteration)
- [ ] Tasks ordered by dependency
- [ ] Each task has clear, verifiable acceptance criteria
- [ ] Parent-child dependencies set for all tasks
- [ ] Priorities set correctly (P1 for work, P2 for containers)
- [ ] User stories from PRD included as tasks

## Post-Creation Verification

```bash
br ready                    # Verify unblocked tasks
br graph                    # Visualize dependencies
br sync --flush-only        # Export to JSONL
git add .beads/             # Stage changes
git commit -m "Add beads tasks for [feature]"
```
