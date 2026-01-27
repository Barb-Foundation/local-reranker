---
name: beads-task-creator
description: "Convert PRDs to beads tasks for autonomous agent execution. Use when you have an existing PRD and need to create beads issues/tasks from it. Triggers on: create beads from this PRD, generate tasks from PRD, convert PRD to beads."
---

# Beads Task Creator

Converts PRDs into beads tasks for autonomous agent execution using beads_rust.

---

## The Job

Take a PRD (markdown file) and create beads tasks with proper hierarchy, dependencies, and priorities.

---

## Task Hierarchy

Beads uses three primary types for PRD-based work:

### 1. Epic (Level 1)
**Type:** `epic`
**When to use:** Large initiatives spanning multiple phases
**Example:** "Batch Processing Refactoring"
**Parent:** None (top-level container)

```bash
br create --title="Batch Processing Refactoring" --type=epic --priority=2
```

### 2. Feature (Level 2)
**Type:** `feature`
**When to use:** Major phases or deliverables from PRD sections
**Example:** "Phase 1: Remove Unused Async Complexity"
**Parent:** Epic (if exists)
**Dependencies:** Use `blocks` for sequential phases

```bash
br create --title="Phase 1: Remove Async" --type=feature --priority=1
br dep add bd-<phase2> bd-<phase1>  # Phase 2 blocked by Phase 1
```

### 3. Task (Level 3)
**Type:** `task`
**When to use:** Individual implementation work items
**Example:** "Remove rerank_async() from protocol"
**Parent:** Feature (parent-child dependency)

```bash
br create --title="Remove async method" --type=task --priority=1
```

---

## Task Size: The Number One Rule

**Each task must be completable in ONE agent iteration (one context window).**

Agents spawn fresh sessions with no memory of previous work. If a task is too big, the agent runs out of context before finishing.

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

---

## Task Ordering: Dependencies First

Tasks should be ordered so earlier tasks don't depend on later ones.

**Correct order:**
1. Remove unused code (protocol changes)
2. Remove unused code (implementations)
3. Create new shared components
4. Refactor existing code to use new components
5. Add tests

**Wrong order:**
1. Refactor to use new component (component doesn't exist yet)
2. Create the new component

---

## Priority Guidelines

- **P0 (0):** Critical bug fixes, security issues
- **P1 (1):** Implementation work, core features (default for tasks)
- **P2 (2):** Epic/feature containers, user stories, metrics tracking
- **P3 (3):** Nice-to-have improvements
- **P4 (4):** Backlog items, future considerations

```bash
# Implementation tasks
br create --title="Remove async method" --type=task --priority=1

# Epic/feature containers  
br create --title="Batch Processing Refactoring" --type=epic --priority=2

# User stories from PRD
br create --title="Clear and simple batch processing" --type=task --priority=2
```

---

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

---

## Acceptance Criteria: Must Be Verifiable

Each task should have clear, verifiable completion criteria in the description.

### Good criteria (verifiable):
- "Remove rerank_async() method from reranker.py (lines 34-45)"
- "Line count reduced from 567 to ≤180 lines"
- "All existing tests pass: pytest tests/test_reranker.py"
- "Type hints added to all new functions"
- "Run ruff and mypy with no errors"

### Bad criteria (vague):
- "Works correctly"
- "Code is clean"
- "Good performance"
- "Handles edge cases"

### Always include where applicable:
```
- Run linter: ruff check .
- Run type checker: mypy .
- Run tests: pytest
```

---

## Conversion Rules

1. **Each PRD section becomes a feature** (Phase 1, Phase 2, etc.)
2. **Each functional requirement becomes a task**
3. **User stories become tasks** with descriptive titles (P2 priority)
4. **Sequential features use `blocks` dependencies**
5. **All child tasks use `parent-child` dependencies**
6. **Set priority based on work type** (P1 for implementation, P2 for containers/stories)

---

## Example Conversion

**Input PRD Section:**
```markdown
### Phase 1: Remove Unused Async Complexity

1. **FR-1.1**: Remove `rerank_async()` method definition from `reranker.py` protocol (lines 34-45)
2. **FR-1.2**: Remove `rerank_async_final()` method definition from `reranker.py` protocol (lines 47-56)
3. **FR-1.3**: Remove `rerank_async()` implementation from `reranker_pytorch.py` (lines 256-439)
```

**Output Beads Tasks:**

```bash
# Step 1: Create epic (if not exists)
br create --title="Batch Processing Refactoring" --type=epic --priority=2
# → bd-15z

# Step 2: Create Phase 1 feature
br create --title="Phase 1: Remove Unused Async Complexity" \
  --type=feature --priority=1
# → bd-3sn

# Step 3: Set parent relationship
br dep add bd-3sn bd-15z

# Step 4: Create implementation tasks
br create --title="Remove async method from reranker.py protocol" \
  --type=task --priority=1 \
  --description="Remove rerank_async() method definition from reranker.py protocol (lines 34-45). Acceptance: Method removed, imports cleaned."
# → bd-1f6

br dep add bd-1f6 bd-3sn

br create --title="Remove async_final method from reranker.py protocol" \
  --type=task --priority=1 \
  --description="Remove rerank_async_final() method definition from reranker.py protocol (lines 47-56). Acceptance: Method removed, imports cleaned."
# → bd-xxx

br dep add bd-xxx bd-3sn

br create --title="Remove async methods from PyTorch backend" \
  --type=task --priority=1 \
  --description="Remove rerank_async() (lines 256-439) and rerank_async_final() (lines 441-566) from reranker_pytorch.py. Acceptance: Both methods removed, ~200 lines deleted."
# → bd-3nr

br dep add bd-3nr bd-3sn
```

---

## Workflow for Task Creation

### Step 1: Read the PRD
Analyze the PRD and identify:
- Epic title (overall initiative)
- Phases or major sections (become features)
- Individual functional requirements (become tasks)
- User stories (become tasks, P2 priority)
- Non-goals (don't create tasks for these)

### Step 2: Create Hierarchy
```bash
# Create epic first
br create --title="[PRD Title]" --type=epic --priority=2

# Create features for each phase
br create --title="Phase 1: [Phase Name]" --type=feature --priority=1

# Set blocking between sequential phases
br dep add bd-<phase2> bd-<phase1>

# Create tasks under each feature
br create --title="[Specific task]" --type=task --priority=1
```

### Step 3: Verify with `br ready`
```bash
br ready  # Should show unblocked tasks ready for implementation
```

---

## Checklist Before Creating Tasks

Before creating beads tasks, verify:

- [ ] Epic created as top-level container (if large initiative)
- [ ] Features created for each major phase/section
- [ ] Sequential features have `blocks` dependencies
- [ ] Tasks are small enough (completable in one iteration)
- [ ] Tasks ordered by dependency (remove before add, schema before UI)
- [ ] Each task has clear, verifiable acceptance criteria in description
- [ ] Parent-child dependencies set for all tasks
- [ ] Priorities set correctly (P1 for work, P2 for containers)
- [ ] User stories from PRD included as tasks (P2 priority)

---

## Common Patterns

### Pattern: Sequential Phases
```bash
# Phase 1, 2, 3 execute sequentially
br dep add bd-phase2 bd-phase1
br dep add bd-phase3 bd-phase2
```

### Pattern: Multiple Backends
```bash
# Same task for different backends (parallel execution)
br create --title="Remove async from PyTorch" --type=task --priority=1
br create --title="Remove async from MLX" --type=task --priority=1
# No dependency between them - can be done in parallel
```

### Pattern: Implementation then Tests
```bash
# Tests depend on implementation
br dep add bd-tests bd-implementation
```

### Pattern: User Stories
```bash
# User stories track acceptance criteria
br create --title="Clear and simple batch processing" \
  --type=task --priority=2 \
  --description="As a developer maintaining the codebase, I should see clear, simple batch processing logic in reranker files, not 500+ lines of duplicated complex batching code."
```

---

## Session Protocol

**After creating tasks:**

```bash
br ready                    # Verify unblocked tasks
br sync --flush-only        # Export to JSONL
git add .beads/             # Stage beads changes
git commit -m "Add beads tasks for [feature]"
```
