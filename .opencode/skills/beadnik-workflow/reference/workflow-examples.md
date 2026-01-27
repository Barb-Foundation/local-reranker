# Workflow Examples

Concrete examples of using the orchestrator system for various scenarios.

## Example 1: Complete Epic Execution

**Scenario:** Execute all tasks in the Batch Processing Refactoring epic.

### Setup

```bash
# Verify tasks exist
br ready
# Shows 14 ready tasks

# Check PRD exists
ls spec/prd-batch-refactoring.md
```

### Execution

```bash
@orchestrator implement epic bd-15z
```

### Expected Flow

```
[16:00:00] Orchestrator started
[16:00:01] Reading PRD: spec/prd-batch-refactoring.md
[16:00:02] Reading progress.txt
[16:00:03] Running: br ready
[16:00:04] Found 14 ready tasks

[16:00:05] Spawning task-worker for bd-1f6
           Task: Remove async method from protocol
[16:05:00] ✓ bd-1f6 complete
[16:05:01] Running: br close bd-1f6

[16:05:02] Spawning task-worker for bd-3nr
           Task: Remove async methods from PyTorch backend
[16:12:00] ✓ bd-3nr complete
[16:12:01] Running: br close bd-3nr

[16:12:02] Spawning task-worker for bd-35n
           Task: Remove async methods from MLX backend
[16:18:00] ✓ bd-35n complete
[16:18:01] Running: br close bd-35n

... (continues through all 14 tasks)

[18:30:00] No more ready tasks
[18:30:01] Committing changes
[18:30:05] Running: br sync --flush-only
[18:30:06] Session complete

<branch_complete>
Epic: Batch Processing Refactoring
Tasks Completed: 14
Duration: 2h 30m
Status: All quality checks passed
</branch_complete>
```

### Post-Execution

```bash
# Review what was done
cat progress.txt
git log --oneline -15

# Verify all tasks closed
br list --status=open
# Should show 0 tasks

# Check line count reduction
wc -l src/local_reranker/reranker_pytorch.py
# Was: 567, Now: 165
```

## Example 2: Partial Execution (Resuming)

**Scenario:** Orchestrator was interrupted, need to resume.

### Current State

```bash
br list --status=open
# Shows 8 remaining tasks

cat progress.txt | tail -30
# Shows 6 tasks already completed
```

### Resume

```bash
@orchestrator implement epic bd-15z
```

### Expected Behavior

Orchestrator:
1. Reads progress.txt (sees 6 tasks done)
2. Runs `br ready` (sees 8 tasks)
3. Continues from where it left off
4. Only processes remaining 8 tasks

### Result

```
[10:00:00] Orchestrator started
[10:00:01] Reading progress.txt
[10:00:02] Found 6 completed tasks
[10:00:03] Running: br ready
[10:00:04] Found 8 ready tasks
[10:00:05] Resuming from task 7...

[10:00:06] Spawning task-worker for bd-12f...
...
[12:00:00] All tasks complete
```

## Example 3: Handling a Failing Task

**Scenario:** One task fails quality checks.

### Execution

```bash
@orchestrator implement epic bd-15z
```

### Failure Scenario

```
[16:00:00] Orchestrator started

[16:00:05] Spawning task-worker for bd-1f6
[16:05:00] ✓ bd-1f6 complete

[16:05:01] Spawning task-worker for bd-3nr
[16:12:00] ✗ bd-3nr FAILED
[16:12:00] Error: mypy check failed
           Details in progress.txt
[16:12:01] Stopping execution

Orchestrator stopped due to task failure.
Check progress.txt for details.
Fix the issue and re-run to continue.
```

### Troubleshooting

```bash
# Check what failed
cat progress.txt | grep -A 20 "bd-3nr"
# Shows:
# ### 2026-01-27 - bd-3nr - Remove async methods from PyTorch
# **Status:** ❌ FAILED
# **Error:** mypy: src/local_reranker/reranker_pytorch.py:45 error

# Fix the issue
# (edit files as needed)

# Verify fix
mypy src/local_reranker/reranker_pytorch.py
# ✅ Success

# Manually close the task
br close bd-3nr

# Resume orchestration
@orchestrator implement epic bd-15z
```

## Example 4: Parallel vs Sequential Tasks

### Sequential Example (Phase Dependencies)

**PRD Structure:**
```
Phase 1: Setup (must complete first)
  → Phase 2: Implementation (depends on Phase 1)
    → Phase 3: Testing (depends on Phase 2)
```

**Beads Setup:**
```bash
# Phase 2 blocked by Phase 1
br dep add bd-phase2 bd-phase1

# Phase 3 blocked by Phase 2
br dep add bd-phase3 bd-phase2
```

**Execution:**
```bash
@orchestrator implement epic bd-xxx
```

**Result:**
```
Orchestrator processes sequentially:
1. Phase 1 tasks (all complete)
2. Phase 2 tasks (unblocked after Phase 1)
3. Phase 3 tasks (unblocked after Phase 2)
```

### Parallel Example (Independent Tasks)

**PRD Structure:**
```
Backend A Refactoring (independent)
Backend B Refactoring (independent)
```

**Beads Setup:**
```bash
# No dependencies between them
br create --title="Refactor Backend A" --type=task
br create --title="Refactor Backend B" --type=task
# No br dep add commands
```

**Execution:**
```bash
@orchestrator implement epic bd-xxx
```

**Result:**
```
Orchestrator processes in priority order:
1. Backend A task
2. Backend B task
# (Could be parallel in future enhancement)
```

## Example 5: Adding Tasks Mid-Execution

**Scenario:** Discover new work while orchestrating.

### Initial State

```bash
@orchestrator implement epic bd-15z
# Processing Phase 1 tasks...
```

### Discovery

Task worker finds additional cleanup needed:

```markdown
### 2026-01-27 - bd-3nr - Remove async methods
**New Patterns Discovered:**
- Found unused helper function in utils.py

**Suggested New Task:**
Remove unused helper `async_helper()` from utils.py
```

### Adding Task

```bash
# In another terminal
br create --title="Remove unused async_helper from utils" \
  --type=task --priority=1 \
  --description="Remove async_helper() function from utils.py"
# → bd-new1

# Set parent
br dep add bd-new1 bd-phase1-feature
```

### Orchestrator Behavior

Orchestrator will:
1. Continue current task
2. Check `br ready` after each task
3. See new task bd-new1
4. Process it when ready

## Example 6: Post-Session Review

**Scenario:** Epic completed, reviewing results.

### Review Progress

```bash
# Read session log
cat progress.txt
```

**Output:**
```markdown
## Session: Batch Processing Refactoring
Epic: bd-15z

---

## Codebase Patterns
- Use simple list operations instead of ResultAggregator
- Keep batch processing sequential for GPU vectorization
- Always run ruff/mypy/pytest before committing

---

## Task Log

### 2026-01-27 - bd-1f6 - Remove async from protocol
**Status:** ✅ Complete
**Files:** reranker.py (-22 lines)
**Patterns:** Protocol cleanup checklist

### 2026-01-27 - bd-3nr - Remove async from PyTorch
**Status:** ✅ Complete
**Files:** reranker_pytorch.py (-184 lines)
**Gotchas:** AsyncGenerator also used in models.py

... (12 more tasks)

### 2026-01-27 - bd-3jm - Validate code reduction
**Status:** ✅ Complete
**Results:**
- PyTorch: 567 → 165 lines (71% reduction)
- MLX: 723 → 208 lines (71% reduction)
- All quality checks pass
```

### Review Git History

```bash
git log --oneline -15
```

**Output:**
```
abc1234 feat: bd-3jm - Validate code reduction
abc1233 feat: bd-15f - Run full test suite
abc1232 feat: bd-1vw - Create BatchProcessor unit tests
...
abc1221 feat: bd-1f6 - Remove async method from protocol
abc1220 feat: Session setup - Add orchestrator config
```

### Review AGENTS.md

```bash
cat AGENTS.md | grep -A 20 "Patterns from"
```

**Output:**
```markdown
## Patterns from Batch Processing Refactoring

### ResultAggregator Replacement
When removing ResultAggregator:
1. Replace with simple list.extend()
2. Use sorted() with key=lambda
3. Slice with [:top_n]

### Async Cleanup Checklist
Before removing async methods:
1. Check protocol definition
2. Check all implementations
3. Check for import side effects
4. Run grep for method name
```

### Verify Completion

```bash
# Check all tasks closed
br list --status=open | wc -l
# → 0

# Check line counts
wc -l src/local_reranker/reranker_pytorch.py src/local_reranker/reranker_mlx.py
# → 165 reranker_pytorch.py
# → 208 reranker_mlx.py

# Run final quality check
ruff check . && mypy . && pytest
# ✅ All pass
```

## Example 7: Multi-Epic Coordination

**Scenario:** Multiple epics with shared dependencies.

### Setup

```bash
# Epic 1: Core Refactoring
br create --title="Core Refactoring" --type=epic
# → bd-epic1

# Epic 2: API Improvements  
br create --title="API Improvements" --type=epic
# → bd-epic2

# Shared dependency: Core must complete first
br dep add bd-epic2 bd-epic1
```

### Execution Strategy

```bash
# Start with Epic 1
@orchestrator implement epic bd-epic1

# ... wait for completion ...

# Then Epic 2
@orchestrator implement epic bd-epic2
```

### Alternative: Manual Dependency Management

If epics don't have explicit dependencies:

```bash
# Run both independently
@orchestrator implement epic bd-epic1 &
@orchestrator implement epic bd-epic2 &

# (Note: Requires future parallel support)
```

## Example 8: Custom Task Worker

**Scenario:** Need specialized task worker for specific work.

### Create Custom Agent

`.opencode/agents/task-worker-security.md`:
```yaml
---
description: "Task worker specialized for security-related changes"
mode: subagent
hidden: true
maxSteps: 50
tools:
  write: true
  edit: true
  bash: true
permissions:
  bash:
    "bandit *": allow
    "safety *": allow
---

# Security Task Worker

Execute security-related tasks with extra checks.

Additional quality checks:
- bandit (security linter)
- safety (dependency check)
```

### Usage

Modify orchestrator to spawn different workers based on task type:

```yaml
# In orchestrator.md, add logic:
# IF task.title contains "security":
#   Spawn @task-worker-security
# ELSE:
#   Spawn @task-worker
```

### Execution

```bash
@orchestrator implement epic bd-security-audit
```

## Best Practices Summary

1. **Always check `br ready`** before starting
2. **Review progress.txt** after completion
3. **Fix failures immediately** - don't accumulate
4. **Add tasks as discovered** - don't lose work
5. **Commit regularly** - orchestrator does this automatically
6. **Update AGENTS.md** - task workers handle this
7. **Verify quality** - all checks must pass
8. **Keep tasks small** - one iteration per task
