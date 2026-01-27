---
name: beadnik-workflow
description: "Manage and execute beads tasks using the beadnik orchestrator and task-worker agent system. Use when implementing beads tasks autonomously with beadnik, spawning subagents for task execution, tracking progress in progress.txt, or converting PRDs to executable task hierarchies. Triggers on: beadnik, @orchestrator implement, run beads tasks, execute epic, convert PRD to beads, autonomous task execution."
---

# Beadnik Workflow

Manage autonomous execution of beads tasks using an orchestrator that spawns specialized task-worker subagents.

## Quick Start

```bash
# Start autonomous execution of an epic
@orchestrator implement epic bd-xxx

# Or reference by PRD
@orchestrator implement prd spec/prd-feature.md
```

## System Overview

This workflow uses three components:

1. **Orchestrator Agent** - Coordinates execution
2. **Task Worker Subagents** - Execute individual tasks (read PRD for context)
3. **Beads + Progress.txt** - Track work and capture learnings

```
Orchestrator monitors br ready
    ↓ Spawns @task-worker with PRD path
Task Worker reads PRD → implements task + docs
    ↓ Returns status
Orchestrator updates beads, continues
```

## When to Use

Use this skill when:
- ✅ You have beads tasks ready to implement
- ✅ You want hands-off autonomous execution
- ✅ Tasks are small and well-defined
- ✅ Quality gates (ruff/mypy/pytest) should be enforced
- ✅ You want automatic progress tracking

## Workflow Steps

### Step 1: Ensure Tasks Exist

Before orchestration, tasks must be created. See `./reference/task-creation-from-prd.md` for detailed guidance.

Quick check:
```bash
br ready  # Should show unblocked tasks
```

### Step 2: Start Orchestrator

```bash
@orchestrator implement epic bd-xxx
```

The orchestrator will:
1. Read the PRD
2. Read progress.txt
3. Monitor `br ready` queue
4. Spawn task workers for each task
5. Update beads status
6. Commit changes when complete

### Step 3: Monitor Progress

Track execution in `progress.txt`:

```markdown
### 2026-01-27 - bd-xxx - Task Title
**Status:** ✅ Complete
**Files Modified:**
- src/file.py: specific changes
**Patterns Discovered:**
- New pattern for future use
**Gotchas:**
- Issue and resolution
```

### Step 4: Review Learnings

After completion, check:
- `progress.txt` - Full session log
- `AGENTS.md` - Consolidated patterns
- Git commits - All changes tracked

## Agent Responsibilities

### Orchestrator (@orchestrator)
- Reads PRD and progress.txt
- Monitors `br ready` queue
- Spawns ONE task worker at a time
- Updates beads: `br close <id>`
- Commits all changes at end

**Does NOT:**
- Implement code
- Write to progress.txt
- Update AGENTS.md

### Task Worker (@task-worker - hidden)
- Reads PRD file (provided by orchestrator) for full context
- Executes ONE beads task
- Runs quality checks (ruff, mypy, pytest)
- Appends to progress.txt
- Updates AGENTS.md with patterns
- Returns: `{"status": "success", "task_id": "bd-xxx"}`

**MUST:**
- Read PRD to understand full context
- Pass all quality checks
- Document all changes
- Capture learnings

## Quality Gates

Every task worker runs:

```bash
ruff check .      # Linting
mypy .            # Type checking
pytest            # Tests (if exist)
```

**All must pass before task is marked complete.**

## Configuration Files

Agent configurations are in `.opencode/agents/`:

- `orchestrator.md` - Primary coordination agent
- `task-worker.md` - Task execution subagent (hidden)

See `./reference/orchestrator-usage.md` and `./reference/task-worker-reference.md` for details.

## Common Patterns

See `./reference/workflow-examples.md` for concrete examples including:
- Full epic execution
- Handling task failures
- Parallel vs sequential tasks
- Post-session review

## Troubleshooting

### No tasks ready?
```bash
br ready              # Check ready queue
br blocked            # Check blocked tasks
br show bd-xxx        # Check specific task
```

### Task worker failed?
- Check `progress.txt` for error details
- Fix issue manually
- Re-run: `@orchestrator implement epic bd-xxx`

### Need to pause?
- Orchestrator processes sequentially
- It will stop if a task fails
- Resume by re-invoking

## Session End Protocol

When orchestrator completes:

```bash
# Already done by orchestrator:
git add .
git commit -m "feat: complete [epic name]"
br sync --flush-only

# You should verify:
git log --oneline -5    # Review commits
cat progress.txt        # Review session log
```

## Documentation References

- `./reference/task-creation-from-prd.md` - Convert PRDs to beads tasks
- `./reference/orchestrator-usage.md` - Orchestrator agent details
- `./reference/task-worker-reference.md` - Task worker agent details
- `./reference/workflow-examples.md` - Concrete usage examples
- `ORCHESTRATOR.md` - Project-specific documentation
- `AGENTS.md` - Project instructions (updated by task workers)
