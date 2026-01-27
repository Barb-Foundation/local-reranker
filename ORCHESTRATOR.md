# Beads Orchestrator System

This project uses an orchestrator-based workflow where a primary agent coordinates task execution by spawning specialized subagents.

## Architecture

```
Orchestrator Agent (@orchestrator)
    ↓ Spawns with PRD path
Task Worker Subagent (@task-worker)
    ↓ Reads PRD → Implements task
progress.txt & AGENTS.md updated
```

## How It Works

1. **Orchestrator** monitors `br ready` queue
2. For each ready task, spawns a **Task Worker** subagent (provides PRD file path)
3. **Task Worker** reads PRD, implements task, runs quality checks, updates documentation
4. **Orchestrator** updates beads status and continues

## Starting a Session

```bash
# Start orchestrator for an epic
@orchestrator implement epic bd-15z

# Or by PRD
@orchestrator implement prd spec/prd-batch-refactoring.md
```

## Agent Roles

### Orchestrator
- Monitors beads ready queue
- Spawns task workers
- Updates task status
- Handles session completion

### Task Worker
- Reads PRD file for full context (provided by orchestrator)
- Executes ONE task completely
- Runs quality checks (ruff, mypy, pytest)
- Updates progress.txt
- Updates AGENTS.md with patterns

## Progress Tracking

All work is documented in `progress.txt`:

```markdown
## Session: [Date] - [Epic]
PRD: spec/prd-*.md
Epic: bd-xxx

---

## Codebase Patterns
- Pattern 1: Description

---

## Task Log
### [Date/Time] - bd-xxx - [Title]
**Status:** ✅ Complete
**Files Modified:** ...
**Patterns Discovered:** ...
```

## Configuration

Agent configs are in `.opencode/agents/`:
- `orchestrator.md` - Primary coordination agent
- `task-worker.md` - Task execution subagent

## Benefits

- **Isolation**: Each task gets fresh context
- **Quality Gates**: Built-in ruff/mypy/pytest requirements
- **Documentation**: Automatic progress capture
- **Knowledge**: Patterns extracted to AGENTS.md
