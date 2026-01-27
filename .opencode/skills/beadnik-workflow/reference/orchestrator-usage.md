# Orchestrator Usage Guide

Complete reference for the orchestrator agent that coordinates beads task execution.

## Overview

The orchestrator is a primary agent that manages autonomous execution of beads tasks by spawning task-worker subagents.

## When to Invoke

Invoke the orchestrator when:
- You have beads tasks ready in `br ready`
- You want hands-off execution
- Tasks are well-defined with acceptance criteria

## Invocation Commands

```bash
# By epic ID
@orchestrator implement epic bd-15z

# By PRD file
@orchestrator implement prd spec/prd-batch-refactoring.md
```

## Orchestrator Workflow

### Phase 1: Initialization

1. **Read PRD** - Loads requirements and context
2. **Read progress.txt** - Checks for existing session
3. **Run `br ready`** - Identifies available tasks
4. **Initialize tracking** - Sets up session monitoring

### Phase 2: Task Execution Loop

```
WHILE tasks in br ready:
  1. Pick highest priority task
  2. Spawn @task-worker with context
  3. Wait for completion status
  4. IF success:
       - Run: br close <task_id>
       - Continue to next task
  5. IF failure:
       - Report error
       - Stop execution
```

### Phase 3: Session Completion

1. **Detect no more ready tasks**
2. **Commit all changes**:
   ```bash
   git add .
   git commit -m "feat: complete [epic name]"
   ```
3. **Sync beads**:
   ```bash
   br sync --flush-only
   ```
4. **Report completion**:
   ```
   <branch_complete>
   Epic: [name]
   Tasks Completed: [count]
   Duration: [time]
   </branch_complete>
   ```

## Context Provided to Task Workers

When spawning a task worker, the orchestrator provides:

```
Task: bd-xxx
Title: [task title]
Description: [full description with acceptance criteria]
PRD Context: [relevant PRD section]
Codebase Patterns: [from progress.txt]
```

## What Orchestrator Does NOT Do

❌ **Does NOT implement code** - Delegates to task workers  
❌ **Does NOT write to progress.txt** - Task workers handle documentation  
❌ **Does NOT update AGENTS.md** - Task workers capture patterns  
❌ **Does NOT parse complex reports** - Task workers return simple JSON status  

## Monitoring Progress

While orchestrator runs:

```bash
# In another terminal, watch progress
tail -f progress.txt

# Or check beads status
br ready
br list --status=in_progress
```

## Handling Failures

### Task Worker Fails

If a task worker returns `{"status": "failed"}`:

1. Orchestrator stops execution
2. Error details in `progress.txt`
3. Fix issue manually
4. Re-run: `@orchestrator implement epic bd-xxx`

### Recovery Strategy

```bash
# Check what failed
cat progress.txt | grep -A 10 "FAILED"

# Fix the issue manually
# (edit files as needed)

# Update beads if task is partially done
br update bd-xxx --status closed

# Resume orchestration
@orchestrator implement epic bd-xxx
```

## Configuration

Agent configuration in `.opencode/agents/orchestrator.md`:

```yaml
---
description: "Coordinates beads task execution by monitoring ready queue and spawning task-worker subagents"
mode: primary
model: anthropic/claude-sonnet-4-20250514
temperature: 0.2
tools:
  write: true
  edit: true
  bash: true
  read: true
  task: true
permissions:
  task:
    "*": deny
    "task-worker": allow
---
```

Key settings:
- **mode: primary** - Can be invoked directly by user
- **task permissions** - Only allowed to spawn task-worker
- **temperature: 0.2** - Focused, deterministic behavior

## Session Artifacts

After orchestration completes, you'll have:

1. **progress.txt** - Complete session log
2. **Updated AGENTS.md** - Consolidated patterns
3. **Git commits** - All changes committed
4. **Closed beads tasks** - All tasks marked complete

## Example Session Flow

```bash
# User starts orchestration
$ @orchestrator implement epic bd-15z

# Orchestrator output:
[16:00:00] Reading PRD: spec/prd-batch-refactoring.md
[16:00:01] Found 14 ready tasks
[16:00:02] Spawning task-worker for bd-1f6...
[16:05:30] ✓ bd-1f6 complete
[16:05:31] Spawning task-worker for bd-3nr...
[16:12:15] ✓ bd-3nr complete
...
[18:30:00] All tasks complete
[18:30:01] Committing changes...
[18:30:05] Session complete

<branch_complete>
Epic: Batch Processing Refactoring
Tasks Completed: 14
Duration: 2h 30m
</branch_complete>
```

## Best Practices

1. **Ensure tasks exist** before starting - run `br ready`
2. **Review progress.txt** after completion
3. **Verify commits** with `git log`
4. **Check AGENTS.md** for new patterns
5. **Re-run if interrupted** - orchestrator is idempotent

## Troubleshooting

### Orchestrator won't start?
- Check `.opencode/agents/orchestrator.md` exists
- Verify no syntax errors in agent config
- Ensure `br` CLI is available

### Tasks not being picked up?
```bash
br ready              # Check ready queue
br show bd-xxx        # Check specific task status
```

### Want to stop mid-session?
- Press Ctrl+C to interrupt
- Orchestrator commits work completed so far
- Resume by re-invoking

## Comparison to Manual Execution

| Aspect | Manual | Orchestrator |
|--------|--------|--------------|
| Context | Accumulates | Fresh per task |
| Quality Gates | Optional | Enforced |
| Documentation | Manual | Automatic |
| Pattern Capture | Manual | Automatic |
| Time | Variable | Consistent |

## Integration with Beads

The orchestrator integrates tightly with beads:

- Reads from `br ready` queue
- Updates status via `br close`
- Syncs with `br sync --flush-only`
- Respects dependencies automatically
