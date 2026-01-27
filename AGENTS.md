# Ignore Patterns
attic/
docs/

# Coding
- Use Python 3.12 features and syntax
- Follow PEP 8 style guide for Python code
- Use type hints everywhere possible
- Use list, dict, and set comprehensions when appropriate for concise and readable code.
- Prefer pathlib over os.path for file system operation
- Use explicit exception handling. Catch specific exceptions rather than using bare except clauses
- Keep functions and methods small and focused on a single task
- Use docstrings for all public modules, functions, classes, and methods
- Use dataclasses for data containers when appropriate
- Prefer composition over inheritance where possible
- Use logging for debugging and monitoring
- Use meaningful variable and method names

# Development
- Use pytest for unit testing
- Do not create tests unless requested by the user
- Use uv for dependency management

## Using uv 
- Always use uv: Dependencies, running, venv, python
- use 'uv add <dependency name>' to add dependencies
- use 'uv remove <dependency name>' to remove dependencies
- Create uv scripts for running scripts in pyproject.toml [project.scripts]
- Use hatchling as the build-system

## Running
- Ensure activate the venv before spawning a new console session: `source .venv/bin/activate`
- Use uv scripts to run something

# Using Tools (MCP)
- Use the deepwiki tool to ask questions about a library
- Use the context7 tool to look up the documentation of a library.
- Use the tavily tool to search for information on the web and extract content form websites.

<!-- br-agent-instructions-v1 -->

---

## Beads Workflow Integration

This project uses [beads_rust](https://github.com/Dicklesworthstone/beads_rust) (`br`) for issue tracking. Issues are stored in `.beads/` and tracked in git.

### Essential Commands

```bash
# View ready issues (unblocked, not deferred)
br ready              # or: bd ready

# List and search
br list --status=open # All open issues
br show <id>          # Full issue details with dependencies
br search "keyword"   # Full-text search

# Create and update
br create --title="..." --type=task --priority=2
br update <id> --status=in_progress
br close <id> --reason="Completed"
br close <id1> <id2>  # Close multiple issues at once

# Sync with git
br sync --flush-only  # Export DB to JSONL
br sync --status      # Check sync status
```

### Workflow Pattern

1. **Start**: Run `br ready` to find actionable work
2. **Claim**: Use `br update <id> --status=in_progress`
3. **Work**: Implement the task
4. **Complete**: Use `br close <id>`
5. **Sync**: Always run `br sync --flush-only` at session end

### Key Concepts

- **Dependencies**: Issues can block other issues. `br ready` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers 0-4, not words)
- **Types**: task, bug, feature, epic, question, docs
- **Blocking**: `br dep add <issue> <depends-on>` to add dependencies

### Task Hierarchy (from PRD)

When implementing from a PRD, use this hierarchy:

**Level 1 - Epic** (`epic`): The overall initiative
- Example: "Batch Processing Refactoring"
- Top-level container for major work

**Level 2 - Features** (`feature`): Major phases/deliverables  
- Example: "Phase 1: Remove Unused Async Complexity"
- Children of the epic
- Use `blocks` dependencies for sequential phases

**Level 3 - Tasks** (`task`): Implementation work
- Example: "Remove rerank_async() from protocol"
- Children of features
- Individual units of work

**Quick Reference:**
```bash
# Create epic for overall initiative
br create --title="Batch Processing Refactoring" --type=epic --priority=2

# Create features for phases (blocking sequential phases)
br create --title="Phase 1: Remove Async" --type=feature --priority=1
br dep add bd-<phase2> bd-<phase1>  # Phase 2 blocks on Phase 1

# Create tasks under features
br create --title="Remove async method" --type=task --priority=1
```

### Session Protocol

**Before ending any session, run this checklist:**

```bash
git status              # Check what changed
git add <files>         # Stage code changes
br sync --flush-only    # Export beads changes to JSONL
git commit -m "..."     # Commit everything
git push                # Push to remote
```

### Best Practices

- Check `br ready` at session start to find available work
- Update status as you work (in_progress → closed)
- Create new issues with `br create` when you discover tasks
- Use descriptive titles and set appropriate priority/type
- Always sync before ending session

### Orchestrator Mode (Autonomous Execution)

For hands-off execution of beads tasks, use the orchestrator system:

```bash
@orchestrator implement epic bd-15z
```

**How it works:**
1. Orchestrator monitors `br ready` queue
2. Spawns @task-worker subagent for each ready task
3. Task worker implements, tests, and documents
4. Orchestrator updates beads status
5. Continues until all tasks complete

**Benefits:**
- Each task gets isolated context (fresh subagent)
- Built-in quality gates (ruff, mypy, pytest)
- Automatic progress tracking in progress.txt
- Pattern extraction to AGENTS.md

**Documentation:**
- `ORCHESTRATOR.md` - Full system documentation
- `.opencode/agents/orchestrator.md` - Orchestrator agent config
- `.opencode/agents/task-worker.md` - Task worker agent config

<!-- end-br-agent-instructions -->
