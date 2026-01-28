[PRD]
# PRD: Fix Duplicate README Content

## Overview
Remove duplicate "Running the Server" section from README.md. The section appears twice (lines 116-133 and 134-151), causing confusion and redundancy.

## Goals
- Eliminate duplicate "Running the Server" section
- Maintain clear, single source of truth for server startup instructions
- Preserve all accurate information from both duplicates

## Quality Gates

Documentation-only change - no automated quality gates required.

## User Stories

### US-001: Remove duplicate "Running the Server" section
**Description:** As a reader, I want to see each section only once so that the README is concise and easy to follow.

**Acceptance Criteria:**
- [ ] Remove duplicate "Running the Server" section (lines 134-151 or 116-133)
- [ ] Keep one complete "Running the Server" section that includes:
  - Method 1: Using CLI (Recommended)
  - Method 2: Using uvicorn directly
  - Method 3: Using uv run
- [ ] Ensure model download information is preserved
- [ ] Verify no information is lost in the consolidation

## Functional Requirements
- FR-1: Retain Method 1 (CLI) instructions with examples
- FR-2: Retain Method 2 (uvicorn) instructions
- FR-3: Retain Method 3 (uv run) instructions
- FR-4: Retain model download information for both PyTorch and MLX

## Non-Goals
- Restructuring other sections
- Rewriting or rewording existing content (unless to consolidate)
- Adding new content
- Formatting changes unrelated to duplicate removal

## Success Metrics
- README has exactly one "Running the Server" section
- No information is lost compared to current version
- Section appears at most once in the document

## Open Questions
- None
[/PRD]
