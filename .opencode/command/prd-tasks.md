---
agent: build
description: create tasks using beads
---
# Rule: Generating a Task List from User Requirements

## Goal

To guide an AI assistant in creating a detailed, step-by-step task list in Markdown format based on user requirements, feature requests, or existing documentation. The task list should guide a developer through implementation.

## Process

1.  **Receive Requirements:** The user provides a feature request, task description, or points to existing documentation
2.  **Analyze Requirements:** The AI analyzes the functional requirements, user needs, and implementation scope from the provided information
3.  **Phase 1: Generate Parent Tasks:** Based on the requirements analysis, create the file and generate the main, high-level tasks required to implement the feature. **IMPORTANT: Always include task "Create feature branch" as the first task, unless the user specifically requests not to create a branch.** Use your judgement on how many additional high-level tasks to use. It's likely to be about 5. Present these tasks to the user in the specified format (without sub-tasks yet). Inform the user: "I have generated the high-level tasks based on your requirements. Ready to generate the sub-tasks? Respond with 'Go' to proceed."
4.  **Wait for Confirmation:** Pause and wait for the user to respond with "Go".
5.  **Phase 2: Generate Sub-Tasks:** Once the user confirms, break down each parent task into smaller, actionable sub-tasks necessary to complete the parent task. Ensure sub-tasks logically follow from the parent task and cover the implementation details implied by the requirements.
6.  **Identify Relevant Files:** Based on the tasks and requirements, identify potential files that will need to be created or modified. List these under the `Relevant Files` section, including corresponding test files if applicable.
7.  **Generate Final Output:** Combine the parent tasks, sub-tasks, relevant files, and notes into the final Markdown structure.

## Beads
Create the tasks with beads. Try to figure out the depednecies and the priorities

### Hierarchy
When creating tasks from a PRD, follow this hierarchy structure:

**Level 1: Epic** (if large initiative)
- Type: `epic` 
- Scope: The overall initiative from the PRD title (e.g., "Batch Processing Refactoring")
- Parent: None (top-level)

**Level 2: Features** (major deliverables)
- Type: `feature`
- Scope: Major phases or deliverables from the PRD (e.g., "Phase 1: Remove Unused Async Complexity")
- Parent: The epic (if exists)
- Dependencies: Set blocking relationships between sequential features

**Level 3: Tasks** (implementation work)
- Type: `task`
- Scope: Individual implementation steps (e.g., "Remove rerank_async() method from protocol")
- Parent: The corresponding feature
- Dependencies: Parent-child relationships for grouping

**Example Structure:**
```
epic: Batch Processing Refactoring
├── feature: Phase 1 - Remove Unused Async Complexity
│   ├── task: Remove async method from protocol
│   ├── task: Remove async methods from PyTorch backend
│   └── task: Remove async methods from MLX backend
├── feature: Phase 2 - Simplify Core Processing
│   └── task: Create helper methods...
└── feature: Phase 3 - Extract Commonality
    └── task: Create BatchProcessor class
```

**Notes:**
- User stories from the PRD can be created as `task` type with descriptive titles
- Use `blocks` dependencies between sequential features (Phase 1 blocks Phase 2, etc.)
- Use `parent-child` dependencies to group tasks under their feature
- Priority: P1 for implementation work, P2 for user stories/metrics tracking 

## Interaction Model

The process explicitly requires a pause after generating parent tasks to get user confirmation ("Go") before proceeding to generate the detailed sub-tasks. This ensures the high-level plan aligns with user expectations before diving into details.

## Target Audience

Assume the primary reader of the task list is a **junior developer** who will implement the feature.