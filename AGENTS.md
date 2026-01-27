# Ignore Patterns
.attic/
docs/

## Coding
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

## Development
- Use pytest for unit testing
- Do not create tests unless requested by the user

## Python dev-tools
- use ruff to lint
- use pyright as a lsp
- Use uv for dependency management and running tools

### Using uv 
- Always use uv: Dependencies, running, venv, python
- use 'uv add <dependency name>' to add dependencies
- use 'uv remove <dependency name>' to remove dependencies
- Create uv scripts for running scripts in pyproject.toml [project.scripts]
- Use hatchling as the build-system

## Running
- Ensure activate the venv before spawning a new console session: `source .venv/bin/activate`
- Use uv scripts to run something

## Using Agent Tools (MCP)
- Use the deepwiki tool to ask questions about a library
- Use the context7 tool to look up the documentation of a library.
- Use the tavily tool to search for information on the web and extract content form websites.