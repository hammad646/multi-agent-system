# Python style
- Python 3.11+, full type hints, pydantic v2 for structured data, pydantic-settings for config.
- async for I/O; `tenacity` for retries; dependency injection for clients and repositories.
- Small functions; docstrings on public functions and every MCP tool.
- structlog for logs. No wildcard imports. No unnecessary global state.
- Do not create one giant file; follow the layout in SPEC.md section 4.
- requirements.txt uses compatible ranges; freeze the working set to `requirements.lock` after phase 1.
