# Contributing to AIterate

Thanks for helping improve AIterate.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,providers,tracking,managed-secrets]"
cd frontend
npm install
```

## Checks

```bash
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check .
cd frontend
npm run build
```

## Pull Requests

- Keep changes focused.
- Add or update tests for behavior changes.
- Do not commit provider keys, tokens, local databases, or generated frontend build output.
- Use the mock provider for tests unless a test explicitly mocks a live provider adapter.
