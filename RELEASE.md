# Aiterate Release Checklist

Use this checklist for the first public release.

## 1. Pick Final Public Coordinates

- Confirm the PyPI name is available:

```bash
python -m pip index versions aiterate
```

If PyPI says no matching distribution is found, the name is likely available. The final authority is
the upload attempt. If the name is taken, change `project.name` in `pyproject.toml` before release.

- Create the public GitHub repo, for example:

```bash
gh repo create AthiraSPillai/aiterate --public --source . --remote origin
```

- Update `pyproject.toml` URLs if your final org/repo differs.

## 2. Clean Local Generated State

Do not publish local runtime files:

```bash
Remove-Item -Recurse -Force .aiterate, .pytest_tmp, frontend\dist -ErrorAction SilentlyContinue
```

## 3. Run Release Checks

```bash
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check .
cd frontend
npm run build
cd ..
```

Review dependency licensing before publishing:

```bash
Get-Content THIRD_PARTY_NOTICES.md
```

Pay special attention to optional extras such as `postgres`, provider SDKs, tracking SDKs, and
managed-secret adapters.

## 4. Build Python Distribution

```bash
.venv\Scripts\python -m pip install --upgrade build twine
.venv\Scripts\python -m build
.venv\Scripts\python -m twine check dist/*
```

## 5. Smoke Test The Wheel

Use a fresh virtual environment:

```bash
python -m venv .venv-release-test
.venv-release-test\Scripts\python -m pip install (Get-ChildItem dist\aiterate-*-py3-none-any.whl | Select-Object -Last 1).FullName
.venv-release-test\Scripts\aiterate --help
```

Run a local mock optimization:

```bash
.venv-release-test\Scripts\aiterate optimize --name smoke-test --data examples\raw_support_notes.txt --policy examples\policies.yml
```

## 6. Publish To TestPyPI First

Preferred path: TestPyPI Trusted Publisher with GitHub Actions.

On TestPyPI, add a pending GitHub publisher with:

- Project name: `aiterate`
- Owner: `AthiraSPillai`
- Repository name: `aiterate`
- Workflow name: `publish-testpypi.yml`
- Environment name: `testpypi`

Then run the GitHub Actions workflow **Publish to TestPyPI** manually from the repository Actions
tab. The workflow uses OIDC trusted publishing and does not need a TestPyPI password or API token.

Manual fallback: create a TestPyPI API token at https://test.pypi.org/manage/account/token/.

```bash
.venv\Scripts\python -m twine upload --repository testpypi dist/*
```

Install from TestPyPI in a fresh environment:

```bash
python -m venv .venv-testpypi
.venv-testpypi\Scripts\python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ aiterate
.venv-testpypi\Scripts\aiterate --help
```

## 7. Publish To PyPI

Preferred path: PyPI Trusted Publisher on the GitHub release workflow.

On PyPI, configure a trusted publisher for:

- Project name: `aiterate`
- Owner: `AthiraSPillai`
- Repository name: `aiterate`
- Workflow name: `publish.yml`

Then publish a GitHub release. The workflow **Publish Python Package** uploads to PyPI without a
password or API token.

Manual fallback: create a PyPI API token at https://pypi.org/manage/account/token/.

```bash
.venv\Scripts\python -m twine upload dist/*
```

## 8. GitHub Public Page

Before announcing:

- Set repo description: `AI Artifact Lifecycle Management for prompt and skill change management.`
- Add topics: `ai`, `prompt-engineering`, `prompt-optimization`, `prompt-management`, `llm-evals`, `regression-testing`, `agents`, `mlflow`, `langsmith`, `fastapi`, `react`, `pypi`.
- Enable Issues and Discussions.
- Add branch protection for `main`.
- Configure TestPyPI and PyPI trusted publishing.
- Create a GitHub release tagged with the package version, for example `v0.1.1`.

## Known V1 Gaps To Be Honest About

- The optimizer is text-artifact optimization, not model-weight fine-tuning.
- GitHub and Bitbucket PR publishing require server-side tokens and existing branches.
- Live model eval requires real provider credentials and may incur model costs.
- RBAC is API-key/JWT boundary ready, not a full hosted SSO product yet.
- The frontend is included as source in the repo, not bundled into the Python wheel.
