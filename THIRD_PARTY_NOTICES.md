# Third-Party Notices

This file is a practical summary for AIterate maintainers. It is not legal advice.

## Default Python Dependencies

The default `aiterate` package is intended to use permissive open-source dependencies, primarily
MIT, BSD-3-Clause, Apache-2.0, or similar licenses.

Notable default dependencies include FastAPI, Uvicorn, Pydantic, Typer, Rich, PyYAML,
Cryptography, SQLAlchemy, Alembic, PyJWT, Requests, python-multipart, and SkillOpt.

SkillOpt is the Microsoft-originated optimization framework used by AIterate's
`SkillOptInspiredOptimizer` for edit/update/gate mechanics. Local package metadata for
`skillopt==0.1.0` reports the license as MIT.

## Optional Extras

Some integrations are optional and should be reviewed by organizations before enabling them:

- `postgres`: installs `psycopg[binary]`. Local metadata reports `LGPL-3.0-only`, so this extra is
  intentionally not part of the default install.
- `providers`: installs provider SDKs for OpenAI, Anthropic, AWS Bedrock, and LiteLLM.
- `tracking`: installs MLflow and LangSmith integrations.
- `managed-secrets`: installs Vault, AWS, Azure, and GCP secret-manager clients.
- `dev`: installs build/test/release tooling and Pillow for regenerating README assets.

## Frontend Dependencies

The frontend direct dependencies are permissively licensed at the time this notice was written:

- React: MIT
- React DOM: MIT
- Vite: MIT
- Vite React plugin: MIT
- lucide-react: ISC

## External Service Terms

AIterate integrates with external services but does not grant rights to use those services. Users
must comply with the terms for their own configured providers and infrastructure, including:

- OpenAI
- Azure OpenAI
- Anthropic
- AWS Bedrock
- Google/Gemini-compatible APIs through LiteLLM
- MLflow
- LangSmith
- GitHub
- Bitbucket
- Vault, AWS Secrets Manager, Azure Key Vault, and GCP Secret Manager

## Maintainer Checklist

Before public releases:

- Run dependency license review for direct and transitive dependencies.
- Keep `psycopg` in the optional `postgres` extra unless legal review approves default inclusion.
- Do not vendor SDK source code unless its license and notice requirements are reviewed.
- Keep generated assets under project ownership or document their source/license.
- Re-run npm audit and Python dependency checks before publishing.
