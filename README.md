# AIterate

**AI Artifact Lifecycle Management from raw data and policies.**

AIterate helps teams turn messy source material into production-ready prompts and agent skills.
Give it raw data, policies, examples, and priorities; AIterate creates, optimizes, validates,
approves, versions, tracks, and promotes the result.

Manual AI artifact changes do not scale well. Teams lose track of why a prompt or skill changed,
which policy or dataset triggered the change, whether old behavior regressed, and which version is
safe to promote. AIterate turns those edits into a repeatable lifecycle with eval checks,
regression signals, approvals, and traceable versions.

![AIterate workflow](docs/assets/aiterate-workflow.gif)

## Why Use AIterate?

- Create prompts and skills from raw data, examples, policies, and rubrics.
- Import common formats: text, CSV, JSON, YAML, and YML.
- Replace ad hoc AI artifact edits with reproducible optimization runs.
- Optimize artifacts using weighted policy priorities.
- Evaluate artifacts with native checks for policy coverage, JSON shape, similarity, grounding,
  uncertainty handling, prompt injection, and PII leakage.
- Catch regressions when prompts, policies, data, or target models change.
- Version every accepted prompt or skill.
- Trace which data, policy, provider, and run produced each version.
- Start from an existing baseline prompt/skill or generate one from raw data.
- Use OpenAI, Azure OpenAI, AWS Bedrock, or other providers through LiteLLM.
- Track experiments with MLflow, with optional LangSmith support.
- Run locally with SQLite metadata storage, then move to Postgres for production.

## Install

```bash
pip install aiterate
```

Optional provider and tracking integrations:

```bash
pip install "aiterate[providers,tracking]"
```

Postgres-backed production installs:

```bash
pip install "aiterate[postgres]"
```

Managed secrets integrations:

```bash
pip install "aiterate[managed-secrets]"
```

AIterate supports Python **3.11, 3.12, and 3.13**.

## Quickstart

Choose the path that fits your audience:

- **UI workflow** for product, policy, and operations users.
- **CLI workflow** for developers and automation.
- **Notebook/Python workflow** for data scientists, AI engineers, and backend-only usage.

All three workflows can create, optimize, version, and trace prompts or agent skills.

## UI Workflow

Start the backend:

```bash
uvicorn aiterate.api.main:app --reload
```

Start the web app:

```bash
cd frontend
npm run dev
```

Open the local UI, paste or upload raw data, enter policies, choose a provider, and run the
optimization. Provider credentials can be pasted for one run or saved encrypted server-side.

The UI supports:

- file upload for text, CSV, JSON, YAML, YML, and Markdown
- automatic data-category detection
- optional baseline prompt/skill input for existing production artifacts
- policy weight editing or equal weighting
- native eval checks for regression-sensitive behavior, safety, grounding, and output shape
- separate optimizer model and target model selection
- safe provider credential entry with hidden keys
- MLflow/LangSmith tracking selection
- Git artifact tracking and promotion PR workflow scaffolding
- GitHub and Bitbucket promotion PR publishing when server credentials are configured
- optimization run results with accepted and rejected candidate decisions
- native eval report with pass rate, failed checks, and suggested prompt/skill changes
- validate and approve flow before creating a promotion PR

Typical use cases:

- a support prompt needs to change after a policy update, but the team wants to catch citation,
  escalation, and tone regressions before promotion
- a skill needs to be generated from messy notes and reviewed as a versioned artifact
- platform teams need to compare prompt versions or model targets with the same eval rubric
- governance teams need proof of which data, policy, model, and approval produced a prompt

## CLI Workflow

Create a raw data file:

```text
Customers ask support agents to summarize account changes, explain policy limits,
and cite the source policy. Responses must be concise and escalate when confidence is low.
```

Create a policy file:

```yaml
policies:
  - id: cite_sources
    text: Always cite the policy or dataset section used to answer.
    weight: 0.35
  - id: concise
    text: Keep answers under 180 words unless the user asks for detail.
    weight: 0.20
  - id: escalate_uncertainty
    text: Escalate to a human reviewer when source data is incomplete or contradictory.
    weight: 0.45
```

Run an optimization:

```bash
aiterate optimize --name support-agent --data raw_support_notes.txt --policy policies.yml
```

If you already have a prompt or skill, use it as the starting baseline:

```bash
aiterate optimize --name support-agent --data raw_support_notes.txt --baseline current_prompt.md --policy policies.yml
```

If `--baseline` is omitted, AIterate creates the initial baseline from raw data and
policies. The CLI defaults to a local mock provider so developers can test automation before adding
model credentials.

Create a skill instead of a prompt:

```bash
aiterate optimize \
  --name support-skill \
  --kind skill \
  --data raw_support_notes.txt \
  --policy policies.yml
```

Use a configured provider:

```bash
aiterate optimize \
  --name support-agent \
  --provider openai \
  --model gpt-4.1 \
  --data raw_support_notes.txt \
  --policy policies.yml
```

Run native eval checks in CI or locally:

```bash
aiterate eval \
  --artifact prompt.md \
  --data raw_support_notes.txt \
  --policy policies.yml \
  --min-score 0.75
```

## Notebook Or Python Workflow

Use AIterate directly from a notebook, script, or backend job:

```python
from pathlib import Path

from aiterate.domain import OptimizationRequest, PriorityRule, ProviderConfig, ProviderKind
from aiterate.sdk import AIterateClient

raw_data = Path("raw_support_notes.txt").read_text()

policies = [
    PriorityRule(
        id="cite_sources",
        text="Always cite the policy or dataset section used to answer.",
        weight=0.35,
    ),
    PriorityRule(
        id="concise",
        text="Keep answers under 180 words unless the user asks for detail.",
        weight=0.20,
    ),
    PriorityRule(
        id="escalate_uncertainty",
        text="Escalate to a human reviewer when source data is incomplete or contradictory.",
        weight=0.45,
    ),
]

client = AIterateClient()

run = client.optimize(
    OptimizationRequest(
        name="support-agent",
        raw_data=raw_data,
        policies=policies,
        provider=ProviderConfig(
            kind=ProviderKind.MOCK,
            model="mock-optimizer",
        ),
        iterations=3,
    )
)

print(run.best_version.content)
print(run.best_version.score)
```

Switch to OpenAI, Azure OpenAI, or AWS Bedrock by changing the provider config:

```python
ProviderConfig(kind=ProviderKind.OPENAI, model="gpt-4.1")
ProviderConfig(kind=ProviderKind.AZURE_OPENAI, model="gpt-4.1", deployment="my-deployment")
ProviderConfig(kind=ProviderKind.AWS_BEDROCK, model="anthropic.claude-3-5-sonnet-20240620-v1:0")
```

## Backend-Only API Workflow

Run the API:

```bash
uvicorn aiterate.api.main:app --reload
```

Submit an optimization request:

```bash
curl -X POST http://127.0.0.1:8000/v1/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "name": "support-agent",
    "raw_data": "Support answers must cite sources and escalate uncertainty.",
    "policies": [
      {
        "id": "cite",
        "text": "Always cite sources.",
        "weight": 0.5
      },
      {
        "id": "escalate",
        "text": "Escalate incomplete data.",
        "weight": 0.5
      }
    ],
    "provider": {
      "kind": "mock",
      "model": "mock-optimizer"
    },
    "iterations": 3
  }'
```

## Supported Data Formats

AIterate accepts plain text plus structured files. Structured data can use any of these top-level
arrays:

- `cases`
- `examples`
- `data`
- `records`

Example JSON:

```json
{
  "cases": [
    {
      "input": "Summarize the refund policy.",
      "expected": "Answer concisely and cite the source."
    }
  ]
}
```

Example CSV:

```csv
input,expected
Summarize the refund policy.,Answer concisely and cite the source.
Data is incomplete.,Escalate uncertainty.
```

## Model Providers

AIterate supports first-class provider configuration for:

- OpenAI
- Anthropic
- Azure OpenAI
- AWS Bedrock
- LiteLLM-compatible providers

Typical environment variables:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AWS_REGION=us-east-1
AWS_PROFILE=...
```

## Tracking

AIterate can record optimization runs, scores, artifacts, and lineage in MLflow. LangSmith support is
available for teams that use it for LLM observability.

```bash
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Background Jobs

For production-style runs, queue optimizer work and process it with the worker:

```bash
aiterate migrate
aiterate worker
```

The API also exposes `/v1/optimization-jobs`, `/v1/jobs/{job_id}`, and an admin-only
`/v1/jobs/run-next` endpoint for controlled worker execution.

## Secrets And Integrations

Long-lived keys can be added through the UI in v1. Paste a key once, and the backend stores it in
encrypted database-backed secret storage. The secret value is never returned to the browser after
save; the UI only shows configured status and a fingerprint.

For production, replace local encrypted storage with a managed secrets provider such as Vault, AWS
Secrets Manager, Azure Key Vault, or GCP Secret Manager.

Common backend variables:

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AWS_PROFILE=...
AWS_REGION=us-east-1
MLFLOW_TRACKING_URI=http://localhost:5000
LANGSMITH_API_KEY=...
GITHUB_TOKEN=...
GITHUB_APP_ID=...
BITBUCKET_TOKEN=...
AIT_SECRET_PROVIDER=database
```

For production, set `AIT_SECRET_PROVIDER` to `vault`, `aws`, `azure`, or `gcp` and configure the
matching backend variables. Run database/tracking connections over TLS.

## Auth And RBAC

Local development runs with auth disabled. For shared environments, enable bearer-token auth:

```bash
AIT_AUTH_ENABLED=true
AIT_ADMIN_API_KEY=<admin-api-key>
AIT_JWT_SECRET=<jwt-signing-secret>
```

Admin users can save secrets and run worker/admin actions. Editor users can run optimizations,
compare models, approve runs, and publish PRs. Viewer users can read runs and job status.

## Production Persistence

AIterate defaults to local SQLite for a fast single-user quickstart. Use Postgres for run history,
jobs, audit logs, and encrypted secret metadata in production.

```bash
AIT_DATABASE_URL=postgresql+psycopg://aiterate:aiterate@localhost:5432/aiterate
AIT_SECRET_KEY=<fernet-key>
AIT_ENABLE_LOCAL_GIT=false
```

Apply migrations before starting production services:

```bash
aiterate migrate
```

Generate a Fernet key:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Local browser draft persistence is not used. Git artifact writes are disabled by default; use the Git
PR workflow for promotion.

## Status

AIterate is early open-source software. The package is designed for local experimentation first, with
production and enterprise integrations built into the roadmap.
