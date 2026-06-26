# Aiterate

**AI Artifact Lifecycle Management from raw data and policies.**

Aiterate helps teams turn messy source material into production-ready prompts and agent skills.
Give it raw data, policies, examples, and priorities; Aiterate creates, optimizes, validates,
approves, versions, tracks, and promotes the result.

Manual AI artifact changes do not scale well. Teams lose track of why a prompt or skill changed,
which policy or dataset triggered the change, whether old behavior regressed, and which version is
safe to promote. Aiterate turns those edits into a repeatable lifecycle with eval checks,
regression signals, approvals, and traceable versions.

![Aiterate workflow](docs/assets/aiterate-workflow.gif)

## Why Use Aiterate?

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
- Compare approved artifacts across model providers using the same prompt and rubric.
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

Aiterate supports Python **3.11, 3.12, and 3.13**.

## Quickstart

### 5-Minute Demo

Try Aiterate with the built-in sample flow first. You do not need cloud model keys for the first
run.

```bash
git clone https://github.com/AthiraSPillai/aiterate.git
cd aiterate
docker compose up --build
```

Open `http://localhost:5173`, then:

1. Go to **Import context** and click **Load sample project**.
2. Review the separated **Data / Examples**, **Policies**, and **Knowledge Base** context.
3. Go to **Configure models** and keep the local/mock setup for a no-key demo, or save a provider
   credential for OpenAI, Anthropic, Azure OpenAI, AWS Bedrock, or LiteLLM.
4. Go to **Run optimizer** and click **Run optimizer**.
5. Open **Review and approve** to see score progress, accepted versions, rejected attempts, eval
   insights, and the best prompt/skill.
6. Click **Approve best version**.
7. Use **Export** to download a promotion package, or configure **Create Git PR** when your Git token
   is ready.

What you should see in five minutes:

- a generated or improved prompt/skill from raw source material
- visible train/test split and policy weights
- version progress with scores and diffs
- eval insights showing what worked, what failed, and what to change next
- approval metadata and a promotion package with raw data, policy, knowledge, hashes, and lineage

Prefer the CLI from the cloned repo? Install locally and run the same kind of no-key optimization:

```bash
python -m pip install -e .
aiterate optimize \
  --name support-agent \
  --data examples/raw_support_notes.txt \
  --policy examples/policies.yml
```

### Choose A Workflow

Choose the path that fits your audience:

- **UI workflow** for product, policy, and operations users.
- **CLI workflow** for developers and automation.
- **Notebook/Python workflow** for data scientists, AI engineers, and backend-only usage.

All three workflows can create, optimize, version, and trace prompts or agent skills.

## UI Workflow

Start the production-style backend stack with Docker:

```bash
docker compose up --build
```

Open `http://localhost:5173`. The API container serves the built React UI, so the default Docker
flow does not need a separate Node container.

For active UI development with hot reload, run the web app locally instead:

```bash
uvicorn aiterate.api.main:app --reload
```

```bash
cd frontend
npm install
npm run dev
```

Open the local UI, import context, choose models, and run the optimization. Provider and tracking
credentials can be pasted once and saved encrypted server-side. Saved credentials are shown only as
configured status and fingerprints, never as full secret values.

The UI supports:

- separate context lanes for **Data / Examples**, **Policies**, and **Knowledge Base / References**
- multi-file upload for text, CSV, JSON, YAML, YML, and Markdown
- automatic context detection and policy extraction from uploaded files
- optional baseline prompt/skill input for existing production artifacts
- visible train/test split controls so users know what is optimized and what is held out
- run controls for optimization depth, iterations, promotion threshold, spend cap, and repeatable seed
- optional target-model validation on holdout examples before approval
- policy weight editing, equal weighting, and regression-oriented eval criteria
- native eval checks for regression-sensitive behavior, safety, grounding, and output shape
- separate optimizer and target model selection, with separate credentials when providers differ
- provider readiness testing before a run starts
- optional MLflow/LangSmith tracking, including URI/endpoint, project, and token setup
- Git artifact tracking and promotion PR workflow scaffolding
- per-project Git settings for tracking, remote, PR workflow, and base branch
- GitHub and Bitbucket promotion PR publishing when server credentials are configured
- a Run History dashboard with approved-artifact badges, clickable run details, delete confirmation, and project cleanup
- optimization run results with candidates for approval and attempts not used
- visual version progress with score deltas across accepted versions
- clickable rejected-attempt details with the proposed content, score, gate decision, and diff
- native eval report with pass rate, failed checks, and suggested prompt/skill changes
- manual approval flow for the best version before creating a promotion PR
- model comparison for any approved historical artifact, with live eval mode when provider calls are enabled
- promotion packages that include the approved artifact, run JSON, and metadata for data, policy,
  raw source snapshots, knowledge sources, model/provider lineage, eval results, accepted versions,
  rejected attempts, approval, and promotion destination settings
- promotion PRs include human-readable source snapshots plus immutable hash-addressed copies under
  `aiterate/immutable/sources/<kind>/<sha256>/`, with DVC pointer files under `aiterate/dvc/<run_id>/`

Model comparison can run in two modes. The default offline mode estimates prompt/model fit from the
same approved artifact, policy rubric, source data, and selected model profile without provider
cost. Enable live eval to call the selected providers on holdout examples before making a final
production model decision.

Typical use cases:

- a support prompt needs to change after a policy update, but the team wants to catch citation,
  escalation, and tone regressions before promotion
- a skill needs to be generated from messy notes and reviewed as a versioned artifact
- platform teams need to compare prompt versions or model targets with the same eval rubric
- teams want to reopen an approved artifact later and compare it across newer or cheaper models
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

If `--baseline` is omitted, Aiterate creates the initial baseline from raw data and
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

Use Aiterate directly from a notebook, script, or backend job:

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

Aiterate accepts plain text plus structured files. The UI separates uploaded material into three
roles:

- **Data / Examples** become training and validation cases for optimization and regression testing.
- **Policies** become weighted rules, acceptance criteria, and scoring signals.
- **Knowledge Base / References** become grounding context for the generated artifact.

Structured data can use any of these top-level
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

Aiterate supports first-class provider configuration for:

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

Aiterate can record optimization runs, scores, artifacts, and lineage in MLflow. LangSmith support is
available for teams that use it for LLM observability. Tracking is optional in the guided workflow;
users can run locally without it and add tracking later.

```bash
MLFLOW_TRACKING_URI=http://host.docker.internal:5000
MLFLOW_TRACKING_TOKEN=...
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=...
```

For Docker Compose, `http://host.docker.internal:5000` points from the Aiterate API container back
to an MLflow server running on your machine. If MLflow runs as its own Compose or Kubernetes service,
replace it with that service URL.

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
MLFLOW_TRACKING_URI=http://host.docker.internal:5000
LANGSMITH_API_KEY=...
GITHUB_TOKEN=...
GITHUB_APP_ID=...
GITHUB_OAUTH_CLIENT_ID=...
GITHUB_OAUTH_CLIENT_SECRET=...
BITBUCKET_TOKEN=...
BITBUCKET_OAUTH_CLIENT_ID=...
BITBUCKET_OAUTH_CLIENT_SECRET=...
AIT_SECRET_PROVIDER=database
```

For Git PR publishing, the UI can use browser-based GitHub or Bitbucket OAuth when the OAuth client
ID/secret variables are configured. Manual tokens remain available as an encrypted fallback for
self-hosted and restricted enterprise environments.

Promotion PRs write the approved artifact, redacted run metadata, source manifest, raw source
snapshots, immutable content-addressed source copies, and DVC pointer files. Aiterate generates
promotion branches automatically with names like `aiterate/promote-art-...`; users only choose the
PR base branch. The raw snapshots are easy for reviewers to read; the immutable paths and hashes make
it clear which exact data, policies, and knowledge sources produced the approved artifact. Teams with
larger datasets can wire the emitted `.dvc` files to their own DVC remote or use Git LFS for
`aiterate/sources/**` and `aiterate/immutable/sources/**`.

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

Aiterate defaults to local SQLite for a fast single-user quickstart. Use Postgres for run history,
jobs, audit logs, and encrypted secret metadata in production.

Docker Compose local/self-hosted runs set `AIT_AUTO_GENERATE_SECRET_KEY=true` by default. On first
start, Aiterate creates a Fernet key in the persisted `.aiterate` volume and reuses it on later
starts. This avoids first-run setup friction while keeping saved UI credentials encrypted.

```bash
AIT_DATABASE_URL=postgresql+psycopg://aiterate:aiterate@localhost:5432/aiterate
AIT_SECRET_KEY=<fernet-key>
AIT_AUTO_GENERATE_SECRET_KEY=false
AIT_TRUST_ENV_PROXY=false
AIT_ENABLE_LOCAL_GIT=false
```

For production, set `AIT_SECRET_KEY` yourself or use a managed secret provider. Keep the same key for
the lifetime of saved credentials; changing it prevents existing encrypted credentials from being
decrypted.

By default, native provider calls ignore `HTTP_PROXY` and `HTTPS_PROXY` from the host environment so
broken local proxy variables do not cause provider connection failures. Set `AIT_TRUST_ENV_PROXY=true`
when your enterprise network requires those proxy variables.

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

Aiterate is early open-source software. The package is designed for local experimentation first, with
production and enterprise integrations built into the roadmap.
