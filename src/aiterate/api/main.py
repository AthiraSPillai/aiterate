from __future__ import annotations

import shutil
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import SecretStr

from aiterate import __version__
from aiterate.audit import AuditLogger
from aiterate.auth import CurrentUser, Role, require_role
from aiterate.config import settings
from aiterate.domain import ModelComparisonRequest, ModelComparisonResponse, OptimizationRequest, OptimizationRun
from aiterate.jobs import JobEnvelope, JobStore, run_one_optimization_job
from aiterate.optimizer import SkillOptInspiredOptimizer, compare_models
from aiterate.pr_publishers import PullRequestSpec, build_pull_request_publisher
from aiterate.run_store import RunStore
from aiterate.secrets import SecretInput, SecretMetadata, SecretStore

app = FastAPI(
    title="AIterate API",
    version=__version__,
    description="Create, optimize, version, and trace AI prompts and agent skills.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/v1/optimize", response_model=OptimizationRun)
def optimize(
    request: OptimizationRequest,
    user: CurrentUser = Depends(require_role(Role.EDITOR)),
) -> OptimizationRun:
    request = _with_stored_provider_secret(request)
    run = SkillOptInspiredOptimizer().optimize(request)
    RunStore().append(run)
    AuditLogger().log(
        "optimization.run_completed",
        user,
        "run",
        run.id,
        {"artifact_id": run.artifact_id, "name": run.name, "mode": "synchronous"},
    )
    return run


@app.post("/v1/optimization-jobs", response_model=JobEnvelope)
def enqueue_optimization_job(
    request: OptimizationRequest,
    user: CurrentUser = Depends(require_role(Role.EDITOR)),
) -> JobEnvelope:
    request = _with_stored_provider_secret(request)
    job = JobStore().enqueue("optimization", request.model_dump(mode="json"))
    AuditLogger().log(
        "optimization.job_queued",
        user,
        "job",
        job.id,
        {"name": request.name, "provider": request.provider.kind.value},
    )
    return job


@app.post("/v1/jobs/run-next", response_model=JobEnvelope | None)
def run_next_job(user: CurrentUser = Depends(require_role(Role.ADMIN))) -> JobEnvelope | None:
    job = run_one_optimization_job()
    if job:
        AuditLogger().log("job.executed", user, "job", job.id, {"status": job.status.value})
    return job


@app.get("/v1/jobs/{job_id}", response_model=JobEnvelope)
def get_job(job_id: str, user: CurrentUser = Depends(require_role(Role.VIEWER))) -> JobEnvelope:
    try:
        return JobStore().get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


@app.get("/v1/runs")
def runs(user: CurrentUser = Depends(require_role(Role.VIEWER))) -> dict[str, list[dict[str, Any]]]:
    _ = user
    return RunStore().list_grouped()


@app.get("/v1/providers")
def providers() -> dict[str, list[str]]:
    return {
        "native": ["openai", "anthropic", "azure_openai", "aws_bedrock"],
        "compatibility": ["litellm"],
    }


@app.post("/v1/compare-models", response_model=ModelComparisonResponse)
def compare_model_outputs(
    request: ModelComparisonRequest,
    user: CurrentUser = Depends(require_role(Role.EDITOR)),
) -> ModelComparisonResponse:
    if request.execute_live:
        request = request.model_copy(
            update={
                "model_a": _with_stored_provider_secret_for_config(request.model_a),
                "model_b": _with_stored_provider_secret_for_config(request.model_b),
            }
        )
    response = compare_models(request)
    AuditLogger().log(
        "evaluation.model_compare",
        user,
        "prompt",
        response.prompt_hash,
        {"execute_live": request.execute_live, "winner": response.winner},
    )
    return response


@app.get("/v1/integrations/status")
def integration_status() -> dict[str, Any]:
    stored = {secret.name: secret for secret in SecretStore().list()}
    return {
        "providers": {
            "openai": {"configured": bool(settings.openai_api_key or stored.get("OPENAI_API_KEY")), "secret_display": "hidden"},
            "anthropic": {"configured": bool(settings.anthropic_api_key or stored.get("ANTHROPIC_API_KEY")), "secret_display": "hidden"},
            "azure_openai": {
                "configured": bool((settings.azure_openai_api_key or stored.get("AZURE_OPENAI_API_KEY")) and settings.azure_openai_endpoint),
                "secret_display": "hidden",
            },
            "aws_bedrock": {
                "configured": bool(settings.aws_profile or settings.aws_region),
                "secret_display": "hidden",
            },
            "litellm": {"configured": True, "secret_display": "provider-managed"},
        },
        "tracking": {
            "mlflow": {"configured": bool(settings.mlflow_tracking_uri or stored.get("MLFLOW_TRACKING_URI")), "uri_set": bool(settings.mlflow_tracking_uri or stored.get("MLFLOW_TRACKING_URI"))},
            "langsmith": {"configured": bool(settings.langsmith_api_key or stored.get("LANGSMITH_API_KEY")), "secret_display": "hidden"},
        },
        "versioning": {
            "git": {"available": bool(shutil.which("git")), "tracking_supported": True},
            "github_pr": {
                "available": bool(settings.github_token or settings.github_app_id or stored.get("GITHUB_TOKEN")),
                "message": "Set GITHUB_TOKEN or configure a GitHub App server-side to enable live PRs.",
            },
            "bitbucket_pr": {
                "available": bool(settings.bitbucket_token or stored.get("BITBUCKET_TOKEN")),
                "message": "Set BITBUCKET_TOKEN server-side to enable Bitbucket PR publishing.",
            },
        },
        "storage": {
            "postgres_ready": True,
            "encryption": "Use TLS for database connections and an external secrets manager for provider keys in production.",
        },
    }


@app.get("/v1/integrations/setup")
def integration_setup() -> dict[str, Any]:
    stored = {secret.name: secret for secret in SecretStore().list()}
    return {
        "secret_policy": {
            "where": "server",
            "provider": settings.secret_provider,
            "message": "Long-lived provider, Git, MLflow, and LangSmith secrets can be configured through the UI. Values are encrypted locally and never returned to the browser after save.",
        },
        "providers": [
            {"name": "OpenAI", "secret_name": "OPENAI_API_KEY", "env": ["OPENAI_API_KEY"], "configured": bool(settings.openai_api_key or stored.get("OPENAI_API_KEY"))},
            {"name": "Anthropic", "secret_name": "ANTHROPIC_API_KEY", "env": ["ANTHROPIC_API_KEY"], "configured": bool(settings.anthropic_api_key or stored.get("ANTHROPIC_API_KEY"))},
            {
                "name": "Azure OpenAI",
                "secret_name": "AZURE_OPENAI_API_KEY",
                "env": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"],
                "configured": bool((settings.azure_openai_api_key or stored.get("AZURE_OPENAI_API_KEY")) and settings.azure_openai_endpoint),
            },
            {"name": "AWS Bedrock", "secret_name": "AWS_PROFILE", "env": ["AWS_PROFILE", "AWS_REGION"], "configured": bool(settings.aws_profile or settings.aws_region)},
        ],
        "tracking": [
            {"name": "MLflow", "secret_name": "MLFLOW_TRACKING_URI", "env": ["MLFLOW_TRACKING_URI"], "configured": bool(settings.mlflow_tracking_uri or stored.get("MLFLOW_TRACKING_URI"))},
            {"name": "LangSmith", "secret_name": "LANGSMITH_API_KEY", "env": ["LANGSMITH_API_KEY"], "configured": bool(settings.langsmith_api_key or stored.get("LANGSMITH_API_KEY"))},
        ],
        "git": [
            {"name": "GitHub", "secret_name": "GITHUB_TOKEN", "env": ["GITHUB_TOKEN", "GITHUB_APP_ID"], "configured": bool(settings.github_token or settings.github_app_id or stored.get("GITHUB_TOKEN"))},
            {"name": "Bitbucket", "secret_name": "BITBUCKET_TOKEN", "env": ["BITBUCKET_TOKEN"], "configured": bool(settings.bitbucket_token or stored.get("BITBUCKET_TOKEN"))},
        ],
        "stored_secrets": [secret.model_dump() for secret in stored.values()],
    }


@app.get("/v1/secrets", response_model=list[SecretMetadata])
def list_secrets(user: CurrentUser = Depends(require_role(Role.ADMIN))) -> list[SecretMetadata]:
    _ = user
    return SecretStore().list()


@app.post("/v1/secrets", response_model=SecretMetadata)
def upsert_secret(
    secret: SecretInput,
    user: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> SecretMetadata:
    metadata = SecretStore().upsert(secret)
    AuditLogger().log(
        "secret.upserted",
        user,
        "secret",
        secret.name,
        {"integration": secret.integration, "fingerprint": metadata.fingerprint},
    )
    return metadata


@app.delete("/v1/secrets/{name}")
def delete_secret(
    name: str,
    user: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> dict[str, Any]:
    deleted = SecretStore().delete(name)
    if not deleted:
        raise HTTPException(status_code=404, detail="Secret not found.")
    AuditLogger().log("secret.deleted", user, "secret", name, {})
    return {"status": "deleted", "name": name}


@app.post("/v1/git/pull-request")
def create_pull_request(
    payload: dict[str, Any],
    user: CurrentUser = Depends(require_role(Role.EDITOR)),
) -> dict[str, Any]:
    provider = str(payload.get("provider") or "github").lower()
    spec = PullRequestSpec(
        title=payload.get("title") or "Promote AIterate artifact",
        body=payload.get("body") or "Promote approved AIterate artifact version.",
        source_branch=payload.get("source_branch") or payload.get("head") or "aiterate-promotion",
        target_branch=payload.get("target_branch") or payload.get("base") or "main",
        owner=payload.get("owner"),
        repo=payload.get("repo"),
        workspace=payload.get("workspace"),
        repo_slug=payload.get("repo_slug"),
    )
    result = build_pull_request_publisher(provider).publish(spec)
    AuditLogger().log(
        "git.pull_request_requested",
        user,
        provider,
        spec.repo or spec.repo_slug or "unknown",
        {"status": result.get("status"), "title": spec.title},
    )
    return result


@app.post("/v1/runs/{run_id}/approve")
def approve_run(
    run_id: str,
    payload: dict[str, Any],
    user: CurrentUser = Depends(require_role(Role.EDITOR)),
) -> dict[str, Any]:
    AuditLogger().log(
        "run.approved",
        user,
        "run",
        run_id,
        {"artifact_id": payload.get("artifact_id"), "version_id": payload.get("version_id")},
    )
    return {
        "status": "approved",
        "run_id": run_id,
        "artifact_id": payload.get("artifact_id"),
        "version_id": payload.get("version_id"),
        "approved_by": payload.get("approved_by") or "local-user",
        "message": "Best version approved for promotion.",
    }


def _with_stored_provider_secret(request: OptimizationRequest) -> OptimizationRequest:
    if request.provider.api_key is not None:
        return request
    secret_name = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "azure_openai": "AZURE_OPENAI_API_KEY",
    }.get(request.provider.kind.value)
    if not secret_name:
        return request
    value = SecretStore().get_value(secret_name)
    if not value:
        return request
    provider = request.provider.model_copy(update={"api_key": SecretStr(value)})
    return request.model_copy(update={"provider": provider})


def _with_stored_provider_secret_for_config(provider_config):
    if provider_config.api_key is not None:
        return provider_config
    secret_name = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "azure_openai": "AZURE_OPENAI_API_KEY",
    }.get(provider_config.kind.value)
    if not secret_name:
        return provider_config
    value = SecretStore().get_value(secret_name)
    if not value:
        return provider_config
    return provider_config.model_copy(update={"api_key": SecretStr(value)})
