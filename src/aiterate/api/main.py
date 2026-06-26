from __future__ import annotations

import os
import secrets as py_secrets
import shutil
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import SecretStr
from sqlalchemy import select

from aiterate import __version__
from aiterate.audit import AuditLogger
from aiterate.auth import CurrentUser, Role, require_role
from aiterate.config import settings
from aiterate.db import ModelCatalogRecord, session_scope
from aiterate.domain import (
    ModelComparisonRequest,
    ModelComparisonResponse,
    OptimizationRequest,
    OptimizationRun,
    ProjectSettings,
    ProviderConfig,
)
from aiterate.jobs import JobEnvelope, JobStore, run_one_optimization_job
from aiterate.optimizer import SkillOptInspiredOptimizer, compare_models
from aiterate.pricing import list_model_prices
from aiterate.project_settings import ProjectSettingsStore
from aiterate.pr_publishers import build_pull_request_publisher, spec_from_payload
from aiterate.providers import build_provider
from aiterate.providers.base import ProviderError
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

_frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/v1/optimize", response_model=OptimizationRun)
def optimize(
    request: OptimizationRequest,
    user: CurrentUser = Depends(require_role(Role.EDITOR)),
) -> OptimizationRun:
    request = _with_stored_provider_secret(request)
    request = _with_stored_tracking_secret(request)
    provider_name = request.provider.kind.value
    try:
        run = SkillOptInspiredOptimizer().optimize(request)
    except ProviderError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "provider_error",
                "section": "models",
                "provider": provider_name,
                "secret_name": _secret_name_for_provider(provider_name),
                "message": str(exc),
            },
        ) from exc
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
    request = _with_stored_tracking_secret(request)
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


@app.get("/v1/runs/{run_id}", response_model=OptimizationRun)
def get_run(run_id: str, user: CurrentUser = Depends(require_role(Role.VIEWER))) -> OptimizationRun:
    _ = user
    try:
        return RunStore().get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc


@app.delete("/v1/runs/{run_id}")
def delete_run(
    run_id: str,
    user: CurrentUser = Depends(require_role(Role.EDITOR)),
) -> dict[str, Any]:
    return _delete_run(run_id, user)


@app.post("/v1/runs/{run_id}/delete")
def delete_run_fallback(
    run_id: str,
    user: CurrentUser = Depends(require_role(Role.EDITOR)),
) -> dict[str, Any]:
    return _delete_run(run_id, user)


@app.post("/v1/runs/{run_id}")
def delete_run_post_compat(
    run_id: str,
    payload: dict[str, Any] | None = None,
    user: CurrentUser = Depends(require_role(Role.EDITOR)),
) -> dict[str, Any]:
    if (payload or {}).get("action") != "delete":
        raise HTTPException(status_code=405, detail="Method Not Allowed")
    return _delete_run(run_id, user)


def _delete_run(run_id: str, user: CurrentUser) -> dict[str, Any]:
    try:
        deleted = RunStore().delete(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found.") from exc
    AuditLogger().log(
        "run.deleted",
        user,
        "run",
        run_id,
        {"artifact_id": deleted.artifact_id, "name": deleted.name},
    )
    return {"status": "deleted", "id": run_id, "name": deleted.name}


@app.post("/v1/projects/{project_name}/delete")
def delete_project_runs(
    project_name: str,
    user: CurrentUser = Depends(require_role(Role.EDITOR)),
) -> dict[str, Any]:
    try:
        deleted = RunStore().delete_by_name(project_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Project has no saved runs.") from exc
    AuditLogger().log(
        "project.runs_deleted",
        user,
        "project",
        project_name,
        {"run_count": len(deleted), "artifact_ids": [run.artifact_id for run in deleted]},
    )
    return {"status": "deleted", "name": project_name, "deleted_runs": len(deleted)}


@app.get("/v1/projects/{project_name}/settings", response_model=ProjectSettings)
def get_project_settings(
    project_name: str,
    user: CurrentUser = Depends(require_role(Role.VIEWER)),
) -> ProjectSettings:
    _ = user
    return ProjectSettingsStore().get(project_name.strip())


@app.put("/v1/projects/{project_name}/settings", response_model=ProjectSettings)
def save_project_settings(
    project_name: str,
    payload: ProjectSettings,
    user: CurrentUser = Depends(require_role(Role.EDITOR)),
) -> ProjectSettings:
    settings_payload = payload.model_copy(update={"project_name": project_name.strip()})
    saved = ProjectSettingsStore().upsert(settings_payload)
    AuditLogger().log(
        "project.settings_saved",
        user,
        "project",
        saved.project_name,
        {
            "enable_git_tracking": saved.enable_git_tracking,
            "enable_promotion_pr_workflow": saved.enable_promotion_pr_workflow,
            "artifact_branch": saved.artifact_branch,
            "pr_base_branch": saved.pr_base_branch,
            "git_remote_configured": bool(saved.git_remote),
        },
    )
    return saved


@app.get("/v1/providers")
def providers() -> dict[str, list[str]]:
    return {
        "native": ["openai", "anthropic", "azure_openai", "aws_bedrock"],
        "compatibility": ["litellm"],
    }


@app.post("/v1/providers/test")
def test_provider(
    config: ProviderConfig,
    user: CurrentUser = Depends(require_role(Role.EDITOR)),
) -> dict[str, Any]:
    config = _with_stored_provider_secret_for_config(config)
    try:
        output = build_provider(config).generate(
            "You are a provider readiness checker. Reply with a short success confirmation.",
            "Confirm this provider connection is ready for AIterate.",
        )
    except ProviderError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "provider_error",
                "section": "models",
                "provider": config.kind.value,
                "secret_name": _secret_name_for_provider(config.kind.value),
                "message": str(exc),
            },
        ) from exc
    AuditLogger().log(
        "provider.tested",
        user,
        "provider",
        config.kind.value,
        {"model": config.model, "status": "ready"},
    )
    return {
        "status": "ready",
        "provider": config.kind.value,
        "model": config.model,
        "message": output[:240] or "Provider connection is ready.",
    }


@app.get("/v1/model-catalog")
def model_catalog(include_live: bool = False) -> dict[str, list[dict[str, Any]]]:
    with session_scope() as session:
        _seed_model_catalog(session)
        records = session.scalars(
            select(ModelCatalogRecord)
            .where(ModelCatalogRecord.enabled == 1)
            .order_by(ModelCatalogRecord.provider, ModelCatalogRecord.sort_order, ModelCatalogRecord.label)
        ).all()
    catalog: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        catalog.setdefault(record.provider, []).append(
            {
                "id": record.model_id,
                "label": record.label,
                "recommended_for": record.recommended_for,
                "source": "catalog",
            }
        )
    if include_live:
        _merge_live_model_catalog(catalog)
    return catalog


def _seed_model_catalog(session) -> None:
    defaults = {
        "openai": [
            {"id": "gpt-5.5", "label": "GPT-5.5", "recommended_for": ["optimizer", "target"]},
            {"id": "gpt-5.4", "label": "GPT-5.4", "recommended_for": ["optimizer", "target"]},
            {"id": "gpt-5.4-mini", "label": "GPT-5.4 mini", "recommended_for": ["optimizer", "target"]},
            {"id": "gpt-5.4-nano", "label": "GPT-5.4 nano", "recommended_for": ["optimizer", "target"]},
            {"id": "gpt-4.1", "label": "GPT-4.1", "recommended_for": ["optimizer", "target"]},
            {"id": "gpt-4.1-mini", "label": "GPT-4.1 mini", "recommended_for": ["optimizer", "target"]},
            {"id": "gpt-4o", "label": "GPT-4o", "recommended_for": ["optimizer", "target"]},
            {"id": "gpt-4o-mini", "label": "GPT-4o mini", "recommended_for": ["optimizer", "target"]},
        ],
        "anthropic": [
            {"id": "claude-opus-4-1-20250805", "label": "Claude Opus 4.1", "recommended_for": ["optimizer", "target"]},
            {"id": "claude-opus-4-20250514", "label": "Claude Opus 4", "recommended_for": ["optimizer", "target"]},
            {"id": "claude-sonnet-4-20250514", "label": "Claude Sonnet 4", "recommended_for": ["optimizer", "target"]},
            {"id": "claude-3-7-sonnet-latest", "label": "Claude 3.7 Sonnet", "recommended_for": ["optimizer", "target"]},
            {"id": "claude-3-7-sonnet-20250219", "label": "Claude 3.7 Sonnet 20250219", "recommended_for": ["optimizer", "target"]},
            {"id": "claude-3-5-sonnet-latest", "label": "Claude 3.5 Sonnet", "recommended_for": ["optimizer", "target"]},
            {"id": "claude-3-5-sonnet-20241022", "label": "Claude 3.5 Sonnet 20241022", "recommended_for": ["optimizer", "target"]},
            {"id": "claude-3-5-haiku-latest", "label": "Claude 3.5 Haiku", "recommended_for": ["optimizer", "target"]},
            {"id": "claude-3-5-haiku-20241022", "label": "Claude 3.5 Haiku 20241022", "recommended_for": ["optimizer", "target"]},
            {"id": "claude-3-opus-latest", "label": "Claude 3 Opus", "recommended_for": ["optimizer", "target"]},
            {"id": "claude-3-opus-20240229", "label": "Claude 3 Opus 20240229", "recommended_for": ["optimizer", "target"]},
            {"id": "claude-3-haiku-20240307", "label": "Claude 3 Haiku 20240307", "recommended_for": ["optimizer", "target"]},
        ],
        "azure_openai": [
            {"id": "gpt-5.5", "label": "GPT-5.5 deployment", "recommended_for": ["optimizer", "target"]},
            {"id": "gpt-5.4", "label": "GPT-5.4 deployment", "recommended_for": ["optimizer", "target"]},
            {"id": "gpt-5.4-mini", "label": "GPT-5.4 mini deployment", "recommended_for": ["optimizer", "target"]},
            {"id": "gpt-4.1", "label": "GPT-4.1 deployment", "recommended_for": ["optimizer", "target"]},
            {"id": "gpt-4.1-mini", "label": "GPT-4.1 mini deployment", "recommended_for": ["optimizer", "target"]},
            {"id": "gpt-4o", "label": "GPT-4o deployment", "recommended_for": ["optimizer", "target"]},
        ],
        "aws_bedrock": [
            {"id": "anthropic.claude-opus-4-1-20250805-v1:0", "label": "Claude Opus 4.1", "recommended_for": ["optimizer", "target"]},
            {"id": "anthropic.claude-opus-4-20250514-v1:0", "label": "Claude Opus 4", "recommended_for": ["optimizer", "target"]},
            {"id": "anthropic.claude-sonnet-4-20250514-v1:0", "label": "Claude Sonnet 4", "recommended_for": ["optimizer", "target"]},
            {"id": "anthropic.claude-3-7-sonnet-20250219-v1:0", "label": "Claude 3.7 Sonnet", "recommended_for": ["optimizer", "target"]},
            {"id": "anthropic.claude-3-5-sonnet-20240620-v1:0", "label": "Claude 3.5 Sonnet", "recommended_for": ["optimizer", "target"]},
            {"id": "anthropic.claude-3-5-sonnet-20241022-v2:0", "label": "Claude 3.5 Sonnet v2", "recommended_for": ["optimizer", "target"]},
            {"id": "anthropic.claude-3-haiku-20240307-v1:0", "label": "Claude 3 Haiku", "recommended_for": ["optimizer", "target"]},
            {"id": "amazon.nova-pro-v1:0", "label": "Amazon Nova Pro", "recommended_for": ["optimizer", "target"]},
            {"id": "amazon.nova-micro-v1:0", "label": "Amazon Nova Micro", "recommended_for": ["optimizer", "target"]},
            {"id": "amazon.nova-lite-v1:0", "label": "Amazon Nova Lite", "recommended_for": ["optimizer", "target"]},
            {"id": "meta.llama3-1-405b-instruct-v1:0", "label": "Llama 3.1 405B Instruct", "recommended_for": ["optimizer", "target"]},
            {"id": "meta.llama3-1-70b-instruct-v1:0", "label": "Llama 3.1 70B Instruct", "recommended_for": ["optimizer", "target"]},
            {"id": "meta.llama3-1-8b-instruct-v1:0", "label": "Llama 3.1 8B Instruct", "recommended_for": ["optimizer", "target"]},
            {"id": "mistral.mistral-large-2407-v1:0", "label": "Mistral Large", "recommended_for": ["optimizer", "target"]},
            {"id": "mistral.mixtral-8x7b-instruct-v0:1", "label": "Mixtral 8x7B Instruct", "recommended_for": ["optimizer", "target"]},
            {"id": "cohere.command-r-plus-v1:0", "label": "Command R+", "recommended_for": ["optimizer", "target"]},
            {"id": "cohere.command-r-v1:0", "label": "Command R", "recommended_for": ["optimizer", "target"]},
            {"id": "ai21.jamba-1-5-large-v1:0", "label": "Jamba 1.5 Large", "recommended_for": ["optimizer", "target"]},
            {"id": "ai21.jamba-1-5-mini-v1:0", "label": "Jamba 1.5 Mini", "recommended_for": ["optimizer", "target"]},
        ],
        "litellm": [
            {"id": "openai/gpt-5.5", "label": "OpenAI GPT-5.5 via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "openai/gpt-5.4", "label": "OpenAI GPT-5.4 via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "openai/gpt-5.4-mini", "label": "OpenAI GPT-5.4 mini via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "anthropic/claude-opus-4-1-20250805", "label": "Claude Opus 4.1 via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "anthropic/claude-sonnet-4-20250514", "label": "Claude Sonnet 4 via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "anthropic/claude-3-7-sonnet-latest", "label": "Claude 3.7 Sonnet via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "anthropic/claude-3-5-sonnet-latest", "label": "Claude 3.5 Sonnet via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "anthropic/claude-3-5-haiku-latest", "label": "Claude 3.5 Haiku via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "gemini/gemini-1.5-pro", "label": "Gemini 1.5 Pro via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "gemini/gemini-1.5-flash", "label": "Gemini 1.5 Flash via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "gemini/gemini-2.0-flash", "label": "Gemini 2.0 Flash via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "gemini/gemini-2.5-pro", "label": "Gemini 2.5 Pro via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "bedrock/anthropic.claude-opus-4-1-20250805-v1:0", "label": "Bedrock Claude Opus 4.1 via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "bedrock/anthropic.claude-sonnet-4-20250514-v1:0", "label": "Bedrock Claude Sonnet 4 via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "bedrock/amazon.nova-pro-v1:0", "label": "Bedrock Nova Pro via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "mistral/mistral-large-latest", "label": "Mistral Large via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "cohere/command-r-plus", "label": "Command R+ via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "groq/llama-3.1-70b-versatile", "label": "Groq Llama 3.1 70B via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "ollama/llama3.1", "label": "Ollama Llama 3.1 via LiteLLM", "recommended_for": ["optimizer", "target"]},
            {"id": "openai/gpt-4.1", "label": "OpenAI GPT-4.1 via LiteLLM", "recommended_for": ["optimizer", "target"]},
        ],
    }
    for provider, models in defaults.items():
        for index, model in enumerate(models):
            record = ModelCatalogRecord(
                id=f"{provider}:{model['id']}",
                provider=provider,
                model_id=model["id"],
                label=model["label"],
                recommended_for=model["recommended_for"],
                sort_order=index,
            )
            session.merge(record)


def _merge_live_model_catalog(catalog: dict[str, list[dict[str, Any]]]) -> None:
    for provider, models in _discover_live_models().items():
        existing_ids = {model["id"] for model in catalog.setdefault(provider, [])}
        for model in models:
            if model["id"] in existing_ids:
                continue
            catalog[provider].append(model)
            existing_ids.add(model["id"])
        catalog[provider].sort(key=lambda model: (model.get("source") != "catalog", model["id"]))


def _discover_live_models() -> dict[str, list[dict[str, Any]]]:
    store = SecretStore()
    stored = {secret.name: secret for secret in store.list()}
    discovered: dict[str, list[dict[str, Any]]] = {}
    openai_key = settings.openai_api_key or store.get_value("OPENAI_API_KEY")
    if _secret_configured("OPENAI_API_KEY", settings.openai_api_key, store, stored):
        discovered["openai"] = _discover_openai_models(openai_key)
    anthropic_key = settings.anthropic_api_key or store.get_value("ANTHROPIC_API_KEY")
    if _secret_configured("ANTHROPIC_API_KEY", settings.anthropic_api_key, store, stored):
        discovered["anthropic"] = _discover_anthropic_models(anthropic_key)
    if settings.aws_profile or os.environ.get("AWS_ACCESS_KEY_ID"):
        discovered["aws_bedrock"] = _discover_bedrock_models()
    return {provider: models for provider, models in discovered.items() if models}


def _discover_openai_models(api_key: str | None) -> list[dict[str, Any]]:
    if not api_key:
        return []
    try:
        from openai import OpenAI

        response = OpenAI(api_key=api_key, timeout=5.0, max_retries=0).models.list()
        ids = [model.id for model in response.data if getattr(model, "id", None)]
    except Exception:  # noqa: BLE001
        return []
    return [_model_option(model_id, "live") for model_id in sorted(set(ids))]


def _discover_anthropic_models(api_key: str | None) -> list[dict[str, Any]]:
    if not api_key:
        return []
    try:
        from anthropic import Anthropic

        response = Anthropic(api_key=api_key, timeout=5.0, max_retries=0).models.list()
        data = getattr(response, "data", response)
        ids = [model.id for model in data if getattr(model, "id", None)]
    except Exception:  # noqa: BLE001
        return []
    return [_model_option(model_id, "live") for model_id in sorted(set(ids))]


def _discover_bedrock_models() -> list[dict[str, Any]]:
    try:
        import boto3

        session_kwargs = {"profile_name": settings.aws_profile} if settings.aws_profile else {}
        session = boto3.Session(**session_kwargs)
        client = session.client("bedrock", region_name=settings.aws_region)
        response = client.list_foundation_models()
        ids = [
            summary["modelId"]
            for summary in response.get("modelSummaries", [])
            if summary.get("modelId")
        ]
    except Exception:  # noqa: BLE001
        return []
    return [_model_option(model_id, "live") for model_id in sorted(set(ids))]


def _model_option(model_id: str, source: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "label": model_id,
        "recommended_for": ["optimizer", "target"],
        "source": source,
    }


@app.get("/v1/model-prices")
def model_prices() -> dict[str, list[dict[str, Any]]]:
    return list_model_prices()


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
    store = SecretStore()
    stored = {secret.name: secret for secret in store.list()}
    return {
        "providers": {
            "openai": {
                "configured": _secret_configured("OPENAI_API_KEY", settings.openai_api_key, store, stored),
                "secret_display": "hidden",
            },
            "anthropic": {
                "configured": _secret_configured("ANTHROPIC_API_KEY", settings.anthropic_api_key, store, stored),
                "secret_display": "hidden",
            },
            "azure_openai": {
                "configured": bool(
                    _secret_configured("AZURE_OPENAI_API_KEY", settings.azure_openai_api_key, store, stored)
                    and settings.azure_openai_endpoint
                ),
                "secret_display": "hidden",
            },
            "aws_bedrock": {
                "configured": bool(settings.aws_profile or settings.aws_region),
                "secret_display": "hidden",
            },
            "litellm": {"configured": True, "secret_display": "provider-managed"},
        },
        "tracking": {
            "mlflow": {
                "configured": bool(settings.mlflow_tracking_uri),
                "endpoint_configured": bool(settings.mlflow_tracking_uri),
                "credential_configured": _secret_configured(
                    "MLFLOW_TRACKING_TOKEN",
                    settings.mlflow_tracking_token,
                    store,
                    stored,
                ),
                "uri_set": bool(settings.mlflow_tracking_uri),
            },
            "langsmith": {
                "configured": bool(settings.langsmith_endpoint) and _secret_configured(
                    "LANGSMITH_API_KEY",
                    settings.langsmith_api_key,
                    store,
                    stored,
                ),
                "endpoint_configured": bool(settings.langsmith_endpoint),
                "credential_configured": _secret_configured(
                    "LANGSMITH_API_KEY",
                    settings.langsmith_api_key,
                    store,
                    stored,
                ),
                "secret_display": "hidden",
            },
        },
        "versioning": {
            "git": {"available": bool(shutil.which("git")), "tracking_supported": True},
            "github_pr": {
                "available": bool(settings.github_app_id or _secret_configured("GITHUB_TOKEN", settings.github_token, store, stored)),
                "browser_auth": bool(settings.github_oauth_client_id and settings.github_oauth_client_secret),
                "message": (
                    "Browser OAuth is available."
                    if settings.github_oauth_client_id and settings.github_oauth_client_secret
                    else "Save a GitHub token to enable live PRs. Browser OAuth requires backend OAuth app credentials."
                ),
            },
            "bitbucket_pr": {
                "available": _secret_configured("BITBUCKET_TOKEN", settings.bitbucket_token, store, stored),
                "browser_auth": bool(settings.bitbucket_oauth_client_id and settings.bitbucket_oauth_client_secret),
                "message": (
                    "Browser OAuth is available."
                    if settings.bitbucket_oauth_client_id and settings.bitbucket_oauth_client_secret
                    else "Save a Bitbucket token to enable live PRs. Browser OAuth requires backend OAuth app credentials."
                ),
            },
        },
        "storage": {
            "postgres_ready": True,
            "encryption": "Use TLS for database connections and an external secrets manager for provider keys in production.",
        },
    }


@app.get("/v1/integrations/setup")
def integration_setup() -> dict[str, Any]:
    store = SecretStore()
    stored = {secret.name: secret for secret in store.list()}
    return {
        "secret_policy": {
            "where": "server",
            "provider": settings.secret_provider,
            "message": "Long-lived provider, Git, MLflow, and LangSmith secrets can be configured through the UI. Values are encrypted locally and never returned to the browser after save.",
        },
        "providers": [
            {
                "name": "OpenAI",
                "secret_name": "OPENAI_API_KEY",
                "env": ["OPENAI_API_KEY"],
                "configured": _secret_configured("OPENAI_API_KEY", settings.openai_api_key, store, stored),
            },
            {
                "name": "Anthropic",
                "secret_name": "ANTHROPIC_API_KEY",
                "env": ["ANTHROPIC_API_KEY"],
                "configured": _secret_configured("ANTHROPIC_API_KEY", settings.anthropic_api_key, store, stored),
            },
            {
                "name": "Azure OpenAI",
                "secret_name": "AZURE_OPENAI_API_KEY",
                "env": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_VERSION"],
                "configured": bool(
                    _secret_configured("AZURE_OPENAI_API_KEY", settings.azure_openai_api_key, store, stored)
                    and settings.azure_openai_endpoint
                ),
            },
            {"name": "AWS Bedrock", "secret_name": "AWS_PROFILE", "env": ["AWS_PROFILE", "AWS_REGION"], "configured": bool(settings.aws_profile or settings.aws_region)},
        ],
        "tracking": [
            {
                "name": "MLflow access token",
                "secret_name": "MLFLOW_TRACKING_TOKEN",
                "env": ["MLFLOW_TRACKING_TOKEN"],
                "configured": _secret_configured(
                    "MLFLOW_TRACKING_TOKEN",
                    settings.mlflow_tracking_token,
                    store,
                    stored,
                ),
            },
            {
                "name": "LangSmith API key",
                "secret_name": "LANGSMITH_API_KEY",
                "env": ["LANGSMITH_API_KEY"],
                "configured": _secret_configured("LANGSMITH_API_KEY", settings.langsmith_api_key, store, stored),
            },
        ],
        "git": [
            {
                "name": "GitHub",
                "secret_name": "GITHUB_TOKEN",
                "env": ["GITHUB_TOKEN", "GITHUB_APP_ID"],
                "browser_auth": bool(settings.github_oauth_client_id and settings.github_oauth_client_secret),
                "configured": bool(settings.github_app_id or _secret_configured("GITHUB_TOKEN", settings.github_token, store, stored)),
            },
            {
                "name": "Bitbucket",
                "secret_name": "BITBUCKET_TOKEN",
                "env": ["BITBUCKET_TOKEN"],
                "browser_auth": bool(settings.bitbucket_oauth_client_id and settings.bitbucket_oauth_client_secret),
                "configured": _secret_configured("BITBUCKET_TOKEN", settings.bitbucket_token, store, stored),
            },
        ],
        "stored_secrets": [_stored_secret_payload(secret, store) for secret in stored.values()],
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
    _validate_secret_input(secret)
    secret = secret.model_copy(update={"name": secret.name.strip().upper(), "value": secret.value.strip()})
    metadata = SecretStore().upsert(secret)
    AuditLogger().log(
        "secret.upserted",
        user,
        "secret",
        secret.name,
        {"integration": secret.integration, "fingerprint": metadata.fingerprint},
    )
    return metadata


def _validate_secret_input(secret: SecretInput) -> None:
    name = secret.name.strip().upper()
    value = secret.value.strip()
    if name.endswith(("API_KEY", "TOKEN")):
        if _secret_value_is_placeholder(name, value):
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_secret",
                    "section": "credentials",
                    "integration": secret.integration,
                    "secret_name": secret.name,
                    "message": f"{secret.integration} needs the actual secret value. Paste the key value for {secret.name}.",
                },
            )


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
    provider, spec = spec_from_payload(payload)
    result = build_pull_request_publisher(provider).publish(spec)
    AuditLogger().log(
        "git.pull_request_requested",
        user,
        provider,
        spec.repo or spec.repo_slug or "unknown",
        {"status": result.get("status"), "title": spec.title},
    )
    return result


@app.get("/v1/git/auth/{provider}/start")
def start_git_browser_auth(
    provider: str,
    user: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> dict[str, Any]:
    _ = user
    provider = provider.lower()
    config = _git_oauth_config(provider)
    if not config:
        return {
            "status": "not_configured",
            "message": f"{provider.title()} browser auth needs OAuth app credentials in the backend environment.",
        }
    state = _create_oauth_state(provider)
    params = {
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "state": state,
        "scope": config["scope"],
    }
    return {"status": "ready", "auth_url": f"{config['authorize_url']}?{urlencode(params)}"}


@app.get("/v1/git/auth/{provider}/callback", include_in_schema=False)
def complete_git_browser_auth(provider: str, code: str, state: str) -> HTMLResponse:
    provider = provider.lower()
    config = _git_oauth_config(provider)
    if not config or not _consume_oauth_state(provider, state):
        return HTMLResponse(
            _oauth_result_page("Git auth failed", "Invalid or expired browser auth request."),
            status_code=400,
        )
    token = _exchange_git_oauth_code(provider, code, config)
    if not token:
        return HTMLResponse(
            _oauth_result_page("Git auth failed", "Could not exchange auth code for an access token."),
            status_code=400,
        )
    secret_name = "GITHUB_TOKEN" if provider == "github" else "BITBUCKET_TOKEN"
    integration = "GitHub" if provider == "github" else "Bitbucket"
    SecretStore().upsert(SecretInput(name=secret_name, integration=integration, value=token))
    return HTMLResponse(
        _oauth_result_page(
            f"{integration} connected",
            "Credential saved encrypted. You can close this tab and refresh AIterate.",
        )
    )


@app.post("/v1/runs/{run_id}/approve")
def approve_run(
    run_id: str,
    payload: dict[str, Any],
    user: CurrentUser = Depends(require_role(Role.EDITOR)),
) -> dict[str, Any]:
    approval = {
        "status": "approved",
        "run_id": run_id,
        "artifact_id": payload.get("artifact_id"),
        "version_id": payload.get("version_id"),
        "approved_by": payload.get("approved_by") or "local-user",
        "message": "Best version approved for promotion.",
    }
    try:
        RunStore().approve(run_id, approval)
    except KeyError:
        # Keep the endpoint useful for external approval systems that call it
        # before the run has been imported into this Aiterate instance.
        pass
    AuditLogger().log(
        "run.approved",
        user,
        "run",
        run_id,
        {"artifact_id": payload.get("artifact_id"), "version_id": payload.get("version_id")},
    )
    return approval


def _with_stored_provider_secret(request: OptimizationRequest) -> OptimizationRequest:
    provider = _with_stored_provider_secret_for_config(request.provider)
    target_provider = (
        _with_stored_provider_secret_for_config(request.target_provider)
        if request.target_provider is not None
        else None
    )
    return request.model_copy(update={"provider": provider, "target_provider": target_provider})


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


def _with_stored_tracking_secret(request: OptimizationRequest) -> OptimizationRequest:
    if request.tracker_api_key is not None:
        return request
    secret_name = {
        "mlflow": "MLFLOW_TRACKING_TOKEN",
        "langsmith": "LANGSMITH_API_KEY",
    }.get(request.tracker.value)
    if not secret_name:
        return request
    value = SecretStore().get_value(secret_name)
    if not value:
        return request
    return request.model_copy(update={"tracker_api_key": SecretStr(value)})


def _secret_name_for_provider(provider: str) -> str | None:
    return {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "azure_openai": "AZURE_OPENAI_API_KEY",
        "aws_bedrock": "AWS_PROFILE",
    }.get(provider)


def _git_oauth_config(provider: str) -> dict[str, str] | None:
    redirect_uri = f"{settings.public_base_url.rstrip('/')}/v1/git/auth/{provider}/callback"
    if provider == "github" and settings.github_oauth_client_id and settings.github_oauth_client_secret:
        return {
            "client_id": settings.github_oauth_client_id,
            "client_secret": settings.github_oauth_client_secret,
            "authorize_url": "https://github.com/login/oauth/authorize",
            "token_url": "https://github.com/login/oauth/access_token",
            "scope": "repo",
            "redirect_uri": redirect_uri,
        }
    if provider == "bitbucket" and settings.bitbucket_oauth_client_id and settings.bitbucket_oauth_client_secret:
        return {
            "client_id": settings.bitbucket_oauth_client_id,
            "client_secret": settings.bitbucket_oauth_client_secret,
            "authorize_url": "https://bitbucket.org/site/oauth2/authorize",
            "token_url": "https://bitbucket.org/site/oauth2/access_token",
            "scope": "repository pullrequest",
            "redirect_uri": redirect_uri,
        }
    return None


def _oauth_state_dir() -> Path:
    path = settings.storage_dir / "oauth-states"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _create_oauth_state(provider: str) -> str:
    token = py_secrets.token_urlsafe(32)
    (_oauth_state_dir() / token).write_text(f"{provider}:{int(time.time())}", encoding="utf-8")
    return token


def _consume_oauth_state(provider: str, state: str) -> bool:
    path = _oauth_state_dir() / state
    if not path.exists():
        return False
    raw = path.read_text(encoding="utf-8")
    path.unlink(missing_ok=True)
    stored_provider, _, created_at = raw.partition(":")
    if stored_provider != provider:
        return False
    return int(time.time()) - int(created_at or "0") <= 600


def _exchange_git_oauth_code(provider: str, code: str, config: dict[str, str]) -> str | None:
    try:
        if provider == "bitbucket":
            response = requests.post(
                config["token_url"],
                auth=(config["client_id"], config["client_secret"]),
                data={"grant_type": "authorization_code", "code": code, "redirect_uri": config["redirect_uri"]},
                headers={"Accept": "application/json"},
                timeout=20,
            )
        else:
            response = requests.post(
                config["token_url"],
                json={
                    "client_id": config["client_id"],
                    "client_secret": config["client_secret"],
                    "code": code,
                    "redirect_uri": config["redirect_uri"],
                },
                headers={"Accept": "application/json"},
                timeout=20,
            )
        payload = response.json()
    except Exception:  # noqa: BLE001
        return None
    if not response.ok:
        return None
    return payload.get("access_token")


def _oauth_result_page(title: str, message: str) -> str:
    return f"""<!doctype html>
<html lang="en">
  <head><meta charset="utf-8"><title>{title}</title></head>
  <body style="font-family: system-ui, sans-serif; padding: 32px;">
    <h1>{title}</h1>
    <p>{message}</p>
  </body>
</html>"""


def _secret_configured(
    name: str,
    env_value: str | None,
    store: SecretStore,
    stored: dict[str, SecretMetadata],
) -> bool:
    if _secret_value_usable(name, env_value):
        return True
    if name not in stored:
        return False
    return _secret_value_usable(name, store.get_value(name))


def _stored_secret_payload(secret: SecretMetadata, store: SecretStore) -> dict[str, Any]:
    payload = secret.model_dump()
    payload["valid"] = _secret_value_usable(secret.name, store.get_value(secret.name))
    return payload


def _secret_value_usable(name: str, value: str | None) -> bool:
    if not value:
        return False
    cleaned = value.strip()
    if not cleaned:
        return False
    if any(character.isspace() for character in cleaned):
        return False
    if not name.endswith(("API_KEY", "TOKEN")):
        return True
    return not _secret_value_is_placeholder(name, cleaned)


def _secret_value_is_placeholder(name: str, value: str | None) -> bool:
    if not value:
        return True
    cleaned = value.strip()
    if not cleaned:
        return True
    upper_value = cleaned.upper()
    placeholders = {
        name,
        "YOUR_API_KEY",
        "YOUR_TOKEN",
        "PASTE_KEY_HERE",
        "PASTE_TOKEN_HERE",
    }
    return upper_value in placeholders or upper_value.endswith("_API_KEY")


if _frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=_frontend_dist / "assets"), name="frontend-assets")

    @app.get("/")
    def frontend_index() -> FileResponse:
        return FileResponse(_frontend_dist / "index.html")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend_fallback(path: str) -> FileResponse:
        if path.startswith("v1/"):
            raise HTTPException(status_code=404, detail="API route not found.")
        return FileResponse(_frontend_dist / "index.html")
